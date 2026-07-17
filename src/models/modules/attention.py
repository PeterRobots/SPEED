import torch
import torch.nn.functional as F
from torch import nn

try:
    from xformers.ops import memory_efficient_attention as _xformers_attention
except (ImportError, OSError):
    _xformers_attention = None


def memory_efficient_attention(q, k, v, scale):
    if _xformers_attention is not None and q.is_cuda:
        try:
            return _xformers_attention(q, k, v, scale=scale)
        except NotImplementedError:
            pass

    # PyTorch SDPA uses [B, H, N, D], while xFormers uses [B, N, H, D].
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=scale,
    )
    return output.transpose(1, 2)


def apply_rope(x, freqs_cis):
    if freqs_cis is None:
        return x
    b, n, h, d = x.shape
    if freqs_cis.ndim == 2:  # [N, D]
        freqs_cis = freqs_cis.unsqueeze(0).expand(b, -1, -1)  # [B, N, D]
    elif freqs_cis.ndim == 3 and freqs_cis.shape[0] != b:
        raise ValueError(f"freqs_cis batch dim {freqs_cis.shape[0]} != input batch {b}")
    x = x.permute(0, 2, 1, 3)
    x_rotated = torch.view_as_complex(x.float().reshape(b, h, n, d // 2, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    x_out = x_out.permute(0, 2, 1, 3)
    return x_out.type_as(x)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Attention(nn.Module):
    def __init__(self, dim=768, head_dim=64):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = (dim // self.num_heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.q_norm = LlamaRMSNorm(head_dim)
        self.k_norm = LlamaRMSNorm(head_dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, y, rotary_pos_emb_x, rotary_pos_emb_y):
        b, n, d = x.shape
        q = apply_rope(self.q_norm(self.to_q(x).reshape(b, n, self.num_heads, d // self.num_heads)), rotary_pos_emb_x)
        x = torch.cat((x, y), dim=1)
        rotary_pos_emb_x = torch.cat((rotary_pos_emb_x, rotary_pos_emb_y), dim=0)
        k = apply_rope(self.k_norm(self.to_k(x).reshape(b, x.shape[1], self.num_heads, d // self.num_heads)), rotary_pos_emb_x)
        v = self.to_v(x).reshape(b, x.shape[1], self.num_heads, d // self.num_heads)
        o = memory_efficient_attention(q, k, v, scale=self.scale)
        x = self.to_out(o.reshape(b, n, d))
        return x

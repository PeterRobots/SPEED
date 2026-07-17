import torch
from torch import nn
from .attention import Attention


def modulate(x, shift, scale):
    if shift is not None and scale is not None:
        return x * (1 + scale) + shift
    return x


def gate(x, g):
    if g is not None:
        return x * g
    return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x = self.fc2(x1 * torch.sigmoid(1.702 * x1))
        return x


class SpeedBlock(nn.Module):
    def __init__(self, dim=768, head_dim=64, adaLN_embed_dim=128):
        super().__init__()
        self.self_attn = Attention(dim, head_dim)
        self.mlp = SwiGLUFFN(dim)
        self.norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(adaLN_embed_dim, 6 * dim),
        )

    def forward(self, query_tokens, cond_tokens, c, rotary_pos_emb_x=None, rotary_pos_emb_y=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        query_tokens = query_tokens + gate(self.self_attn(modulate(self.norm_self_attn(query_tokens), shift_msa, scale_msa),
                                                          y=cond_tokens, rotary_pos_emb_x=rotary_pos_emb_x, rotary_pos_emb_y=rotary_pos_emb_y), gate_msa)
        query_tokens = query_tokens + gate(self.mlp(modulate(self.norm_mlp(query_tokens), shift_mlp, scale_mlp)), gate_mlp)
        return query_tokens


class FinalLayer(nn.Module):
    def __init__(self, dim, final_dim, adaLN_embed_dim=128):
        super().__init__()
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(dim, final_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaLN_embed_dim, 2 * dim, bias=True)
        )

    def forward(self, tokens, c):
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        tokens = modulate(self.norm_out(tokens), shift, scale)
        return self.proj_out(tokens)

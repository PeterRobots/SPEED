import math
import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, adaLN_embed_dim=128, frequency_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, adaLN_embed_dim, bias=True),
        )
        self.frequency_dim_half = frequency_dim // 2

    def forward(self, m, max_period=10000):
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=self.frequency_dim_half, dtype=torch.float32) / self.frequency_dim_half
        ).to(device=m.device)
        args = m[:, None].float() * freqs[None]
        m_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        m_emb = self.mlp(m_freq)
        return m_emb


class RopeEmbedder(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(1024)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
            Args:
                index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, fhw, device="cuda"):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        frame, height, width = fhw
        rope_key = (frame, height, width, str(device))
        if rope_key not in self.rope_cache:
            vid_freqs = []
            freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            if self.scale_rope:
                freqs_height = torch.cat(
                    [
                        freqs_neg[1][-(height - height // 2):],
                        freqs_pos[1][:height // 2]
                    ],
                    dim=0
                )
                freqs_height = freqs_height.view(1, height, 1, -1).expand(1, height, width, -1)
                freqs_width = torch.cat(
                    [
                        freqs_neg[2][-(width - width // 2):],
                        freqs_pos[2][:width // 2]
                    ],
                    dim=0
                )
                freqs_width = freqs_width.view(1, 1, width, -1).expand(1, height, width, -1)
            else:
                freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(1, height, width, -1)
                freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(1, height, width, -1)
            seq_lens = height * width
            for f in range(frame):
                freqs_frame = freqs_pos[0][f:f+1].view(1, 1, 1, -1).expand(1, height, width, -1)
                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                vid_freqs.append(freqs)

            vid_freqs = torch.cat(vid_freqs, dim=0)
            self.rope_cache[rope_key] = vid_freqs.clone().contiguous()

        vid_freqs = self.rope_cache[rope_key]
        return vid_freqs

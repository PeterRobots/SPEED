import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .modules import RopeEmbedder, TimestepEmbedder
from .modules.block import SpeedBlock, FinalLayer


def pad_to_divisor(x, divisor):
    b, c, h, w = x.shape
    pad_h = (-h) % divisor
    pad_w = (-w) % divisor
    if pad_h > 0 or pad_w > 0:
        pad_mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return x, (h, w)


def unpad_to_orig(x, orig_size):
    h, w = orig_size
    return x[..., :h, :w]


class SpeedStage(nn.Module):
    def __init__(self, in_dim, in_dim_cond, out_dim, hidden_dim, head_dim, depth,
                 adaLN_embed_dim=128):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.in_proj_cond = nn.Linear(in_dim_cond, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.blocks = nn.ModuleList([
            SpeedBlock(hidden_dim, head_dim, adaLN_embed_dim)
            for _ in range(depth)
        ])

    def forward(self, x, cond, t, pos, cond_pos):
        x_tok = self.in_proj(x)
        cond_tok = self.in_proj_cond(cond)

        for blk in self.blocks:
            x_tok = blk(x_tok, cond_tok, c=t, rotary_pos_emb_x=pos, rotary_pos_emb_y=cond_pos)

        x_tok = self.out_proj(x_tok)
        return x_tok


class SpeedDiT(nn.Module):
    def __init__(self, hidden_dim=768, head_dim=64, depths=(2, 6, 4), patch_sizes=(64, 32, 16)):
        super().__init__()
        if len(depths) != 3 or len(patch_sizes) != 3:
            raise ValueError("SPEED expects exactly three depths and three patch sizes.")
        if any(patch_sizes[i] % patch_sizes[i + 1] != 0 for i in range(len(patch_sizes) - 1)):
            raise ValueError("Each patch size must be divisible by the next smaller patch size.")

        self.patch_sizes = patch_sizes
        self.scales = [patch_sizes[i] // patch_sizes[i+1] for i in range(len(patch_sizes) - 1)]

        # Stages
        stage_1_in_dim = 3 * patch_sizes[0] * patch_sizes[0]
        self.stage1 = SpeedStage(stage_1_in_dim, stage_1_in_dim, hidden_dim, hidden_dim, head_dim, depths[0])
        self.stage2 = SpeedStage(hidden_dim // (self.scales[0] * self.scales[0]), 3 * patch_sizes[1] * patch_sizes[1], hidden_dim, hidden_dim, head_dim, depths[1])
        self.stage3 = SpeedStage(hidden_dim // (self.scales[1] * self.scales[1]), 3 * patch_sizes[2] * patch_sizes[2], hidden_dim, hidden_dim, head_dim, depths[2])

        self.final_layer = FinalLayer(hidden_dim, 3 * patch_sizes[2] * patch_sizes[2])
        self.timestep_embedder = TimestepEmbedder(hidden_dim)
        self.rope_embedder = RopeEmbedder(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)

        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation
        for block in self.stage1.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.stage2.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.stage3.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.proj_out.weight, 0)
        nn.init.constant_(self.final_layer.proj_out.bias, 0)

    def _get_pos(self, h, w, device):
        pos = self.rope_embedder([3, h, w], device=device)
        pos_0, cur_pos, pos_1 = pos.chunk(3, dim=0)
        cond_pos = torch.cat([pos_0, pos_1], dim=0)
        return cur_pos, cond_pos

    def forward(self, noisy_frames, cond_frames, timestep):
        t = self.timestep_embedder(timestep)
        ps = self.patch_sizes
        s = self.scales

        noisy_frames, orig_size = pad_to_divisor(noisy_frames, ps[0])
        cond_frames, _ = pad_to_divisor(cond_frames, ps[0])

        h, w = noisy_frames.shape[-2:]

        x_1 = rearrange(noisy_frames, "b c (ph p1) (pw p2) -> b (ph pw) (c p1 p2)", p1=ps[0], p2=ps[0])
        c_1 = rearrange(cond_frames, "(p0 b) c (ph p1) (pw p2) -> b (p0 ph pw) (c p1 p2)", p0=2, p1=ps[0], p2=ps[0])
        pos, c_pos = self._get_pos(h // ps[0], w // ps[0], x_1.device)
        x_stage1 = self.stage1(x_1, c_1, t, pos, c_pos)

        x_2= rearrange(x_stage1, "b (ph pw) (c p1 p2) -> b (ph p1 pw p2) c", ph=h // ps[0], pw=w // ps[0], p1=s[0], p2=s[0])
        c_2 = rearrange(cond_frames, "(p0 b) c (ph p1) (pw p2) -> b (p0 ph pw) (c p1 p2)", p0=2, p1=ps[1], p2=ps[1])
        pos, c_pos = self._get_pos(h // ps[1], w // ps[1], x_2.device)
        x_stage2 = self.stage2(x_2, c_2, t, pos, c_pos)

        x_3 = rearrange(x_stage2, "b (ph pw) (c p1 p2) -> b (ph p1 pw p2) c", ph=h // ps[1], pw=w // ps[1], p1=s[1], p2=s[1])
        c_3 = rearrange(cond_frames, "(p0 b) c (ph p1) (pw p2) -> b (p0 ph pw) (c p1 p2)", p0=2, p1=ps[2], p2=ps[2])
        pos, c_pos = self._get_pos(h // ps[2], w // ps[2], x_3.device)
        x_stage3 = self.stage3(x_3, c_3, t, pos, c_pos)

        out = rearrange(self.final_layer(x_stage3, t), "b (ph pw) (c p1 p2) -> b c (ph p1) (pw p2)", ph=h // ps[2], pw=w // ps[2], p1=ps[2], p2=ps[2])

        out = unpad_to_orig(out, orig_size)
        return out

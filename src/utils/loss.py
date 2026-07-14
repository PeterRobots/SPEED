from lpips import LPIPS
from DISTS_pytorch import DISTS
from collections import defaultdict

import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


def freeze_module(module):
    module.eval()
    module.requires_grad_(False)
    return module


class LossMeter:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, name, value):
        self.sum[name] += value
        self.count[name] += 1

    def average(self):
        return {k: self.sum[k] / self.count[k] for k in self.sum}

    def reset(self):
        self.sum.clear()
        self.count.clear()


class LossFunction(nn.Module):
    def __init__(self,
                 lpips_loss_weight=1.0, dists_loss_weight=1.0, style_loss_weight=20.0,
                 lpips_loss_start=0, dists_loss_start=0, style_loss_start=0):
        super().__init__()
        self.lpips_loss_weight = lpips_loss_weight
        self.dists_loss_weight = dists_loss_weight
        self.style_loss_weight = style_loss_weight
        self.lpips_loss_start = lpips_loss_start
        self.dists_loss_start = dists_loss_start
        self.style_loss_start = style_loss_start

        if lpips_loss_weight > 0:
            self.lpips = freeze_module(LPIPS())
        if dists_loss_weight > 0:
            self.dists = freeze_module(DISTS())
        if style_loss_weight > 0:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            self.vgg_feats = nn.ModuleList([
                nn.Sequential(*vgg.features[:4]),
                nn.Sequential(*vgg.features[4:9]),
                nn.Sequential(*vgg.features[9:14]),
                nn.Sequential(*vgg.features[14:23]),
                nn.Sequential(*vgg.features[23:32]),
            ])
            freeze_module(self.vgg_feats)
            self.register_buffer("vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.alpha_l = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]

        self.target_size = 256

    def preprocess_for_perceptual(self, x):
        h, w = x.shape[-2:]
        if h < w:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))
        else:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))

        if h != new_h or w != new_w:
            x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return x

    def get_vgg_features(self, x):
        # [-1,1] -> [0,1] -> normalize
        x = (x + 1) / 2
        x = (x - self.vgg_mean) / self.vgg_std
        feats = []
        out = x
        for layer in self.vgg_feats:
            out = layer(out)
            feats.append(out)
        return feats

    def gram(self, feats):
        # feats: list of [B,C,H,W]
        if not isinstance(feats, list):
            feats = [feats]
        grams = []
        for f in feats:
            f_norm = f / 255.
            g = torch.einsum('b c h w, b d h w -> b c d', f_norm, f_norm)
            grams.append(g)
        return grams

    def forward(self, pred, gt, train_step, fm_loss):
        loss = fm_loss
        log_data = {}
        if self.lpips_loss_weight > 0 and train_step >= self.lpips_loss_start:
            lpips_loss = self.lpips(pred, gt).mean() * self.lpips_loss_weight
            log_data["loss_lpips"] = lpips_loss
            loss = loss + lpips_loss
        if self.dists_loss_weight > 0 and train_step >= self.dists_loss_start:
            dists_loss = self.dists((pred + 1.0) / 2.0, (gt + 1.0) / 2.0, require_grad=True, batch_average=True) * self.dists_loss_weight
            log_data["loss_dists"] = dists_loss
            loss = loss + dists_loss
        if self.style_loss_weight > 0 and train_step >= self.style_loss_start:
            pred = self.preprocess_for_perceptual(pred)
            gt = self.preprocess_for_perceptual(gt)
            pred_feats = self.get_vgg_features(pred)
            gt_feats = self.get_vgg_features(gt)
            pred_grams = self.gram(pred_feats)
            gt_grams = self.gram(gt_feats)
            style_loss = sum(((p-g)**2).mean()*a for p,g,a in zip(pred_grams, gt_grams, self.alpha_l))
            style_loss = style_loss * self.style_loss_weight
            log_data["loss_style"] = style_loss
            loss = loss + style_loss

        return {
            "loss_total": loss,
            "log_data": log_data
        }

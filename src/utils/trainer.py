import random

import torch
import torch.nn.functional as F

from .scheduler import FlowMatchScheduler


class Trainer:
    def __init__(self, loss_function, **kwargs):
        self.loss_function = loss_function
        self.visual_results_num = kwargs.get("visual_results_num", 4)
        self.base_resolution = kwargs.get("base_resolution", 128)
        self.small_method = kwargs.get("small_method", "crop")
        self.device = kwargs.get("device", "cuda:0")

        self.noise_scheduler = FlowMatchScheduler()
        self.noise_scheduler.set_timesteps(num_inference_steps=1000, shift=10, training=True)

        self.inv_127_5 = 1.0 / 127.5

    def drift_aware_timestep_sampling(self, p):
        p = min(p, 0.9999)
        t = torch.rand(1, device=self.device)
        s_p = 1.0 / (1.0 - p)
        t_prime = (s_p * t) / (1.0 + (s_p - 1.0) * t)
        return t_prime

    def random_small(self, frames, method=None):
        b, f, c, h, w = frames.shape
        small_h = random.randint(self.base_resolution, h)
        small_w = random.randint(self.base_resolution, w)
        method = method or self.small_method
        if method == "crop":
            h_start = random.randint(0, h - small_h)
            w_start = random.randint(0, w - small_w)
            return frames[:, :, :, h_start: h_start + small_h, w_start: w_start + small_w]
        elif method == "resize":
            return F.interpolate(frames.reshape(-1, c, h, w), size=(small_h, small_w), mode='bilinear', align_corners=False).reshape(b, f, c, small_h, small_w)
        else:
            raise ValueError(f"Invalid method: {method}. Must be 'crop' or 'resize'")

    def patch_split(self, frames):
        b, t, c, h, w = frames.shape
        if h > 1280 or w > 1280:
            mid_h = h // 2
            mid_w = w // 2
            top_left = frames[..., :mid_h, :mid_w]
            top_right = frames[..., :mid_h, mid_w:]
            bottom_left = frames[..., mid_h:, :mid_w]
            bottom_right = frames[..., mid_h:, mid_w:]
            frames = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=1)
            frames = frames.flatten(0, 1)
        return frames

    def patch_merge(self, frames):
        current_b, c, h, w = frames.shape
        if current_b % 4 != 0:
            raise ValueError(f"Batch size {current_b} is not divisible by 4, cannot merge patches.")

        target_b = current_b // 4
        frames = frames.view(target_b, 4, c, h, w)
        top_row = torch.cat([frames[:, 0], frames[:, 1]], dim=-1)
        bottom_row = torch.cat([frames[:, 2], frames[:, 3]], dim=-1)
        frames = torch.cat([top_row, bottom_row], dim=-2)
        return frames

    def preprocess_frames(self, frames, is_val=False):
        if not is_val:
            frames = self.random_small(frames)
        timestep_id = torch.tensor([0], dtype=torch.long)
        frames = frames * self.inv_127_5 - 1.
        frame_data = {
            "frame_0": frames[:, 0],
            "frame_1": frames[:, 1],
            "frame_t": frames[:, 2],
        }
        patched_frames = frames
        frame_0, frame_1, frame_t = patched_frames[:, 0], patched_frames[:, 1], patched_frames[:, 2]
        cond_frames = torch.cat((frame_0, frame_1), dim=0)

        noise = torch.randn_like(frame_t)
        timestep = self.noise_scheduler.timesteps[timestep_id].to(frame_t)
        noisy_frames = noise

        frame_data.update({
            "target": self.noise_scheduler.training_target(frame_t, noise, is_x=True),
            "noise": noise,
        })
        training_inputs = {
            "noisy_frames": noisy_frames,
            "cond_frames": cond_frames,
            "timestep": timestep,
        }
        return training_inputs, frame_data

    def train_one_step(self, model, frames, train_step):
        training_inputs, frame_data = self.preprocess_frames(frames)
        pred = model(**training_inputs)
        fm_loss_pix = torch.abs(pred - frame_data["frame_t"]).mean()

        frame_data["pred_frame_t"] = pred
        loss_data = self.loss_function(frame_data["pred_frame_t"], frame_data["frame_t"], train_step, fm_loss_pix)
        loss_data["log_data"]["loss_pix"] = fm_loss_pix
        return loss_data, frame_data

    @torch.no_grad()
    def validate_one_step(self, model, frames):
        val_inputs, frame_data = self.preprocess_frames(frames, is_val=True)
        with torch.no_grad():
            pred = model(**val_inputs)
        return pred, frame_data["frame_t"]

    def visualize_frame_data(self, frame_data):
        nrow = 3
        visual_frame_0, visual_frame_1 = frame_data["frame_0"][:self.visual_results_num], frame_data["frame_1"][:self.visual_results_num]
        blended_input = visual_frame_0 * 0.5 + visual_frame_1 * 0.5
        visual_frame_t = frame_data["frame_t"][:self.visual_results_num]
        visual_pred_frame_t = frame_data["pred_frame_t"][:self.visual_results_num]
        visualization = (torch.stack((blended_input, visual_frame_t, visual_pred_frame_t), dim=1).cpu() / 2. + 0.5).flatten(0, 1)
        return [visualization], nrow


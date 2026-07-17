import random

import torch
import torch.nn.functional as F

from .scheduler import FlowMatchScheduler


class Trainer:
    DEFAULT_SHIFT_SCHEDULE = (
        (0, 1.0),
        (5000, 3.0),
        (10000, 5.0),
        (20000, 7.0),
        (50000, 10.0),
    )

    def __init__(self, loss_function, **kwargs):
        self.loss_function = loss_function
        self.visual_results_num = kwargs.get("visual_results_num", 4)
        self.base_resolution = kwargs.get("base_resolution", 128)
        self.small_method = kwargs.get("small_method", "crop")
        self.eval_patch_size = self._positive_int(kwargs.get("eval_patch_size", 1280), "eval_patch_size")
        self.device = kwargs.get("device", "cuda:0")

        self.num_train_timesteps = self._positive_int(kwargs.get("dts_num_train_timesteps", 1000), "dts_num_train_timesteps")
        self.dts_mode = str(kwargs.get("dts_mode", "step_shift")).lower()
        if self.dts_mode not in {"smooth_shift", "step_shift"}:
            raise ValueError("dts_mode must be 'smooth_shift' (smooth DTS) or 'step_shift' (segmented DTS)")

        self.dts_shift_schedule = self._parse_shift_schedule(kwargs.get("dts_shift_schedule"))
        self.dts_pure_noise_step = self._optional_int(kwargs.get("dts_pure_noise_step", 100000))
        self.dts_progress_steps = self._positive_int(
            kwargs.get("dts_progress_steps", self.dts_pure_noise_step or 100000),
            "dts_progress_steps",
        )
        self.dts_max_p = float(kwargs.get("dts_max_p", 0.9999))
        if not 0.0 <= self.dts_max_p < 1.0:
            raise ValueError("dts_max_p must be in [0, 1)")
        self.dts_smooth_shift = float(kwargs.get("dts_smooth_shift", 1.0))

        self.noise_scheduler = FlowMatchScheduler(num_train_timesteps=self.num_train_timesteps)
        self._noise_scheduler_shift = None
        initial_shift = self.dts_smooth_shift if self.dts_mode == "smooth_shift" else self.dts_shift_schedule[0][1]
        self._set_noise_scheduler_shift(initial_shift)

        self.inv_127_5 = 1.0 / 127.5

    @staticmethod
    def _positive_int(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _optional_int(value):
        return None if value is None else int(value)

    def _parse_shift_schedule(self, shift_schedule):
        if shift_schedule is None:
            shift_schedule = self.DEFAULT_SHIFT_SCHEDULE

        parsed = []
        for item in shift_schedule:
            start_step, shift = (item["step"], item["shift"]) if isinstance(item, dict) else item
            start_step, shift = int(start_step), float(shift)
            if start_step < 0:
                raise ValueError("dts_shift_schedule steps must be non-negative")
            if shift <= 0:
                raise ValueError("dts_shift_schedule shifts must be positive")
            parsed.append((start_step, shift))

        if not parsed:
            raise ValueError("dts_shift_schedule must not be empty")

        parsed.sort(key=lambda x: x[0])
        if parsed[0][0] > 0:
            parsed.insert(0, (0, parsed[0][1]))
        return parsed

    def _set_noise_scheduler_shift(self, shift):
        shift = float(shift)
        if self._noise_scheduler_shift != shift:
            self.noise_scheduler.set_timesteps(
                num_inference_steps=self.num_train_timesteps,
                shift=shift,
                training=True,
            )
            self._noise_scheduler_shift = shift

    def _shift_for_step(self, train_step):
        return next(
            shift for start_step, shift in reversed(self.dts_shift_schedule)
            if train_step >= start_step
        )

    def _dts_progress(self, train_step):
        progress = float(train_step) / float(self.dts_progress_steps)
        return min(max(progress, 0.0), self.dts_max_p)

    def drift_aware_timestep_sampling(self, p, device=None):
        p = min(max(float(p), 0.0), self.dts_max_p)
        t = torch.rand(1, device=device or self.device)
        s_p = 1.0 / (1.0 - p)
        return (s_p * t) / (1.0 + (s_p - 1.0) * t)

    def _sigma_from_timestep(self, timestep, reference):
        timestep = timestep.detach().cpu()
        timestep_id = torch.argmin((self.noise_scheduler.timesteps - timestep).abs())
        return self.noise_scheduler.sigmas[timestep_id].to(device=reference.device, dtype=reference.dtype)

    def _add_noise(self, frame_t, noise, timestep):
        sigma = self._sigma_from_timestep(timestep, frame_t)
        return (1.0 - sigma) * frame_t + sigma * noise

    def _pure_noise_training_inputs(self, frame_t, noise):
        return torch.full(
            (1,),
            float(self.num_train_timesteps),
            device=frame_t.device,
            dtype=frame_t.dtype,
        ), noise

    def _sample_smooth_shift_dts_inputs(self, frame_t, noise, train_step):
        self._set_noise_scheduler_shift(self.dts_smooth_shift)
        progress = self._dts_progress(train_step)
        timestep = self.drift_aware_timestep_sampling(progress, frame_t.device) * self.num_train_timesteps
        timestep = timestep.to(dtype=frame_t.dtype)
        return timestep, self._add_noise(frame_t, noise, timestep)

    def _sample_step_shift_dts_inputs(self, frame_t, noise, train_step):
        if self.dts_pure_noise_step is not None and train_step >= self.dts_pure_noise_step:
            return self._pure_noise_training_inputs(frame_t, noise)

        shift = self._shift_for_step(train_step)
        self._set_noise_scheduler_shift(shift)
        timestep_id = torch.randint(0, len(self.noise_scheduler.timesteps), (1,))
        timestep = self.noise_scheduler.timesteps[timestep_id].to(frame_t)
        return timestep, self._add_noise(frame_t, noise, timestep)

    def sample_training_inputs(self, frame_t, noise, train_step, is_val=False):
        if is_val:
            return self._pure_noise_training_inputs(frame_t, noise)

        train_step = 0 if train_step is None else int(train_step)
        if self.dts_mode == "smooth_shift":
            return self._sample_smooth_shift_dts_inputs(frame_t, noise, train_step)
        return self._sample_step_shift_dts_inputs(frame_t, noise, train_step)

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

    def _pad_frames_to_even_size(self, frames):
        b, t, c, h, w = frames.shape
        pad_h = h % 2
        pad_w = w % 2
        if not (pad_h or pad_w):
            return frames, h, w

        padded = F.pad(frames.reshape(b * t, c, h, w), (0, pad_w, 0, pad_h), mode="replicate")
        return padded.reshape(b, t, c, h + pad_h, w + pad_w), h, w

    @staticmethod
    def _split_quadrants(frames, mid_h, mid_w):
        return torch.stack((
            frames[..., :mid_h, :mid_w],
            frames[..., :mid_h, mid_w:],
            frames[..., mid_h:, :mid_w],
            frames[..., mid_h:, mid_w:],
        ), dim=1).flatten(0, 1)

    @staticmethod
    def _merge_quadrants(frames, target_b):
        frames = frames.view(target_b, 4, *frames.shape[1:])
        top_row = torch.cat((frames[:, 0], frames[:, 1]), dim=-1)
        bottom_row = torch.cat((frames[:, 2], frames[:, 3]), dim=-1)
        return torch.cat((top_row, bottom_row), dim=-2)

    def patch_split(self, frames, return_meta=False):
        patch_meta = []
        while frames.shape[-2] > self.eval_patch_size or frames.shape[-1] > self.eval_patch_size:
            frames, original_h, original_w = self._pad_frames_to_even_size(frames)
            b, t, c, h, w = frames.shape
            patch_meta.append({
                "batch_size": b,
                "height": original_h,
                "width": original_w,
            })
            frames = self._split_quadrants(frames, h // 2, w // 2)

        if return_meta:
            return frames, patch_meta
        return frames

    def patch_merge(self, frames, patch_meta=None):
        if patch_meta is None:
            current_b, _, h, w = frames.shape
            if current_b % 4 != 0:
                raise ValueError(f"Batch size {current_b} is not divisible by 4, cannot merge patches.")
            patch_meta = [{
                "batch_size": current_b // 4,
                "height": h * 2,
                "width": w * 2,
            }]

        for meta in reversed(patch_meta):
            target_b = meta["batch_size"]
            if frames.shape[0] != target_b * 4:
                raise ValueError(f"Cannot merge {frames.shape[0]} patches into batch size {target_b}.")
            frames = self._merge_quadrants(frames, target_b)
            frames = frames[..., :meta["height"], :meta["width"]]

        return frames

    def preprocess_frames(self, frames, is_val=False, train_step=None):
        if not is_val:
            frames = self.random_small(frames)
        frames = frames * self.inv_127_5 - 1.
        frame_data = {
            "frame_0": frames[:, 0],
            "frame_1": frames[:, 1],
            "frame_t": frames[:, 2],
        }
        patched_frames, patch_meta = self.patch_split(frames, return_meta=True) if is_val else (frames, [])

        frame_0, frame_1, frame_t = patched_frames[:, 0], patched_frames[:, 1], patched_frames[:, 2]
        cond_frames = torch.cat((frame_0, frame_1), dim=0)

        noise = torch.randn_like(frame_t)
        timestep, noisy_frames = self.sample_training_inputs(frame_t, noise, train_step, is_val=is_val)

        frame_data.update({
            "target": self.noise_scheduler.training_target(frame_t, noise, is_x=True),
            "noise": noise,
            "patch_meta": patch_meta,
        })
        training_inputs = {
            "noisy_frames": noisy_frames,
            "cond_frames": cond_frames,
            "timestep": timestep,
        }
        return training_inputs, frame_data

    def train_one_step(self, model, frames, train_step):
        training_inputs, frame_data = self.preprocess_frames(frames, train_step=train_step)
        pred = model(**training_inputs)
        fm_loss_pix = torch.abs(pred - frame_data["frame_t"]).mean()

        frame_data["pred_frame_t"] = pred
        loss_data = self.loss_function(frame_data["pred_frame_t"], frame_data["frame_t"], train_step, fm_loss_pix)
        loss_data["log_data"]["loss_pix"] = fm_loss_pix
        return loss_data, frame_data

    @torch.no_grad()
    def validate_one_step(self, model, frames):
        val_inputs, frame_data = self.preprocess_frames(frames, is_val=True)
        pred = model(**val_inputs)
        pred = self.patch_merge(pred, frame_data["patch_meta"])
        return pred, frame_data["frame_t"]

    def visualize_frame_data(self, frame_data):
        nrow = 3
        visual_frame_0, visual_frame_1 = frame_data["frame_0"][:self.visual_results_num], frame_data["frame_1"][:self.visual_results_num]
        blended_input = visual_frame_0 * 0.5 + visual_frame_1 * 0.5
        visual_frame_t = frame_data["frame_t"][:self.visual_results_num]
        visual_pred_frame_t = frame_data["pred_frame_t"][:self.visual_results_num]
        visualization = (torch.stack((blended_input, visual_frame_t, visual_pred_frame_t), dim=1).cpu() / 2. + 0.5).flatten(0, 1)
        return [visualization], nrow

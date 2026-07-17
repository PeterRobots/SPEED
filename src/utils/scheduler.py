import torch


class FlowMatchScheduler:
    DEFAULT_SHIFT = 3
    SIGMA_MIN = 0.0
    SIGMA_MAX = 1.0

    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        self.training = False

    def set_training_weight(self):
        steps = 1000
        weights = torch.exp(-2 * ((self.timesteps - steps / 2) / steps) ** 2)
        weights = weights - weights.min()
        weights = weights * (steps / weights.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(self, num_inference_steps=100, training=False, shift=None):
        shift = self.DEFAULT_SHIFT if shift is None else shift
        sigmas = torch.linspace(self.SIGMA_MAX, self.SIGMA_MIN, num_inference_steps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = sigmas * self.num_train_timesteps
        self.training = training
        if training:
            self.set_training_weight()

    @staticmethod
    def _as_cpu_timestep(timestep):
        if isinstance(timestep, torch.Tensor):
            return timestep.detach().cpu()
        return timestep

    @staticmethod
    def _match_reference(value, reference):
        if reference is None:
            return value
        return value.to(device=reference.device, dtype=reference.dtype)

    def _timestep_id(self, timestep):
        timestep = self._as_cpu_timestep(timestep)
        return torch.argmin((self.timesteps - timestep).abs())

    def _sigma_by_id(self, timestep_id, reference=None):
        return self._match_reference(self.sigmas[timestep_id], reference)

    def _sigma(self, timestep, reference=None):
        return self._sigma_by_id(self._timestep_id(timestep), reference)

    def step(self, model_output, timestep, sample, to_final=False):
        timestep_id = self._timestep_id(timestep)
        sigma = self._sigma_by_id(timestep_id, sample)
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = torch.zeros_like(sigma)
        else:
            sigma_next = self._sigma_by_id(timestep_id + 1, sample)
        return sample + model_output * (sigma_next - sigma)

    def return_to_timestep(self, timestep, sample, sample_stablized):
        return (sample - sample_stablized) / self._sigma(timestep, sample)

    def add_noise(self, original_samples, noise, timestep):
        sigma = self._sigma(timestep, original_samples)
        return (1 - sigma) * original_samples + sigma * noise

    def convert_to_v(self, pred_x, noisy_frames, timestep):
        return (noisy_frames - pred_x) / self._sigma(timestep, noisy_frames)

    def training_target(self, sample, noise, is_x=False):
        return sample if is_x else noise - sample

    def training_weight(self, timestep):
        return self.linear_timesteps_weights[self._timestep_id(timestep)]

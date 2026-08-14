"""Policy-observation delay, noise, sample-hold, and link-jitter models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def expand_observation_parameter(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar, per-environment, or per-channel values to an observation batch."""

    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1).expand_as(reference)
    if tensor.ndim == 1:
        if tensor.shape[0] == reference.shape[0]:
            return tensor.reshape(-1, 1).expand_as(reference)
        if tensor.shape[0] == reference.shape[1]:
            return tensor.reshape(1, -1).expand_as(reference)
    if tensor.ndim == 2:
        if tensor.shape == (reference.shape[0], 1) or tensor.shape == (1, reference.shape[1]):
            return tensor.expand_as(reference)
        if tensor.shape == reference.shape:
            return tensor
    raise ValueError(f"Cannot broadcast value with shape {tuple(tensor.shape)} to {tuple(reference.shape)}.")


def _observation_group_indices(
    selector: slice | int | Sequence[int],
    obs_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(selector, slice):
        return torch.arange(obs_dim, dtype=torch.long, device=device)[selector]
    if isinstance(selector, int):
        return torch.tensor([selector], dtype=torch.long, device=device)
    return torch.as_tensor(list(selector), dtype=torch.long, device=device)


def build_observation_group_parameter(
    group_values: Mapping[str, torch.Tensor | float | Sequence[float]],
    group_slices: Mapping[str, slice | int | Sequence[int]],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Expand semantic observation-group values to the 30-D policy input."""

    result = torch.zeros_like(reference)
    obs_dim = reference.shape[1]
    for group_name, value in group_values.items():
        if group_name not in group_slices:
            known = ", ".join(sorted(group_slices))
            raise ValueError(f"Unknown observation group {group_name!r}. Known groups: {known}.")
        indices = _observation_group_indices(group_slices[group_name], obs_dim, reference.device)
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.ndim == 0:
            result[:, indices] = tensor
        elif tensor.ndim == 1 and tensor.shape[0] == indices.numel():
            result[:, indices] = tensor.reshape(1, -1)
        elif tensor.ndim == 1 and tensor.shape[0] == reference.shape[0]:
            result[:, indices] = tensor.reshape(-1, 1)
        elif tensor.ndim == 2 and tensor.shape == (reference.shape[0], indices.numel()):
            result[:, indices] = tensor
        elif tensor.ndim == 2 and tensor.shape == (1, indices.numel()):
            result[:, indices] = tensor
        else:
            raise ValueError(
                f"Cannot broadcast observation group {group_name!r} with shape {tuple(tensor.shape)} "
                f"to {(reference.shape[0], indices.numel())}."
            )
    return result


class ObservationDelayBuffer:
    """Per-environment fixed-step communication-delay buffer."""

    def __init__(self, num_envs: int, obs_dim: int, max_delay_steps: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.history_length = self.max_delay_steps + 1
        self.history_index = 0
        self.history = torch.zeros(self.history_length, num_envs, obs_dim, device=device)
        self.valid_counts = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: list | torch.Tensor) -> None:
        self.history[:, env_ids, :] = 0.0
        self.valid_counts[env_ids] = 0

    def update(self, obs: torch.Tensor, delay_steps: torch.Tensor | int) -> torch.Tensor:
        self.history[self.history_index] = obs
        self.valid_counts = torch.clamp(self.valid_counts + 1, max=self.history_length)
        delays = torch.as_tensor(delay_steps, dtype=torch.long, device=obs.device)
        if delays.ndim == 0:
            delays = delays.repeat(self.num_envs)
        delays = torch.clamp(delays.reshape(self.num_envs), 0, self.max_delay_steps)
        delays = torch.minimum(delays, torch.clamp(self.valid_counts - 1, min=0))
        indices = (self.history_index - delays) % self.history_length
        env_indices = torch.arange(self.num_envs, dtype=torch.long, device=obs.device)
        delayed = self.history[indices, env_indices]
        self.history_index = (self.history_index + 1) % self.history_length
        return delayed


class ObservationFilterState:
    """Sample hold, packet loss, low-pass filtering, white noise, and bias drift."""

    def __init__(self, num_envs: int, obs_dim: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.previous_measurement = torch.zeros(num_envs, obs_dim, device=device)
        self.bias_drift = torch.zeros_like(self.previous_measurement)
        self.step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.has_measurement = torch.zeros_like(self.previous_measurement, dtype=torch.bool)

    def reset(self, env_ids: list | torch.Tensor) -> None:
        self.previous_measurement[env_ids] = 0.0
        self.bias_drift[env_ids] = 0.0
        self.step_counts[env_ids] = 0
        self.has_measurement[env_ids] = False

    def update(
        self,
        obs: torch.Tensor,
        fixed_bias: torch.Tensor | float,
        noise_std: torch.Tensor | float,
        update_period_steps: torch.Tensor | int = 1,
        dropout_probability: torch.Tensor | float = 0.0,
        lowpass_alpha: torch.Tensor | float = 1.0,
        bias_drift_std: torch.Tensor | float = 0.0,
        dt: float = 1.0,
    ) -> torch.Tensor:
        periods = torch.as_tensor(update_period_steps, dtype=torch.long, device=obs.device)
        if periods.ndim == 0:
            periods = periods.repeat(self.num_envs)
        periods = torch.clamp(periods.reshape(self.num_envs), min=1)

        drift_std = torch.clamp(expand_observation_parameter(bias_drift_std, obs), min=0.0)
        if torch.any(drift_std > 0.0):
            self.bias_drift += torch.randn_like(obs) * drift_std * (dt**0.5)
        noise = torch.clamp(expand_observation_parameter(noise_std, obs), min=0.0)
        raw = obs + expand_observation_parameter(fixed_bias, obs) + self.bias_drift
        if torch.any(noise > 0.0):
            raw += torch.randn_like(obs) * noise

        alpha = torch.clamp(expand_observation_parameter(lowpass_alpha, obs), 0.0, 1.0)
        previous = torch.where(self.has_measurement, self.previous_measurement, raw)
        filtered = alpha * raw + (1.0 - alpha) * previous
        due = (self.step_counts % periods == 0).unsqueeze(-1)
        lost = torch.rand_like(obs) < torch.clamp(
            expand_observation_parameter(dropout_probability, obs), 0.0, 1.0
        )
        accept = due & (~lost | ~self.has_measurement)
        measurement = torch.where(accept, filtered, self.previous_measurement)
        self.previous_measurement.copy_(measurement)
        self.has_measurement |= accept
        self.step_counts += 1
        return measurement


def apply_observation_sensor_model(
    obs: torch.Tensor,
    delay_buffer: ObservationDelayBuffer,
    delay_steps: torch.Tensor | int,
    noise_std: torch.Tensor | float,
    bias: torch.Tensor | float,
    filter_state: ObservationFilterState | None = None,
    update_period_steps: torch.Tensor | int = 1,
    dropout_probability: torch.Tensor | float = 0.0,
    lowpass_alpha: torch.Tensor | float = 1.0,
    bias_drift_std: torch.Tensor | float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Apply the configured communication and observation transport chain."""

    delayed = delay_buffer.update(obs, delay_steps)
    if filter_state is None:
        noise = torch.clamp(expand_observation_parameter(noise_std, delayed), min=0.0)
        if torch.any(noise > 0.0):
            delayed = delayed + torch.randn_like(delayed) * noise
        return delayed + expand_observation_parameter(bias, delayed)
    return filter_state.update(
        delayed,
        bias,
        noise_std,
        update_period_steps,
        dropout_probability,
        lowpass_alpha,
        bias_drift_std,
        dt,
    )

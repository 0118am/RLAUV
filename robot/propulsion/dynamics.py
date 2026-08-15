"""Stateful command transport and first-order thruster response."""

from __future__ import annotations

import torch

class FirstOrderThrusterResponse:
    """Filter T60 body-force targets with a per-environment motor time constant."""

    def __init__(
        self,
        num_envs: int,
        num_thrusters: int,
        time_constant_s: torch.Tensor | float,
        device: torch.device,
    ) -> None:
        if num_envs <= 0 or num_thrusters <= 0:
            raise ValueError("num_envs and num_thrusters must be positive.")
        self.num_envs = int(num_envs)
        self.num_thrusters = int(num_thrusters)
        self.device = torch.device(device)
        self.output_forces_b = torch.zeros(
            (self.num_envs, self.num_thrusters, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_update_time_s = torch.full(
            (self.num_envs,),
            torch.nan,
            dtype=torch.float32,
            device=self.device,
        )
        self.set_time_constants(time_constant_s)

    def set_time_constants(self, time_constant_s: torch.Tensor | float) -> None:
        """Set one shared constant or one constant per vectorized environment."""

        value = torch.as_tensor(time_constant_s, dtype=torch.float32, device=self.device)
        if value.ndim > 1 or (value.ndim == 1 and value.shape != (self.num_envs,)):
            raise ValueError("time_constant_s must be scalar or have shape (num_envs,).")
        self.time_constant_s = value.clone()

    def reset(self, env_ids: list | torch.Tensor | None = None) -> None:
        """Clear the realized force and timing state for selected environments."""

        selected = slice(None) if env_ids is None else env_ids
        self.output_forces_b[selected] = 0.0
        self.last_update_time_s[selected] = torch.nan

    def advance(self, target_forces_b: torch.Tensor, time_s: torch.Tensor | float) -> torch.Tensor:
        """Advance the exact discrete first-order response to ``time_s``."""

        expected_shape = (self.num_envs, self.num_thrusters, 3)
        if target_forces_b.shape != expected_shape:
            raise ValueError(f"target_forces_b must have shape {expected_shape}.")

        current_time_s = torch.as_tensor(
            time_s,
            dtype=target_forces_b.dtype,
            device=target_forces_b.device,
        )
        if current_time_s.ndim == 0:
            current_time_s = current_time_s.repeat(self.num_envs)
        elif current_time_s.shape != (self.num_envs,):
            raise ValueError("time_s must be scalar or have shape (num_envs,).")

        previous_time_s = self.last_update_time_s.to(
            dtype=target_forces_b.dtype,
            device=target_forces_b.device,
        )
        elapsed_s = torch.where(
            torch.isfinite(previous_time_s),
            torch.clamp(current_time_s - previous_time_s, min=0.0),
            torch.zeros_like(current_time_s),
        )
        time_constant_s = self.time_constant_s.to(
            dtype=target_forces_b.dtype,
            device=target_forces_b.device,
        )
        if time_constant_s.ndim == 0:
            time_constant_s = time_constant_s.repeat(self.num_envs)
        decay = torch.exp(-elapsed_s / torch.clamp(time_constant_s, min=1.0e-6))
        decay = torch.where(time_constant_s <= 0.0, torch.zeros_like(decay), decay)
        blend = decay.reshape(self.num_envs, 1, 1)
        self.output_forces_b = self.output_forces_b * blend + (1.0 - blend) * target_forces_b
        self.last_update_time_s[:] = current_time_s
        return self.output_forces_b


class ThrusterCommandProcessor:
    """Apply command delay, dropouts, rate limits, and quantization."""

    def __init__(
        self,
        num_envs: int,
        num_thrusters: int,
        max_delay_steps: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.num_thrusters = num_thrusters
        self.device = device
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.history_length = self.max_delay_steps + 1
        self.history_index = 0
        self.history = torch.zeros(
            (self.history_length, self.num_envs, self.num_thrusters),
            dtype=torch.float32,
            device=self.device,
        )
        self.rate_limited_state = torch.zeros(
            (self.num_envs, self.num_thrusters),
            dtype=torch.float32,
            device=self.device,
        )
        self._env_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: list | torch.Tensor | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self.history[:, selected, :] = 0.0
        self.rate_limited_state[selected, :] = 0.0
        if env_ids is None:
            self.history_index = 0

    def process(
        self,
        commands: torch.Tensor,
        delay_steps: torch.Tensor | int,
        max_rate: torch.Tensor | float,
        dt: torch.Tensor | float,
        command_resolution: torch.Tensor | float = 0.0,
        dropout_probability: torch.Tensor | float = 0.0,
        *,
        dropout_enabled: bool | None = None,
    ) -> torch.Tensor:
        expected_shape = (self.num_envs, self.num_thrusters)
        if commands.shape != expected_shape:
            raise ValueError(f"commands must have shape {expected_shape}.")
        self.history[self.history_index, :, :] = commands

        delay_steps = torch.as_tensor(delay_steps, dtype=torch.long, device=commands.device)
        if delay_steps.ndim == 0:
            delay_steps = delay_steps.repeat(self.num_envs)
        delay_steps = torch.clamp(delay_steps.reshape(self.num_envs), min=0, max=self.max_delay_steps)

        delayed_indices = (self.history_index - delay_steps) % self.history_length
        delayed_cmd = self.history[delayed_indices, self._env_indices, :]
        self.history_index = (self.history_index + 1) % self.history_length

        dropout_probability = torch.clamp(
            _expand_env_thruster_value(dropout_probability, commands),
            min=0.0,
            max=1.0,
        )
        if dropout_enabled is None:
            dropout_enabled = bool(torch.any(dropout_probability > 0.0))
        if dropout_enabled:
            dropout_mask = torch.rand_like(commands) < dropout_probability
            delayed_cmd = torch.where(dropout_mask, self.rate_limited_state, delayed_cmd)

        rate = _expand_env_thruster_value(max_rate, commands)
        dt_tensor = torch.as_tensor(dt, dtype=commands.dtype, device=commands.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.reshape(1, 1)
        elif dt_tensor.ndim == 1:
            dt_tensor = dt_tensor.reshape(self.num_envs, 1)
        max_delta = torch.clamp(rate, min=0.0) * dt_tensor

        delta = delayed_cmd - self.rate_limited_state
        limited_cmd = self.rate_limited_state + torch.clamp(delta, -max_delta, max_delta)
        processed_cmd = torch.where(rate <= 0.0, delayed_cmd, limited_cmd)

        resolution = torch.clamp(_expand_env_thruster_value(command_resolution, commands), min=0.0)
        quantized_cmd = torch.round(processed_cmd / torch.clamp(resolution, min=1.0e-6)) * resolution
        self.rate_limited_state = torch.where(resolution > 0.0, quantized_cmd, processed_cmd)
        self.rate_limited_state = torch.clamp(self.rate_limited_state, min=-1.0, max=1.0)
        return self.rate_limited_state


def _expand_env_thruster_value(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        if tensor.shape[0] == reference.shape[0]:
            return tensor.reshape(reference.shape[0], 1)
        if tensor.shape[0] == reference.shape[1]:
            return tensor.reshape(1, reference.shape[1])
    if tensor.ndim == 2:
        if tensor.shape == (reference.shape[0], 1):
            return tensor
        if tensor.shape == (1, reference.shape[1]):
            return tensor
    if tensor.shape == reference.shape:
        return tensor
    raise ValueError(f"Cannot broadcast value with shape {tuple(tensor.shape)} to {tuple(reference.shape)}.")

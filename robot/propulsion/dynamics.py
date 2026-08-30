"""Fixed command delay and stateful first-order thruster response."""

from __future__ import annotations

import torch


class FixedStepCommandDelay:
    """Delay normalized thruster commands by an exact number of physics steps."""

    def __init__(
        self,
        num_envs: int,
        num_thrusters: int,
        delay_steps: int,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0 or num_thrusters <= 0:
            raise ValueError("num_envs and num_thrusters must be positive.")
        if int(delay_steps) != delay_steps or delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer.")
        self.num_envs = int(num_envs)
        self.num_thrusters = int(num_thrusters)
        self.delay_steps = int(delay_steps)
        self.device = torch.device(device)
        self.write_index = 0
        self.command_history = torch.zeros(
            (max(self.delay_steps, 1), self.num_envs, self.num_thrusters),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, env_ids: list | torch.Tensor | None) -> None:
        """Fill selected command pipelines with neutral commands."""

        selected = slice(None) if env_ids is None else env_ids
        self.command_history[:, selected] = 0.0

    def advance(self, commands: torch.Tensor) -> torch.Tensor:
        """Record current commands and return commands from ``delay_steps`` ago."""

        expected_shape = (self.num_envs, self.num_thrusters)
        if commands.shape != expected_shape:
            raise ValueError(f"commands must have shape {expected_shape}.")
        if self.delay_steps == 0:
            return commands
        delayed = self.command_history[self.write_index].clone()
        self.command_history[self.write_index].copy_(commands)
        self.write_index = (self.write_index + 1) % self.delay_steps
        return delayed


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

    def reset(
        self,
        env_ids: list | torch.Tensor | None,
        *,
        time_s: float,
    ) -> None:
        """Clear realized force and anchor selected environments at ``time_s``."""

        selected = slice(None) if env_ids is None else env_ids
        self.output_forces_b[selected] = 0.0
        self.last_update_time_s[selected] = float(time_s)

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
        elapsed_s = torch.clamp(current_time_s - previous_time_s, min=0.0)
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

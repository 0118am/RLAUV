"""Deterministically delayed fused-state sensor used by the policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoseSensorMeasurement:
    position_w: torch.Tensor
    quaternion_wxyz: torch.Tensor
    linear_velocity_b: torch.Tensor
    angular_velocity_b: torch.Tensor


class DelayedPoseSensor:
    """Return an exact rigid-body state from the configured delay horizon."""

    def __init__(
        self,
        num_envs: int,
        delay_steps: int,
        device: torch.device | str,
    ) -> None:
        self.num_envs = int(num_envs)
        self.delay_steps = int(delay_steps)
        self.device = torch.device(device)
        self.history_length = self.delay_steps + 1
        self.write_index = 0

        self.position_history = torch.zeros(
            (self.history_length, self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.quaternion_history = torch.zeros(
            (self.history_length, self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.quaternion_history[..., 0] = 1.0
        self.linear_velocity_history = torch.zeros_like(self.position_history)
        self.angular_velocity_history = torch.zeros_like(self.position_history)

    def record(
        self,
        position_w: torch.Tensor,
        quaternion_wxyz: torch.Tensor,
        linear_velocity_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
    ) -> None:
        """Record one 100 Hz truth sample before the next physics step."""

        self.position_history[self.write_index].copy_(position_w)
        self.quaternion_history[self.write_index].copy_(quaternion_wxyz)
        self.linear_velocity_history[self.write_index].copy_(linear_velocity_b)
        self.angular_velocity_history[self.write_index].copy_(angular_velocity_b)
        self.write_index = (self.write_index + 1) % self.history_length

    def reset(
        self,
        env_ids: torch.Tensor,
        position_w: torch.Tensor,
        quaternion_wxyz: torch.Tensor,
        linear_velocity_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
    ) -> None:
        """Fill selected histories with the new episode's exact initial state."""

        self.position_history[:, env_ids] = position_w.unsqueeze(0)
        self.quaternion_history[:, env_ids] = quaternion_wxyz.unsqueeze(0)
        self.linear_velocity_history[:, env_ids] = linear_velocity_b.unsqueeze(0)
        self.angular_velocity_history[:, env_ids] = angular_velocity_b.unsqueeze(0)

    def measure(self) -> PoseSensorMeasurement:
        """Return the exact state stored ``delay_steps`` physics samples ago."""

        read_index = (self.write_index - self.delay_steps) % self.history_length
        return PoseSensorMeasurement(
            self.position_history[read_index],
            self.quaternion_history[read_index],
            self.linear_velocity_history[read_index],
            self.angular_velocity_history[read_index],
        )

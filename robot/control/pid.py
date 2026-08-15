"""Robot-owned six-degree-of-freedom PID tracking controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from robot.control.allocation import NonlinearThrusterAllocator
from robot.control.trajectory.observation_contract import (
    BASE_OBSERVATION_DIM,
    OBSERVATION_FIELD_SLICES,
    TRAJECTORY_OBSERVATION,
)


@dataclass(frozen=True)
class PIDGains:
    position_kp: Sequence[float] = (20.0, 20.0, 25.0)
    position_ki: Sequence[float] = (0.5, 0.5, 0.8)
    velocity_kd: Sequence[float] = (15.0, 15.0, 18.0)
    attitude_kp: Sequence[float] = (8.0, 8.0, 6.0)
    attitude_ki: Sequence[float] = (0.2, 0.2, 0.15)
    angular_velocity_kd: Sequence[float] = (3.0, 3.0, 2.5)


class PIDTrajectoryController:
    """Convert tracking observations to a desired wrench, then allocate it."""

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        thruster_positions_b: torch.Tensor,
        thruster_force_curve_coefficients: torch.Tensor,
        mass_kg: float | torch.Tensor,
        gains: PIDGains = PIDGains(),
        integral_position_limit_m_s: float = 2.0,
        integral_attitude_limit_rad_s: float = 1.0,
        allocation_iterations: int = 16,
        allocation_damping: float = 1.0e-3,
        allocation_tolerance: float = 1.0e-4,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        device = thruster_positions_b.device
        dtype = thruster_positions_b.dtype
        self.position_scale_m = TRAJECTORY_OBSERVATION.field(
            "position_error_b"
        ).physical_scale
        self.linear_velocity_scale_mps = TRAJECTORY_OBSERVATION.field(
            "linear_velocity_error_b"
        ).physical_scale
        self.angular_velocity_scale_radps = TRAJECTORY_OBSERVATION.field(
            "angular_velocity_b"
        ).physical_scale
        self.linear_acceleration_scale_mps2 = TRAJECTORY_OBSERVATION.field(
            "target_linear_acceleration_b"
        ).physical_scale
        self.integral_position_limit = float(integral_position_limit_m_s)
        self.integral_attitude_limit = float(integral_attitude_limit_rad_s)

        mass = torch.as_tensor(mass_kg, device=device, dtype=dtype)
        if mass.numel() not in (1, self.num_envs):
            raise ValueError("mass_kg must be scalar or contain one value per environment.")
        self.mass_kg = mass.reshape(-1, 1) if mass.numel() > 1 else mass.reshape(1, 1)
        self.position_integral = torch.zeros(self.num_envs, 3, device=device, dtype=dtype)
        self.attitude_integral = torch.zeros_like(self.position_integral)
        self.position_kp = torch.as_tensor(gains.position_kp, device=device, dtype=dtype)
        self.position_ki = torch.as_tensor(gains.position_ki, device=device, dtype=dtype)
        self.velocity_kd = torch.as_tensor(gains.velocity_kd, device=device, dtype=dtype)
        self.attitude_kp = torch.as_tensor(gains.attitude_kp, device=device, dtype=dtype)
        self.attitude_ki = torch.as_tensor(gains.attitude_ki, device=device, dtype=dtype)
        self.angular_velocity_kd = torch.as_tensor(
            gains.angular_velocity_kd,
            device=device,
            dtype=dtype,
        )
        self.allocator = NonlinearThrusterAllocator(
            num_envs=self.num_envs,
            thruster_positions_b=thruster_positions_b,
            thruster_force_curve_coefficients=thruster_force_curve_coefficients,
            iterations=allocation_iterations,
            damping=allocation_damping,
            tolerance=allocation_tolerance,
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        self.position_integral[env_ids] = 0.0
        self.attitude_integral[env_ids] = 0.0

    def allocate_wrench(self, desired_wrench_b: torch.Tensor) -> torch.Tensor:
        """Allocate a desired body wrench with the configured thruster model."""

        return self.allocator.allocate(desired_wrench_b)

    @staticmethod
    def _quaternion_rotation_vector(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion_wxyz / torch.linalg.vector_norm(
            quaternion_wxyz,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-8)
        vector = quaternion[:, 1:4]
        vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vector_norm, quaternion[:, 0:1].clamp_min(0.0))
        return vector / vector_norm.clamp_min(1.0e-8) * angle

    def desired_wrench(
        self,
        observations: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PID/feed-forward body wrench before allocation."""

        obs = observations if isinstance(observations, torch.Tensor) else observations["policy"]
        if obs.shape[-1] < BASE_OBSERVATION_DIM:
            raise ValueError(
                f"Trajectory observation must have at least {BASE_OBSERVATION_DIM} values, "
                f"got {obs.shape[-1]}."
            )
        current = obs[:, :BASE_OBSERVATION_DIM]
        position_error = (
            current[:, OBSERVATION_FIELD_SLICES["position_error_b"]]
            * self.position_scale_m
        )
        velocity_error = (
            current[:, OBSERVATION_FIELD_SLICES["linear_velocity_error_b"]]
            * self.linear_velocity_scale_mps
        )
        attitude_error = self._quaternion_rotation_vector(
            current[:, OBSERVATION_FIELD_SLICES["attitude_error_quat"]]
        )
        angular_velocity_error = (
            current[:, OBSERVATION_FIELD_SLICES["target_angular_velocity_b"]]
            - current[:, OBSERVATION_FIELD_SLICES["angular_velocity_b"]]
        ) * self.angular_velocity_scale_radps
        target_acceleration = (
            current[:, OBSERVATION_FIELD_SLICES["target_linear_acceleration_b"]]
            * self.linear_acceleration_scale_mps2
        )

        self.position_integral.add_(position_error * self.dt).clamp_(
            -self.integral_position_limit,
            self.integral_position_limit,
        )
        self.attitude_integral.add_(attitude_error * self.dt).clamp_(
            -self.integral_attitude_limit,
            self.integral_attitude_limit,
        )
        force_b = (
            self.position_kp * position_error
            + self.position_ki * self.position_integral
            + self.velocity_kd * velocity_error
            + self.mass_kg * target_acceleration
        )
        torque_b = (
            self.attitude_kp * attitude_error
            + self.attitude_ki * self.attitude_integral
            + self.angular_velocity_kd * angular_velocity_error
        )
        return torch.cat((force_b, torque_b), dim=-1)

    def __call__(
        self,
        observations: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        return self.allocator.allocate(self.desired_wrench(observations))

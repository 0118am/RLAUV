"""Six-degree-of-freedom PID tracking baseline for the eight-thruster AUV."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import torch

from robot.propulsion.thrusters import (
    measured_thruster_body_forces,
    measured_thruster_force_jacobian,
    reduce_point_forces_to_wrench,
)
from robot.dynamics.parameters import AUV


@dataclass(frozen=True)
class PIDGains:
    position_kp: Sequence[float] = (20.0, 20.0, 25.0)
    position_ki: Sequence[float] = (0.5, 0.5, 0.8)
    velocity_kd: Sequence[float] = (15.0, 15.0, 18.0)
    attitude_kp: Sequence[float] = (8.0, 8.0, 6.0)
    attitude_ki: Sequence[float] = (0.2, 0.2, 0.15)
    angular_velocity_kd: Sequence[float] = (3.0, 3.0, 2.5)


class PIDTrajectoryController:
    """Map normalized tracking observations to bounded T1...T8 commands.

    The allocator solves the measured nonlinear three-component force curves
    directly.  A thruster therefore has neither a separate direction nor a
    command polarity in this controller: both axial and off-axis force signs
    are already present in its measured coefficients.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        thruster_positions_b: torch.Tensor,
        thruster_force_curve_coefficients: torch.Tensor,
        position_scale_m: float,
        linear_velocity_scale_mps: float,
        angular_velocity_scale_radps: float,
        linear_acceleration_scale_mps2: float,
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
        self.position_scale_m = float(position_scale_m)
        self.linear_velocity_scale_mps = float(linear_velocity_scale_mps)
        self.angular_velocity_scale_radps = float(angular_velocity_scale_radps)
        self.linear_acceleration_scale_mps2 = float(linear_acceleration_scale_mps2)
        self.integral_position_limit = float(integral_position_limit_m_s)
        self.integral_attitude_limit = float(integral_attitude_limit_rad_s)
        self.allocation_iterations = int(allocation_iterations)
        self.allocation_damping = float(allocation_damping)
        self.allocation_tolerance = float(allocation_tolerance)
        if self.allocation_iterations < 1:
            raise ValueError("allocation_iterations must be positive.")
        if self.allocation_damping <= 0.0 or self.allocation_tolerance <= 0.0:
            raise ValueError("allocation damping and tolerance must be positive.")

        device = thruster_positions_b.device
        dtype = thruster_positions_b.dtype
        self.thruster_positions_b = thruster_positions_b.to(device=device, dtype=dtype)
        self.thruster_force_curve_coefficients = thruster_force_curve_coefficients.to(
            device=device,
            dtype=dtype,
        )
        expected_thruster_count = len(AUV.thruster_labels)
        if self.thruster_positions_b.shape != (expected_thruster_count, 3):
            raise ValueError(f"thruster_positions_b must have shape ({expected_thruster_count}, 3).")
        if self.thruster_force_curve_coefficients.shape != (expected_thruster_count, 4, 3):
            raise ValueError(
                f"thruster_force_curve_coefficients must have shape ({expected_thruster_count}, 4, 3)."
            )

        mass = torch.as_tensor(mass_kg, device=device, dtype=dtype)
        if mass.numel() not in (1, self.num_envs):
            raise ValueError("mass_kg must be scalar or contain one value per environment.")
        self.mass_kg = mass.reshape(-1, 1) if mass.numel() > 1 else mass.reshape(1, 1)

        characteristic_length = torch.linalg.vector_norm(self.thruster_positions_b, dim=-1).max().clamp_min(1.0e-3)
        self._wrench_weights = torch.cat(
            (
                torch.ones(3, device=device, dtype=dtype),
                torch.ones(3, device=device, dtype=dtype) / characteristic_length,
            )
        )
        branch_ids = torch.arange(1 << expected_thruster_count, device=device).unsqueeze(-1)
        bit_ids = torch.arange(expected_thruster_count, device=device)
        self._branch_signs = torch.where(
            torch.bitwise_and(torch.bitwise_right_shift(branch_ids, bit_ids), 1).bool(),
            torch.ones((), device=device, dtype=dtype),
            -torch.ones((), device=device, dtype=dtype),
        )
        endpoint_forces = measured_thruster_body_forces(
            self._branch_signs,
            self.thruster_force_curve_coefficients,
        )
        branch_positions = self.thruster_positions_b.unsqueeze(0).expand_as(endpoint_forces)
        branch_columns = torch.cat(
            (
                endpoint_forces,
                torch.cross(branch_positions, endpoint_forces, dim=-1),
            ),
            dim=-1,
        ).transpose(-1, -2)
        self._branch_seed_pinv = torch.linalg.pinv(
            self._wrench_weights.reshape(1, 6, 1) * branch_columns
        )
        self._allocation_identity = torch.eye(expected_thruster_count, device=device, dtype=dtype)
        self._line_search_scales = torch.tensor(
            (1.0, 0.5, 0.25, 0.125, 0.0),
            device=device,
            dtype=dtype,
        )
        self._allocation_row_indices = torch.arange(
            self._branch_signs.shape[0] * self.num_envs,
            device=device,
        )
        self._environment_indices = torch.arange(self.num_envs, device=device)
        self._active_command_floor = min(
            float(AUV.thruster_pwm_deadband_us / AUV.thruster_pwm_half_range_us) + 1.0e-3,
            1.0,
        )
        self.position_integral = torch.zeros(self.num_envs, 3, device=device, dtype=dtype)
        self.attitude_integral = torch.zeros_like(self.position_integral)
        self.position_kp = torch.as_tensor(gains.position_kp, device=device, dtype=dtype)
        self.position_ki = torch.as_tensor(gains.position_ki, device=device, dtype=dtype)
        self.velocity_kd = torch.as_tensor(gains.velocity_kd, device=device, dtype=dtype)
        self.attitude_kp = torch.as_tensor(gains.attitude_kp, device=device, dtype=dtype)
        self.attitude_ki = torch.as_tensor(gains.attitude_ki, device=device, dtype=dtype)
        self.angular_velocity_kd = torch.as_tensor(gains.angular_velocity_kd, device=device, dtype=dtype)

    def reset(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        self.position_integral[env_ids] = 0.0
        self.attitude_integral[env_ids] = 0.0

    def _realized_wrench(self, commands: torch.Tensor) -> torch.Tensor:
        forces = measured_thruster_body_forces(
            commands,
            self.thruster_force_curve_coefficients,
        )
        positions = self.thruster_positions_b.unsqueeze(0).expand(commands.shape[0], -1, -1)
        return reduce_point_forces_to_wrench(positions, forces)

    def _wrench_jacobian(self, commands: torch.Tensor) -> torch.Tensor:
        force_jacobian = measured_thruster_force_jacobian(
            commands,
            self.thruster_force_curve_coefficients,
        )
        positions = self.thruster_positions_b.unsqueeze(0).expand(commands.shape[0], -1, -1)
        torque_jacobian = torch.cross(positions, force_jacobian, dim=-1)
        return torch.cat((force_jacobian, torque_jacobian), dim=-1).transpose(-1, -2)

    def _allocation_cost(self, commands: torch.Tensor, desired_wrench_b: torch.Tensor) -> torch.Tensor:
        residual = (self._realized_wrench(commands) - desired_wrench_b) * self._wrench_weights
        return torch.sum(residual.square(), dim=-1)

    def allocate_wrench(self, desired_wrench_b: torch.Tensor) -> torch.Tensor:
        """Return bounded normalized commands for a desired body wrench.

        A weighted projected Levenberg--Marquardt solve handles the measured
        piecewise-quadratic vector force curves.  All positive/negative branch
        combinations are solved in parallel because a branch cannot be crossed
        reliably through the measured zero-Jacobian deadband.  Each branch seed
        starts just outside that deadband and the lowest-cost bounded solution
        is returned.
        """

        if desired_wrench_b.ndim != 2 or desired_wrench_b.shape != (self.num_envs, 6):
            raise ValueError(f"desired_wrench_b must have shape ({self.num_envs}, 6).")
        desired = desired_wrench_b.to(
            device=self.thruster_positions_b.device,
            dtype=self.thruster_positions_b.dtype,
        )
        weighted_desired = desired * self._wrench_weights
        nonzero_request = torch.linalg.vector_norm(weighted_desired, dim=-1, keepdim=True) > self.allocation_tolerance
        if not bool(torch.any(nonzero_request)):
            return torch.zeros(
                (self.num_envs, self.thruster_positions_b.shape[0]),
                device=desired.device,
                dtype=desired.dtype,
            )

        seed_magnitudes = torch.einsum(
            "bij,nj->bni",
            self._branch_seed_pinv,
            weighted_desired,
        ).clamp(0.0, 1.0)
        seed_magnitudes = self._active_command_floor + (
            1.0 - self._active_command_floor
        ) * seed_magnitudes
        commands = self._branch_signs.unsqueeze(1) * seed_magnitudes
        commands = torch.where(
            nonzero_request.unsqueeze(0),
            commands,
            torch.zeros_like(commands),
        )
        branch_count = commands.shape[0]
        commands = commands.reshape(-1, commands.shape[-1])
        branch_desired = desired.unsqueeze(0).expand(branch_count, -1, -1).reshape(-1, 6)

        for _ in range(self.allocation_iterations):
            realized = self._realized_wrench(commands)
            weighted_residual = (realized - branch_desired) * self._wrench_weights
            residual_norm = torch.linalg.vector_norm(weighted_residual, dim=-1).reshape(
                branch_count,
                self.num_envs,
            )
            if bool(torch.all(torch.any(residual_norm <= self.allocation_tolerance, dim=0))):
                break
            jacobian = self._wrench_jacobian(commands) * self._wrench_weights.reshape(1, 6, 1)
            jacobian_t = jacobian.transpose(-1, -2)
            normal_matrix = torch.bmm(jacobian_t, jacobian) + self.allocation_damping * self._allocation_identity
            normal_rhs = -torch.bmm(jacobian_t, weighted_residual.unsqueeze(-1)).squeeze(-1)
            step = torch.linalg.solve(normal_matrix, normal_rhs.unsqueeze(-1)).squeeze(-1).clamp(-0.5, 0.5)

            candidates = torch.clamp(
                commands.unsqueeze(0) + self._line_search_scales.reshape(-1, 1, 1) * step.unsqueeze(0),
                -1.0,
                1.0,
            )
            flat_candidates = candidates.reshape(-1, commands.shape[-1])
            repeated_desired = branch_desired.unsqueeze(0).expand(candidates.shape[0], -1, -1).reshape(-1, 6)
            costs = self._allocation_cost(flat_candidates, repeated_desired).reshape(
                candidates.shape[0],
                commands.shape[0],
            )
            best = torch.argmin(costs, dim=0)
            commands = candidates[best, self._allocation_row_indices]

        final_costs = self._allocation_cost(commands, branch_desired).reshape(branch_count, self.num_envs)
        best_branch = torch.argmin(final_costs, dim=0)
        commands = commands.reshape(branch_count, self.num_envs, -1)[best_branch, self._environment_indices]
        return torch.where(nonzero_request, commands, torch.zeros_like(commands)).clamp(-1.0, 1.0)

    @staticmethod
    def _quaternion_rotation_vector(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion_wxyz / torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True).clamp_min(1e-8)
        vector = quaternion[:, 1:4]
        vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vector_norm, quaternion[:, 0:1].clamp_min(0.0))
        return vector / vector_norm.clamp_min(1e-8) * angle

    def __call__(self, observations: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        # Isaac's RSL-RL wrapper returns a TensorDict, which supports string
        # indexing but is not a builtin dict.  Distinguish the raw-tensor API
        # first and treat every other accepted container as mapping-like.
        obs = observations if isinstance(observations, torch.Tensor) else observations["policy"]
        current = obs[:, :30]
        position_error = current[:, 0:3] * self.position_scale_m
        velocity_error = current[:, 6:9] * self.linear_velocity_scale_mps
        attitude_error = self._quaternion_rotation_vector(current[:, 9:13])
        angular_velocity_error = (current[:, 16:19] - current[:, 13:16]) * self.angular_velocity_scale_radps
        target_acceleration = current[:, 19:22] * self.linear_acceleration_scale_mps2

        self.position_integral.add_(position_error * self.dt).clamp_(
            -self.integral_position_limit, self.integral_position_limit
        )
        self.attitude_integral.add_(attitude_error * self.dt).clamp_(
            -self.integral_attitude_limit, self.integral_attitude_limit
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
        return self.allocate_wrench(torch.cat((force_b, torque_b), dim=-1))

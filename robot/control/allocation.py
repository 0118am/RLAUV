"""Nonlinear allocation from desired body wrench to measured T60 commands."""

from __future__ import annotations

import torch

from robot.dynamics.parameters import AUV
from robot.propulsion.curves import (
    measured_thruster_body_forces,
    measured_thruster_force_jacobian,
    reduce_point_forces_to_wrench,
)


class NonlinearThrusterAllocator:
    """Solve the measured piecewise-quadratic vector force curves."""

    def __init__(
        self,
        *,
        num_envs: int,
        thruster_positions_b: torch.Tensor,
        thruster_force_curve_coefficients: torch.Tensor,
        iterations: int = 16,
        damping: float = 1.0e-3,
        tolerance: float = 1.0e-4,
    ) -> None:
        self.num_envs = int(num_envs)
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.tolerance = float(tolerance)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.damping <= 0.0 or self.tolerance <= 0.0:
            raise ValueError("damping and tolerance must be positive.")

        device = thruster_positions_b.device
        dtype = thruster_positions_b.dtype
        self.thruster_positions_b = thruster_positions_b.to(device=device, dtype=dtype)
        self.force_curve_coefficients = thruster_force_curve_coefficients.to(
            device=device,
            dtype=dtype,
        )
        thruster_count = len(AUV.thruster_labels)
        if self.thruster_positions_b.shape != (thruster_count, 3):
            raise ValueError(f"thruster_positions_b must have shape ({thruster_count}, 3).")
        if self.force_curve_coefficients.shape != (thruster_count, 4, 3):
            raise ValueError(
                f"thruster_force_curve_coefficients must have shape ({thruster_count}, 4, 3)."
            )

        characteristic_length = torch.linalg.vector_norm(
            self.thruster_positions_b, dim=-1
        ).max().clamp_min(1.0e-3)
        self._wrench_weights = torch.cat(
            (
                torch.ones(3, device=device, dtype=dtype),
                torch.ones(3, device=device, dtype=dtype) / characteristic_length,
            )
        )
        branch_ids = torch.arange(1 << thruster_count, device=device).unsqueeze(-1)
        bit_ids = torch.arange(thruster_count, device=device)
        self._branch_signs = torch.where(
            torch.bitwise_and(torch.bitwise_right_shift(branch_ids, bit_ids), 1).bool(),
            torch.ones((), device=device, dtype=dtype),
            -torch.ones((), device=device, dtype=dtype),
        )
        endpoint_forces = measured_thruster_body_forces(
            self._branch_signs,
            self.force_curve_coefficients,
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
        self._identity = torch.eye(thruster_count, device=device, dtype=dtype)
        self._line_search_scales = torch.tensor(
            (1.0, 0.5, 0.25, 0.125, 0.0),
            device=device,
            dtype=dtype,
        )
        self._row_indices = torch.arange(
            self._branch_signs.shape[0] * self.num_envs,
            device=device,
        )
        self._environment_indices = torch.arange(self.num_envs, device=device)
        self._active_command_floor = min(
            float(AUV.thruster_pwm_deadband_us / AUV.thruster_pwm_half_range_us) + 1.0e-3,
            1.0,
        )

    def _realized_wrench(self, commands: torch.Tensor) -> torch.Tensor:
        forces = measured_thruster_body_forces(commands, self.force_curve_coefficients)
        positions = self.thruster_positions_b.unsqueeze(0).expand(commands.shape[0], -1, -1)
        return reduce_point_forces_to_wrench(positions, forces)

    def _wrench_jacobian(self, commands: torch.Tensor) -> torch.Tensor:
        force_jacobian = measured_thruster_force_jacobian(
            commands,
            self.force_curve_coefficients,
        )
        positions = self.thruster_positions_b.unsqueeze(0).expand(commands.shape[0], -1, -1)
        torque_jacobian = torch.cross(positions, force_jacobian, dim=-1)
        return torch.cat((force_jacobian, torque_jacobian), dim=-1).transpose(-1, -2)

    def _cost(self, commands: torch.Tensor, desired_wrench_b: torch.Tensor) -> torch.Tensor:
        residual = (self._realized_wrench(commands) - desired_wrench_b) * self._wrench_weights
        return torch.sum(residual.square(), dim=-1)

    def allocate(self, desired_wrench_b: torch.Tensor) -> torch.Tensor:
        """Return bounded normalized commands for ``(num_envs, 6)`` body wrenches."""

        if desired_wrench_b.shape != (self.num_envs, 6):
            raise ValueError(f"desired_wrench_b must have shape ({self.num_envs}, 6).")
        desired = desired_wrench_b.to(
            device=self.thruster_positions_b.device,
            dtype=self.thruster_positions_b.dtype,
        )
        weighted_desired = desired * self._wrench_weights
        nonzero_request = (
            torch.linalg.vector_norm(weighted_desired, dim=-1, keepdim=True) > self.tolerance
        )
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
        commands = torch.where(nonzero_request.unsqueeze(0), commands, torch.zeros_like(commands))
        branch_count = commands.shape[0]
        commands = commands.reshape(-1, commands.shape[-1])
        branch_desired = desired.unsqueeze(0).expand(branch_count, -1, -1).reshape(-1, 6)

        for _ in range(self.iterations):
            realized = self._realized_wrench(commands)
            weighted_residual = (realized - branch_desired) * self._wrench_weights
            residual_norm = torch.linalg.vector_norm(weighted_residual, dim=-1).reshape(
                branch_count,
                self.num_envs,
            )
            if bool(torch.all(torch.any(residual_norm <= self.tolerance, dim=0))):
                break
            jacobian = self._wrench_jacobian(commands) * self._wrench_weights.reshape(1, 6, 1)
            jacobian_t = jacobian.transpose(-1, -2)
            normal_matrix = (
                torch.bmm(jacobian_t, jacobian) + self.damping * self._identity
            )
            normal_rhs = -torch.bmm(
                jacobian_t,
                weighted_residual.unsqueeze(-1),
            ).squeeze(-1)
            step = torch.linalg.solve(normal_matrix, normal_rhs.unsqueeze(-1)).squeeze(-1)
            step = step.clamp(-0.5, 0.5)

            candidates = torch.clamp(
                commands.unsqueeze(0)
                + self._line_search_scales.reshape(-1, 1, 1) * step.unsqueeze(0),
                -1.0,
                1.0,
            )
            flat_candidates = candidates.reshape(-1, commands.shape[-1])
            repeated_desired = branch_desired.unsqueeze(0).expand(
                candidates.shape[0], -1, -1
            ).reshape(-1, 6)
            costs = self._cost(flat_candidates, repeated_desired).reshape(
                candidates.shape[0],
                commands.shape[0],
            )
            best = torch.argmin(costs, dim=0)
            commands = candidates[best, self._row_indices]

        final_costs = self._cost(commands, branch_desired).reshape(
            branch_count,
            self.num_envs,
        )
        best_branch = torch.argmin(final_costs, dim=0)
        commands = commands.reshape(branch_count, self.num_envs, -1)[
            best_branch,
            self._environment_indices,
        ]
        return torch.where(nonzero_request, commands, torch.zeros_like(commands)).clamp(
            -1.0,
            1.0,
        )

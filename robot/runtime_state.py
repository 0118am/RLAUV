"""Explicit robot runtime tensors independent of simulator APIs."""

from __future__ import annotations

import torch

from robot.dynamics.parameters import AUVModel
from robot.dynamics.rigid_body import inertia_matrix_tensor, principal_inertia_and_axes
from robot.propulsion.curves import (
    get_thruster_axes,
    get_thruster_positions,
    measured_thruster_body_forces,
    reduce_point_forces_to_wrench,
)
from robot.propulsion.dynamics import FixedStepCommandDelay, FirstOrderThrusterResponse
from robot.propulsion.effects import (
    calculate_axial_inflow_thrust_scale,
    calculate_thruster_wake_interaction_scale,
)
from robot.sensors import DelayedPoseSensor


class RobotRuntimeState:
    """All mutable vehicle and actuator state."""

    def __init__(
        self,
        cfg,
        *,
        model: AUVModel,
        num_envs: int,
        action_dim: int,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.thruster_com_offsets = get_thruster_positions(self.device)
        self.thruster_axes_b = get_thruster_axes(self.device)
        self.num_thrusters = int(self.thruster_com_offsets.shape[0])
        if action_dim != self.num_thrusters:
            raise ValueError(f"Expected {self.num_thrusters} actions, got {action_dim}.")
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_axes_b = self.thruster_axes_b.unsqueeze(0).repeat(self.num_envs, 1, 1)

        principal, axes = principal_inertia_and_axes(cfg.inertia_diag, self.device)
        nominal_body_inertia = inertia_matrix_tensor(cfg.inertia_diag, self.device)
        self.nominal_principal_inertia = principal
        self.nominal_principal_axes = axes
        self.nominal_body_inertia = nominal_body_inertia
        self.inertia_principal_moments = principal.reshape(1, 3).repeat(self.num_envs, 1)
        self.inertia_principal_axes = axes.reshape(1, 3, 3).repeat(self.num_envs, 1, 1)
        self.inertia_body_matrices = nominal_body_inertia.reshape(1, 3, 3).repeat(
            self.num_envs, 1, 1
        )
        self.masses = torch.full((self.num_envs, 1), float(cfg.mass), device=self.device)
        self.center_of_mass_offsets = torch.as_tensor(
            cfg.center_of_mass_offset, dtype=torch.float32, device=self.device
        ).reshape(1, 3).repeat(self.num_envs, 1)
        self.com_to_cob_offsets = torch.as_tensor(
            cfg.com_to_cob_offset, dtype=torch.float32, device=self.device
        ).reshape(1, 3).repeat(self.num_envs, 1)
        self.volumes = torch.full(
            (self.num_envs, 1), float(cfg.volume), dtype=torch.float32, device=self.device
        )

        self.thruster_response = FirstOrderThrusterResponse(
            self.num_envs, self.num_thrusters, cfg.dyn_time_constant, self.device
        )
        self.thruster_command_delay = FixedStepCommandDelay(
            self.num_envs,
            self.num_thrusters,
            cfg.thruster_command_delay_steps,
            self.device,
        )
        self.thruster_force_curve_coefficients = torch.as_tensor(
            model.thruster_force_curve_coefficients, dtype=torch.float32, device=self.device
        )
        self.pose_sensor = DelayedPoseSensor(
            self.num_envs,
            cfg.pose_sensor_delay_steps,
            self.device,
        )
        self.ones_thrusters = torch.ones(
            (self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device
        )
        endpoint_commands = torch.tensor(
            [[-1.0] * self.num_thrusters, [1.0] * self.num_thrusters],
            dtype=torch.float32,
            device=self.device,
        )
        endpoint_forces = measured_thruster_body_forces(
            endpoint_commands, self.thruster_force_curve_coefficients
        )
        endpoint_axial_thrust = torch.sum(
            endpoint_forces * self.thruster_axes_b[: endpoint_forces.shape[0]], dim=-1
        )
        self.thruster_wake_reference_force_n = max(
            float(torch.abs(endpoint_axial_thrust).max().item()), 1.0e-6
        )
        self.realized_thruster_force_n = torch.zeros(
            (self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device
        )
        self.realized_thruster_forces_b = torch.zeros(
            (self.num_envs, self.num_thrusters, 3), dtype=torch.float32, device=self.device
        )
        self.realized_thruster_wrench_b = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        initial_common_scale = (
            float(cfg.evaluation_thruster_force_scale)
            if bool(cfg.evaluation_thruster_force_scale_override)
            else 1.0
        )
        self.thruster_force_scale = torch.ones(
            (self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device
        )
        self.common_thruster_force_scale = torch.full(
            (self.num_envs, 1), initial_common_scale, dtype=torch.float32, device=self.device
        )
        self.thruster_time_constant = torch.full(
            (self.num_envs,), float(cfg.dyn_time_constant), dtype=torch.float32, device=self.device
        )
        self.thruster_wake_loss_coefficient = torch.full(
            (self.num_envs,), float(cfg.thruster_wake_loss_coefficient), device=self.device
        )
        self.thruster_response.set_time_constants(self.thruster_time_constant)

    def advance_thruster_forces(
        self, actions: torch.Tensor, *, physics_time_s: float
    ) -> torch.Tensor:
        delayed_commands = self.thruster_command_delay.advance(actions)
        target_forces_b = measured_thruster_body_forces(
            delayed_commands, self.thruster_force_curve_coefficients
        )
        return self.thruster_response.advance(target_forces_b, physics_time_s)

    def compose_thruster_wrench(
        self,
        forces_b: torch.Tensor,
        *,
        relative_velocity_b: torch.Tensor,
        environment_thruster_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forces_b = forces_b * (
            self.thruster_force_scale
            * self.common_thruster_force_scale
            * environment_thruster_scale
        ).unsqueeze(-1)
        axial_thrust = torch.sum(forces_b * self.thruster_axes_b, dim=-1)
        if self.cfg.thruster_inflow_loss_enabled:
            thrust_direction = torch.sign(axial_thrust).unsqueeze(-1) * self.thruster_axes_b
            axial_inflow = torch.sum(
                relative_velocity_b[:, :3].unsqueeze(1) * thrust_direction, dim=-1
            )
            inflow_scale = calculate_axial_inflow_thrust_scale(
                axial_inflow,
                self.cfg.thruster_inflow_loss_coefficient,
                self.cfg.thruster_inflow_reference_speed,
                self.cfg.thruster_inflow_min_scale,
            )
            forces_b = forces_b * inflow_scale.unsqueeze(-1)
        if self.cfg.thruster_wake_interaction_enabled:
            wake_scale = calculate_thruster_wake_interaction_scale(
                self.thruster_com_offsets,
                self.thruster_axes_b,
                axial_thrust,
                self.cfg.thruster_wake_length,
                self.cfg.thruster_wake_radius,
                self.thruster_wake_loss_coefficient,
                self.cfg.thruster_wake_expansion_rate,
                self.cfg.thruster_wake_min_scale,
                self.thruster_wake_reference_force_n,
            )
            forces_b = forces_b * wake_scale.unsqueeze(-1)
        self.realized_thruster_forces_b[:] = forces_b
        self.realized_thruster_force_n[:] = torch.linalg.vector_norm(forces_b, dim=-1)
        thruster_offsets_from_current_com = (
            self.thruster_com_offsets - self.center_of_mass_offsets.unsqueeze(1)
        )
        wrench = reduce_point_forces_to_wrench(thruster_offsets_from_current_com, forces_b)
        self.realized_thruster_wrench_b[:] = wrench
        return wrench[:, :3], wrench[:, 3:]

    def reset_dynamic_buffers(self, env_ids: torch.Tensor, *, physics_time_s: float) -> None:
        self.thruster_command_delay.reset(env_ids)
        self.thruster_response.reset(env_ids, time_s=physics_time_s)
        self.realized_thruster_force_n[env_ids] = 0.0
        self.realized_thruster_forces_b[env_ids] = 0.0
        self.realized_thruster_wrench_b[env_ids] = 0.0

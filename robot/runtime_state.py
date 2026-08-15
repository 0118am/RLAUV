"""Explicit robot runtime tensors independent of simulator APIs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robot.dynamics.parameters import AUVModel
from robot.dynamics.rigid_body import physx_principal_inertia_and_com_quat_xyzw
from robot.dynamics.tether import (
    calculate_multisegment_tether_wrench,
    update_rate_limited_winch_slack_length,
)
from robot.propulsion.curves import (
    get_thruster_positions,
    measured_thruster_body_forces,
    reduce_point_forces_to_wrench,
)
from robot.propulsion.dynamics import FirstOrderThrusterResponse, ThrusterCommandProcessor
from robot.propulsion.effects import (
    calculate_axial_inflow_thrust_scale,
    calculate_thruster_wake_interaction_scale,
    calculate_voltage_thrust_scale,
)


@dataclass(frozen=True)
class PayloadHydrodynamicScale:
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    added_mass: torch.Tensor


class RobotRuntimeState:
    """All mutable vehicle, actuator, battery, and tether state."""

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
        self.num_thrusters = int(self.thruster_com_offsets.shape[0])
        if action_dim != self.num_thrusters:
            raise ValueError(f"Expected {self.num_thrusters} actions, got {action_dim}.")
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)

        principal, axes = physx_principal_inertia_and_com_quat_xyzw(cfg.inertia_diag, self.device)
        self.nominal_principal_inertia = principal
        self.nominal_principal_axes_xyzw = axes
        self.inertia_principal_moments = principal.reshape(1, 3).repeat(self.num_envs, 1)
        self.inertia_principal_axes_xyzw = axes.reshape(1, 4).repeat(self.num_envs, 1)
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
        self.thruster_force_curve_coefficients = torch.as_tensor(
            model.thruster_force_curve_coefficients, dtype=torch.float32, device=self.device
        )
        delay_range = cfg.domain_randomization.thruster_command_delay_steps_range
        self.thruster_command_processor = ThrusterCommandProcessor(
            self.num_envs,
            self.num_thrusters,
            max(int(cfg.thruster_command_delay_steps), int(delay_range[1])),
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
        self.thruster_wake_reference_force_n = max(
            float(torch.linalg.vector_norm(endpoint_forces, dim=-1).max().item()), 1.0e-6
        )
        dropout_range = cfg.domain_randomization.thruster_command_dropout_probability_range
        self.thruster_command_dropout_enabled = (
            float(cfg.thruster_command_dropout_probability) > 0.0
            or max(float(value) for value in dropout_range) > 0.0
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
        initial_scale = (
            float(cfg.evaluation_thruster_force_scale)
            if bool(cfg.evaluation_thruster_force_scale_override)
            else 1.0
        )
        self.thruster_force_scale = torch.full(
            (self.num_envs, self.num_thrusters), initial_scale, device=self.device
        )
        self.thruster_time_constant = torch.full(
            (self.num_envs,), float(cfg.dyn_time_constant), dtype=torch.float32, device=self.device
        )
        self.thruster_delay_steps = torch.full(
            (self.num_envs,), int(cfg.thruster_command_delay_steps), dtype=torch.long, device=self.device
        )
        self.thruster_max_command_rate = torch.full(
            (self.num_envs, 1), float(cfg.thruster_max_command_rate), device=self.device
        )
        self.thruster_command_resolution = torch.full(
            (self.num_envs, 1), float(cfg.thruster_command_resolution), device=self.device
        )
        self.thruster_command_dropout_probability = torch.full(
            (self.num_envs, 1), float(cfg.thruster_command_dropout_probability), device=self.device
        )
        self.thruster_wake_loss_coefficient = torch.full(
            (self.num_envs,), float(cfg.thruster_wake_loss_coefficient), device=self.device
        )
        self.battery_initial_voltage = torch.full(
            (self.num_envs, 1), float(cfg.battery_voltage), device=self.device
        )
        self.battery_voltage = self.battery_initial_voltage.clone()
        self.battery_voltage_drop_per_s = torch.full(
            (self.num_envs, 1), float(cfg.battery_voltage_drop_per_s), device=self.device
        )
        self.battery_voltage_scale = torch.ones_like(self.battery_voltage)
        self.tether_slack_length = torch.full(
            (self.num_envs, 1), float(cfg.tether_slack_length), device=self.device
        )
        self.thruster_response.set_time_constants(self.thruster_time_constant)

        from robot.randomization.rigid_body import initialize_payload_domain

        initialize_payload_domain(self, cfg)

    def advance_battery(self, episode_time_s: torch.Tensor) -> None:
        self.battery_voltage[:] = torch.clamp(
            self.battery_initial_voltage - self.battery_voltage_drop_per_s * episode_time_s,
            min=float(self.cfg.battery_min_voltage),
        )
        self.battery_voltage_scale[:] = calculate_voltage_thrust_scale(
            self.battery_voltage,
            self.cfg.battery_voltage_nominal,
            self.cfg.battery_voltage_thrust_exponent,
            self.cfg.battery_min_voltage,
        ).to(device=self.device, dtype=torch.float32)

    def advance_thruster_forces(
        self, actions: torch.Tensor, *, physics_time_s: float, physics_dt: float
    ) -> torch.Tensor:
        commands = self.thruster_command_processor.process(
            actions,
            self.thruster_delay_steps,
            self.thruster_max_command_rate,
            physics_dt,
            self.thruster_command_resolution,
            self.thruster_command_dropout_probability,
            dropout_enabled=self.thruster_command_dropout_enabled,
        )
        targets = measured_thruster_body_forces(commands, self.thruster_force_curve_coefficients)
        return self.thruster_response.advance(targets, physics_time_s)

    @staticmethod
    def _thruster_axes(forces_b: torch.Tensor) -> torch.Tensor:
        magnitudes = torch.linalg.vector_norm(forces_b, dim=-1, keepdim=True)
        return forces_b / magnitudes.clamp_min(1.0e-8)

    def compose_thruster_wrench(
        self,
        forces_b: torch.Tensor,
        *,
        relative_velocity_b: torch.Tensor,
        environment_thruster_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forces_b = forces_b * (
            self.thruster_force_scale * self.battery_voltage_scale * environment_thruster_scale
        ).unsqueeze(-1)
        needs_axes = self.cfg.thruster_inflow_loss_enabled or self.cfg.thruster_wake_interaction_enabled
        axes = self._thruster_axes(forces_b) if needs_axes else None
        if self.cfg.thruster_inflow_loss_enabled:
            axial_inflow = torch.sum(relative_velocity_b[:, :3].unsqueeze(1) * axes, dim=-1)
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
                axes,
                torch.linalg.vector_norm(forces_b, dim=-1),
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
        wrench = reduce_point_forces_to_wrench(self.thruster_com_offsets, forces_b)
        self.realized_thruster_wrench_b[:] = wrench
        return wrench[:, :3], wrench[:, 3:]

    def compose_tether_wrench(
        self,
        kinematics,
        *,
        water_current_w: torch.Tensor,
        gravity_w: torch.Tensor,
        physics_dt: float,
        additional_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if not self.cfg.tether_enabled:
            return zeros, zeros
        if self.cfg.tether_winch_enabled:
            self.tether_slack_length[:] = update_rate_limited_winch_slack_length(
                self.tether_slack_length,
                self.cfg.tether_winch_target_length,
                self.cfg.tether_winch_reel_speed,
                physics_dt,
                self.cfg.tether_winch_min_length,
                self.cfg.tether_winch_max_length,
            )
        anchor_local = torch.as_tensor(
            self.cfg.tether_anchor_pos_w,
            dtype=kinematics.root_position_w.dtype,
            device=self.device,
        ).reshape(1, 3)
        anchor_w = kinematics.scene_origins_w + anchor_local
        from environment.hydrodynamics.tensor_ops import quat_apply_wxyz, quat_conjugate_wxyz

        force_w, torque_b = calculate_multisegment_tether_wrench(
            kinematics.root_position_w,
            kinematics.root_quat_w,
            kinematics.root_linear_velocity_w,
            water_current_w,
            anchor_w,
            self.cfg.tether_attach_offset_b,
            self.tether_slack_length,
            self.cfg.tether_stiffness,
            self.cfg.tether_damping,
            self.cfg.tether_drag_coeff,
            self.cfg.tether_num_segments,
            self.cfg.tether_segment_diameter,
            self.cfg.tether_segment_density,
            self.cfg.tether_segment_buoyancy_density,
            gravity_w,
            quat_conjugate_wxyz,
            quat_apply_wxyz,
        )
        return force_w * additional_scale, torque_b * additional_scale

    def reset_dynamic_buffers(self, env_ids: torch.Tensor) -> None:
        self.thruster_response.reset(env_ids)
        self.thruster_command_processor.reset(env_ids)
        self.realized_thruster_force_n[env_ids] = 0.0
        self.realized_thruster_forces_b[env_ids] = 0.0
        self.realized_thruster_wrench_b[env_ids] = 0.0

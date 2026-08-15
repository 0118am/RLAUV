"""Thruster, tether, and final wrench composition for the Isaac AUV."""

from __future__ import annotations

from typing import Tuple

import torch

from environment.hydrodynamics.models import (
    quat_apply_wxyz as quat_apply,
    quat_conjugate_wxyz as quat_conjugate,
)
from robot.dynamics.tether import (
    calculate_multisegment_tether_wrench,
    update_rate_limited_winch_slack_length,
)
from robot.propulsion.curves import measured_thruster_body_forces, reduce_point_forces_to_wrench
from robot.propulsion.effects import (
    calculate_axial_inflow_thrust_scale,
    calculate_reaction_torques,
    calculate_thruster_wake_interaction_scale,
)

from .hydrodynamic_state import EffectiveHydrodynamicState


class AUVForceCompositionMixin:
    """Compose actuator, tether, and hydrodynamic contributions into one wrench."""

    def _calculate_tether_wrench(self, water_current_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.tether_enabled:
            return self._runtime_zeros_env_3, self._runtime_zeros_env_3
        if water_current_w.ndim == 1:
            water_current_w = water_current_w.reshape(1, 3).repeat(self.num_envs, 1)
        if self.cfg.tether_winch_enabled:
            self.tether_slack_length[:] = update_rate_limited_winch_slack_length(
                self.tether_slack_length,
                self.cfg.tether_winch_target_length,
                self.cfg.tether_winch_reel_speed,
                self.physics_dt,
                self.cfg.tether_winch_min_length,
                self.cfg.tether_winch_max_length,
            )
        anchor_local = torch.as_tensor(
            self.cfg.tether_anchor_pos_w,
            dtype=self._robot.data.root_pos_w.dtype,
            device=self.device,
        ).reshape(1, 3)
        anchor_w = self.scene.env_origins + anchor_local
        force_w, torque_b = calculate_multisegment_tether_wrench(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
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
            self._gravity_w,
            quat_conjugate,
            quat_apply,
        )
        curriculum_scale = self._additional_hydrodynamics_scale()
        return force_w * curriculum_scale, torque_b * curriculum_scale

    @staticmethod
    def _calculate_thruster_axes_b(thruster_forces_b: torch.Tensor) -> torch.Tensor:
        """Return instantaneous measured force directions for optional effects."""

        magnitudes = torch.linalg.vector_norm(thruster_forces_b, dim=-1, keepdim=True)
        return thruster_forces_b / magnitudes.clamp_min(1.0e-8)

    def _calculate_thruster_inflow_scale(
        self,
        relative_velocity_b: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        axial_inflow_along_axis = torch.sum(
            relative_velocity_b[:, 0:3].unsqueeze(1) * thruster_axes_b,
            dim=-1,
        )
        return calculate_axial_inflow_thrust_scale(
            axial_inflow_along_axis,
            self.cfg.thruster_inflow_loss_coefficient,
            self.cfg.thruster_inflow_reference_speed,
            self.cfg.thruster_inflow_min_scale,
        )

    def _calculate_thruster_wake_scale(
        self,
        thruster_magnitudes: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.thruster_wake_interaction_enabled:
            return self._runtime_ones_thrusters
        return calculate_thruster_wake_interaction_scale(
            self.thruster_com_offsets,
            thruster_axes_b,
            thruster_magnitudes,
            self.cfg.thruster_wake_length,
            self.cfg.thruster_wake_radius,
            self.thruster_wake_loss_coefficient,
            self.cfg.thruster_wake_expansion_rate,
            self.cfg.thruster_wake_min_scale,
            self._thruster_wake_reference_force_n,
        )

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        coords /= torch.norm(coords, dim=1).unsqueeze(1)
        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1 / 3)
        return radii * coords

    def _advance_thruster_forces(
        self,
        actions: torch.Tensor,
        physics_time_s: float,
    ) -> torch.Tensor:
        """Apply command transport, the measured curve, and motor lag."""

        commands = self.thruster_command_processor.process(
            actions,
            self.thruster_delay_steps,
            self.thruster_max_command_rate,
            self.physics_dt,
            self.thruster_command_resolution,
            self.thruster_command_dropout_probability,
            dropout_enabled=self._thruster_command_dropout_enabled,
        )
        target_forces_b = measured_thruster_body_forces(
            commands,
            self._thruster_force_curve_coefficients,
        )
        return self.thruster_response.advance(target_forces_b, physics_time_s)

    def _compose_thruster_wrench(
        self,
        thruster_forces_b: torch.Tensor,
        effective_hydrodynamics: EffectiveHydrodynamicState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply actuator/environment effects and reduce point forces at COM."""

        common_scale = (
            self.thruster_force_scale
            * self._battery_voltage_scale
            * effective_hydrodynamics.thruster_scale
        )
        thruster_forces_b = thruster_forces_b * common_scale.unsqueeze(-1)
        needs_axes = (
            self.cfg.thruster_inflow_loss_enabled
            or self.cfg.thruster_wake_interaction_enabled
            or self._thruster_reaction_torque_enabled
        )
        thruster_axes_b = self._calculate_thruster_axes_b(thruster_forces_b) if needs_axes else None
        if self.cfg.thruster_inflow_loss_enabled:
            inflow_scale = self._calculate_thruster_inflow_scale(
                effective_hydrodynamics.relative_velocity_b,
                thruster_axes_b,
            )
            thruster_forces_b = thruster_forces_b * inflow_scale.unsqueeze(-1)
        if self.cfg.thruster_wake_interaction_enabled:
            wake_scale = self._calculate_thruster_wake_scale(
                torch.linalg.vector_norm(thruster_forces_b, dim=-1),
                thruster_axes_b,
            )
            thruster_forces_b = thruster_forces_b * wake_scale.unsqueeze(-1)
        magnitudes = torch.linalg.vector_norm(thruster_forces_b, dim=-1)
        self.realized_thruster_forces_b[:] = thruster_forces_b
        self.realized_thruster_force_n[:] = magnitudes
        wrench_b = reduce_point_forces_to_wrench(self.thruster_com_offsets, thruster_forces_b)
        forces_b, torques_b = wrench_b[:, 0:3], wrench_b[:, 3:6]
        if self._thruster_reaction_torque_enabled:
            assert thruster_axes_b is not None
            torques_b = torques_b + calculate_reaction_torques(
                magnitudes,
                thruster_axes_b,
                self.thruster_reaction_torque_coeff,
                self._thruster_spin_directions,
            ).sum(dim=-2)
        return forces_b, torques_b

    def _relative_acceleration_for_added_mass(
        self,
        effective_hydrodynamics: EffectiveHydrodynamicState,
    ) -> torch.Tensor | None:
        scale = float(getattr(self.cfg, "added_mass_inertia_scale", 1.0))
        if scale <= 0.0:
            return None
        return self._update_relative_acceleration_b(
            effective_hydrodynamics.relative_velocity_b
        ) * scale

    def _compose_fluid_wrench(
        self,
        effective_hydrodynamics: EffectiveHydrodynamicState,
        relative_acceleration_b: torch.Tensor | None,
        physics_time_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Fossen terms and the separately managed residual wrench."""

        volumes = self.volumes * effective_hydrodynamics.buoyancy_scale
        forces_b, torques_b = self.force_calculation_functions.calculate_fossen_fluid_forces(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._gravity_w,
            self.cfg.water_rho,
            volumes,
            self.com_to_cob_offsets,
            effective_hydrodynamics.linear_damping,
            effective_hydrodynamics.quadratic_damping,
            effective_hydrodynamics.water_current_w,
            effective_hydrodynamics.added_mass,
            relative_acceleration_b,
            False,
            self.high_order_residual_added_mass_factor,
            self.high_order_residual_linear_damping_factor,
            self.high_order_residual_quadratic_damping_factor,
            self.high_order_residual_cubic_damping_factor,
            added_mass_enabled=self._added_mass_enabled,
            relative_velocity_b=effective_hydrodynamics.relative_velocity_b,
        )
        if self.physx_hydrodynamic_wrench_manager.enabled:
            residual_wrench_b = self.physx_hydrodynamic_wrench_manager.compute_wrench(
                effective_hydrodynamics.relative_velocity_b,
                relative_acceleration_b,
                physics_time_s,
            )
            forces_b = forces_b + residual_wrench_b[:, 0:3]
            torques_b = torques_b + residual_wrench_b[:, 3:6]
        return forces_b, torques_b

    def _compute_dynamics(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compose thruster, fluid, and tether wrenches for one physics step."""

        physics_time_s = self._sim_step_counter * self.physics_dt
        raw_thruster_forces_b = self._advance_thruster_forces(actions, physics_time_s)
        effective_hydrodynamics = self._calculate_effective_hydrodynamic_state()
        self._store_effective_hydrodynamic_state(effective_hydrodynamics)
        self._pending_critic_hydrodynamic_env_ids = None
        thruster_forces, thruster_torques = self._compose_thruster_wrench(
            raw_thruster_forces_b,
            effective_hydrodynamics,
        )
        relative_acceleration_b = self._relative_acceleration_for_added_mass(effective_hydrodynamics)
        fluid_forces, fluid_torques = self._compose_fluid_wrench(
            effective_hydrodynamics,
            relative_acceleration_b,
            physics_time_s,
        )
        tether_forces, tether_torques = self._calculate_tether_wrench(
            effective_hydrodynamics.water_current_w
        )
        return (
            fluid_forces + thruster_forces + tether_forces,
            fluid_torques + thruster_torques + tether_torques,
        )

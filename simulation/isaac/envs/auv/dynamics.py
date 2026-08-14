"""Isaac-state to wrench bridge for the AUV validation vehicle.

This module obtains live state from IsaacLab, applies configuration switches,
and composes model outputs. Pure thrust, hydrodynamic, current, and tether
equations live in the shared ``robot`` and ``environment`` domains and are
called here rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

from .domain_randomization.current import update_smooth_current
from environment.water.current_fields import (
    calculate_periodic_water_current,
    calculate_trilinear_current_field,
)
from environment.water.pool_effects import (
    RectangularSloshingState,
    calculate_free_surface_scales,
    calculate_pool_boundary_scales,
    calculate_rectangular_pool_sloshing_state,
)
from environment.hydrodynamics.models import (
    calculate_speed_dependent_damping_scale,
    scale_hydrodynamic_coefficients,
)
from robot.propulsion.thrusters import (
    calculate_axial_inflow_thrust_scale,
    calculate_reaction_torques,
    calculate_thruster_wake_interaction_scale,
    calculate_voltage_thrust_scale,
    measured_thruster_body_forces,
    reduce_point_forces_to_wrench,
)
from robot.dynamics.tether import calculate_multisegment_tether_wrench, update_rate_limited_winch_slack_length

def _nominal_hydro_coeff_tensor(values, device: torch.device, name: str) -> torch.Tensor:
    """Normalize 6-DOF hydrodynamic coefficients to a single-env tensor."""

    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1 and tensor.shape[0] == 6:
        return tensor.reshape(1, 6)
    if tensor.ndim == 2:
        if tensor.shape == (6, 6):
            return tensor.reshape(1, 6, 6)
        if tensor.shape == (1, 6):
            return tensor
    if tensor.ndim == 3 and tensor.shape == (1, 6, 6):
        return tensor
    raise ValueError(f"{name} must be a 6-vector or 6x6 matrix, got shape {tuple(tensor.shape)}.")


def _repeat_hydro_coeff_for_envs(nominal: torch.Tensor, num_envs: int) -> torch.Tensor:
    repeats = (num_envs,) + tuple(1 for _ in range(nominal.ndim - 1))
    return nominal.repeat(repeats)


@dataclass
class EffectiveHydrodynamicState:
    """Effective quantities shared by the force path and asymmetric Critic."""

    water_current_w: torch.Tensor
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    added_mass: torch.Tensor
    buoyancy_scale: torch.Tensor
    thruster_scale: torch.Tensor


class AUVDynamicsMixin:
    """Composes configured environmental effects into body force and torque."""

    def _additional_hydrodynamics_scale(self) -> float:
        """Return the curriculum multiplier for modeled pool-only effects."""

        if not self._domain_randomization_enabled():
            return 1.0
        scales = getattr(
            self.cfg.domain_randomization,
            "additional_hydrodynamics_scale_by_stage",
            [1.0],
        )
        return float(scales[self._get_disturbance_curriculum_stage()])

    def _update_smooth_water_current(self) -> None:
        update_smooth_current(self)

    def _calculate_water_current_w(
        self,
        sloshing_state: RectangularSloshingState | None = None,
        *,
        local_positions: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        additional_scale: float | None = None,
    ) -> torch.Tensor:
        if local_positions is None:
            local_positions = self._robot.data.root_pos_w - self.scene.env_origins
            if env_ids is not None:
                local_positions = local_positions[env_ids]
        total_current_w = self.water_current_w if env_ids is None else self.water_current_w[env_ids]
        if additional_scale is None:
            additional_scale = self._additional_hydrodynamics_scale()
        if self.cfg.water_current_periodic_enabled and additional_scale != 0.0:
            periodic_current_w = calculate_periodic_water_current(
                self._sim_step_counter * self.physics_dt,
                self._periodic_current_amplitude_w,
                self._periodic_current_period_s,
                self._periodic_current_phase_rad,
            )
            total_current_w = total_current_w + additional_scale * periodic_current_w
        if self.cfg.water_current_field_enabled:
            field_current_w = calculate_trilinear_current_field(
                local_positions,
                self._current_field_bounds,
                self._current_field_shape,
                self._current_field_values,
                validate=False,
            )
            total_current_w = total_current_w + field_current_w
        if sloshing_state is None:
            sloshing_state = self._calculate_surface_sloshing_state(
                local_positions,
                env_ids=env_ids,
                additional_scale=additional_scale,
            )
        if env_ids is None:
            self._current_free_surface_z[:] = sloshing_state.surface_z
        else:
            self._current_free_surface_z[env_ids] = sloshing_state.surface_z
        return total_current_w + sloshing_state.orbital_velocity_w

    def _calculate_surface_sloshing_state(
        self,
        local_positions: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        additional_scale: float | None = None,
    ) -> RectangularSloshingState:
        if additional_scale is None:
            additional_scale = self._additional_hydrodynamics_scale()
        if not self.cfg.free_surface_sloshing_enabled or additional_scale == 0.0:
            count = local_positions.shape[0]
            zeros = self._runtime_zeros_env_1[:count]
            return RectangularSloshingState(
                surface_z=self._runtime_flat_surface_z[:count],
                elevation_up_m=zeros,
                orbital_velocity_w=self._runtime_zeros_env_3[:count],
                angular_frequencies_rad_s=self._runtime_empty_frequencies,
            )
        return calculate_rectangular_pool_sloshing_state(
            local_positions,
            self._sim_step_counter * self.physics_dt,
            self.cfg.free_surface_z,
            self._sloshing_pool_bounds,
            self.cfg.free_surface_sloshing_water_depth,
            self._sloshing_mode_numbers,
            self._sloshing_amplitudes_m * additional_scale,
            self._sloshing_phases_rad,
            float(self._gravity_magnitude),
            self.cfg.free_surface_sloshing_depth_axis_sign,
            angular_frequencies_rad_s=self._sloshing_angular_frequencies_rad_s,
            validate=False,
        )

    def _update_relative_acceleration_b(self, water_current_w: torch.Tensor) -> torch.Tensor:
        """Estimate filtered body-frame ``dot(nu_r)`` for added-mass inertia."""

        nu_r = self.force_calculation_functions.calculate_relative_velocity(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            water_current_w,
        )
        if nu_r.shape[0] != water_current_w.shape[0]:
            raise RuntimeError("Relative-acceleration state and water-current batch sizes differ.")
        has_previous = self._has_previous_nu_r.unsqueeze(-1)
        previous_nu_r = torch.where(has_previous, self._previous_nu_r, nu_r)
        raw_nu_r_dot = (nu_r - previous_nu_r) / max(float(self.physics_dt), 1.0e-6)

        alpha = self._added_mass_accel_filter_alpha
        self._filtered_nu_r_dot[:] = alpha * raw_nu_r_dot + (1.0 - alpha) * self._filtered_nu_r_dot
        self._filtered_nu_r_dot[:] = torch.where(
            has_previous,
            self._filtered_nu_r_dot,
            torch.zeros_like(self._filtered_nu_r_dot),
        )
        self._previous_nu_r[:] = nu_r
        self._has_previous_nu_r[:] = True
        return self._filtered_nu_r_dot

    def _update_battery_voltage_scale(self) -> torch.Tensor:
        episode_time = (
            self.episode_length_buf.to(dtype=torch.float32).reshape(self.num_envs, 1)
            * self.physics_dt
            * self.cfg.decimation
        )
        self.battery_voltage[:] = torch.clamp(
            self.battery_initial_voltage - self.battery_voltage_drop_per_s * episode_time,
            min=self.cfg.battery_min_voltage,
        )
        scale = calculate_voltage_thrust_scale(
            self.battery_voltage,
            self.cfg.battery_voltage_nominal,
            self.cfg.battery_voltage_thrust_exponent,
            self.cfg.battery_min_voltage,
        )
        return scale.to(device=self.device, dtype=torch.float32)

    def _calculate_pool_boundary_scales(
        self,
        local_positions: torch.Tensor | None = None,
        *,
        additional_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if local_positions is None:
            local_positions = self._robot.data.root_pos_w - self.scene.env_origins
        ones = self._runtime_ones_env_1[: local_positions.shape[0]]
        if additional_scale is None:
            additional_scale = self._additional_hydrodynamics_scale()
        if not self.cfg.pool_boundary_effects_enabled or additional_scale == 0.0:
            return ones, ones, ones

        scales = calculate_pool_boundary_scales(
            local_positions,
            self._pool_bounds,
            self.cfg.pool_boundary_effect_distance,
            self.cfg.pool_boundary_damping_scale,
            self.cfg.pool_boundary_added_mass_scale,
            self.cfg.pool_boundary_thrust_scale,
        )
        return tuple(1.0 + additional_scale * (value - 1.0) for value in scales)

    def _calculate_free_surface_scales(
        self,
        sloshing_state: RectangularSloshingState | None = None,
        *,
        local_positions: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        additional_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if local_positions is None:
            local_positions = self._robot.data.root_pos_w - self.scene.env_origins
            if env_ids is not None:
                local_positions = local_positions[env_ids]
        count = local_positions.shape[0]
        ones_6 = self._runtime_ones_env_6[:count]
        ones_1 = self._runtime_ones_env_1[:count]
        if additional_scale is None:
            additional_scale = self._additional_hydrodynamics_scale()
        if not self.cfg.free_surface_effects_enabled or additional_scale == 0.0:
            return ones_6, ones_6, ones_1, ones_1

        if sloshing_state is None:
            sloshing_state = self._calculate_surface_sloshing_state(
                local_positions,
                env_ids=env_ids,
                additional_scale=additional_scale,
            )
        if env_ids is None:
            self._current_free_surface_z[:] = sloshing_state.surface_z
        else:
            self._current_free_surface_z[env_ids] = sloshing_state.surface_z
        scales = calculate_free_surface_scales(
            local_positions,
            sloshing_state.surface_z,
            self.cfg.free_surface_effect_distance,
            self.cfg.free_surface_heave_damping_scale,
            self.cfg.free_surface_roll_pitch_damping_scale,
            self.cfg.free_surface_added_mass_scale,
            self.cfg.free_surface_buoyancy_scale,
            self.cfg.free_surface_thrust_scale,
            self.cfg.free_surface_sloshing_depth_axis_sign,
        )
        return tuple(1.0 + additional_scale * (value - 1.0) for value in scales)

    def _calculate_speed_dependent_damping_scales(
        self,
        water_current_w: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ones = self._runtime_ones_env_6[: water_current_w.shape[0]]
        if not self.cfg.speed_dependent_damping_enabled:
            return ones, ones

        has_linear_curve = len(self.cfg.linear_damping_speed_scales) > 0
        has_quadratic_curve = len(self.cfg.quadratic_damping_speed_scales) > 0
        if not has_linear_curve and not has_quadratic_curve:
            raise ValueError(
                "At least one damping speed scale curve must be provided when "
                "speed_dependent_damping_enabled=True."
            )

        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        if env_ids is not None:
            root_quat_w = root_quat_w[env_ids]
            root_lin_vel_b = root_lin_vel_b[env_ids]
            root_ang_vel_b = root_ang_vel_b[env_ids]
        nu_r = self.force_calculation_functions.calculate_relative_velocity(
            root_quat_w,
            root_lin_vel_b,
            root_ang_vel_b,
            water_current_w,
        )
        if nu_r.shape[0] != water_current_w.shape[0]:
            raise RuntimeError("Speed-dependent damping state and water-current batch sizes differ.")
        linear_scale = ones
        quadratic_scale = ones
        if has_linear_curve:
            linear_scale = calculate_speed_dependent_damping_scale(
                nu_r,
                self._damping_speed_points,
                self._linear_damping_speed_scales,
                validate=False,
            )
        if has_quadratic_curve:
            quadratic_scale = calculate_speed_dependent_damping_scale(
                nu_r,
                self._damping_speed_points,
                self._quadratic_damping_speed_scales,
                validate=False,
            )
        return (
            linear_scale
            * (
                self.damping_speed_linear_randomization_scale
                if env_ids is None
                else self.damping_speed_linear_randomization_scale[env_ids]
            ),
            quadratic_scale
            * (
                self.damping_speed_quadratic_randomization_scale
                if env_ids is None
                else self.damping_speed_quadratic_randomization_scale[env_ids]
            ),
        )

    def _calculate_effective_hydrodynamic_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> EffectiveHydrodynamicState:
        """Evaluate the force-path state once for all or selected environments."""

        local_positions = self._robot.data.root_pos_w - self.scene.env_origins
        if env_ids is not None:
            local_positions = local_positions[env_ids]
        additional_scale = self._additional_hydrodynamics_scale()
        sloshing_state = self._calculate_surface_sloshing_state(
            local_positions,
            env_ids=env_ids,
            additional_scale=additional_scale,
        )
        water_current_w = self._calculate_water_current_w(
            sloshing_state,
            local_positions=local_positions,
            env_ids=env_ids,
            additional_scale=additional_scale,
        )
        pool_damping_scale, pool_added_mass_scale, pool_thruster_scale = (
            self._calculate_pool_boundary_scales(
                local_positions,
                additional_scale=additional_scale,
            )
        )
        (
            surface_damping_scale,
            surface_added_mass_scale,
            surface_buoyancy_scale,
            surface_thruster_scale,
        ) = self._calculate_free_surface_scales(
            sloshing_state,
            local_positions=local_positions,
            env_ids=env_ids,
            additional_scale=additional_scale,
        )
        linear_speed_scale, quadratic_speed_scale = self._calculate_speed_dependent_damping_scales(
            water_current_w,
            env_ids=env_ids,
        )
        linear_damping = self.linear_damping if env_ids is None else self.linear_damping[env_ids]
        quadratic_damping = self.quadratic_damping if env_ids is None else self.quadratic_damping[env_ids]
        added_mass = self.added_mass_diag if env_ids is None else self.added_mass_diag[env_ids]
        damping_scale = pool_damping_scale * surface_damping_scale
        return EffectiveHydrodynamicState(
            water_current_w=water_current_w,
            linear_damping=scale_hydrodynamic_coefficients(
                linear_damping,
                damping_scale * linear_speed_scale,
            ),
            quadratic_damping=scale_hydrodynamic_coefficients(
                quadratic_damping,
                damping_scale * quadratic_speed_scale,
            ),
            added_mass=scale_hydrodynamic_coefficients(
                added_mass,
                pool_added_mass_scale * surface_added_mass_scale,
            ),
            buoyancy_scale=surface_buoyancy_scale,
            thruster_scale=pool_thruster_scale * surface_thruster_scale,
        )

    def _store_effective_hydrodynamic_state(
        self,
        state: EffectiveHydrodynamicState,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None or self._effective_hydrodynamic_state is None:
            self._effective_hydrodynamic_state = state
        else:
            cached = self._effective_hydrodynamic_state
            # Identity stage-zero states are views of preallocated tensors.
            # Clone only reset rows so an in-place scatter never aliases its
            # source; the full physics-step cache remains allocation-free.
            cached.water_current_w[env_ids] = state.water_current_w.clone()
            cached.linear_damping[env_ids] = state.linear_damping.clone()
            cached.quadratic_damping[env_ids] = state.quadratic_damping.clone()
            cached.added_mass[env_ids] = state.added_mass.clone()
            cached.buoyancy_scale[env_ids] = state.buoyancy_scale.clone()
            cached.thruster_scale[env_ids] = state.thruster_scale.clone()

    def _effective_hydrodynamic_state_for_critic(self) -> EffectiveHydrodynamicState:
        """Return cached force-path state, refreshing only freshly reset rows."""

        if self._effective_hydrodynamic_state is None:
            self._store_effective_hydrodynamic_state(self._calculate_effective_hydrodynamic_state())
        elif self._pending_critic_hydrodynamic_env_ids is not None:
            env_ids = self._pending_critic_hydrodynamic_env_ids
            self._store_effective_hydrodynamic_state(
                self._calculate_effective_hydrodynamic_state(env_ids),
                env_ids,
            )
        self._pending_critic_hydrodynamic_env_ids = None
        return self._effective_hydrodynamic_state

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

    def _calculate_thruster_axial_inflow(
        self,
        water_current_w: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        nu_r = self.force_calculation_functions.calculate_relative_velocity(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            water_current_w,
        )
        relative_linvel_b = nu_r[:, 0:3]
        return torch.sum(relative_linvel_b.unsqueeze(1) * thruster_axes_b, dim=-1)

    def _calculate_thruster_inflow_scale(
        self,
        thruster_magnitudes: torch.Tensor,
        water_current_w: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.thruster_inflow_loss_enabled:
            return self._runtime_ones_thrusters

        axial_inflow_along_axis = self._calculate_thruster_axial_inflow(water_current_w, thruster_axes_b)
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

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms

        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1/3)

        return radii * coords

    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamics from normalized measured T60 commands.

        ``-1`` maps to 1300 µs and ``+1`` maps to 1700 µs.  Each T1...T8
        command evaluates one measured body-frame ``(Fx, Fy, Fz)`` curve;
        there is no separate command polarity or fixed thrust direction.

        Args:
            actions (torch.Tensor): Actions shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Forces sent to the simulation
            [torch.Tensor]: Torques sent to the simulation
        """

        if self._debug: print("actions: ", actions)

        thruster_commands = actions

        if self._debug: print("thruster commands: ", thruster_commands)

        thruster_commands = self.thruster_command_processor.update(
            thruster_commands,
            self.thruster_delay_steps,
            self.thruster_max_command_rate,
            self.physics_dt,
            self.thruster_command_resolution,
            self.thruster_command_dropout_probability,
            dropout_enabled=self._thruster_command_dropout_enabled,
        )

        thruster_force_cmd_b = measured_thruster_body_forces(
            thruster_commands,
            self._thruster_force_curve_coefficients,
        )

        # Update first-order thrust dynamics at the actual physics clock.
        # DirectRLEnv may run several physics steps per policy step (decimation),
        # so episode_length_buf * dt would under-count time and freeze dynamics
        # inside the decimation loop.
        physics_time = self._sim_step_counter * self.physics_dt
        thruster_forces_b = self.thruster_dynamics.update(thruster_force_cmd_b, physics_time)

        voltage_scale = self._update_battery_voltage_scale()
        effective_hydrodynamics = self._calculate_effective_hydrodynamic_state()
        self._store_effective_hydrodynamic_state(effective_hydrodynamics)
        self._pending_critic_hydrodynamic_env_ids = None
        water_current_w = effective_hydrodynamics.water_current_w
        common_thruster_scale = (
            self.thruster_force_scale
            * voltage_scale
            * effective_hydrodynamics.thruster_scale
        )
        thruster_forces_b = thruster_forces_b * common_thruster_scale.unsqueeze(-1)
        thruster_magnitudes = torch.linalg.vector_norm(thruster_forces_b, dim=-1)
        thruster_axes_b = self._calculate_thruster_axes_b(thruster_forces_b)
        inflow_scale = self._calculate_thruster_inflow_scale(
            thruster_magnitudes,
            water_current_w,
            thruster_axes_b,
        )
        thruster_forces_b = thruster_forces_b * inflow_scale.unsqueeze(-1)
        thruster_magnitudes = torch.linalg.vector_norm(thruster_forces_b, dim=-1)

        wake_scale = self._calculate_thruster_wake_scale(thruster_magnitudes, thruster_axes_b)
        thruster_forces_b = thruster_forces_b * wake_scale.unsqueeze(-1)
        thruster_magnitudes = torch.linalg.vector_norm(thruster_forces_b, dim=-1)
        # Persist the exact actuator force passed to the wrench composition.
        # This is exposed only to the asymmetric training Critic; the Actor
        # receives the deployable, rate-limited normalized command instead.
        self.realized_thruster_forces_b[:] = thruster_forces_b
        self.realized_thruster_force_n[:] = thruster_magnitudes

        # IsaacLab receives one body-frame wrench at the RigidObject COM, so
        # reduce the eight point forces here.  The COM-relative arms enter
        # exactly once through r x F; do not pass positions downstream.
        thruster_wrench_b = reduce_point_forces_to_wrench(self.thruster_com_offsets, thruster_forces_b)
        thruster_forces = thruster_wrench_b[:, 0:3]
        thruster_torques = thruster_wrench_b[:, 3:6]
        if self._thruster_reaction_torque_enabled:
            thruster_torques = thruster_torques + calculate_reaction_torques(
                thruster_magnitudes,
                thruster_axes_b,
                self.thruster_reaction_torque_coeff,
                self._thruster_spin_directions,
            ).sum(dim=-2)

        added_mass_inertia_scale = float(getattr(self.cfg, "added_mass_inertia_scale", 1.0))
        relative_acceleration_b = self._update_relative_acceleration_b(water_current_w)
        if added_mass_inertia_scale <= 0.0:
            relative_acceleration_b = None
        else:
            relative_acceleration_b = relative_acceleration_b * added_mass_inertia_scale

        ## Calculate hydrodynamics
        if self._debug: print("gravity magnitude: ", self._gravity_magnitude)
        volumes = self.volumes * effective_hydrodynamics.buoyancy_scale
        fluid_forces, fluid_torques = self.force_calculation_functions.calculate_fossen_fluid_forces(
          self._robot.data.root_quat_w,
          self._robot.data.root_lin_vel_b,
          self._robot.data.root_ang_vel_b,
          self._gravity_w,
          self.cfg.water_rho,
          volumes,
          self.com_to_cob_offsets,
          effective_hydrodynamics.linear_damping,
          effective_hydrodynamics.quadratic_damping,
          water_current_w,
          effective_hydrodynamics.added_mass,
          relative_acceleration_b,
          # High-order terms are managed separately and added below as an
          # explicit external PhysX wrench. This avoids double application.
          False,
          self.high_order_residual_added_mass_factor,
          self.high_order_residual_linear_damping_factor,
          self.high_order_residual_quadratic_damping_factor,
          self.high_order_residual_cubic_damping_factor,
          added_mass_enabled=self._added_mass_enabled,
        )

        nu_relative_b = self.force_calculation_functions.calculate_relative_velocity(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            water_current_w,
        )
        high_order_wrench_b = self.physx_hydrodynamic_wrench_manager.compute_wrench(
            nu_relative_b,
            relative_acceleration_b,
            physics_time,
        )
        fluid_forces = fluid_forces + high_order_wrench_b[:, 0:3]
        fluid_torques = fluid_torques + high_order_wrench_b[:, 3:6]
        if self._debug: print("fluid forces: ", fluid_forces)
        if self._debug: print("fluid torques: ", fluid_torques)

        if self._debug: print("thruster forces: ", thruster_forces)
        if self._debug: print("thruster torques: ", thruster_torques)

        tether_forces, tether_torques = self._calculate_tether_wrench(water_current_w)

        forces = fluid_forces + thruster_forces + tether_forces
        torques = fluid_torques + thruster_torques + tether_torques

        if self._debug: print("final forces", forces)
        if self._debug: print("final torques", torques)

        return forces, torques

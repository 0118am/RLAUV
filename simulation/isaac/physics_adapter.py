"""Isaac-state to wrench and PhysX bridge for the AUV validation vehicle.

This module obtains live state from IsaacLab, applies configuration switches,
and composes model outputs. Pure thrust, hydrodynamic, current, and tether
equations live in the shared ``robot`` and ``environment`` domains and are
called here rather than reimplemented.
"""

from __future__ import annotations

import torch

from environment.randomization.current import update_smooth_current
from environment.hydrodynamics.current_fields import (
    calculate_periodic_water_current,
    calculate_trilinear_current_field,
)
from environment.hydrodynamics.pool_effects import (
    RectangularSloshingState,
    calculate_free_surface_scales,
    calculate_pool_boundary_scales,
    calculate_rectangular_pool_sloshing_state,
)
from environment.hydrodynamics.models import (
    calculate_speed_dependent_damping_scale,
    scale_hydrodynamic_coefficients,
)
from robot.propulsion.effects import calculate_voltage_thrust_scale
from .dynamics_state import AUVDynamicsStateMixin
from .force_composition import AUVForceCompositionMixin
from .hydrodynamic_state import (
    EffectiveHydrodynamicState,
    PhysxHydrodynamicWrenchCfg,
    PhysxHydrodynamicWrenchManager,
)


class AUVDynamicsMixin(AUVForceCompositionMixin, AUVDynamicsStateMixin):
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
        update_smooth_current(
            self,
            enabled=self._domain_randomization_feature_enabled("current"),
        )

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
        if not self.cfg.free_surface_sloshing_enabled or additional_scale == 0.0:
            return total_current_w
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

    def _update_relative_acceleration_b(self, nu_r: torch.Tensor) -> torch.Tensor:
        """Estimate filtered body-frame ``dot(nu_r)`` for added-mass inertia."""

        if nu_r.shape != self._previous_nu_r.shape:
            raise RuntimeError("Relative-acceleration state has an unexpected batch shape.")
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

    def _update_battery_voltage_scale(self) -> None:
        episode_time = (
            self.episode_length_buf.to(dtype=torch.float32).reshape(self.num_envs, 1)
            * self.physics_dt
            * self.cfg.decimation
        )
        self.battery_voltage[:] = torch.clamp(
            self.battery_initial_voltage - self.battery_voltage_drop_per_s * episode_time,
            min=self.cfg.battery_min_voltage,
        )
        self._battery_voltage_scale[:] = calculate_voltage_thrust_scale(
            self.battery_voltage,
            self.cfg.battery_voltage_nominal,
            self.cfg.battery_voltage_thrust_exponent,
            self.cfg.battery_min_voltage,
        ).to(device=self.device, dtype=torch.float32)

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
        nu_r: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ones = self._runtime_ones_env_6[: nu_r.shape[0]]
        if not self.cfg.speed_dependent_damping_enabled:
            return ones, ones

        has_linear_curve = len(self.cfg.linear_damping_speed_scales) > 0
        has_quadratic_curve = len(self.cfg.quadratic_damping_speed_scales) > 0
        if not has_linear_curve and not has_quadratic_curve:
            raise ValueError(
                "At least one damping speed scale curve must be provided when "
                "speed_dependent_damping_enabled=True."
            )

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

    def _hydrodynamic_local_positions(
        self,
        env_ids: torch.Tensor | None,
        additional_scale: float,
    ) -> torch.Tensor:
        spatial_effects_active = bool(
            self.cfg.water_current_field_enabled
            or additional_scale != 0.0
            and (
                self.cfg.free_surface_sloshing_enabled
                or self.cfg.pool_boundary_effects_enabled
                or self.cfg.free_surface_effects_enabled
            )
        )
        count = self.num_envs if env_ids is None else env_ids.numel()
        if not spatial_effects_active:
            return self._runtime_zeros_env_3[:count]
        positions = self._robot.data.root_pos_w - self.scene.env_origins
        return positions if env_ids is None else positions[env_ids]

    def _selected_root_kinematics(
        self,
        env_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = (
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )
        if env_ids is None:
            return root_state
        return tuple(value[env_ids] for value in root_state)

    def _effective_hydrodynamic_coefficients(
        self,
        relative_velocity_b: torch.Tensor,
        sloshing_state: RectangularSloshingState,
        local_positions: torch.Tensor,
        env_ids: torch.Tensor | None,
        additional_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pool_damping, pool_added_mass, pool_thruster = self._calculate_pool_boundary_scales(
            local_positions,
            additional_scale=additional_scale,
        )
        surface_damping, surface_added_mass, buoyancy, surface_thruster = (
            self._calculate_free_surface_scales(
                sloshing_state,
                local_positions=local_positions,
                env_ids=env_ids,
                additional_scale=additional_scale,
            )
        )
        linear_scale, quadratic_scale = self._calculate_speed_dependent_damping_scales(
            relative_velocity_b,
            env_ids=env_ids,
        )
        linear = self.linear_damping if env_ids is None else self.linear_damping[env_ids]
        quadratic = self.quadratic_damping if env_ids is None else self.quadratic_damping[env_ids]
        added_mass = self.added_mass_diag if env_ids is None else self.added_mass_diag[env_ids]
        boundary_effects_active = bool(
            additional_scale != 0.0
            and (self.cfg.pool_boundary_effects_enabled or self.cfg.free_surface_effects_enabled)
        )
        damping_effects_active = bool(
            self.cfg.speed_dependent_damping_enabled or boundary_effects_active
        )
        if boundary_effects_active:
            linear_scale = linear_scale * pool_damping * surface_damping
            quadratic_scale = quadratic_scale * pool_damping * surface_damping
            added_mass = scale_hydrodynamic_coefficients(
                added_mass,
                pool_added_mass * surface_added_mass,
            )
        if damping_effects_active:
            linear = scale_hydrodynamic_coefficients(linear, linear_scale)
            quadratic = scale_hydrodynamic_coefficients(quadratic, quadratic_scale)
        thruster = (
            pool_thruster * surface_thruster
            if boundary_effects_active
            else self._runtime_ones_env_1[: relative_velocity_b.shape[0]]
        )
        return linear, quadratic, added_mass, buoyancy, thruster

    def _calculate_effective_hydrodynamic_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> EffectiveHydrodynamicState:
        """Evaluate the force-path state once for all or selected environments."""

        additional_scale = self._additional_hydrodynamics_scale()
        local_positions = self._hydrodynamic_local_positions(env_ids, additional_scale)
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
        root_quat_w, root_lin_vel_b, root_ang_vel_b = self._selected_root_kinematics(env_ids)
        relative_velocity_b = self.force_calculation_functions.calculate_relative_velocity(
            root_quat_w,
            root_lin_vel_b,
            root_ang_vel_b,
            water_current_w,
        )
        (
            linear_damping,
            quadratic_damping,
            added_mass,
            buoyancy_scale,
            thruster_scale,
        ) = self._effective_hydrodynamic_coefficients(
            relative_velocity_b,
            sloshing_state,
            local_positions,
            env_ids,
            additional_scale,
        )
        return EffectiveHydrodynamicState(
            water_current_w=water_current_w,
            relative_velocity_b=relative_velocity_b,
            linear_damping=linear_damping,
            quadratic_damping=quadratic_damping,
            added_mass=added_mass,
            buoyancy_scale=buoyancy_scale,
            thruster_scale=thruster_scale,
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
            # Identity fast paths may cache aliases of nominal/runtime state.
            # Detach the full cache before the rare reset-row scatter so base
            # coefficients and current state are never mutated indirectly.
            for name in (
                "water_current_w",
                "relative_velocity_b",
                "linear_damping",
                "quadratic_damping",
                "added_mass",
                "buoyancy_scale",
                "thruster_scale",
            ):
                target = getattr(cached, name).clone()
                target[env_ids] = getattr(state, name)
                setattr(cached, name, target)

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

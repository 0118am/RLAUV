"""Explicit hydrodynamic evaluation over an EnvironmentRuntimeState."""

from __future__ import annotations

import torch

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
from environment.hydrodynamics.tensor_ops import (
    calculate_speed_dependent_damping_scale,
    scale_hydrodynamic_coefficients,
)
from .effective_state import BodyKinematics, EffectiveHydrodynamicState
from .state import _EnvironmentRuntimeBuffers


class EnvironmentRuntimeState(_EnvironmentRuntimeBuffers):
    def advance_smooth_current(self, *, stage: int, enabled: bool, policy_dt: float) -> None:
        from environment.randomization.current import update_smooth_current
        update_smooth_current(self, self.cfg, stage, policy_dt, enabled=enabled)

    def calculate_surface_sloshing_state(
        self, local_positions, *, env_ids, additional_scale: float, sim_time_s: float
    ) -> RectangularSloshingState:
        count = local_positions.shape[0]
        if not self.cfg.free_surface_sloshing_enabled or additional_scale == 0.0:
            state = RectangularSloshingState(
                self.flat_surface_z[:count],
                self.zeros_env_1[:count],
                self.zeros_env_3[:count],
                self.sloshing_angular_frequencies_rad_s,
            )
        else:
            state = calculate_rectangular_pool_sloshing_state(
                local_positions, sim_time_s, self.cfg.free_surface_z, self.sloshing_pool_bounds,
                self.cfg.free_surface_sloshing_water_depth, self.sloshing_mode_numbers,
                self.sloshing_amplitudes_m, self.sloshing_phases_rad,
                float(self.gravity_magnitude), self.cfg.free_surface_sloshing_depth_axis_sign,
                angular_frequencies_rad_s=self.sloshing_angular_frequencies_rad_s, validate=False,
            )
            if additional_scale != 1.0:
                flat = self.flat_surface_z[:count]
                state = RectangularSloshingState(
                    flat + additional_scale * (state.surface_z - flat),
                    additional_scale * state.elevation_up_m,
                    additional_scale * state.orbital_velocity_w,
                    state.angular_frequencies_rad_s,
                )
        if env_ids is None:
            self.current_free_surface_z[:] = state.surface_z
        else:
            self.current_free_surface_z[env_ids] = state.surface_z
        return state

    def calculate_water_current_w(
        self, positions, sloshing, *, env_ids, additional_scale: float, sim_time_s: float
    ):
        current = self.water_current_w if env_ids is None else self.water_current_w[env_ids]
        if self.cfg.water_current_periodic_enabled and additional_scale != 0.0:
            current = current + additional_scale * calculate_periodic_water_current(
                sim_time_s, self.periodic_current_amplitude_w,
                self.periodic_current_period_s, self.periodic_current_phase_rad,
            )
        if self.cfg.water_current_field_enabled:
            current = current + calculate_trilinear_current_field(
                positions, self.current_field_bounds, self.current_field_shape,
                self.current_field_values, validate=False,
            )
        if self.cfg.free_surface_sloshing_enabled and additional_scale != 0.0:
            current = current + sloshing.orbital_velocity_w
        return current

    def update_relative_acceleration(self, nu_r, *, physics_dt: float):
        raw = (nu_r - self.previous_nu_r) / physics_dt
        valid = self.has_previous_nu_r.unsqueeze(-1)
        raw = torch.where(valid, raw, torch.zeros_like(raw))
        alpha = self.added_mass_accel_filter_alpha
        filtered = alpha * self.filtered_nu_r_dot + (1.0 - alpha) * raw
        self.filtered_nu_r_dot[:] = torch.where(valid, filtered, torch.zeros_like(filtered))
        self.previous_nu_r[:] = nu_r
        self.has_previous_nu_r[:] = True
        return self.filtered_nu_r_dot

    def _pool_scales(self, positions, additional_scale):
        ones = self.ones_env_1[: positions.shape[0]]
        if not self.cfg.pool_boundary_effects_enabled or additional_scale == 0.0:
            return ones, ones, ones
        scales = calculate_pool_boundary_scales(
            positions, self.pool_bounds, self.cfg.pool_boundary_effect_distance,
            self.cfg.pool_boundary_damping_scale, self.cfg.pool_boundary_added_mass_scale,
            self.cfg.pool_boundary_thrust_scale,
        )
        return tuple(1.0 + additional_scale * (value - 1.0) for value in scales)

    def _surface_scales(self, positions, sloshing, additional_scale):
        count = positions.shape[0]
        if not self.cfg.free_surface_effects_enabled or additional_scale == 0.0:
            return self.ones_env_6[:count], self.ones_env_6[:count], self.ones_env_1[:count], self.ones_env_1[:count]
        scales = calculate_free_surface_scales(
            positions, sloshing.surface_z, self.cfg.free_surface_effect_distance,
            self.cfg.free_surface_heave_damping_scale, self.cfg.free_surface_roll_pitch_damping_scale,
            self.cfg.free_surface_added_mass_scale, self.cfg.free_surface_buoyancy_scale,
            self.cfg.free_surface_thrust_scale, self.cfg.free_surface_sloshing_depth_axis_sign,
        )
        return tuple(1.0 + additional_scale * (value - 1.0) for value in scales)

    def _speed_scales(self, nu_r, env_ids):
        linear = quadratic = self.ones_env_6[: nu_r.shape[0]]
        if self.cfg.speed_dependent_damping_enabled:
            if not len(self.cfg.linear_damping_speed_scales) and not len(
                self.cfg.quadratic_damping_speed_scales
            ):
                raise ValueError(
                    "speed_dependent_damping_enabled requires at least one damping scale table."
                )
            if len(self.cfg.linear_damping_speed_scales):
                linear = calculate_speed_dependent_damping_scale(
                    nu_r, self.damping_speed_points, self.linear_damping_speed_scales, validate=False
                )
            if len(self.cfg.quadratic_damping_speed_scales):
                quadratic = calculate_speed_dependent_damping_scale(
                    nu_r, self.damping_speed_points, self.quadratic_damping_speed_scales, validate=False
                )
        linear_random = self.damping_speed_linear_randomization_scale
        quadratic_random = self.damping_speed_quadratic_randomization_scale
        if env_ids is not None:
            linear_random, quadratic_random = linear_random[env_ids], quadratic_random[env_ids]
        return linear * linear_random, quadratic * quadratic_random

    def calculate_effective_state(
        self, kinematics: BodyKinematics, *, sim_time_s: float, additional_scale: float,
        env_ids: torch.Tensor | None = None,
    ) -> EffectiveHydrodynamicState:
        positions = kinematics.root_position_local_w
        root_quat = kinematics.root_quat_w
        linear_velocity = kinematics.root_linear_velocity_b
        angular_velocity = kinematics.root_angular_velocity_b
        if env_ids is not None:
            positions, root_quat = positions[env_ids], root_quat[env_ids]
            linear_velocity, angular_velocity = linear_velocity[env_ids], angular_velocity[env_ids]
        sloshing = self.calculate_surface_sloshing_state(
            positions, env_ids=env_ids, additional_scale=additional_scale, sim_time_s=sim_time_s
        )
        current = self.calculate_water_current_w(
            positions, sloshing, env_ids=env_ids,
            additional_scale=additional_scale, sim_time_s=sim_time_s,
        )
        nu_r = self.force_models.calculate_relative_velocity(
            root_quat, linear_velocity, angular_velocity, current
        )
        pool_damping, pool_added_mass, pool_thruster = self._pool_scales(positions, additional_scale)
        surface_damping, surface_added_mass, buoyancy, surface_thruster = self._surface_scales(
            positions, sloshing, additional_scale
        )
        linear_scale, quadratic_scale = self._speed_scales(nu_r, env_ids)
        linear = self.linear_damping if env_ids is None else self.linear_damping[env_ids]
        quadratic = self.quadratic_damping if env_ids is None else self.quadratic_damping[env_ids]
        added_mass = self.added_mass if env_ids is None else self.added_mass[env_ids]
        boundary_active = additional_scale != 0.0 and (
            self.cfg.pool_boundary_effects_enabled or self.cfg.free_surface_effects_enabled
        )
        if boundary_active:
            linear_scale = linear_scale * pool_damping * surface_damping
            quadratic_scale = quadratic_scale * pool_damping * surface_damping
            added_mass = scale_hydrodynamic_coefficients(
                added_mass, pool_added_mass * surface_added_mass
            )
        if self.cfg.speed_dependent_damping_enabled or boundary_active:
            linear = scale_hydrodynamic_coefficients(linear, linear_scale)
            quadratic = scale_hydrodynamic_coefficients(quadratic, quadratic_scale)
        thruster = pool_thruster * surface_thruster if boundary_active else self.ones_env_1[: nu_r.shape[0]]
        state = EffectiveHydrodynamicState(
            current, nu_r, linear, quadratic, added_mass, buoyancy, thruster
        )
        if env_ids is None:
            self.effective_state = state
        return state

    def compose_fluid_wrench(
        self, kinematics, effective, *, volumes, com_to_cob_offsets, relative_acceleration_b
    ):
        return self.force_models.calculate_fossen_fluid_forces(
            kinematics.root_quat_w, kinematics.root_linear_velocity_b,
            kinematics.root_angular_velocity_b, self.gravity_w, self.cfg.water_rho,
            volumes * effective.buoyancy_scale, com_to_cob_offsets,
            effective.linear_damping, effective.quadratic_damping,
            effective.water_current_w, effective.added_mass, relative_acceleration_b,
            added_mass_enabled=self.added_mass_enabled,
            relative_velocity_b=effective.relative_velocity_b,
        )


__all__ = ["EnvironmentRuntimeState"]

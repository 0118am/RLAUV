"""Explicit hydrodynamic evaluation over an EnvironmentRuntimeState."""

from __future__ import annotations

import torch

from common.tensor_math import quat_apply_wxyz
from environment.hydrodynamics.current_fields import (
    calculate_periodic_water_current,
    calculate_trilinear_current_field,
)
from environment.hydrodynamics.models import HydrodynamicForceModels
from environment.hydrodynamics.pool_effects import (
    RectangularSloshingState,
    calculate_free_surface_scales,
    calculate_pool_boundary_scales,
    calculate_rectangular_pool_sloshing_state,
    rectangular_sloshing_mode_frequencies,
)
from environment.hydrodynamics.tensor_ops import (
    calculate_speed_dependent_damping_scale,
    scale_hydrodynamic_coefficients,
)
from environment.randomization.current import update_smooth_current
from .effective_state import (
    BodyKinematics,
    EffectiveHydrodynamicState,
    _nominal_hydro_coeff_tensor,
    _repeat_hydro_coeff_for_envs,
)


class EnvironmentRuntimeState:
    """Mutable current, pool-effect, and hydrodynamic runtime state."""

    def __init__(self, cfg, *, num_envs, device, gravity_w, pool_center_local) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.gravity_w = gravity_w.to(device=self.device, dtype=torch.float32)
        self.gravity_magnitude = self.gravity_w.norm()
        self.force_models = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.zeros_env_1 = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        self.zeros_env_3 = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.ones_env_1 = torch.ones_like(self.zeros_env_1)
        self.ones_env_6 = torch.ones(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self.flat_surface_z = torch.full_like(
            self.zeros_env_1, float(cfg.free_surface_z)
        )
        self.pool_bounds = torch.as_tensor(
            cfg.pool_bounds, dtype=torch.float32, device=self.device
        )
        self.pool_center_local = torch.as_tensor(
            pool_center_local, dtype=torch.float32, device=self.device
        )
        self.pool_half_extents = torch.stack(
            (
                0.5 * (self.pool_bounds[1] - self.pool_bounds[0]),
                0.5 * (self.pool_bounds[3] - self.pool_bounds[2]),
                0.5 * (self.pool_bounds[5] - self.pool_bounds[4]),
            )
        ).clamp_min(1.0e-6)
        self.body_half_extents_b = 0.5 * torch.as_tensor(
            cfg.body_bounds_size_m, dtype=torch.float32, device=self.device
        )
        self.current_free_surface_z = self.flat_surface_z.clone()

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.periodic_current_amplitude_w = tensor(
            cfg.water_current_periodic_amplitude_w
        )
        self.periodic_current_period_s = tensor(cfg.water_current_periodic_period_s)
        self.periodic_current_phase_rad = tensor(cfg.water_current_periodic_phase_rad)
        self.current_field_bounds = tensor(cfg.water_current_field_bounds)
        self.current_field_shape = tuple(
            int(value) for value in cfg.water_current_field_shape
        )
        self.current_field_values = tensor(cfg.water_current_field_values)
        self.damping_speed_points = tensor(cfg.damping_speed_points)

        def damping_scales(values):
            values_tensor = tensor(values)
            if values_tensor.ndim == 1 and values_tensor.numel():
                return values_tensor.reshape(-1, 1).expand(-1, 6)
            return values_tensor

        self.linear_damping_speed_scales = damping_scales(
            cfg.linear_damping_speed_scales
        )
        self.quadratic_damping_speed_scales = damping_scales(
            cfg.quadratic_damping_speed_scales
        )
        self.sloshing_pool_bounds = tensor(cfg.free_surface_sloshing_pool_bounds)
        self.sloshing_mode_numbers = tensor(cfg.free_surface_sloshing_mode_numbers)
        self.sloshing_amplitudes_m = tensor(cfg.free_surface_sloshing_amplitudes_m)
        self.sloshing_phases_rad = tensor(cfg.free_surface_sloshing_phases_rad)
        self.sloshing_angular_frequencies_rad_s = (
            rectangular_sloshing_mode_frequencies(
                self.sloshing_pool_bounds,
                cfg.free_surface_sloshing_water_depth,
                self.sloshing_mode_numbers,
                float(self.gravity_magnitude),
                dtype=torch.float32,
                device=self.device,
            )
            if cfg.free_surface_sloshing_enabled
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )

        self.nominal_linear_damping = _nominal_hydro_coeff_tensor(
            cfg.linear_damping, self.device, "linear_damping"
        )
        self.nominal_quadratic_damping = _nominal_hydro_coeff_tensor(
            cfg.quadratic_damping, self.device, "quadratic_damping"
        )
        self.nominal_fluid_added_mass = _nominal_hydro_coeff_tensor(
            cfg.added_mass, self.device, "added_mass"
        )
        self.fluid_added_mass_enabled = bool(
            torch.any(self.nominal_fluid_added_mass != 0.0).item()
        )
        self.linear_damping = _repeat_hydro_coeff_for_envs(
            self.nominal_linear_damping, self.num_envs
        )
        self.quadratic_damping = _repeat_hydro_coeff_for_envs(
            self.nominal_quadratic_damping, self.num_envs
        )
        self.fluid_added_mass = _repeat_hydro_coeff_for_envs(
            self.nominal_fluid_added_mass, self.num_envs
        )
        self.linear_damping_randomization_scale = self.ones_env_1.clone()
        self.quadratic_damping_randomization_scale = self.ones_env_1.clone()
        self.fluid_added_mass_randomization_scale = self.ones_env_6.clone()
        self.damping_speed_linear_randomization_scale = self.ones_env_6.clone()
        self.damping_speed_quadratic_randomization_scale = self.ones_env_6.clone()
        self.nominal_water_current_w = tensor(cfg.water_current_w).reshape(1, 3)
        self.water_current_w = self.nominal_water_current_w.repeat(self.num_envs, 1)
        self.water_current_mean_w = self.water_current_w.clone()
        current_std = float(cfg.evaluation_current_variation_std)
        if cfg.evaluation_current_override:
            horizontal_limit = float(
                torch.linalg.vector_norm(self.nominal_water_current_w[0, :2])
            ) + 3.0 * current_std
            vertical_limit = abs(float(self.nominal_water_current_w[0, 2])) + 1.5 * current_std
        else:
            horizontal_limit = vertical_limit = 0.0
        self.water_current_horizontal_max = torch.full(
            (self.num_envs,), horizontal_limit, dtype=torch.float32, device=self.device
        )
        self.water_current_vertical_max = torch.full(
            (self.num_envs,), vertical_limit, dtype=torch.float32, device=self.device
        )
        self.water_current_tau = torch.full(
            (self.num_envs,),
            float(cfg.evaluation_current_tau),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_current_velocity_b = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self.current_acceleration_b = torch.zeros_like(
            self.previous_current_velocity_b
        )
        self.generalized_acceleration_b = torch.zeros_like(
            self.previous_current_velocity_b
        )
        self.has_previous_current_velocity = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.effective_state = None

    def reset_kinematic_history(self, env_ids: torch.Tensor) -> None:
        self.previous_current_velocity_b[env_ids] = 0.0
        self.current_acceleration_b[env_ids] = 0.0
        self.generalized_acceleration_b[env_ids] = 0.0
        self.has_previous_current_velocity[env_ids] = False

    def advance_smooth_current(self, *, stage: int, enabled: bool, policy_dt: float) -> None:
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

    def update_current_acceleration(self, velocity_b, nu_r, *, physics_dt: float):
        """Differentiate only the prescribed current, never vehicle feedback."""

        current_velocity_b = velocity_b - nu_r
        raw = (current_velocity_b - self.previous_current_velocity_b) / physics_dt
        valid = self.has_previous_current_velocity.unsqueeze(-1)
        self.current_acceleration_b[:] = torch.where(valid, raw, torch.zeros_like(raw))
        self.previous_current_velocity_b[:] = current_velocity_b
        self.has_previous_current_velocity[:] = True
        return self.current_acceleration_b

    def body_half_extents_world(self, root_quat):
        """Return the current world-axis AABB half extents of the rigid body."""

        count = root_quat.shape[0]
        body_axes = torch.eye(3, dtype=root_quat.dtype, device=root_quat.device).reshape(
            1, 3, 3
        ).repeat(count, 1, 1)
        rotated_axes = quat_apply_wxyz(
            root_quat.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4),
            body_axes.reshape(-1, 3),
        ).reshape(count, 3, 3)
        return torch.sum(
            torch.abs(rotated_axes) * self.body_half_extents_b.reshape(1, 3, 1), dim=1
        )

    def _pool_scales(self, positions, body_half_extents_w, additional_scale):
        ones = self.ones_env_1[: positions.shape[0]]
        if not self.cfg.pool_boundary_effects_enabled or additional_scale == 0.0:
            return ones, ones, ones
        scales = calculate_pool_boundary_scales(
            positions, body_half_extents_w, self.pool_bounds, self.cfg.pool_boundary_effect_distance,
            self.cfg.pool_boundary_damping_scale, self.cfg.pool_boundary_added_mass_scale,
            self.cfg.pool_boundary_thrust_scale,
        )
        return tuple(1.0 + additional_scale * (value - 1.0) for value in scales)

    def _surface_scales(self, positions, body_half_extents_w, sloshing, additional_scale):
        count = positions.shape[0]
        if not self.cfg.free_surface_effects_enabled or additional_scale == 0.0:
            return self.ones_env_6[:count], self.ones_env_6[:count], self.ones_env_1[:count], self.ones_env_1[:count]
        nearest_surface_positions = positions.clone()
        nearest_surface_positions[:, 2] -= (
            float(self.cfg.free_surface_sloshing_depth_axis_sign) * body_half_extents_w[:, 2]
        )
        scales = calculate_free_surface_scales(
            nearest_surface_positions, sloshing.surface_z, self.cfg.free_surface_effect_distance,
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
        body_half_extents_w = self.body_half_extents_world(root_quat)
        pool_damping, pool_added_mass, pool_thruster = self._pool_scales(
            positions, body_half_extents_w, additional_scale
        )
        surface_damping, surface_added_mass, buoyancy, surface_thruster = self._surface_scales(
            positions, body_half_extents_w, sloshing, additional_scale
        )
        linear_scale, quadratic_scale = self._speed_scales(nu_r, env_ids)
        linear = self.linear_damping if env_ids is None else self.linear_damping[env_ids]
        quadratic = self.quadratic_damping if env_ids is None else self.quadratic_damping[env_ids]
        fluid_added_mass = (
            self.fluid_added_mass
            if env_ids is None
            else self.fluid_added_mass[env_ids]
        )
        boundary_active = additional_scale != 0.0 and (
            self.cfg.pool_boundary_effects_enabled or self.cfg.free_surface_effects_enabled
        )
        if boundary_active:
            linear_scale = linear_scale * pool_damping * surface_damping
            quadratic_scale = quadratic_scale * pool_damping * surface_damping
            fluid_added_mass = scale_hydrodynamic_coefficients(
                fluid_added_mass, pool_added_mass * surface_added_mass
            )
        if self.cfg.speed_dependent_damping_enabled or boundary_active:
            linear = scale_hydrodynamic_coefficients(linear, linear_scale)
            quadratic = scale_hydrodynamic_coefficients(quadratic, quadratic_scale)
        thruster = pool_thruster * surface_thruster if boundary_active else self.ones_env_1[: nu_r.shape[0]]
        state = EffectiveHydrodynamicState(
            water_current_w=current,
            relative_velocity_b=nu_r,
            linear_damping=linear,
            quadratic_damping=quadratic,
            fluid_added_mass=fluid_added_mass,
            buoyancy_scale=buoyancy,
            thruster_scale=thruster,
        )
        if env_ids is None:
            self.effective_state = state
        return state

    def compose_fluid_wrench(self, kinematics, effective, *, volumes, com_to_cob_offsets):
        return self.force_models.calculate_fossen_fluid_forces(
            kinematics.root_quat_w, kinematics.root_linear_velocity_b,
            kinematics.root_angular_velocity_b, self.gravity_w, self.cfg.water_rho,
            volumes * effective.buoyancy_scale, com_to_cob_offsets,
            effective.linear_damping, effective.quadratic_damping,
            effective.water_current_w, effective.fluid_added_mass,
            fluid_added_mass_enabled=self.fluid_added_mass_enabled,
            relative_velocity_b=effective.relative_velocity_b,
        )


__all__ = ["EnvironmentRuntimeState"]

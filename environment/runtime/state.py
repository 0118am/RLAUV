"""Explicit water and hydrodynamic runtime state with no simulator dependency."""

from __future__ import annotations

import torch

from environment.hydrodynamics.models import HydrodynamicForceModels
from environment.hydrodynamics.pool_effects import rectangular_sloshing_mode_frequencies
from .effective_state import _nominal_hydro_coeff_tensor, _repeat_hydro_coeff_for_envs


class _EnvironmentRuntimeBuffers:
    """All mutable current, pool-effect, and hydrodynamic tensors."""

    def __init__(self, cfg, *, num_envs, device, gravity_w, pool_center_local) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.gravity_w = gravity_w.to(device=self.device, dtype=torch.float32)
        self.gravity_magnitude = self.gravity_w.norm()
        self.force_models = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.zeros_env_1 = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.zeros_env_3 = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ones_env_1 = torch.ones_like(self.zeros_env_1)
        self.ones_env_6 = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.flat_surface_z = torch.full_like(self.zeros_env_1, float(cfg.free_surface_z))
        self.pool_bounds = torch.as_tensor(cfg.pool_bounds, dtype=torch.float32, device=self.device)
        self.pool_center_local = torch.as_tensor(pool_center_local, dtype=torch.float32, device=self.device)
        self.pool_half_extents = torch.stack((
            0.5 * (self.pool_bounds[1] - self.pool_bounds[0]),
            0.5 * (self.pool_bounds[3] - self.pool_bounds[2]),
            0.5 * (self.pool_bounds[5] - self.pool_bounds[4]),
        )).clamp_min(1.0e-6)
        self.current_free_surface_z = torch.full_like(self.zeros_env_1, float(cfg.free_surface_z))

        self.periodic_current_amplitude_w = torch.as_tensor(
            cfg.water_current_periodic_amplitude_w, dtype=torch.float32, device=self.device
        )
        self.periodic_current_period_s = torch.as_tensor(
            cfg.water_current_periodic_period_s, dtype=torch.float32, device=self.device
        )
        self.periodic_current_phase_rad = torch.as_tensor(
            cfg.water_current_periodic_phase_rad, dtype=torch.float32, device=self.device
        )
        self.current_field_bounds = torch.as_tensor(
            cfg.water_current_field_bounds, dtype=torch.float32, device=self.device
        )
        self.current_field_shape = tuple(int(value) for value in cfg.water_current_field_shape)
        self.current_field_values = torch.as_tensor(
            cfg.water_current_field_values, dtype=torch.float32, device=self.device
        )
        self.damping_speed_points = torch.as_tensor(
            cfg.damping_speed_points, dtype=torch.float32, device=self.device
        )

        def damping_scales(values):
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            return tensor.reshape(-1, 1).expand(-1, 6) if tensor.ndim == 1 and tensor.numel() else tensor

        self.linear_damping_speed_scales = damping_scales(cfg.linear_damping_speed_scales)
        self.quadratic_damping_speed_scales = damping_scales(cfg.quadratic_damping_speed_scales)
        self.sloshing_pool_bounds = torch.as_tensor(
            cfg.free_surface_sloshing_pool_bounds, dtype=torch.float32, device=self.device
        )
        self.sloshing_mode_numbers = torch.as_tensor(
            cfg.free_surface_sloshing_mode_numbers, dtype=torch.float32, device=self.device
        )
        self.sloshing_amplitudes_m = torch.as_tensor(
            cfg.free_surface_sloshing_amplitudes_m, dtype=torch.float32, device=self.device
        )
        self.sloshing_phases_rad = torch.as_tensor(
            cfg.free_surface_sloshing_phases_rad, dtype=torch.float32, device=self.device
        )
        if cfg.free_surface_sloshing_enabled:
            self.sloshing_angular_frequencies_rad_s = rectangular_sloshing_mode_frequencies(
                self.sloshing_pool_bounds,
                cfg.free_surface_sloshing_water_depth,
                self.sloshing_mode_numbers,
                float(self.gravity_magnitude),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.sloshing_angular_frequencies_rad_s = torch.empty(0, dtype=torch.float32, device=self.device)

        self.nominal_linear_damping = _nominal_hydro_coeff_tensor(
            cfg.linear_damping, self.device, "linear_damping"
        )
        self.nominal_quadratic_damping = _nominal_hydro_coeff_tensor(
            cfg.quadratic_damping, self.device, "quadratic_damping"
        )
        self.nominal_added_mass = _nominal_hydro_coeff_tensor(
            cfg.added_mass_diag, self.device, "added_mass_diag"
        )
        self.added_mass_enabled = bool(torch.any(self.nominal_added_mass != 0.0).item())
        self.linear_damping = _repeat_hydro_coeff_for_envs(self.nominal_linear_damping, self.num_envs)
        self.quadratic_damping = _repeat_hydro_coeff_for_envs(self.nominal_quadratic_damping, self.num_envs)
        self.added_mass = _repeat_hydro_coeff_for_envs(self.nominal_added_mass, self.num_envs)
        self.added_mass_randomization_scale = self.ones_env_6.clone()
        self.damping_speed_linear_randomization_scale = self.ones_env_6.clone()
        self.damping_speed_quadratic_randomization_scale = self.ones_env_6.clone()
        self.nominal_water_current_w = torch.as_tensor(
            cfg.water_current_w, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.water_current_w = self.nominal_water_current_w.repeat(self.num_envs, 1)
        self.water_current_mean_w = self.water_current_w.clone()
        current_std = float(cfg.evaluation_current_variation_std)
        if cfg.evaluation_current_override:
            horizontal_limit = float(torch.linalg.vector_norm(self.nominal_water_current_w[0, :2])) + 3.0 * current_std
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
            (self.num_envs,), float(cfg.evaluation_current_tau), dtype=torch.float32, device=self.device
        )
        self.added_mass_accel_filter_alpha = min(max(float(cfg.added_mass_accel_filter_alpha), 0.0), 1.0)
        self.previous_nu_r = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.filtered_nu_r_dot = torch.zeros_like(self.previous_nu_r)
        self.has_previous_nu_r = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.effective_state = None

    def reset_acceleration(self, env_ids: torch.Tensor) -> None:
        self.previous_nu_r[env_ids] = 0.0
        self.filtered_nu_r_dot[env_ids] = 0.0
        self.has_previous_nu_r[env_ids] = False

    def apply_payload_hydrodynamic_scale(
        self, env_ids, *, linear_damping, quadratic_damping, added_mass
    ) -> None:
        from environment.hydrodynamics.tensor_ops import scale_hydrodynamic_coefficients

        self.linear_damping[env_ids] = scale_hydrodynamic_coefficients(
            self.linear_damping[env_ids], linear_damping
        )
        self.quadratic_damping[env_ids] = scale_hydrodynamic_coefficients(
            self.quadratic_damping[env_ids], quadratic_damping
        )
        self.added_mass[env_ids] = scale_hydrodynamic_coefficients(
            self.added_mass[env_ids], added_mass
        )


__all__ = ["_EnvironmentRuntimeBuffers"]

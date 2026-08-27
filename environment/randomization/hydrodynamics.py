"""Damping, speed-curve, and added-mass DR feature."""

from __future__ import annotations

import torch

from common.random_sampling import sample_bounded_normal
from environment.hydrodynamics.tensor_ops import (
    mean_one_lognormal_scale,
    scale_hydrodynamic_coefficients,
)


def reset_hydrodynamics(state, cfg, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore profile hydrodynamics and sample selected residual uncertainty."""

    count = len(env_ids)
    state.linear_damping[env_ids] = state.nominal_linear_damping
    state.quadratic_damping[env_ids] = state.nominal_quadratic_damping
    state.fluid_added_mass[env_ids] = state.nominal_fluid_added_mass
    state.linear_damping_randomization_scale[env_ids] = 1.0
    state.quadratic_damping_randomization_scale[env_ids] = 1.0
    state.fluid_added_mass_randomization_scale[env_ids] = 1.0
    state.damping_speed_linear_randomization_scale[env_ids] = 1.0
    state.damping_speed_quadratic_randomization_scale[env_ids] = 1.0
    if not enabled:
        return

    linear_damping_log_std = (
        cfg.domain_randomization.linear_damping_log_std_by_stage[stage]
    )
    if linear_damping_log_std > 0.0:
        linear_scale = mean_one_lognormal_scale(
            torch.randn((count, 1), dtype=torch.float32, device=state.device),
            linear_damping_log_std,
        )
        state.linear_damping_randomization_scale[env_ids] = linear_scale
        state.linear_damping[env_ids] = scale_hydrodynamic_coefficients(
            state.linear_damping[env_ids], linear_scale
        )

    quadratic_damping_log_std = (
        cfg.domain_randomization.quadratic_damping_log_std_by_stage[stage]
    )
    if quadratic_damping_log_std > 0.0:
        quadratic_scale = mean_one_lognormal_scale(
            torch.randn((count, 1), dtype=torch.float32, device=state.device),
            quadratic_damping_log_std,
        )
        state.quadratic_damping_randomization_scale[env_ids] = quadratic_scale
        state.quadratic_damping[env_ids] = scale_hydrodynamic_coefficients(
            state.quadratic_damping[env_ids], quadratic_scale
        )

    fluid_added_mass_log_std = (
        cfg.domain_randomization.fluid_added_mass_log_std_by_stage[stage]
    )
    if fluid_added_mass_log_std > 0.0:
        latent = torch.randn((count, 6), dtype=torch.float32, device=state.device)
        scale = mean_one_lognormal_scale(latent, fluid_added_mass_log_std)
        state.fluid_added_mass_randomization_scale[env_ids] = scale
        # Congruence scaling retains the CFD matrix's symmetry, coupling zeros,
        # and positive definiteness while varying all six inertial directions.
        state.fluid_added_mass[env_ids] = scale_hydrodynamic_coefficients(
            state.fluid_added_mass[env_ids], scale
        )

    linear_range = cfg.domain_randomization.damping_speed_linear_scale_range
    if linear_range is not None:
        state.damping_speed_linear_randomization_scale[env_ids] = sample_bounded_normal(
            linear_range[0], linear_range[1], (count, 1), state.device
        ).repeat(1, 6)
    quadratic_range = cfg.domain_randomization.damping_speed_quadratic_scale_range
    if quadratic_range is not None:
        state.damping_speed_quadratic_randomization_scale[env_ids] = sample_bounded_normal(
            quadratic_range[0], quadratic_range[1], (count, 1), state.device
        ).repeat(1, 6)

"""Damping, speed-curve, and added-mass DR feature."""

from __future__ import annotations

import torch

from environment.hydrodynamics.tensor_ops import mean_one_lognormal_scale, scale_hydrodynamic_coefficients
from environment.profiles.random_sampling import sample_bounded_normal, sample_symmetric_bounded_normal


def reset_hydrodynamics(state, cfg, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore profile hydrodynamics and sample selected residual uncertainty."""

    count = len(env_ids)
    state.linear_damping[env_ids] = state.nominal_linear_damping
    state.quadratic_damping[env_ids] = state.nominal_quadratic_damping
    state.added_mass[env_ids] = state.nominal_added_mass
    state.added_mass_randomization_scale[env_ids] = 1.0
    state.damping_speed_linear_randomization_scale[env_ids] = 1.0
    state.damping_speed_quadratic_randomization_scale[env_ids] = 1.0
    if not enabled:
        return

    damping_scale = cfg.domain_randomization.damping_scale_by_stage[stage]
    if damping_scale > 0.0:
        damping_shape = (count, 6) if state.linear_damping.ndim == 2 else (count, 1, 1)
        multiplier = 1.0 + sample_symmetric_bounded_normal(
            damping_scale, damping_shape, state.device
        )
        state.linear_damping[env_ids] = state.nominal_linear_damping * multiplier
        state.quadratic_damping[env_ids] = state.nominal_quadratic_damping * multiplier

    added_mass_log_std = cfg.domain_randomization.added_mass_log_std_by_stage[stage]
    if added_mass_log_std > 0.0:
        latent = torch.randn((count, 6), dtype=torch.float32, device=state.device)
        scale = mean_one_lognormal_scale(latent, added_mass_log_std)
        state.added_mass_randomization_scale[env_ids] = scale
        state.added_mass[env_ids] = scale_hydrodynamic_coefficients(
            state.added_mass[env_ids], scale
        )

    linear_min, linear_max = getattr(
        cfg.domain_randomization, "damping_speed_linear_scale_range", [1.0, 1.0]
    )
    quadratic_min, quadratic_max = getattr(
        cfg.domain_randomization, "damping_speed_quadratic_scale_range", [1.0, 1.0]
    )
    state.damping_speed_linear_randomization_scale[env_ids] = sample_bounded_normal(
        linear_min, linear_max, (count, 1), state.device
    ).repeat(1, 6)
    state.damping_speed_quadratic_randomization_scale[env_ids] = sample_bounded_normal(
        quadratic_min, quadratic_max, (count, 1), state.device
    ).repeat(1, 6)

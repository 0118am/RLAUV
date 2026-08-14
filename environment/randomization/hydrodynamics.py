"""Damping, speed-curve, and added-mass DR feature."""

from __future__ import annotations

import torch

from environment.hydrodynamics.models import mean_one_lognormal_scale, scale_hydrodynamic_coefficients
from environment.profiles.random_sampling import sample_bounded_normal, sample_symmetric_bounded_normal


def reset_hydrodynamics(env, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore profile hydrodynamics and sample selected residual uncertainty."""

    count = len(env_ids)
    env.linear_damping[env_ids] = env._nominal_linear_damping
    env.quadratic_damping[env_ids] = env._nominal_quadratic_damping
    env.added_mass_diag[env_ids] = env._nominal_added_mass_diag
    env.added_mass_randomization_scale[env_ids] = 1.0
    env.damping_speed_linear_randomization_scale[env_ids] = 1.0
    env.damping_speed_quadratic_randomization_scale[env_ids] = 1.0
    if not enabled:
        return

    damping_scale = env.cfg.domain_randomization.damping_scale_by_stage[stage]
    if damping_scale > 0.0:
        damping_shape = (count, 6) if env.linear_damping.ndim == 2 else (count, 1, 1)
        multiplier = 1.0 + sample_symmetric_bounded_normal(damping_scale, damping_shape, env.device)
        env.linear_damping[env_ids] = env._nominal_linear_damping * multiplier
        env.quadratic_damping[env_ids] = env._nominal_quadratic_damping * multiplier

    added_mass_log_std = env.cfg.domain_randomization.added_mass_log_std_by_stage[stage]
    if added_mass_log_std > 0.0:
        latent = torch.randn((count, 6), dtype=torch.float32, device=env.device)
        scale = mean_one_lognormal_scale(latent, added_mass_log_std)
        env.added_mass_randomization_scale[env_ids] = scale
        env.added_mass_diag[env_ids] = scale_hydrodynamic_coefficients(env.added_mass_diag[env_ids], scale)

    linear_min, linear_max = getattr(
        env.cfg.domain_randomization, "damping_speed_linear_scale_range", [1.0, 1.0]
    )
    quadratic_min, quadratic_max = getattr(
        env.cfg.domain_randomization, "damping_speed_quadratic_scale_range", [1.0, 1.0]
    )
    env.damping_speed_linear_randomization_scale[env_ids] = sample_bounded_normal(
        linear_min, linear_max, (count, 1), env.device
    ).repeat(1, 6)
    env.damping_speed_quadratic_randomization_scale[env_ids] = sample_bounded_normal(
        quadratic_min, quadratic_max, (count, 1), env.device
    ).repeat(1, 6)

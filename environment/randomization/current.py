"""Episode and time-varying water-current DR feature."""

from __future__ import annotations

import torch

from common.random_sampling import sample_bounded_normal, sample_isotropic_bounded_normal, sample_symmetric_bounded_normal


def reset_current(state, cfg, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore nominal current then sample an episode-current field if selected."""

    state.water_current_w[env_ids] = state.nominal_water_current_w
    state.water_current_mean_w[env_ids] = state.nominal_water_current_w
    state.water_current_horizontal_max[env_ids] = 0.0
    state.water_current_vertical_max[env_ids] = 0.0
    state.water_current_tau[env_ids] = float(cfg.evaluation_current_tau)
    if cfg.evaluation_current_override:
        variation_std = float(cfg.evaluation_current_variation_std)
        nominal = state.nominal_water_current_w[0]
        state.water_current_horizontal_max[env_ids] = (
            torch.linalg.vector_norm(nominal[0:2]) + 3.0 * variation_std
        )
        state.water_current_vertical_max[env_ids] = abs(float(nominal[2].item())) + 1.5 * variation_std
        state.water_current_tau[env_ids] = float(cfg.evaluation_current_tau)
        return
    if not enabled:
        return

    current_max = cfg.domain_randomization.water_current_max_by_stage[stage]
    vertical_max = cfg.domain_randomization.water_current_vertical_max_by_stage[stage]
    if current_max <= 0.0 and vertical_max <= 0.0:
        return
    count = len(env_ids)
    state.water_current_mean_w[env_ids, 0:2] = sample_isotropic_bounded_normal(
        current_max, count, 2, state.device
    )
    state.water_current_mean_w[env_ids, 2] = sample_symmetric_bounded_normal(
        vertical_max, (count,), state.device
    )
    state.water_current_w[env_ids] = state.water_current_mean_w[env_ids]
    state.water_current_horizontal_max[env_ids] = current_max
    state.water_current_vertical_max[env_ids] = vertical_max
    tau_min, tau_max = cfg.domain_randomization.water_current_tau_range
    state.water_current_tau[env_ids] = sample_bounded_normal(
        tau_min, tau_max, (count,), state.device
    )


def update_smooth_current(
    state, cfg, stage: int, policy_dt: float, *, enabled: bool
) -> None:
    """Advance the bounded, mean-reverting current only when this feature is active."""

    if not enabled:
        return
    if not cfg.domain_randomization.water_current_smooth:
        return

    variation_std = cfg.domain_randomization.water_current_variation_std_by_stage[stage]
    horizontal_max = cfg.domain_randomization.water_current_max_by_stage[stage]
    vertical_max = cfg.domain_randomization.water_current_vertical_max_by_stage[stage]
    if variation_std <= 0.0 and horizontal_max <= 0.0 and vertical_max <= 0.0:
        return

    tau = torch.clamp(state.water_current_tau, min=policy_dt)
    alpha = torch.exp(-policy_dt / tau).unsqueeze(-1)
    noise_scale = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0))
    noise = torch.randn_like(state.water_current_w) * variation_std * noise_scale
    noise[:, 2] *= 0.5
    state.water_current_w[:] = (
        alpha * state.water_current_w + (1.0 - alpha) * state.water_current_mean_w + noise
    )

    xy = state.water_current_w[:, 0:2]
    xy_norm = torch.linalg.norm(xy, dim=1, keepdim=True)
    xy_limit = state.water_current_horizontal_max.unsqueeze(-1)
    xy_scale = torch.clamp(xy_limit / torch.clamp(xy_norm, min=1.0e-6), max=1.0)
    state.water_current_w[:, 0:2] = xy * xy_scale
    state.water_current_w[:, 2] = torch.clamp(
        state.water_current_w[:, 2],
        -state.water_current_vertical_max,
        state.water_current_vertical_max,
    )

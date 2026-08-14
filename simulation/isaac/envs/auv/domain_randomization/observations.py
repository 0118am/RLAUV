"""Policy-observation transport domain randomization."""

from __future__ import annotations

import torch

from environment.profiles.features import domain_randomization_feature_enabled
from environment.profiles.random_sampling import sample_bounded_normal, sample_bounded_normal_integer, sample_symmetric_bounded_normal


def reset_observation_transport(env, env_ids: torch.Tensor) -> None:
    """Restore observation transport, then sample its selected uncertainty."""

    count = len(env_ids)
    env.observation_delay_steps[env_ids] = int(env.cfg.observation_delay_steps)
    env.observation_update_period_steps[env_ids] = int(env.cfg.observation_update_period_steps)
    env._set_fixed_observation_noise(env_ids)
    if not domain_randomization_feature_enabled(env, "observations"):
        return

    noise_min, noise_max = env.cfg.domain_randomization.observation_noise_std_range
    bias_min, bias_max = env.cfg.domain_randomization.observation_bias_range
    delay_min, delay_max = env.cfg.domain_randomization.observation_delay_steps_range
    period_min, period_max = env.cfg.domain_randomization.observation_update_period_steps_range
    dropout_min, dropout_max = env.cfg.domain_randomization.observation_dropout_probability_range
    lowpass_min, lowpass_max = env.cfg.domain_randomization.observation_lowpass_alpha_range
    drift_min, drift_max = env.cfg.domain_randomization.observation_bias_drift_std_range
    env.observation_noise_std[env_ids] = sample_bounded_normal(noise_min, noise_max, (count, 1), env.device)
    if bias_max >= bias_min and bias_max > 0.0:
        magnitude = sample_bounded_normal(bias_min, bias_max, (count, 1), env.device)
        env.observation_bias[env_ids] = sample_symmetric_bounded_normal(
            magnitude, (count, env.cfg.observation_base_dim), env.device
        )
    env.observation_delay_steps[env_ids] = sample_bounded_normal_integer(
        int(delay_min), int(delay_max), (count,), env.device
    )
    env.observation_update_period_steps[env_ids] = sample_bounded_normal_integer(
        int(period_min), int(period_max), (count,), env.device
    )
    env.observation_dropout_probability[env_ids] = sample_bounded_normal(
        dropout_min, dropout_max, (count, 1), env.device
    )
    lowpass = sample_bounded_normal(lowpass_min, lowpass_max, (count, 1), env.device)
    env.observation_lowpass_alpha[env_ids] = torch.clamp(lowpass, min=0.0, max=1.0)
    env.observation_bias_drift_std[env_ids] = sample_bounded_normal(
        drift_min, drift_max, (count, 1), env.device
    )

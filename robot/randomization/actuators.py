"""Per-thruster gain, bandwidth, delay, and wake DR feature."""

from __future__ import annotations

import torch

from environment.profiles.random_sampling import sample_bounded_normal, sample_bounded_normal_integer, sample_symmetric_bounded_normal


def reset_actuators(env, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore nominal actuator chain and sample per-episode uncertainty."""

    count = len(env_ids)
    env.thruster_force_scale[env_ids] = 1.0
    env.thruster_time_constant[env_ids] = env.cfg.dyn_time_constant
    env.thruster_delay_steps[env_ids] = int(env.cfg.thruster_command_delay_steps)
    env.thruster_max_command_rate[env_ids] = env.cfg.thruster_max_command_rate
    env.thruster_command_resolution[env_ids] = env.cfg.thruster_command_resolution
    env.thruster_command_dropout_probability[env_ids] = env.cfg.thruster_command_dropout_probability
    env.thruster_wake_loss_coefficient[env_ids] = env.cfg.thruster_wake_loss_coefficient
    if not enabled:
        if bool(getattr(env.cfg, "evaluation_thruster_force_scale_override", False)):
            env.thruster_force_scale[env_ids] = float(env.cfg.evaluation_thruster_force_scale)
        env.thruster_response.set_time_constants(env.thruster_time_constant)
        return

    force_scale = env.cfg.domain_randomization.thruster_scale_by_stage[stage]
    if force_scale > 0.0:
        env.thruster_force_scale[env_ids] = 1.0 + sample_symmetric_bounded_normal(
            force_scale, (count, env.num_thrusters), env.device
        )
    tau_scale = env.cfg.domain_randomization.thruster_tau_scale_by_stage[stage]
    if tau_scale > 0.0:
        multiplier = 1.0 + sample_symmetric_bounded_normal(tau_scale, (count,), env.device)
        env.thruster_time_constant[env_ids] = torch.clamp(
            env.cfg.dyn_time_constant * multiplier,
            min=0.0,
        )

    delay_min, delay_max = env.cfg.domain_randomization.thruster_command_delay_steps_range
    if delay_max > delay_min:
        env.thruster_delay_steps[env_ids] = sample_bounded_normal_integer(
            int(delay_min), int(delay_max), (count,), env.device
        )
    else:
        env.thruster_delay_steps[env_ids] = int(delay_min)
    rate_min, rate_max = env.cfg.domain_randomization.thruster_max_command_rate_range
    resolution_min, resolution_max = env.cfg.domain_randomization.thruster_command_resolution_range
    dropout_min, dropout_max = env.cfg.domain_randomization.thruster_command_dropout_probability_range
    env.thruster_max_command_rate[env_ids] = sample_bounded_normal(rate_min, rate_max, (count, 1), env.device)
    env.thruster_command_resolution[env_ids] = sample_bounded_normal(
        resolution_min, resolution_max, (count, 1), env.device
    )
    env.thruster_command_dropout_probability[env_ids] = sample_bounded_normal(
        dropout_min, dropout_max, (count, 1), env.device
    )

    wake_min, wake_max = getattr(
        env.cfg.domain_randomization, "thruster_wake_loss_coefficient_scale_range", [1.0, 1.0]
    )
    wake_scale = sample_bounded_normal(wake_min, wake_max, (count,), env.device)
    env.thruster_wake_loss_coefficient[env_ids] = torch.clamp(
        env.cfg.thruster_wake_loss_coefficient * wake_scale, min=0.0
    )
    if bool(getattr(env.cfg, "evaluation_thruster_force_scale_override", False)):
        env.thruster_force_scale[env_ids] = float(env.cfg.evaluation_thruster_force_scale)
    env.thruster_response.set_time_constants(env.thruster_time_constant)

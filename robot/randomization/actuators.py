"""Per-thruster gain, bandwidth, delay, and wake DR feature."""

from __future__ import annotations

import torch

from environment.profiles.random_sampling import sample_bounded_normal, sample_bounded_normal_integer, sample_symmetric_bounded_normal


def reset_actuators(state, cfg, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore nominal actuator chain and sample per-episode uncertainty."""

    count = len(env_ids)
    state.thruster_force_scale[env_ids] = 1.0
    state.thruster_time_constant[env_ids] = cfg.dyn_time_constant
    state.thruster_delay_steps[env_ids] = int(cfg.thruster_command_delay_steps)
    state.thruster_max_command_rate[env_ids] = cfg.thruster_max_command_rate
    state.thruster_command_resolution[env_ids] = cfg.thruster_command_resolution
    state.thruster_command_dropout_probability[env_ids] = cfg.thruster_command_dropout_probability
    state.thruster_wake_loss_coefficient[env_ids] = cfg.thruster_wake_loss_coefficient
    if not enabled:
        if bool(getattr(cfg, "evaluation_thruster_force_scale_override", False)):
            state.thruster_force_scale[env_ids] = float(cfg.evaluation_thruster_force_scale)
        state.thruster_response.set_time_constants(state.thruster_time_constant)
        return

    force_scale = cfg.domain_randomization.thruster_scale_by_stage[stage]
    if force_scale > 0.0:
        state.thruster_force_scale[env_ids] = 1.0 + sample_symmetric_bounded_normal(
            force_scale, (count, state.num_thrusters), state.device
        )
    tau_scale = cfg.domain_randomization.thruster_tau_scale_by_stage[stage]
    if tau_scale > 0.0:
        multiplier = 1.0 + sample_symmetric_bounded_normal(tau_scale, (count,), state.device)
        state.thruster_time_constant[env_ids] = torch.clamp(
            cfg.dyn_time_constant * multiplier,
            min=0.0,
        )

    delay_min, delay_max = cfg.domain_randomization.thruster_command_delay_steps_range
    if delay_max > delay_min:
        state.thruster_delay_steps[env_ids] = sample_bounded_normal_integer(
            int(delay_min), int(delay_max), (count,), state.device
        )
    else:
        state.thruster_delay_steps[env_ids] = int(delay_min)
    rate_min, rate_max = cfg.domain_randomization.thruster_max_command_rate_range
    resolution_min, resolution_max = cfg.domain_randomization.thruster_command_resolution_range
    dropout_min, dropout_max = cfg.domain_randomization.thruster_command_dropout_probability_range
    state.thruster_max_command_rate[env_ids] = sample_bounded_normal(rate_min, rate_max, (count, 1), state.device)
    state.thruster_command_resolution[env_ids] = sample_bounded_normal(
        resolution_min, resolution_max, (count, 1), state.device
    )
    state.thruster_command_dropout_probability[env_ids] = sample_bounded_normal(
        dropout_min, dropout_max, (count, 1), state.device
    )

    wake_min, wake_max = getattr(
        cfg.domain_randomization, "thruster_wake_loss_coefficient_scale_range", [1.0, 1.0]
    )
    wake_scale = sample_bounded_normal(wake_min, wake_max, (count,), state.device)
    state.thruster_wake_loss_coefficient[env_ids] = torch.clamp(
        cfg.thruster_wake_loss_coefficient * wake_scale, min=0.0
    )
    if bool(getattr(cfg, "evaluation_thruster_force_scale_override", False)):
        state.thruster_force_scale[env_ids] = float(cfg.evaluation_thruster_force_scale)
    state.thruster_response.set_time_constants(state.thruster_time_constant)

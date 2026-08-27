"""Common/per-thruster gain, command conditioning, and wake DR feature."""

from __future__ import annotations

import torch

from common.random_sampling import sample_bounded_normal, sample_symmetric_bounded_normal


def reset_actuators(state, cfg, env_ids: torch.Tensor, stage: int, *, enabled: bool) -> None:
    """Restore nominal actuator chain and sample per-episode uncertainty."""

    count = len(env_ids)
    state.thruster_force_scale[env_ids] = 1.0
    state.common_thruster_force_scale[env_ids] = 1.0
    state.thruster_time_constant[env_ids] = cfg.dyn_time_constant
    state.thruster_command_resolution[env_ids] = cfg.thruster_command_resolution
    state.thruster_command_dropout_probability[env_ids] = cfg.thruster_command_dropout_probability
    state.thruster_wake_loss_coefficient[env_ids] = cfg.thruster_wake_loss_coefficient
    if not enabled:
        if cfg.evaluation_thruster_force_scale_override:
            state.common_thruster_force_scale[env_ids] = float(
                cfg.evaluation_thruster_force_scale
            )
        state.thruster_response.set_time_constants(state.thruster_time_constant)
        return

    force_scale = cfg.domain_randomization.thruster_scale_by_stage[stage]
    if force_scale > 0.0:
        state.thruster_force_scale[env_ids] = 1.0 + sample_symmetric_bounded_normal(
            force_scale, (count, state.num_thrusters), state.device
        )
    common_reduction = cfg.domain_randomization.common_thruster_scale_reduction_by_stage[stage]
    if common_reduction > 0.0:
        state.common_thruster_force_scale[env_ids] = sample_bounded_normal(
            1.0 - common_reduction,
            1.0,
            (count, 1),
            state.device,
        )
    time_constant_range = cfg.domain_randomization.thruster_time_constant_range
    if time_constant_range is not None:
        state.thruster_time_constant[env_ids] = sample_bounded_normal(
            time_constant_range[0],
            time_constant_range[1],
            (count,),
            state.device,
        )
    scalar_ranges = (
        ("thruster_command_resolution_range", state.thruster_command_resolution),
        (
            "thruster_command_dropout_probability_range",
            state.thruster_command_dropout_probability,
        ),
    )
    for name, destination in scalar_ranges:
        value_range = getattr(cfg.domain_randomization, name)
        if value_range is not None:
            destination[env_ids] = sample_bounded_normal(
                value_range[0], value_range[1], (count, 1), state.device
            )

    wake_range = cfg.domain_randomization.thruster_wake_loss_coefficient_scale_range
    if wake_range is not None:
        wake_scale = sample_bounded_normal(
            wake_range[0], wake_range[1], (count,), state.device
        )
        state.thruster_wake_loss_coefficient[env_ids] = torch.clamp(
            cfg.thruster_wake_loss_coefficient * wake_scale, min=0.0
        )
    if cfg.evaluation_thruster_force_scale_override:
        state.common_thruster_force_scale[env_ids] = float(
            cfg.evaluation_thruster_force_scale
        )
    state.thruster_response.set_time_constants(state.thruster_time_constant)

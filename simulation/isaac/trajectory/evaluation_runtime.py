"""State sampling and stepping loop for trajectory evaluation."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from isaaclab.utils.math import quat_apply, quat_error_magnitude

from simulation.isaac.trajectory.evaluation_logging import (
    ChunkedTensorLog,
    EVALUATION_LOG_SCHEMA_VERSION,
    evaluation_log_columns,
)


@dataclass(frozen=True)
class TrackingSnapshot:
    target_pos_w: torch.Tensor
    target_lin_vel_w: torch.Tensor
    target_quat_w: torch.Tensor
    target_acceleration_w: torch.Tensor
    target_jerk_w: torch.Tensor
    target_curvature_m_inv: torch.Tensor
    target_orientation_rate_radps: torch.Tensor
    requested_period_s: torch.Tensor
    requested_speed_mps: torch.Tensor
    effective_period_s: torch.Tensor
    retimed: torch.Tensor
    root_pos_w: torch.Tensor
    root_pos_local: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor
    position_error: torch.Tensor
    velocity_error: torch.Tensor
    attitude_error: torch.Tensor
    command_heading_error: torch.Tensor
    motion_sideslip_error: torch.Tensor


@dataclass(frozen=True)
class ActionSnapshot:
    raw: torch.Tensor
    requested: torch.Tensor
    clip_mask: torch.Tensor
    norm: torch.Tensor
    rms: torch.Tensor
    rate_rms: torch.Tensor


@dataclass(frozen=True)
class EvaluationResult:
    log: pd.DataFrame
    any_failure: torch.Tensor
    first_failure_time_s: torch.Tensor
    termination_events: int


def _direction_error(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
    min_norm: float = 1.0e-3,
) -> torch.Tensor:
    norm_a = torch.norm(vector_a, dim=1)
    norm_b = torch.norm(vector_b, dim=1)
    cosine = torch.sum(vector_a * vector_b, dim=1) / torch.clamp(norm_a * norm_b, min=min_norm**2)
    angle = torch.acos(torch.clamp(cosine, min=-1.0, max=1.0))
    valid = (norm_a > min_norm) & (norm_b > min_norm)
    return torch.where(valid, angle, torch.full_like(angle, float("nan")))


def _capture_tracking_snapshot(env: Any) -> TrackingSnapshot:
    target_pos_w, target_lin_vel_w, target_quat_w = env.unwrapped.get_tracking_targets()
    kinematics = env.unwrapped.get_tracking_kinematics()
    robot_data = env.unwrapped._robot.data
    root_pos_w = robot_data.root_pos_w
    root_quat_w = robot_data.root_quat_w
    root_lin_vel_b = robot_data.root_lin_vel_b
    root_lin_vel_w = quat_apply(root_quat_w, root_lin_vel_b)
    body_x_b = torch.zeros_like(root_lin_vel_b)
    body_x_b[:, 0] = 1.0
    nose_direction_w = quat_apply(root_quat_w, body_x_b)

    return TrackingSnapshot(
        target_pos_w=target_pos_w,
        target_lin_vel_w=target_lin_vel_w,
        target_quat_w=target_quat_w,
        target_acceleration_w=kinematics["target_acceleration_w"],
        target_jerk_w=kinematics["target_jerk_w"],
        target_curvature_m_inv=kinematics["target_curvature_m_inv"],
        target_orientation_rate_radps=kinematics["target_orientation_rate_radps"],
        requested_period_s=kinematics["requested_period_s"],
        requested_speed_mps=kinematics["requested_speed_mps"],
        effective_period_s=kinematics["effective_period_s"],
        retimed=kinematics["retimed"],
        root_pos_w=root_pos_w,
        root_pos_local=root_pos_w - env.unwrapped.scene.env_origins,
        root_quat_w=root_quat_w,
        root_lin_vel_w=root_lin_vel_w,
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=robot_data.root_ang_vel_b,
        position_error=torch.norm(target_pos_w - root_pos_w, dim=1),
        velocity_error=torch.norm(target_lin_vel_w - root_lin_vel_w, dim=1),
        attitude_error=quat_error_magnitude(target_quat_w, root_quat_w),
        command_heading_error=_direction_error(nose_direction_w, target_lin_vel_w),
        motion_sideslip_error=_direction_error(nose_direction_w, root_lin_vel_w),
    )


def _sample_actions(
    policy: Any,
    observations: Any,
    active_envs: torch.Tensor,
    previous_actions: torch.Tensor | None,
) -> ActionSnapshot:
    raw = policy(observations)
    requested = torch.where(
        active_envs.unsqueeze(-1),
        torch.clamp(raw, -1.0, 1.0),
        torch.zeros_like(raw),
    )
    delta = torch.zeros_like(requested) if previous_actions is None else requested - previous_actions
    return ActionSnapshot(
        raw=raw,
        requested=requested,
        clip_mask=raw.abs() > 1.0,
        norm=torch.norm(requested, dim=1),
        rms=torch.sqrt(torch.mean(requested.square(), dim=1)),
        rate_rms=torch.sqrt(torch.mean(delta.square(), dim=1)),
    )


def _pre_step_values(env: Any, tracking: TrackingSnapshot, actions: ActionSnapshot) -> torch.Tensor:
    return torch.cat(
        (
            env.unwrapped.water_current_w,
            tracking.target_pos_w,
            tracking.root_pos_w,
            tracking.target_lin_vel_w,
            torch.linalg.vector_norm(tracking.target_lin_vel_w, dim=1, keepdim=True),
            tracking.requested_speed_mps.unsqueeze(-1),
            torch.linalg.vector_norm(tracking.target_acceleration_w, dim=1, keepdim=True),
            torch.linalg.vector_norm(tracking.target_jerk_w, dim=1, keepdim=True),
            tracking.target_curvature_m_inv.unsqueeze(-1),
            tracking.target_orientation_rate_radps.unsqueeze(-1),
            tracking.requested_period_s.unsqueeze(-1),
            tracking.effective_period_s.unsqueeze(-1),
            tracking.retimed.to(dtype=torch.float32).unsqueeze(-1),
            tracking.root_lin_vel_w,
            tracking.root_pos_local,
            tracking.root_quat_w,
            tracking.root_ang_vel_b,
            actions.requested,
            actions.raw,
            actions.clip_mask.to(dtype=torch.float32),
            tracking.position_error.unsqueeze(-1),
            tracking.velocity_error.unsqueeze(-1),
            tracking.attitude_error.unsqueeze(-1),
            tracking.command_heading_error.unsqueeze(-1),
            tracking.motion_sideslip_error.unsqueeze(-1),
            actions.norm.unsqueeze(-1),
            actions.rms.unsqueeze(-1),
            actions.rate_rms.unsqueeze(-1),
            torch.linalg.vector_norm(actions.raw, dim=1, keepdim=True),
            torch.sqrt(torch.mean(actions.raw.square(), dim=1, keepdim=True)),
            actions.clip_mask.to(dtype=torch.float32).mean(dim=1, keepdim=True),
        ),
        dim=1,
    )


def _post_step_actuator_values(
    env: Any,
    requested_actions: torch.Tensor,
    previous_applied_actions: torch.Tensor | None,
    terminated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    applied = env.unwrapped.thruster_command_processor.rate_limited_state
    applied_delta = torch.zeros_like(applied) if previous_applied_actions is None else applied - previous_applied_actions
    requested_to_applied = requested_actions - applied
    thruster_force = env.unwrapped.realized_thruster_force_n
    values = torch.cat(
        (
            applied,
            requested_to_applied,
            thruster_force,
            torch.sqrt(torch.mean(requested_to_applied.square(), dim=1, keepdim=True)),
            torch.sqrt(torch.mean(applied_delta.square(), dim=1, keepdim=True)),
            thruster_force.abs().mean(dim=1, keepdim=True),
            thruster_force.abs().amax(dim=1, keepdim=True),
            env.unwrapped._thrust[:, 0, :],
            env.unwrapped._moment[:, 0, :],
        ),
        dim=1,
    )
    values = torch.where(terminated.unsqueeze(-1), torch.full_like(values, float("nan")), values)
    return values, applied.clone()


def _update_visualizer(
    visualizer: Any,
    active_envs: torch.Tensor,
    tracking: TrackingSnapshot,
    step: int,
    time_s: float,
) -> None:
    if not visualizer.enabled or not bool(torch.any(active_envs)):
        return
    env_id = int(torch.nonzero(active_envs, as_tuple=False)[0, 0].item())
    visualizer.update(
        step,
        time_s,
        tracking.target_pos_w[env_id],
        tracking.root_pos_w[env_id],
        tracking.position_error[env_id].item(),
        tracking.velocity_error[env_id].item(),
    )


def collect_domain_samples(env: Any) -> pd.DataFrame:
    unwrapped = env.unwrapped
    physics_dt = float(unwrapped.cfg.sim.dt)
    columns = (
        "sampled_mass_kg",
        "sampled_volume_m3",
        "payload_sample_index",
        "sampled_center_of_mass_x_m",
        "sampled_center_of_mass_y_m",
        "sampled_center_of_mass_z_m",
        "sampled_com_to_cob_x_m",
        "sampled_com_to_cob_y_m",
        "sampled_com_to_cob_z_m",
        "sampled_principal_inertia_x_kg_m2",
        "sampled_principal_inertia_y_kg_m2",
        "sampled_principal_inertia_z_kg_m2",
        "sampled_linear_damping_l2",
        "sampled_quadratic_damping_l2",
        "sampled_added_mass_l2",
        "sampled_thruster_scale_mean",
        "sampled_thruster_scale_min",
        "sampled_thruster_scale_max",
        "sampled_thruster_time_constant_s",
        "sampled_thruster_delay_steps",
        "sampled_thruster_delay_s",
        "sampled_thruster_max_command_rate_per_s",
        "sampled_thruster_command_resolution_mean",
        "sampled_thruster_command_dropout_probability_mean",
        "sampled_battery_voltage_v",
    )
    delay_steps = unwrapped.thruster_delay_steps.reshape(-1)
    values = torch.cat(
        (
            unwrapped.masses.reshape(-1, 1),
            unwrapped.volumes.reshape(-1, 1),
            unwrapped.payload_sample_indices.to(dtype=torch.float32).unsqueeze(-1),
            unwrapped.center_of_mass_offsets,
            unwrapped.com_to_cob_offsets,
            unwrapped.inertia_principal_moments,
            torch.linalg.vector_norm(unwrapped.linear_damping.reshape(unwrapped.num_envs, -1), dim=1, keepdim=True),
            torch.linalg.vector_norm(unwrapped.quadratic_damping.reshape(unwrapped.num_envs, -1), dim=1, keepdim=True),
            torch.linalg.vector_norm(unwrapped.added_mass_diag.reshape(unwrapped.num_envs, -1), dim=1, keepdim=True),
            unwrapped.thruster_force_scale.mean(dim=1, keepdim=True),
            unwrapped.thruster_force_scale.amin(dim=1, keepdim=True),
            unwrapped.thruster_force_scale.amax(dim=1, keepdim=True),
            unwrapped.thruster_time_constant.reshape(-1, 1),
            delay_steps.to(dtype=torch.float32).unsqueeze(-1),
            delay_steps.to(dtype=torch.float32).unsqueeze(-1) * physics_dt,
            unwrapped.thruster_max_command_rate.reshape(-1, 1),
            unwrapped.thruster_command_resolution.mean(dim=1, keepdim=True),
            unwrapped.thruster_command_dropout_probability.mean(dim=1, keepdim=True),
            unwrapped.battery_voltage.reshape(-1, 1),
        ),
        dim=1,
    ).detach().cpu().numpy()
    frame = pd.DataFrame(values, columns=columns)
    frame["payload_sample_index"] = frame["payload_sample_index"].astype(np.int64)
    frame.insert(0, "domain_randomization_spec_name", unwrapped.cfg.domain_randomization_spec_name or "")
    frame.insert(0, "environment_profile_name", unwrapped.cfg.environment_profile_name)
    frame.insert(0, "seed", int(unwrapped.cfg.seed))
    frame.insert(0, "env_id", np.arange(unwrapped.num_envs, dtype=np.int64))
    return frame


def run_evaluation(
    env: Any,
    policy: Any,
    observations: Any,
    *,
    duration_s: float,
    trajectory: str,
    reward_profile: str,
    disturbance_label: str,
    visualizer: Any,
) -> EvaluationResult:
    step_dt = float(env.unwrapped.cfg.sim.dt) * int(env.unwrapped.cfg.decimation)
    tensor_log = ChunkedTensorLog(evaluation_log_columns(env.num_actions), num_envs=env.unwrapped.num_envs)
    previous_actions = None
    previous_applied_actions = None
    active_envs = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    any_failure = torch.zeros_like(active_envs)
    first_failure_time_s = torch.full(
        (env.unwrapped.num_envs,),
        float("nan"),
        dtype=torch.float32,
        device=env.unwrapped.device,
    )

    for step in range(int(math.ceil(duration_s / step_dt))):
        time_s = step * step_dt
        with torch.inference_mode():
            logging_mask = active_envs.clone()
            tracking = _capture_tracking_snapshot(env)
            _update_visualizer(visualizer, active_envs, tracking, step, time_s)
            actions = _sample_actions(policy, observations, active_envs, previous_actions)
            previous_actions = actions.requested.clone()
            pre_step = _pre_step_values(env, tracking, actions)

            observations, rewards, _, _ = env.step(actions.requested)
            terminated = env.unwrapped.reset_terminated.clone() & active_envs
            any_failure |= terminated
            first_failure_time_s = torch.where(
                terminated & torch.isnan(first_failure_time_s),
                torch.full_like(first_failure_time_s, time_s + step_dt),
                first_failure_time_s,
            )
            post_step, previous_applied_actions = _post_step_actuator_values(
                env,
                actions.requested,
                previous_applied_actions,
                terminated,
            )
            values = torch.cat(
                (
                    pre_step,
                    rewards.unsqueeze(-1),
                    post_step,
                    terminated.to(dtype=torch.float32).unsqueeze(-1),
                ),
                dim=1,
            )
            tensor_log.append(step=step, time_s=time_s, active_mask=logging_mask, values=values)
            active_envs &= ~terminated

    log = tensor_log.finish()
    log.insert(0, "disturbance", disturbance_label or "nominal")
    log.insert(0, "reward_profile", reward_profile)
    log.insert(0, "trajectory", trajectory)
    log.insert(0, "log_schema_version", EVALUATION_LOG_SCHEMA_VERSION)
    return EvaluationResult(
        log=log,
        any_failure=any_failure,
        first_failure_time_s=first_failure_time_s,
        termination_events=int(torch.count_nonzero(any_failure).item()),
    )

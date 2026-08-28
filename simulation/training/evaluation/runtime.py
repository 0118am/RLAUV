"""State sampling and stepping loop for policy evaluation."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.tensor_math import quat_apply_wxyz, quaternion_error_magnitude

EVALUATION_LOG_SCHEMA_VERSION = 9


def indexed_columns(prefix: str, count: int, suffix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{index}{suffix}" for index in range(count))


def evaluation_log_columns(action_dim: int) -> tuple[str, ...]:
    """Return the fixed numeric feature order used by :class:`ChunkedTensorLog`."""

    return (
        "water_current_x", "water_current_y", "water_current_z",
        "desired_x", "desired_y", "desired_z",
        "true_x", "true_y", "true_z",
        "desired_vx", "desired_vy", "desired_vz",
        "target_speed_mps", "requested_speed_mps", "target_acceleration_mps2",
        "target_jerk_mps3", "target_curvature_m_inv", "target_yaw_rate_radps",
        "requested_period_s", "effective_period_s", "trajectory_retimed",
        "true_vx", "true_vy", "true_vz",
        "position_local_x_m", "position_local_y_m", "position_local_z_m",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "angular_velocity_b_x_radps", "angular_velocity_b_y_radps", "angular_velocity_b_z_radps",
        *indexed_columns("action_", action_dim),
        *indexed_columns("latent_policy_mean_", action_dim),
        "position_error", "velocity_error", "attitude_error",
        "nose_to_target_heading_angle_rad", "nose_to_motion_heading_angle_rad",
        "action_norm", "action_rms", "action_rate_rms_per_s", "action_acceleration_rms_per_s2",
        "action_saturation_fraction", "latent_policy_mean_norm", "latent_policy_mean_rms",
        "reward",
        *indexed_columns("realized_thruster_force_", action_dim, "_n"),
        "realized_thruster_force_abs_mean_n", "realized_thruster_force_abs_max_n",
        "thruster_wrench_b_force_x_n", "thruster_wrench_b_force_y_n", "thruster_wrench_b_force_z_n",
        "thruster_wrench_b_torque_x_nm", "thruster_wrench_b_torque_y_nm", "thruster_wrench_b_torque_z_nm",
        "physx_applied_wrench_b_force_x_n", "physx_applied_wrench_b_force_y_n",
        "physx_applied_wrench_b_force_z_n", "physx_applied_wrench_b_torque_x_nm",
        "physx_applied_wrench_b_torque_y_nm", "physx_applied_wrench_b_torque_z_nm",
    )


class ChunkedTensorLog:
    """Transfer one dense tensor per chunk instead of synchronizing per scalar."""

    def __init__(
        self,
        columns: tuple[str, ...],
        *,
        num_envs: int,
        chunk_steps: int = 128,
    ) -> None:
        if num_envs <= 0 or chunk_steps <= 0:
            raise ValueError("num_envs and chunk_steps must be positive.")
        self.columns = columns
        self.num_envs = int(num_envs)
        self.chunk_steps = int(chunk_steps)
        self._payloads: list[torch.Tensor] = []
        self._steps: list[int] = []
        self._times: list[float] = []
        self._frames: list[pd.DataFrame] = []

    def append(
        self,
        *,
        step: int,
        time_s: float,
        active_mask: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if values.shape != (self.num_envs, len(self.columns)):
            raise ValueError(
                f"Expected log tensor {(self.num_envs, len(self.columns))}, got {tuple(values.shape)}."
            )
        if active_mask.shape != (self.num_envs,):
            raise ValueError(f"Expected active mask {(self.num_envs,)}, got {tuple(active_mask.shape)}.")
        # torch.cat owns fresh storage, so later simulator in-place updates
        # cannot mutate buffered history. The final column is the row mask.
        self._payloads.append(
            torch.cat((values, active_mask.to(dtype=values.dtype).unsqueeze(-1)), dim=-1)
        )
        self._steps.append(int(step))
        self._times.append(float(time_s))
        if len(self._payloads) >= self.chunk_steps:
            self.flush()

    def flush(self) -> None:
        if not self._payloads:
            return
        payload = torch.stack(self._payloads).detach().cpu().numpy()
        active = payload[..., -1].astype(bool).reshape(-1)
        features = payload[..., :-1].reshape(-1, len(self.columns))[active]
        step_grid = np.broadcast_to(
            np.asarray(self._steps, dtype=np.int64)[:, None],
            (len(self._steps), self.num_envs),
        ).reshape(-1)[active]
        time_grid = np.broadcast_to(
            np.asarray(self._times, dtype=np.float64)[:, None],
            (len(self._times), self.num_envs),
        ).reshape(-1)[active]
        env_grid = np.broadcast_to(
            np.arange(self.num_envs, dtype=np.int64)[None, :],
            (len(self._steps), self.num_envs),
        ).reshape(-1)[active]
        frame = pd.DataFrame(features, columns=self.columns)
        frame.insert(0, "time", time_grid)
        frame.insert(0, "episode_step", step_grid)
        frame.insert(0, "env_id", env_grid)
        self._frames.append(frame)
        self._payloads.clear()
        self._steps.clear()
        self._times.clear()

    def finish(self) -> pd.DataFrame:
        self.flush()
        if not self._frames:
            return pd.DataFrame(columns=("env_id", "episode_step", "time", *self.columns))
        return pd.concat(self._frames, ignore_index=True)


@dataclass(frozen=True)
class TrackingSnapshot:
    target_pos_w: torch.Tensor
    target_lin_vel_w: torch.Tensor
    target_quat_w: torch.Tensor
    target_acceleration_w: torch.Tensor
    target_jerk_w: torch.Tensor
    target_curvature_m_inv: torch.Tensor
    target_yaw_rate_radps: torch.Tensor
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
    nose_to_target_heading_angle: torch.Tensor
    nose_to_motion_heading_angle: torch.Tensor


@dataclass(frozen=True)
class ActionSnapshot:
    action: torch.Tensor
    latent_mean: torch.Tensor
    norm: torch.Tensor
    rms: torch.Tensor
    rate_rms_per_s: torch.Tensor
    acceleration_rms_per_s2: torch.Tensor
    saturation_fraction: torch.Tensor


@dataclass(frozen=True)
class EvaluationResult:
    log: pd.DataFrame
    any_failure: torch.Tensor
    first_failure_time_s: torch.Tensor
    termination_events: int


def _horizontal_direction_error(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
    min_norm: float = 1.0e-3,
) -> torch.Tensor:
    horizontal_a = vector_a[:, :2]
    horizontal_b = vector_b[:, :2]
    norm_a = torch.norm(horizontal_a, dim=1)
    norm_b = torch.norm(horizontal_b, dim=1)
    cosine = torch.sum(horizontal_a * horizontal_b, dim=1) / torch.clamp(
        norm_a * norm_b,
        min=min_norm**2,
    )
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
    root_lin_vel_w = quat_apply_wxyz(root_quat_w, root_lin_vel_b)
    body_x_b = torch.zeros_like(root_lin_vel_b)
    body_x_b[:, 0] = 1.0
    nose_direction_w = quat_apply_wxyz(root_quat_w, body_x_b)
    target_nose_direction_w = quat_apply_wxyz(target_quat_w, body_x_b)

    return TrackingSnapshot(
        target_pos_w=target_pos_w,
        target_lin_vel_w=target_lin_vel_w,
        target_quat_w=target_quat_w,
        target_acceleration_w=kinematics["target_acceleration_w"],
        target_jerk_w=kinematics["target_jerk_w"],
        target_curvature_m_inv=kinematics["target_curvature_m_inv"],
        target_yaw_rate_radps=kinematics["target_yaw_rate_radps"],
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
        attitude_error=quaternion_error_magnitude(target_quat_w, root_quat_w),
        nose_to_target_heading_angle=_horizontal_direction_error(
            nose_direction_w,
            target_nose_direction_w,
        ),
        nose_to_motion_heading_angle=_horizontal_direction_error(
            nose_direction_w,
            root_lin_vel_w,
        ),
    )


def _sample_actions(
    policy: Any,
    observations: Any,
    active_envs: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_previous_actions: torch.Tensor,
    policy_dt_s: float,
) -> ActionSnapshot:
    action, latent_mean = policy.action_and_latent_mean(observations)
    action = torch.where(
        active_envs.unsqueeze(-1),
        action,
        torch.zeros_like(action),
    )
    latent_mean = torch.where(
        active_envs.unsqueeze(-1),
        latent_mean,
        torch.zeros_like(latent_mean),
    )
    delta = action - previous_actions
    second_delta = action - 2.0 * previous_actions + previous_previous_actions
    return ActionSnapshot(
        action=action,
        latent_mean=latent_mean,
        norm=torch.norm(action, dim=1),
        rms=torch.sqrt(torch.mean(action.square(), dim=1)),
        rate_rms_per_s=torch.sqrt(
            torch.mean((delta / policy_dt_s).square(), dim=1)
        ),
        acceleration_rms_per_s2=torch.sqrt(
            torch.mean((second_delta / (policy_dt_s * policy_dt_s)).square(), dim=1)
        ),
        saturation_fraction=(action.abs() > 0.95).to(dtype=torch.float32).mean(dim=1),
    )


def _transition_end_values(env: Any, tracking: TrackingSnapshot, actions: ActionSnapshot) -> torch.Tensor:
    environment_runtime = env.unwrapped.environment_runtime
    effective_current_w = (
        environment_runtime.effective_state.water_current_w
        if environment_runtime.effective_state is not None
        else environment_runtime.water_current_w
    )
    return torch.cat(
        (
            effective_current_w,
            tracking.target_pos_w,
            tracking.root_pos_w,
            tracking.target_lin_vel_w,
            torch.linalg.vector_norm(tracking.target_lin_vel_w, dim=1, keepdim=True),
            tracking.requested_speed_mps.unsqueeze(-1),
            torch.linalg.vector_norm(tracking.target_acceleration_w, dim=1, keepdim=True),
            torch.linalg.vector_norm(tracking.target_jerk_w, dim=1, keepdim=True),
            tracking.target_curvature_m_inv.unsqueeze(-1),
            tracking.target_yaw_rate_radps.unsqueeze(-1),
            tracking.requested_period_s.unsqueeze(-1),
            tracking.effective_period_s.unsqueeze(-1),
            tracking.retimed.to(dtype=torch.float32).unsqueeze(-1),
            tracking.root_lin_vel_w,
            tracking.root_pos_local,
            tracking.root_quat_w,
            tracking.root_ang_vel_b,
            actions.action,
            actions.latent_mean,
            tracking.position_error.unsqueeze(-1),
            tracking.velocity_error.unsqueeze(-1),
            tracking.attitude_error.unsqueeze(-1),
            tracking.nose_to_target_heading_angle.unsqueeze(-1),
            tracking.nose_to_motion_heading_angle.unsqueeze(-1),
            actions.norm.unsqueeze(-1),
            actions.rms.unsqueeze(-1),
            actions.rate_rms_per_s.unsqueeze(-1),
            actions.acceleration_rms_per_s2.unsqueeze(-1),
            actions.saturation_fraction.unsqueeze(-1),
            torch.linalg.vector_norm(actions.latent_mean, dim=1, keepdim=True),
            torch.sqrt(torch.mean(actions.latent_mean.square(), dim=1, keepdim=True)),
        ),
        dim=1,
    )


def _post_step_propulsion_values(env: Any) -> torch.Tensor:
    """Return realized thruster and PhysX-wrench diagnostics after one policy step."""

    robot = env.unwrapped.robot_runtime
    thruster_force = robot.realized_thruster_force_n
    return torch.cat(
        (
            thruster_force,
            thruster_force.abs().mean(dim=1, keepdim=True),
            thruster_force.abs().amax(dim=1, keepdim=True),
            robot.realized_thruster_wrench_b,
            env.unwrapped._thrust[:, 0, :],
            env.unwrapped._moment[:, 0, :],
        ),
        dim=1,
    )


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
    robot = unwrapped.robot_runtime
    environment = unwrapped.environment_runtime
    columns = (
        "sampled_linear_damping_l2",
        "sampled_quadratic_damping_l2",
        "sampled_fluid_added_mass_scale_surge",
        "sampled_fluid_added_mass_scale_sway",
        "sampled_fluid_added_mass_scale_heave",
        "sampled_fluid_added_mass_scale_roll",
        "sampled_fluid_added_mass_scale_pitch",
        "sampled_fluid_added_mass_scale_yaw",
        "sampled_fluid_added_mass_l2",
        "sampled_thruster_scale_mean",
        "sampled_thruster_scale_min",
        "sampled_thruster_scale_max",
        "sampled_thruster_time_constant_s",
        "pose_sensor_delay_s",
        "sampled_common_thruster_force_scale",
    )
    sensor_delay = torch.full(
        (unwrapped.num_envs, 1), float(unwrapped.cfg.pose_sensor_delay_s), device=robot.device
    )
    values = torch.cat(
        (
            torch.linalg.vector_norm(environment.linear_damping.reshape(unwrapped.num_envs, -1), dim=1, keepdim=True),
            torch.linalg.vector_norm(environment.quadratic_damping.reshape(unwrapped.num_envs, -1), dim=1, keepdim=True),
            environment.fluid_added_mass_randomization_scale,
            torch.linalg.vector_norm(
                environment.fluid_added_mass.reshape(unwrapped.num_envs, -1),
                dim=1,
                keepdim=True,
            ),
            robot.thruster_force_scale.mean(dim=1, keepdim=True),
            robot.thruster_force_scale.amin(dim=1, keepdim=True),
            robot.thruster_force_scale.amax(dim=1, keepdim=True),
            robot.thruster_time_constant.reshape(-1, 1),
            sensor_delay,
            robot.common_thruster_force_scale,
        ),
        dim=1,
    ).detach().cpu().numpy()
    frame = pd.DataFrame(values, columns=columns)
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
    previous_actions = torch.zeros(
        (env.unwrapped.num_envs, env.num_actions),
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    previous_previous_actions = previous_actions.clone()
    active_envs = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    any_failure = torch.zeros_like(active_envs)
    first_failure_time_s = torch.full(
        (env.unwrapped.num_envs,),
        float("nan"),
        dtype=torch.float32,
        device=env.unwrapped.device,
    )

    for step in range(int(math.ceil(duration_s / step_dt))):
        time_s = (step + 1) * step_dt
        with torch.inference_mode():
            actions = _sample_actions(
                policy,
                observations,
                active_envs,
                previous_actions,
                previous_previous_actions,
                step_dt,
            )
            previous_previous_actions = previous_actions.clone()
            previous_actions = actions.action.clone()

            observations, rewards, _, _ = env.step(actions.action)
            terminated = env.unwrapped.reset_terminated.clone() & active_envs
            any_failure |= terminated
            first_failure_time_s = torch.where(
                terminated & torch.isnan(first_failure_time_s),
                torch.full_like(first_failure_time_s, time_s),
                first_failure_time_s,
            )
            active_envs &= ~terminated
            tracking = _capture_tracking_snapshot(env)
            _update_visualizer(visualizer, active_envs, tracking, step + 1, time_s)
            transition_end = _transition_end_values(env, tracking, actions)
            post_step = _post_step_propulsion_values(env)
            values = torch.cat(
                (
                    transition_end,
                    rewards.unsqueeze(-1),
                    post_step,
                ),
                dim=1,
            )
            # DirectRLEnv has already reset terminated environments by this
            # point. Excluding those rows prevents reset poses from being
            # mislabeled as the terminal state of the preceding transition.
            tensor_log.append(
                step=step + 1,
                time_s=time_s,
                active_mask=active_envs,
                values=values,
            )
            if not bool(torch.any(active_envs)):
                break

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

"""Precision trajectory reward and its tensor implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from robot.runtime import T60_RUNTIME


@dataclass(frozen=True)
class TrackingRewardPolicy:
    """Immutable definition of the training reward contract."""

    name: str
    description: str
    position_weight: float
    attitude_recovery_weight: float
    attitude_precision_weight: float
    velocity_weight: float
    angular_velocity_broad_weight: float
    angular_velocity_precision_weight: float
    action_weight: float
    action_rate_weight: float
    action_rate_scale_per_s: float
    control_period_s: float
    position_sigma: float
    attitude_recovery_transition: float
    attitude_recovery_zero: float
    attitude_precision_sigma: float
    velocity_sigma: float
    angular_velocity_broad_sigma: float
    angular_velocity_precision_sigma: float
    termination_penalty: float = 1.0

    @property
    def maximum_positive_reward(self) -> float:
        return (
            self.position_weight
            + self.attitude_recovery_weight
            + self.attitude_precision_weight
            + self.velocity_weight
            + self.angular_velocity_broad_weight
            + self.angular_velocity_precision_weight
        )

    @property
    def minimum_running_reward(self) -> float:
        """Lower bound for processed commands in [-1, 1], excluding termination."""

        maximum_scaled_error = math.pi / self.attitude_recovery_transition
        maximum_huber_loss = maximum_scaled_error - 0.5
        zero_scaled_error = (
            self.attitude_recovery_zero / self.attitude_recovery_transition
        )
        zero_huber_loss = zero_scaled_error - 0.5
        minimum_recovery_reward = self.attitude_recovery_weight * (
            1.0 - maximum_huber_loss / zero_huber_loss
        )
        minimum_precision_reward = self.attitude_precision_weight / (
            1.0 + (math.pi / self.attitude_precision_sigma) ** 2
        )
        maximum_normalized_rate = (
            2.0 / self.control_period_s / self.action_rate_scale_per_s
        )
        return (
            minimum_recovery_reward
            + minimum_precision_reward
            - self.action_weight
            - self.action_rate_weight * maximum_normalized_rate**2
        )


ATTITUDE_RECOVERY_TRANSITION_RAD = math.pi / 18.0
ATTITUDE_RECOVERY_ZERO_RAD = math.pi / 3.0
ATTITUDE_PRECISION_SIGMA_RAD = math.radians(2.5)
POLICY_CONTROL_PERIOD_S = 1.0 / 25.0
ACTION_RATE_SCALE_PER_S = 2.0 / T60_RUNTIME.thruster_time_constant_s


PRECISION_V9 = TrackingRewardPolicy(
    name="precision_v9",
    description=(
        "Equal roll/pitch/yaw attitude rewards for level roll/pitch targets and "
        "trajectory yaw, using signed Huber recovery plus 2.5 degree Cauchy "
        "precision, dual-scale angular velocity, and raw processed-command "
        "magnitude and squared-rate penalties."
    ),
    position_weight=0.35,
    attitude_recovery_weight=0.25,
    attitude_precision_weight=0.25,
    velocity_weight=0.10,
    angular_velocity_broad_weight=0.03,
    angular_velocity_precision_weight=0.02,
    action_weight=0.010,
    action_rate_weight=0.010,
    action_rate_scale_per_s=ACTION_RATE_SCALE_PER_S,
    control_period_s=POLICY_CONTROL_PERIOD_S,
    position_sigma=0.10,
    attitude_recovery_transition=ATTITUDE_RECOVERY_TRANSITION_RAD,
    attitude_recovery_zero=ATTITUDE_RECOVERY_ZERO_RAD,
    attitude_precision_sigma=ATTITUDE_PRECISION_SIGMA_RAD,
    velocity_sigma=0.08,
    angular_velocity_broad_sigma=0.30,
    angular_velocity_precision_sigma=0.15,
)


@torch.jit.script
def cauchy_tolerance(error: torch.Tensor, half_width: float) -> torch.Tensor:
    """Return a bounded score whose value is 0.5 at ``error=half_width``."""

    return 1.0 / (1.0 + (error / half_width) ** 2)


@torch.jit.script
def normalized_huber_attitude_recovery(
    angle_error: torch.Tensor,
    transition: float,
    zero_error: float,
) -> torch.Tensor:
    """Map Huber loss to 1 at zero and 0 at ``zero_error``."""

    scaled_error = angle_error / transition
    huber_loss = torch.where(
        scaled_error <= 1.0,
        0.5 * scaled_error.square(),
        scaled_error - 0.5,
    )
    zero_loss = zero_error / transition - 0.5
    return 1.0 - huber_loss / zero_loss


@torch.jit.script
def level_heading_axis_errors(
    root_quat: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """Return absolute roll, pitch, and wrapped-yaw errors in radians.

    Trajectory guidance always commands zero roll and pitch.  Only target yaw
    is read from ``target_quat``.
    """

    root_quat = root_quat / torch.linalg.vector_norm(
        root_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    target_quat = target_quat / torch.linalg.vector_norm(
        target_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    root_w, root_x, root_y, root_z = root_quat.unbind(dim=-1)
    target_w, target_x, target_y, target_z = target_quat.unbind(dim=-1)

    roll = torch.atan2(
        2.0 * (root_w * root_x + root_y * root_z),
        1.0 - 2.0 * (root_x.square() + root_y.square()),
    )
    pitch = torch.asin(
        torch.clamp(
            2.0 * (root_w * root_y - root_z * root_x),
            min=-1.0,
            max=1.0,
        )
    )
    root_yaw = torch.atan2(
        2.0 * (root_w * root_z + root_x * root_y),
        1.0 - 2.0 * (root_y.square() + root_z.square()),
    )
    target_yaw = torch.atan2(
        2.0 * (target_w * target_z + target_x * target_y),
        1.0 - 2.0 * (target_y.square() + target_z.square()),
    )
    yaw_delta = root_yaw - target_yaw
    yaw_error = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    return torch.stack((roll.abs(), pitch.abs(), yaw_error.abs()), dim=-1)


@torch.jit.script
def compute_tracking_reward_terms(
    rew_scale_pos: float,
    rew_scale_attitude_recovery: float,
    rew_scale_attitude_precision: float,
    rew_scale_track_vel: float,
    rew_scale_angular_velocity_broad: float,
    rew_scale_angular_velocity_precision: float,
    rew_scale_actions: float,
    rew_scale_action_rate: float,
    rew_action_rate_scale_per_s: float,
    policy_dt_s: float,
    rew_pos_sigma: float,
    rew_attitude_recovery_transition: float,
    rew_attitude_recovery_zero: float,
    rew_attitude_precision_sigma: float,
    rew_track_vel_sigma: float,
    rew_angular_velocity_broad_sigma: float,
    rew_angular_velocity_precision_sigma: float,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    target_lin_vel_b: torch.Tensor,
    target_ang_vel_b: torch.Tensor,
    commands: torch.Tensor,
    previous_commands: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return total reward followed by every weighted component."""

    position_error = torch.norm(target_pos_w - root_pos, dim=1)
    attitude_axis_errors = level_heading_axis_errors(root_quat, target_quat_w)
    velocity_error = torch.norm(target_lin_vel_b - root_lin_vel_b, dim=1)
    angular_velocity_error = torch.norm(target_ang_vel_b - root_ang_vel_b, dim=1)

    position_reward = rew_scale_pos * cauchy_tolerance(position_error, rew_pos_sigma)
    attitude_recovery_rewards = (
        rew_scale_attitude_recovery / 3.0
        * normalized_huber_attitude_recovery(
            attitude_axis_errors,
            rew_attitude_recovery_transition,
            rew_attitude_recovery_zero,
        )
    )
    attitude_precision_rewards = rew_scale_attitude_precision / 3.0 * cauchy_tolerance(
        attitude_axis_errors,
        rew_attitude_precision_sigma,
    )
    velocity_reward = rew_scale_track_vel * cauchy_tolerance(
        velocity_error, rew_track_vel_sigma
    )
    angular_velocity_broad_reward = (
        rew_scale_angular_velocity_broad
        * cauchy_tolerance(
            angular_velocity_error,
            rew_angular_velocity_broad_sigma,
        )
    )
    angular_velocity_precision_reward = (
        rew_scale_angular_velocity_precision
        * cauchy_tolerance(
            angular_velocity_error,
            rew_angular_velocity_precision_sigma,
        )
    )

    action_penalty = rew_scale_actions * torch.mean(commands.square(), dim=1)
    action_rate_per_s = (commands - previous_commands) / policy_dt_s
    normalized_action_rate = action_rate_per_s / rew_action_rate_scale_per_s
    action_rate_penalty = rew_scale_action_rate * torch.mean(
        normalized_action_rate.square(), dim=1
    )
    total = (
        position_reward
        + attitude_recovery_rewards.sum(dim=1)
        + attitude_precision_rewards.sum(dim=1)
        + velocity_reward
        + angular_velocity_broad_reward
        + angular_velocity_precision_reward
        - action_penalty
        - action_rate_penalty
    )
    return (
        total,
        position_reward,
        attitude_recovery_rewards,
        attitude_precision_rewards,
        velocity_reward,
        angular_velocity_broad_reward,
        angular_velocity_precision_reward,
        action_penalty,
        action_rate_penalty,
    )


@torch.jit.script
def compute_tracking_rewards(
    rew_scale_pos: float,
    rew_scale_attitude_recovery: float,
    rew_scale_attitude_precision: float,
    rew_scale_track_vel: float,
    rew_scale_angular_velocity_broad: float,
    rew_scale_angular_velocity_precision: float,
    rew_scale_actions: float,
    rew_scale_action_rate: float,
    rew_action_rate_scale_per_s: float,
    policy_dt_s: float,
    rew_pos_sigma: float,
    rew_attitude_recovery_transition: float,
    rew_attitude_recovery_zero: float,
    rew_attitude_precision_sigma: float,
    rew_track_vel_sigma: float,
    rew_angular_velocity_broad_sigma: float,
    rew_angular_velocity_precision_sigma: float,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    target_lin_vel_b: torch.Tensor,
    target_ang_vel_b: torch.Tensor,
    commands: torch.Tensor,
    previous_commands: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the precision reward without duplicating the formula."""

    return compute_tracking_reward_terms(
        rew_scale_pos,
        rew_scale_attitude_recovery,
        rew_scale_attitude_precision,
        rew_scale_track_vel,
        rew_scale_angular_velocity_broad,
        rew_scale_angular_velocity_precision,
        rew_scale_actions,
        rew_scale_action_rate,
        rew_action_rate_scale_per_s,
        policy_dt_s,
        rew_pos_sigma,
        rew_attitude_recovery_transition,
        rew_attitude_recovery_zero,
        rew_attitude_precision_sigma,
        rew_track_vel_sigma,
        rew_angular_velocity_broad_sigma,
        rew_angular_velocity_precision_sigma,
        root_pos,
        root_quat,
        root_lin_vel_b,
        root_ang_vel_b,
        target_pos_w,
        target_quat_w,
        target_lin_vel_b,
        target_ang_vel_b,
        commands,
        previous_commands,
    )[0]


def canonical_tracking_reward_policy_name(name: str) -> str:
    normalized = str(name)
    if normalized != PRECISION_V9.name:
        raise ValueError(
            f"Unknown tracking reward policy {name!r}. "
            f"The configured policy is {PRECISION_V9.name!r}."
        )
    return normalized


def apply_tracking_reward_policy(cfg) -> TrackingRewardPolicy:
    """Apply the selected immutable coefficient set to ``cfg``."""

    name = canonical_tracking_reward_policy_name(cfg.tracking_reward_profile)
    policy = PRECISION_V9
    cfg.tracking_reward_profile = name
    cfg.rew_scale_pos = policy.position_weight
    cfg.rew_scale_attitude_recovery = policy.attitude_recovery_weight
    cfg.rew_scale_attitude_precision = policy.attitude_precision_weight
    cfg.rew_scale_track_vel = policy.velocity_weight
    cfg.rew_scale_angular_velocity_broad = policy.angular_velocity_broad_weight
    cfg.rew_scale_angular_velocity_precision = policy.angular_velocity_precision_weight
    cfg.rew_scale_actions = policy.action_weight
    cfg.rew_scale_action_rate = policy.action_rate_weight
    cfg.rew_action_rate_scale_per_s = policy.action_rate_scale_per_s
    cfg.rew_scale_terminated = policy.termination_penalty
    cfg.rew_pos_sigma = policy.position_sigma
    cfg.rew_attitude_recovery_transition = policy.attitude_recovery_transition
    cfg.rew_attitude_recovery_zero = policy.attitude_recovery_zero
    cfg.rew_attitude_precision_sigma = policy.attitude_precision_sigma
    cfg.rew_track_vel_sigma = policy.velocity_sigma
    cfg.rew_angular_velocity_broad_sigma = policy.angular_velocity_broad_sigma
    cfg.rew_angular_velocity_precision_sigma = policy.angular_velocity_precision_sigma
    return policy


def get_tracking_reward_function(name: str):
    canonical_tracking_reward_policy_name(name)
    return compute_tracking_reward_terms

"""Policy 6: Huber tracking residuals with applied-actuator regularization."""

from __future__ import annotations

import torch

from .base import TrackingRewardPolicy, huber_tracking_score, quaternion_error_magnitude


# The positive optimum remains 1.00, matching policy_5. Unlike Cauchy, Huber
# scores become negative beyond 1.5 transition widths and preserve a linear
# recovery signal for large tracking residuals.
POLICY = TrackingRewardPolicy(
    name="policy_6",
    description="Huber tracking residuals with applied-action L2 effort, smoothness, and safety termination cost.",
    position_weight=0.55,
    attitude_weight=0.15,
    velocity_weight=0.25,
    angular_velocity_weight=0.05,
    forward_alignment_weight=0.0,
    motion_alignment_weight=0.0,
    action_weight=0.02,
    action_rate_weight=0.03,
    # These are Huber transition widths. score(error=sigma) == 0.5.
    position_sigma=0.7,
    attitude_sigma=0.75,
    velocity_sigma=0.35,
    angular_velocity_sigma=0.5,
    action_source="applied",
    termination_penalty=1.0,
    requires_action_rate_limit=True,
)


@torch.jit.script
def compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_track_vel: float,
    rew_scale_ang_vel: float,
    rew_forward: float,
    rew_motion_alignment: float,
    rew_scale_actions: float,
    rew_scale_action_rate: float,
    rew_pos_sigma: float,
    rew_ang_sigma: float,
    rew_track_vel_sigma: float,
    rew_ang_vel_sigma: float,
    rew_forward_min_speed: float,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    target_lin_vel_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    applied_action_rate_limit: torch.Tensor,
) -> torch.Tensor:
    """Reward Huber tracking residuals and physically applied commands."""

    pos_error = torch.norm(target_pos_w - root_pos, dim=1)
    ang_error = quaternion_error_magnitude(target_quat_w[:, :], root_quat[:, :])
    track_vel_error = torch.norm(target_lin_vel_b - root_lin_vel_b, dim=1)
    ang_vel_error = torch.norm(root_ang_vel_b, dim=1)

    rew_pos = rew_scale_pos * huber_tracking_score(pos_error, rew_pos_sigma)
    rew_ang = rew_scale_ang * huber_tracking_score(ang_error, rew_ang_sigma)
    rew_track_vel = rew_scale_track_vel * huber_tracking_score(track_vel_error, rew_track_vel_sigma)
    rew_ang_vel = rew_scale_ang_vel * huber_tracking_score(ang_vel_error, rew_ang_vel_sigma)

    # Keep policy_5's actuator semantics: score only commands that survive
    # delay and rate limiting, never the raw PPO distribution sample.
    rew_action = -rew_scale_actions * torch.mean(torch.square(actions), dim=1)
    applied_delta_normalized = (actions - previous_actions) / torch.clamp(applied_action_rate_limit, min=1.0e-6)
    rew_action_rate = -rew_scale_action_rate * torch.mean(torch.square(applied_delta_normalized), dim=1)
    return rew_pos + rew_ang + rew_track_vel + rew_ang_vel + rew_action + rew_action_rate

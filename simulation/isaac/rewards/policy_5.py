"""Policy 5: compact task reward with applied-actuator regularization."""

from __future__ import annotations

import torch

from .base import TrackingRewardPolicy, cauchy_tolerance, quaternion_error_magnitude


# Algebraic per-policy-step reward upper bound: 1.00 = 0.55 + 0.25 +
# 0.15 + 0.05. It requires perfect tracking and zero applied actuator effort;
# the -1.00 true-termination penalty is separate and timeouts remain neutral.
POLICY = TrackingRewardPolicy(
    name="policy_5",
    description="Compact Cauchy tracking with applied-action effort, smoothness, and safety termination cost.",
    position_weight=0.55,
    attitude_weight=0.15,
    velocity_weight=0.25,
    angular_velocity_weight=0.05,
    # Target attitude already aligns body +X with reference velocity. Keeping
    # these zero avoids double-counting heading and penalizing valid sideslip.
    forward_alignment_weight=0.0,
    motion_alignment_weight=0.0,
    action_weight=0.02,
    action_rate_weight=0.03,
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
    rew_scale_forward: float,
    rew_scale_motion_alignment: float,
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
    """Reward tracking and the physically realized command, not PPO requests."""

    pos_error = torch.norm(target_pos_w - root_pos, dim=1)
    ang_error = quaternion_error_magnitude(target_quat_w[:, :], root_quat[:, :])
    track_vel_error = torch.norm(target_lin_vel_b - root_lin_vel_b, dim=1)
    ang_vel_error = torch.norm(root_ang_vel_b, dim=1)

    rew_pos = rew_scale_pos * cauchy_tolerance(pos_error, rew_pos_sigma)
    rew_ang = rew_scale_ang * cauchy_tolerance(ang_error, rew_ang_sigma)
    rew_track_vel = rew_scale_track_vel * cauchy_tolerance(track_vel_error, rew_track_vel_sigma)
    rew_ang_vel = rew_scale_ang_vel * cauchy_tolerance(ang_vel_error, rew_ang_vel_sigma)

    # Normalize by each environment's actual rate limit over one policy step.
    # An unlimited command rate uses 1.0 (the normalized-command range) as its
    # reference, avoiding the old overloaded zero-rate convention.
    rew_action = -rew_scale_actions * torch.mean(torch.square(actions), dim=1)
    applied_delta_normalized = (actions - previous_actions) / torch.clamp(applied_action_rate_limit, min=1.0e-6)
    rew_action_rate = -rew_scale_action_rate * torch.mean(torch.square(applied_delta_normalized), dim=1)
    return rew_pos + rew_ang + rew_track_vel + rew_ang_vel + rew_action + rew_action_rate

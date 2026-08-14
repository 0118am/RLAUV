"""Policy 2: Gaussian precision tracking inspired by exponential imitation rewards."""

from __future__ import annotations

import torch

from .base import (
    TrackingRewardPolicy,
    body_x_alignment_score,
    gaussian_tolerance,
    quaternion_error_magnitude,
)


# Algebraic per-policy-step reward upper bound: 4.87 = 3.00 + 0.25 +
# 0.80 + 0.02 + 0.40 + 0.40.  It requires every positive tracking/alignment
# term to be one and both non-positive action penalties to be zero; dynamics
# may make that combination infeasible while moving.
POLICY = TrackingRewardPolicy(
    name="policy_2",
    description="Gaussian tracking with command-heading and mandatory actual-motion alignment.",
    position_weight=3.0,
    attitude_weight=0.25,
    velocity_weight=0.8,
    angular_velocity_weight=0.02,
    forward_alignment_weight=0.4,
    motion_alignment_weight=0.4,
    action_weight=0.003,
    action_rate_weight=0.0012,
    position_sigma=0.7,
    attitude_sigma=0.75,
    velocity_sigma=0.35,
    angular_velocity_sigma=0.5,
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
) -> torch.Tensor:
    """Compute a narrow-tailed Gaussian tracking reward and L2 controls."""

    pos_error = torch.norm(target_pos_w - root_pos, dim=1)
    ang_error = quaternion_error_magnitude(target_quat_w[:, :], root_quat[:, :])
    track_vel_error = torch.norm(target_lin_vel_b - root_lin_vel_b, dim=1)
    ang_vel_error = torch.norm(root_ang_vel_b, dim=1)

    rew_pos = rew_scale_pos * gaussian_tolerance(pos_error, rew_pos_sigma)
    rew_ang = rew_scale_ang * gaussian_tolerance(ang_error, rew_ang_sigma)
    rew_track_vel = rew_scale_track_vel * gaussian_tolerance(track_vel_error, rew_track_vel_sigma)
    rew_ang_vel = rew_scale_ang_vel * gaussian_tolerance(ang_vel_error, rew_ang_vel_sigma)
    rew_forward = rew_scale_forward * body_x_alignment_score(
        target_lin_vel_b,
        rew_forward_min_speed,
        1.0,
    )
    rew_motion_alignment = rew_scale_motion_alignment * body_x_alignment_score(
        root_lin_vel_b,
        rew_forward_min_speed,
        0.0,
    )

    rew_action = -rew_scale_actions * torch.norm(actions, dim=1) ** 2
    rew_action_rate = -rew_scale_action_rate * torch.norm(actions - previous_actions, dim=1) ** 2
    return (
        rew_pos
        + rew_ang
        + rew_track_vel
        + rew_ang_vel
        + rew_forward
        + rew_motion_alignment
        + rew_action
        + rew_action_rate
    )

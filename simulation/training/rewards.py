"""Reward policies, selection, and tensor engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrackingRewardPolicy:
    """Immutable definition of one reproducible tracking reward policy."""

    name: str
    description: str
    variant: int
    position_weight: float
    attitude_weight: float
    velocity_weight: float
    angular_velocity_weight: float
    forward_alignment_weight: float
    motion_alignment_weight: float
    action_weight: float
    action_rate_weight: float
    position_sigma: float
    attitude_sigma: float
    velocity_sigma: float
    angular_velocity_sigma: float
    forward_min_speed: float = 1.0e-3
    # Existing policies preserve requested-action semantics. New policies may
    # instead score the post delay/rate-limiter command that the vehicle uses.
    action_source: str = "requested"
    # Positive magnitude subtracted only for true terminations, never timeout.
    termination_penalty: float = 0.0
    requires_action_rate_limit: bool = False

    @property
    def maximum_positive_reward(self) -> float:
        """Maximum positive reward per policy step, excluding penalties."""

        return (
            self.position_weight
            + self.attitude_weight
            + self.velocity_weight
            + self.angular_velocity_weight
            + self.forward_alignment_weight
            + self.motion_alignment_weight
        )


@torch.jit.script
def quaternion_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Return the shortest angular distance between scalar-first quaternions."""

    q1_normalized = q1 / torch.clamp(torch.norm(q1, dim=-1, keepdim=True), min=1.0e-9)
    q2_normalized = q2 / torch.clamp(torch.norm(q2, dim=-1, keepdim=True), min=1.0e-9)
    cosine_half_angle = torch.abs(torch.sum(q1_normalized * q2_normalized, dim=-1))
    return 2.0 * torch.acos(torch.clamp(cosine_half_angle, min=0.0, max=1.0))


@torch.jit.script
def cauchy_tolerance(error: torch.Tensor, half_width: float) -> torch.Tensor:
    """Return a long-tailed score whose value is 0.5 at ``error=half_width``."""

    return 1.0 / (1.0 + (error / half_width) ** 2)


@torch.jit.script
def huber_tracking_score(error: torch.Tensor, transition_width: float) -> torch.Tensor:
    """Return a unit-at-zero Huber score with a linear large-error tail.

    The score is ``0.5`` at ``error=transition_width``, matching the
    half-reward convention of the tolerance kernels. It becomes negative past
    ``1.5 * transition_width`` so large residuals retain a recovery signal.
    """

    normalized_error = torch.abs(error) / transition_width
    huber_loss = torch.where(
        normalized_error <= 1.0,
        0.5 * torch.square(normalized_error),
        normalized_error - 0.5,
    )
    return 1.0 - huber_loss


@torch.jit.script
def gaussian_tolerance(error: torch.Tensor, half_width: float) -> torch.Tensor:
    """Return a Gaussian score whose value is 0.5 at ``error=half_width``."""

    return torch.exp(-0.6931471805599453 * (error / half_width) ** 2)


@torch.jit.script
def body_x_alignment_score(
    velocity_b: torch.Tensor,
    min_speed: float,
    stationary_score: float,
) -> torch.Tensor:
    """Score alignment of a body-frame velocity with the vehicle's +X nose."""

    speed = torch.norm(velocity_b, dim=1)
    cosine = velocity_b[:, 0] / torch.clamp(speed, min=min_speed)
    score = (0.5 * (1.0 + torch.clamp(cosine, min=-1.0, max=1.0))) ** 2
    fallback = torch.full_like(score, stationary_score)
    return torch.where(speed > min_speed, score, fallback)


POLICY_0 = TrackingRewardPolicy(
    name="policy_0",
    description="Original tracking balance plus mandatory nose/actual-motion alignment.",
    variant=0,
    position_weight=3.2,
    attitude_weight=0.25,
    velocity_weight=0.6,
    angular_velocity_weight=0.02,
    forward_alignment_weight=0.0,
    motion_alignment_weight=0.8,
    action_weight=0.003,
    action_rate_weight=0.0012,
    position_sigma=0.7,
    attitude_sigma=0.75,
    velocity_sigma=0.35,
    angular_velocity_sigma=0.5,
)

POLICY_1 = TrackingRewardPolicy(
    name="policy_1",
    description="Balances command-heading, actual-motion alignment, and smooth actuator commands.",
    variant=1,
    position_weight=3.0,
    attitude_weight=0.25,
    velocity_weight=0.8,
    angular_velocity_weight=0.02,
    forward_alignment_weight=0.4,
    motion_alignment_weight=0.4,
    action_weight=0.01,
    action_rate_weight=0.05,
    position_sigma=0.7,
    attitude_sigma=0.75,
    velocity_sigma=0.35,
    angular_velocity_sigma=0.5,
)

POLICY_2 = TrackingRewardPolicy(
    name="policy_2",
    description="Gaussian tracking with command-heading and actual-motion alignment.",
    variant=2,
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

POLICY_3 = TrackingRewardPolicy(
    name="policy_3",
    description="Geometrically couples commanded heading and actual body-x motion alignment.",
    variant=3,
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

POLICY_4 = TrackingRewardPolicy(
    name="policy_4",
    description="Cauchy tracking with normalized L1 anti-chatter penalties.",
    variant=4,
    position_weight=3.0,
    attitude_weight=0.25,
    velocity_weight=0.8,
    angular_velocity_weight=0.02,
    forward_alignment_weight=0.4,
    motion_alignment_weight=0.4,
    action_weight=0.08,
    action_rate_weight=1.6,
    position_sigma=0.7,
    attitude_sigma=0.75,
    velocity_sigma=0.35,
    angular_velocity_sigma=0.5,
)

POLICY_5 = TrackingRewardPolicy(
    name="policy_5",
    description="Compact Cauchy tracking with applied-action regularization and safety cost.",
    variant=5,
    position_weight=0.55,
    attitude_weight=0.15,
    velocity_weight=0.25,
    angular_velocity_weight=0.05,
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

POLICY_6 = TrackingRewardPolicy(
    name="policy_6",
    description="Huber tracking with applied-action regularization and safety cost.",
    variant=6,
    position_weight=0.55,
    attitude_weight=0.15,
    velocity_weight=0.25,
    angular_velocity_weight=0.05,
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

TRACKING_REWARD_POLICIES = {
    policy.name: policy
    for policy in (POLICY_0, POLICY_1, POLICY_2, POLICY_3, POLICY_4, POLICY_5, POLICY_6)
}


@torch.jit.script
def compute_tracking_rewards(
    variant: int,
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
    """Evaluate one selected profile without repeating tracking-error logic."""

    pos_error = torch.norm(target_pos_w - root_pos, dim=1)
    ang_error = quaternion_error_magnitude(target_quat_w, root_quat)
    velocity_error = torch.norm(target_lin_vel_b - root_lin_vel_b, dim=1)
    angular_velocity_error = torch.norm(root_ang_vel_b, dim=1)

    if variant == 2:
        position_score = gaussian_tolerance(pos_error, rew_pos_sigma)
        attitude_score = gaussian_tolerance(ang_error, rew_ang_sigma)
        velocity_score = gaussian_tolerance(velocity_error, rew_track_vel_sigma)
        angular_velocity_score = gaussian_tolerance(
            angular_velocity_error,
            rew_ang_vel_sigma,
        )
    elif variant == 6:
        position_score = huber_tracking_score(pos_error, rew_pos_sigma)
        attitude_score = huber_tracking_score(ang_error, rew_ang_sigma)
        velocity_score = huber_tracking_score(velocity_error, rew_track_vel_sigma)
        angular_velocity_score = huber_tracking_score(
            angular_velocity_error,
            rew_ang_vel_sigma,
        )
    else:
        position_score = cauchy_tolerance(pos_error, rew_pos_sigma)
        attitude_score = cauchy_tolerance(ang_error, rew_ang_sigma)
        velocity_score = cauchy_tolerance(velocity_error, rew_track_vel_sigma)
        angular_velocity_score = cauchy_tolerance(
            angular_velocity_error,
            rew_ang_vel_sigma,
        )

    reward = (
        rew_scale_pos * position_score
        + rew_scale_ang * attitude_score
        + rew_scale_track_vel * velocity_score
        + rew_scale_ang_vel * angular_velocity_score
    )

    if variant in (1, 2, 3, 4):
        target_alignment = body_x_alignment_score(
            target_lin_vel_b,
            rew_forward_min_speed,
            1.0,
        )
        actual_alignment = body_x_alignment_score(
            root_lin_vel_b,
            rew_forward_min_speed,
            0.0,
        )
        if variant == 3:
            target_speed = torch.norm(target_lin_vel_b, dim=1)
            actual_alignment = torch.where(
                target_speed > rew_forward_min_speed,
                actual_alignment,
                torch.ones_like(actual_alignment),
            )
            motion_alignment = torch.sqrt(
                torch.clamp(target_alignment * actual_alignment, min=0.0, max=1.0)
            )
        else:
            motion_alignment = actual_alignment
        reward = (
            reward
            + rew_scale_forward * target_alignment
            + rew_scale_motion_alignment * motion_alignment
        )
    elif variant == 0:
        reward = reward + rew_scale_motion_alignment * body_x_alignment_score(
            root_lin_vel_b,
            rew_forward_min_speed,
            0.0,
        )

    action_delta = actions - previous_actions
    if variant == 4:
        action_penalty = rew_scale_actions * torch.mean(torch.abs(actions), dim=1)
        rate_penalty = (
            rew_scale_action_rate * 0.5 * torch.mean(torch.abs(action_delta), dim=1)
        )
    elif variant in (5, 6):
        normalized_delta = action_delta / torch.clamp(
            applied_action_rate_limit,
            min=1.0e-6,
        )
        action_penalty = rew_scale_actions * torch.mean(torch.square(actions), dim=1)
        rate_penalty = rew_scale_action_rate * torch.mean(
            torch.square(normalized_delta),
            dim=1,
        )
    else:
        action_penalty = rew_scale_actions * torch.norm(actions, dim=1).square()
        rate_penalty = rew_scale_action_rate * torch.norm(action_delta, dim=1).square()
    return reward - action_penalty - rate_penalty


def canonical_tracking_reward_policy_name(name: str) -> str:
    """Validate and return a selectable policy_N name."""

    normalized = str(name)
    if normalized == "custom":
        return normalized
    if normalized not in TRACKING_REWARD_POLICIES:
        choices = ", ".join(available_tracking_reward_policies())
        raise ValueError(f"Unknown tracking reward policy {name!r}. Available policies: {choices}")
    return normalized


def available_tracking_reward_policies() -> tuple[str, ...]:
    """Return canonical selectable policies, including direct-config mode."""

    return (*TRACKING_REWARD_POLICIES.keys(), "custom")


def apply_tracking_reward_policy(cfg) -> TrackingRewardPolicy | None:
    """Apply the selected policy's immutable coefficient set to ``cfg``."""

    name = canonical_tracking_reward_policy_name(cfg.tracking_reward_profile)
    if name == "custom":
        return None

    policy = TRACKING_REWARD_POLICIES[name]
    if policy.action_source not in {"requested", "applied"}:
        raise ValueError(f"Unsupported reward action source {policy.action_source!r} for {policy.name}.")
    cfg.tracking_reward_profile = name
    cfg.rew_scale_pos = policy.position_weight
    cfg.rew_scale_ang = policy.attitude_weight
    cfg.rew_scale_track_vel = policy.velocity_weight
    cfg.rew_scale_ang_vel = policy.angular_velocity_weight
    cfg.rew_scale_forward = policy.forward_alignment_weight
    cfg.rew_scale_motion_alignment = policy.motion_alignment_weight
    cfg.rew_scale_actions = policy.action_weight
    cfg.rew_scale_action_rate = policy.action_rate_weight
    cfg.rew_action_source = policy.action_source
    cfg.rew_scale_terminated = policy.termination_penalty
    cfg.rew_pos_sigma = policy.position_sigma
    cfg.rew_ang_sigma = policy.attitude_sigma
    cfg.rew_track_vel_sigma = policy.velocity_sigma
    cfg.rew_ang_vel_sigma = policy.angular_velocity_sigma
    cfg.rew_forward_min_speed = policy.forward_min_speed
    return policy


def get_tracking_reward_function(name: str):
    """Return the one TorchScript reward engine and the selected variant."""

    canonical = canonical_tracking_reward_policy_name(name)
    if canonical == "custom":
        canonical = "policy_1"
    return compute_tracking_rewards, TRACKING_REWARD_POLICIES[canonical].variant

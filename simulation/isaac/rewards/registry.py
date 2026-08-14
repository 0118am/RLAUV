"""Registry and selection helpers for versioned reward policy files."""

from __future__ import annotations

from .base import TrackingRewardPolicy
from .policy_0 import POLICY as POLICY_0, compute_rewards as compute_policy_0_rewards
from .policy_1 import POLICY as POLICY_1, compute_rewards as compute_policy_1_rewards
from .policy_2 import POLICY as POLICY_2, compute_rewards as compute_policy_2_rewards
from .policy_3 import POLICY as POLICY_3, compute_rewards as compute_policy_3_rewards
from .policy_4 import POLICY as POLICY_4, compute_rewards as compute_policy_4_rewards
from .policy_5 import POLICY as POLICY_5, compute_rewards as compute_policy_5_rewards
from .policy_6 import POLICY as POLICY_6, compute_rewards as compute_policy_6_rewards


TRACKING_REWARD_POLICIES: dict[str, TrackingRewardPolicy] = {
    POLICY_0.name: POLICY_0,
    POLICY_1.name: POLICY_1,
    POLICY_2.name: POLICY_2,
    POLICY_3.name: POLICY_3,
    POLICY_4.name: POLICY_4,
    POLICY_5.name: POLICY_5,
    POLICY_6.name: POLICY_6,
}

TRACKING_REWARD_FUNCTIONS = {
    POLICY_0.name: compute_policy_0_rewards,
    POLICY_1.name: compute_policy_1_rewards,
    POLICY_2.name: compute_policy_2_rewards,
    POLICY_3.name: compute_policy_3_rewards,
    POLICY_4.name: compute_policy_4_rewards,
    POLICY_5.name: compute_policy_5_rewards,
    POLICY_6.name: compute_policy_6_rewards,
}
if TRACKING_REWARD_FUNCTIONS.keys() != TRACKING_REWARD_POLICIES.keys():
    raise RuntimeError("Every registered reward policy must have exactly one reward function.")

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
    """Return the TorchScript reward function owned by the selected policy file."""

    canonical = canonical_tracking_reward_policy_name(name)
    if canonical == "custom":
        return TRACKING_REWARD_FUNCTIONS["policy_1"]
    return TRACKING_REWARD_FUNCTIONS[canonical]

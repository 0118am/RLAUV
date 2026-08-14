"""Public API for versioned AUV reward policy files."""

from .base import (
    TrackingRewardPolicy,
    body_x_alignment_score,
    cauchy_tolerance,
    gaussian_tolerance,
    huber_tracking_score,
    quaternion_error_magnitude,
)
from .registry import (
    LEGACY_REWARD_POLICY_ALIASES,
    TRACKING_REWARD_POLICIES,
    TRACKING_REWARD_PROFILES,
    TrackingRewardProfile,
    apply_tracking_reward_policy,
    apply_tracking_reward_profile,
    available_tracking_reward_policies,
    available_tracking_reward_profiles,
    canonical_tracking_reward_policy_name,
    get_tracking_reward_function,
)

__all__ = [
    "LEGACY_REWARD_POLICY_ALIASES",
    "TRACKING_REWARD_POLICIES",
    "TRACKING_REWARD_PROFILES",
    "TrackingRewardPolicy",
    "TrackingRewardProfile",
    "body_x_alignment_score",
    "cauchy_tolerance",
    "gaussian_tolerance",
    "huber_tracking_score",
    "apply_tracking_reward_policy",
    "apply_tracking_reward_profile",
    "available_tracking_reward_policies",
    "available_tracking_reward_profiles",
    "canonical_tracking_reward_policy_name",
    "get_tracking_reward_function",
    "quaternion_error_magnitude",
]

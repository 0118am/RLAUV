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
    TRACKING_REWARD_POLICIES,
    apply_tracking_reward_policy,
    available_tracking_reward_policies,
    canonical_tracking_reward_policy_name,
    get_tracking_reward_function,
)

__all__ = [
    "TRACKING_REWARD_POLICIES",
    "TrackingRewardPolicy",
    "body_x_alignment_score",
    "cauchy_tolerance",
    "gaussian_tolerance",
    "huber_tracking_score",
    "apply_tracking_reward_policy",
    "available_tracking_reward_policies",
    "canonical_tracking_reward_policy_name",
    "get_tracking_reward_function",
    "quaternion_error_magnitude",
]

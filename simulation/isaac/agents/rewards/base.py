"""Shared types and math used by versioned AUV reward policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrackingRewardPolicy:
    """Immutable definition of one reproducible tracking reward policy."""

    name: str
    description: str
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

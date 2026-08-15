"""Regression check for the reward policy used by the fixed experiment."""

from __future__ import annotations

import torch

from simulation.isaac.rewards.policy_6 import POLICY, compute_rewards


def _reward(
    *,
    position_weight: float,
    position_error: float,
    action_rate_weight: float = 0.0,
    action: float = 0.0,
) -> torch.Tensor:
    zeros3 = torch.zeros((1, 3))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    actions = torch.full((1, 8), action)
    return compute_rewards(
        position_weight,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        action_rate_weight,
        POLICY.position_sigma,
        POLICY.attitude_sigma,
        POLICY.velocity_sigma,
        POLICY.angular_velocity_sigma,
        POLICY.forward_min_speed,
        torch.tensor([[position_error, 0.0, 0.0]]),
        identity,
        zeros3,
        zeros3,
        zeros3,
        identity,
        zeros3,
        actions,
        torch.zeros_like(actions),
        torch.full((1, 1), 0.08),
    )


def test_policy_6_preserves_tracking_and_applied_action_semantics() -> None:
    assert POLICY.maximum_positive_reward == 1.0
    assert POLICY.action_source == "applied"
    assert torch.allclose(
        _reward(position_weight=1.0, position_error=POLICY.position_sigma),
        torch.tensor([0.5]),
    )
    assert torch.allclose(
        _reward(position_weight=0.0, position_error=0.0, action_rate_weight=1.0, action=0.08),
        torch.tensor([-1.0]),
    )

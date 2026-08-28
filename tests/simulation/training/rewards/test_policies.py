"""Regression checks for the precision trajectory reward."""

from __future__ import annotations

import math

import torch

from simulation.training.rewards import (
    ACTION_RATE_SCALE_PER_S,
    ATTITUDE_PRECISION_SIGMA_RAD,
    ATTITUDE_RECOVERY_TRANSITION_RAD,
    ATTITUDE_RECOVERY_ZERO_RAD,
    POLICY_CONTROL_PERIOD_S,
    PRECISION_V9,
    TrackingRewardPolicy,
    compute_tracking_reward_terms,
    level_heading_axis_errors,
    normalized_huber_attitude_recovery,
)


POLICY = PRECISION_V9


def _reward_terms(
    *,
    policy: TrackingRewardPolicy = POLICY,
    position_error: float = 0.0,
    attitude_error_rad: float = 0.0,
    attitude_axis: int = 2,
    angular_velocity_error: float = 0.0,
    action: float = 0.0,
    previous_action: float = 0.0,
    tracking_weights: tuple[float, float, float, float, float, float] | None = None,
    action_weight: float | None = None,
    action_rate_weight: float | None = None,
    policy_dt_s: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if tracking_weights is None:
        tracking_weights = (
            policy.position_weight,
            policy.attitude_recovery_weight,
            policy.attitude_precision_weight,
            policy.velocity_weight,
            policy.angular_velocity_broad_weight,
            policy.angular_velocity_precision_weight,
        )
    if action_weight is None:
        action_weight = policy.action_weight
    if action_rate_weight is None:
        action_rate_weight = policy.action_rate_weight
    if policy_dt_s is None:
        policy_dt_s = policy.control_period_s

    zeros3 = torch.zeros((1, 3))
    half_angle = 0.5 * attitude_error_rad
    root_quaternion = torch.zeros((1, 4))
    root_quaternion[0, 0] = math.cos(half_angle)
    root_quaternion[0, attitude_axis + 1] = math.sin(half_angle)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    actions = torch.full((1, 8), action)
    previous_actions = torch.full((1, 8), previous_action)
    root_angular_velocity = torch.tensor([[angular_velocity_error, 0.0, 0.0]])
    return compute_tracking_reward_terms(
        tracking_weights[0],
        tracking_weights[1],
        tracking_weights[2],
        tracking_weights[3],
        tracking_weights[4],
        tracking_weights[5],
        action_weight,
        action_rate_weight,
        policy.action_rate_scale_per_s,
        policy_dt_s,
        policy.position_sigma,
        policy.attitude_recovery_transition,
        policy.attitude_recovery_zero,
        policy.attitude_precision_sigma,
        policy.velocity_sigma,
        policy.angular_velocity_broad_sigma,
        policy.angular_velocity_precision_sigma,
        torch.tensor([[position_error, 0.0, 0.0]]),
        root_quaternion,
        zeros3,
        root_angular_velocity,
        zeros3,
        identity,
        zeros3,
        zeros3,
        actions,
        previous_actions,
    )


def test_precision_v9_has_declared_bounds_and_perfect_reward() -> None:
    assert POLICY.maximum_positive_reward == 1.0
    maximum_huber_loss = math.pi / POLICY.attitude_recovery_transition - 0.5
    zero_huber_loss = (
        POLICY.attitude_recovery_zero / POLICY.attitude_recovery_transition - 0.5
    )
    maximum_normalized_rate = (
        2.0 / POLICY.control_period_s / POLICY.action_rate_scale_per_s
    )
    expected_minimum = (
        POLICY.attitude_recovery_weight
        * (1.0 - maximum_huber_loss / zero_huber_loss)
        + POLICY.attitude_precision_weight
        / (1.0 + (math.pi / POLICY.attitude_precision_sigma) ** 2)
        - POLICY.action_weight
        - POLICY.action_rate_weight * maximum_normalized_rate**2
    )
    assert math.isclose(POLICY.minimum_running_reward, expected_minimum)
    torch.testing.assert_close(_reward_terms()[0], torch.ones(1))

    penalties_only = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action=1.0,
        previous_action=-1.0,
    )[0]
    torch.testing.assert_close(
        penalties_only,
        torch.tensor(
            [-(POLICY.action_weight + POLICY.action_rate_weight * 2.0**2)]
        ),
    )

def test_position_cauchy_half_width_remains_ten_centimetres() -> None:
    terms = _reward_terms(
        position_error=POLICY.position_sigma,
        tracking_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
    )
    torch.testing.assert_close(terms[0], torch.tensor([0.5]))


def test_attitude_recovery_is_zero_at_sixty_and_signed_beyond() -> None:
    transition = POLICY.attitude_recovery_transition
    zero_error = POLICY.attitude_recovery_zero
    scores = normalized_huber_attitude_recovery(
        torch.tensor(
            [0.0, transition, math.pi / 9.0, zero_error, math.pi / 2.0, math.pi]
        ),
        transition,
        zero_error,
    )
    expected = torch.tensor(
        [
            1.0,
            1.0 - 0.5 / 5.5,
            1.0 - 1.5 / 5.5,
            0.0,
            1.0 - 8.5 / 5.5,
            1.0 - 17.5 / 5.5,
        ]
    )
    torch.testing.assert_close(scores, expected)


def test_attitude_precision_is_split_by_axis_and_half_at_two_point_five_degrees() -> None:
    terms = _reward_terms(
        attitude_error_rad=math.radians(2.5),
        tracking_weights=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
    )
    torch.testing.assert_close(
        terms[3],
        torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]]),
    )
    torch.testing.assert_close(terms[0], torch.tensor([5.0 / 6.0]))


def test_level_heading_reward_separates_roll_pitch_and_yaw() -> None:
    attitude_scores = []
    for axis in range(3):
        terms = _reward_terms(
            attitude_error_rad=math.pi / 18.0,
            attitude_axis=axis,
            tracking_weights=(
                0.0,
                POLICY.attitude_recovery_weight,
                POLICY.attitude_precision_weight,
                0.0,
                0.0,
                0.0,
            ),
            action_weight=0.0,
            action_rate_weight=0.0,
        )
        axis_rewards = terms[2] + terms[3]
        assert axis_rewards[0, axis] < axis_rewards[0, (axis + 1) % 3]
        attitude_scores.append(terms[0])

    torch.testing.assert_close(attitude_scores[0], attitude_scores[1])
    torch.testing.assert_close(attitude_scores[1], attitude_scores[2])
    assert attitude_scores[0] < POLICY.attitude_recovery_weight + POLICY.attitude_precision_weight

    half_target_yaw = torch.tensor(0.45)
    half_root_yaw = torch.tensor(-0.45)
    target = torch.tensor([[torch.cos(half_target_yaw), 0.0, 0.0, torch.sin(half_target_yaw)]])
    root = torch.tensor([[torch.cos(half_root_yaw), 0.0, 0.0, torch.sin(half_root_yaw)]])
    errors = level_heading_axis_errors(root, target)
    torch.testing.assert_close(errors, torch.tensor([[0.0, 0.0, 1.8]]))


def test_action_magnitude_charges_every_nonzero_command() -> None:
    reward = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=1.0,
        action_rate_weight=0.0,
        action=0.1,
        previous_action=0.1,
    )[0]
    torch.testing.assert_close(reward, torch.tensor([-0.01]))


def test_action_rate_penalty_is_invariant_to_policy_period() -> None:
    rate_25_hz = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action=0.2,
        policy_dt_s=0.04,
    )[8]
    rate_50_hz = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action=0.1,
        policy_dt_s=0.02,
    )[8]
    torch.testing.assert_close(rate_25_hz, rate_50_hz)


def test_dual_angular_velocity_terms_use_broad_and_precision_scales() -> None:
    terms = _reward_terms(
        angular_velocity_error=0.3,
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        action_weight=0.0,
        action_rate_weight=0.0,
    )
    torch.testing.assert_close(terms[5], torch.tensor([0.5]))
    torch.testing.assert_close(terms[6], torch.tensor([0.2]))


def test_attitude_thresholds_and_priority_are_defined_in_radians() -> None:
    assert ATTITUDE_RECOVERY_TRANSITION_RAD == math.pi / 18.0
    assert ATTITUDE_RECOVERY_ZERO_RAD == math.pi / 3.0
    assert ATTITUDE_PRECISION_SIGMA_RAD == math.radians(2.5)
    assert POLICY.attitude_recovery_transition == ATTITUDE_RECOVERY_TRANSITION_RAD
    assert POLICY.attitude_recovery_zero == ATTITUDE_RECOVERY_ZERO_RAD
    assert POLICY.attitude_precision_sigma == ATTITUDE_PRECISION_SIGMA_RAD
    assert (
        POLICY.attitude_recovery_weight
        + POLICY.attitude_precision_weight
        + POLICY.angular_velocity_broad_weight
        + POLICY.angular_velocity_precision_weight
    ) == 0.55


def test_lateral_motion_can_reach_maximum_reward_with_fixed_heading() -> None:
    zeros3 = torch.zeros((1, 3))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    lateral_velocity = torch.tensor([[0.0, 0.1, 0.0]])
    actions = torch.zeros((1, 8))
    terms = compute_tracking_reward_terms(
        POLICY.position_weight,
        POLICY.attitude_recovery_weight,
        POLICY.attitude_precision_weight,
        POLICY.velocity_weight,
        POLICY.angular_velocity_broad_weight,
        POLICY.angular_velocity_precision_weight,
        POLICY.action_weight,
        POLICY.action_rate_weight,
        POLICY.action_rate_scale_per_s,
        POLICY.control_period_s,
        POLICY.position_sigma,
        POLICY.attitude_recovery_transition,
        POLICY.attitude_recovery_zero,
        POLICY.attitude_precision_sigma,
        POLICY.velocity_sigma,
        POLICY.angular_velocity_broad_sigma,
        POLICY.angular_velocity_precision_sigma,
        zeros3,
        identity,
        lateral_velocity,
        zeros3,
        zeros3,
        identity,
        lateral_velocity,
        zeros3,
        actions,
        actions,
    )
    torch.testing.assert_close(terms[0], torch.ones(1))


def test_action_rate_reference_scale_matches_t60_time_constant() -> None:
    assert POLICY_CONTROL_PERIOD_S == 0.04
    assert ACTION_RATE_SCALE_PER_S == 25.0

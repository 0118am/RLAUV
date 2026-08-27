"""Regression checks for the precision trajectory reward."""

from __future__ import annotations

import math

import torch

from simulation.training.rewards import (
    ACTION_ACCELERATION_SCALE_PER_S2,
    ACTION_RATE_SCALE_PER_S,
    ATTITUDE_PRECISION_SIGMA_RAD,
    ATTITUDE_RECOVERY_TRANSITION_RAD,
    ATTITUDE_RECOVERY_ZERO_RAD,
    POLICY_CONTROL_PERIOD_S,
    PRECISION_V6,
    TrackingRewardPolicy,
    compute_tracking_reward_terms,
    normalized_huber_attitude_recovery,
)


POLICY = PRECISION_V6


def _reward_terms(
    *,
    policy: TrackingRewardPolicy = POLICY,
    position_error: float = 0.0,
    attitude_error_rad: float = 0.0,
    attitude_axis: int = 2,
    angular_velocity_error: float = 0.0,
    action: float = 0.0,
    previous_action: float = 0.0,
    previous_previous_action: float = 0.0,
    tracking_weights: tuple[float, float, float, float, float, float] | None = None,
    action_weight: float | None = None,
    action_rate_weight: float | None = None,
    action_acceleration_weight: float | None = None,
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
    if action_acceleration_weight is None:
        action_acceleration_weight = policy.action_acceleration_weight
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
    previous_previous_actions = torch.full((1, 8), previous_previous_action)
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
        action_acceleration_weight,
        policy.action_rate_scale_per_s,
        policy.action_acceleration_scale_per_s2,
        policy_dt_s,
        policy.position_sigma,
        policy.attitude_recovery_transition,
        policy.attitude_recovery_zero,
        policy.attitude_precision_sigma,
        policy.velocity_sigma,
        policy.angular_velocity_broad_sigma,
        policy.angular_velocity_precision_sigma,
        policy.action_deadband,
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
        previous_previous_actions,
    )


def test_precision_v6_has_declared_bounds_and_perfect_reward() -> None:
    assert POLICY.action_acceleration_weight == 1.2
    assert POLICY.maximum_positive_reward == 1.0
    maximum_huber_loss = math.pi / POLICY.attitude_recovery_transition - 0.5
    zero_huber_loss = (
        POLICY.attitude_recovery_zero / POLICY.attitude_recovery_transition - 0.5
    )
    maximum_normalized_rate = (
        2.0 / POLICY.control_period_s / POLICY.action_rate_scale_per_s
    )
    maximum_normalized_acceleration = (
        4.0
        / POLICY.control_period_s**2
        / POLICY.action_acceleration_scale_per_s2
    )
    expected_minimum = (
        POLICY.attitude_recovery_weight
        * (1.0 - maximum_huber_loss / zero_huber_loss)
        + POLICY.attitude_precision_weight
        / (1.0 + (math.pi / POLICY.attitude_precision_sigma) ** 2)
        - POLICY.action_weight
        - POLICY.action_rate_weight * maximum_normalized_rate**2
        - POLICY.action_acceleration_weight * maximum_normalized_acceleration**2
    )
    assert math.isclose(POLICY.minimum_running_reward, expected_minimum)
    torch.testing.assert_close(_reward_terms()[0], torch.ones(1))

    penalties_only = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action=1.0,
        previous_action=-1.0,
        previous_previous_action=1.0,
    )[0]
    torch.testing.assert_close(
        penalties_only,
        torch.tensor(
            [
                -(
                    POLICY.action_weight
                    + POLICY.action_rate_weight * 2.0**2
                    + POLICY.action_acceleration_weight * 4.0**2
                )
            ]
        ),
    )

    worst_attitude = _reward_terms(
        attitude_error_rad=math.pi,
        tracking_weights=(
            0.0,
            POLICY.attitude_recovery_weight,
            POLICY.attitude_precision_weight,
            0.0,
            0.0,
            0.0,
        ),
        action=1.0,
        previous_action=-1.0,
        previous_previous_action=1.0,
    )[0]
    torch.testing.assert_close(
        worst_attitude,
        torch.tensor([POLICY.minimum_running_reward]),
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


def test_attitude_precision_cauchy_is_half_reward_at_one_degree() -> None:
    terms = _reward_terms(
        attitude_error_rad=math.pi / 180.0,
        tracking_weights=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
    )
    torch.testing.assert_close(terms[3], torch.tensor([0.5]))
    torch.testing.assert_close(terms[0], torch.tensor([0.5]))


def test_full_quaternion_reward_includes_roll_pitch_and_yaw() -> None:
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
        attitude_scores.append(terms[0])

    torch.testing.assert_close(attitude_scores[0], attitude_scores[1])
    torch.testing.assert_close(attitude_scores[1], attitude_scores[2])
    assert attitude_scores[0] < (
        POLICY.attitude_recovery_weight + POLICY.attitude_precision_weight
    )


def test_hybrid_attitude_reward_penalizes_ninety_degrees() -> None:
    terms = _reward_terms(
        attitude_error_rad=math.pi / 2.0,
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
    expected_recovery = POLICY.attitude_recovery_weight * (1.0 - 8.5 / 5.5)
    normalized_error = (math.pi / 2.0) / POLICY.attitude_precision_sigma
    expected_precision = POLICY.attitude_precision_weight / (1.0 + normalized_error**2)
    torch.testing.assert_close(terms[2], torch.tensor([expected_recovery]))
    torch.testing.assert_close(terms[3], torch.tensor([expected_precision]))
    torch.testing.assert_close(
        terms[0], torch.tensor([expected_recovery + expected_precision])
    )
    assert torch.all(terms[0] < 0.0)


def test_twenty_degree_attitude_earns_less_than_half_attitude_reward() -> None:
    terms = _reward_terms(
        attitude_error_rad=math.pi / 9.0,
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
    maximum_attitude_reward = (
        POLICY.attitude_recovery_weight + POLICY.attitude_precision_weight
    )
    assert torch.all(terms[0] < 0.5 * maximum_attitude_reward)


def test_precision_reward_does_not_charge_stationary_deadband_command() -> None:
    reward = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=1.0,
        action_rate_weight=0.0,
        action=POLICY.action_deadband,
        previous_action=POLICY.action_deadband,
        previous_previous_action=POLICY.action_deadband,
    )[0]
    assert torch.equal(reward, torch.zeros(1))


def test_action_acceleration_penalizes_reversal_but_not_linear_ramp() -> None:
    linear_ramp = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
        action=0.5,
        previous_action=0.0,
        previous_previous_action=-0.5,
    )
    assert torch.equal(linear_ramp[9], torch.zeros(1))

    full_reversal = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
        action=1.0,
        previous_action=-1.0,
        previous_previous_action=1.0,
    )
    torch.testing.assert_close(
        full_reversal[9],
        torch.tensor([POLICY.action_acceleration_weight * 4.0**2]),
    )
    torch.testing.assert_close(full_reversal[0], -full_reversal[9])


def test_action_derivative_penalties_are_invariant_to_policy_period() -> None:
    rate_25_hz = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_acceleration_weight=0.0,
        action=0.2,
        policy_dt_s=0.04,
    )[8]
    rate_50_hz = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_acceleration_weight=0.0,
        action=0.1,
        policy_dt_s=0.02,
    )[8]
    torch.testing.assert_close(rate_25_hz, rate_50_hz)

    acceleration_25_hz = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
        action=0.16,
        policy_dt_s=0.04,
    )[9]
    acceleration_50_hz = _reward_terms(
        tracking_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_weight=0.0,
        action_rate_weight=0.0,
        action=0.04,
        policy_dt_s=0.02,
    )[9]
    torch.testing.assert_close(acceleration_25_hz, acceleration_50_hz)


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
    assert ATTITUDE_PRECISION_SIGMA_RAD == math.pi / 180.0
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
        POLICY.action_acceleration_weight,
        POLICY.action_rate_scale_per_s,
        POLICY.action_acceleration_scale_per_s2,
        POLICY.control_period_s,
        POLICY.position_sigma,
        POLICY.attitude_recovery_transition,
        POLICY.attitude_recovery_zero,
        POLICY.attitude_precision_sigma,
        POLICY.velocity_sigma,
        POLICY.angular_velocity_broad_sigma,
        POLICY.angular_velocity_precision_sigma,
        POLICY.action_deadband,
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
        actions,
    )
    torch.testing.assert_close(terms[0], torch.ones(1))


def test_action_derivative_reference_scales_match_t60_time_constant() -> None:
    assert POLICY_CONTROL_PERIOD_S == 0.04
    assert ACTION_RATE_SCALE_PER_S == 25.0
    assert ACTION_ACCELERATION_SCALE_PER_S2 == 625.0

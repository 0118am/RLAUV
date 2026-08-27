"""Unit checks for level, horizontal-heading AUV attitude commands."""

from __future__ import annotations

import torch

from common.tensor_math import quat_apply_wxyz
from robot.control.trajectory import guidance


def test_level_heading_uses_only_horizontal_velocity():
    velocity = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -3.0],
            [-1.0, 0.0, 0.0],
            [1.0, -2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(len(velocity), 1)

    fixed_attitude = torch.zeros(len(velocity), dtype=torch.bool)
    heading_velocity = guidance.horizontal_heading_velocity(velocity, fixed_attitude)
    quaternion = guidance.quaternion_from_level_heading(heading_velocity, previous)
    body_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).repeat(len(velocity), 1)
    forward_w = quat_apply_wxyz(quaternion, body_x)
    expected = heading_velocity / torch.linalg.vector_norm(
        heading_velocity,
        dim=-1,
        keepdim=True,
    )

    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
    assert torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-12)
    assert torch.equal(quaternion[:, 1:3], torch.zeros_like(quaternion[:, 1:3]))
    assert torch.equal(forward_w[:, 2], torch.zeros_like(forward_w[:, 2]))
    assert torch.allclose(forward_w, expected, atol=1.0e-12)


def test_near_zero_horizontal_velocity_keeps_previous_level_heading():
    half_yaw = torch.tensor(0.25, dtype=torch.float64)
    previous = torch.tensor(
        [[torch.cos(half_yaw), 0.0, 0.0, torch.sin(half_yaw)]],
        dtype=torch.float64,
    )
    velocity = torch.tensor([[1.0e-5, -2.0e-5, 10.0]], dtype=torch.float64)

    quaternion = guidance.quaternion_from_level_heading(
        velocity,
        previous,
        min_horizontal_speed=1.0e-3,
    )

    assert torch.equal(quaternion, previous)


def test_near_zero_horizontal_velocity_removes_previous_roll_and_pitch():
    half_roll = torch.tensor(0.4, dtype=torch.float64)
    half_yaw = torch.tensor(0.3, dtype=torch.float64)
    cos_roll = torch.cos(half_roll)
    sin_roll = torch.sin(half_roll)
    cos_yaw = torch.cos(half_yaw)
    sin_yaw = torch.sin(half_yaw)
    previous = torch.tensor(
        [[
            cos_yaw * cos_roll,
            cos_yaw * sin_roll,
            sin_yaw * sin_roll,
            sin_yaw * cos_roll,
        ]],
        dtype=torch.float64,
    )
    velocity = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

    quaternion = guidance.quaternion_from_level_heading(velocity, previous)

    expected = torch.tensor([[cos_yaw, 0.0, 0.0, sin_yaw]], dtype=torch.float64)
    assert torch.allclose(quaternion, expected, atol=1.0e-12)


def test_reciprocating_velocity_uses_fixed_attitude_through_reversal():
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    velocities = torch.tensor(
        [[0.0, 0.4, 0.0], [0.0, -0.4, 0.0]],
        dtype=torch.float64,
    )
    fixed_attitude = torch.ones(2, dtype=torch.bool)

    heading_velocity = guidance.horizontal_heading_velocity(velocities, fixed_attitude)
    commanded = guidance.quaternion_from_level_heading(
        heading_velocity,
        previous.repeat(2, 1),
    )

    assert torch.equal(heading_velocity, torch.zeros_like(heading_velocity))
    assert torch.equal(commanded, previous.repeat(2, 1))
    angular_velocity = guidance.quaternion_step_angular_velocity_body(
        commanded[:1], commanded[1:], 0.02
    )
    assert torch.equal(angular_velocity, torch.zeros_like(angular_velocity))


def test_quaternion_sign_stays_continuous_across_yaw_wrap():
    epsilon = 1.0e-6
    velocity_before_wrap = torch.tensor([[-1.0, epsilon, 0.0]], dtype=torch.float64)
    velocity_after_wrap = torch.tensor([[-1.0, -epsilon, 0.0]], dtype=torch.float64)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    previous = guidance.quaternion_from_level_heading(velocity_before_wrap, identity)

    current = guidance.quaternion_from_level_heading(velocity_after_wrap, previous)

    assert torch.sum(previous * current).item() > 0.999999


def test_quaternion_step_returns_shortest_body_angular_velocity():
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    half_angle = torch.tensor(torch.pi / 4.0, dtype=torch.float64)
    current = torch.tensor(
        [[torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)]],
        dtype=torch.float64,
    )

    angular_velocity = guidance.quaternion_step_angular_velocity_body(previous, -current, 0.5)

    expected = torch.tensor([[0.0, 0.0, torch.pi]], dtype=torch.float64)
    assert torch.allclose(angular_velocity, expected, atol=1.0e-12)

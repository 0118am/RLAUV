"""Unit checks for velocity-aligned AUV trajectory attitude commands."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[6]
MODULE_PATH = REPO_ROOT / "simulation/isaac/envs/auv/trajectory/guidance.py"
SPEC = importlib.util.spec_from_file_location("trajectory_guidance", MODULE_PATH)
guidance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guidance)


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector_part = quaternion[..., 1:]
    cross_twice = 2.0 * torch.linalg.cross(vector_part, vector)
    return vector + quaternion[..., :1] * cross_twice + torch.linalg.cross(vector_part, cross_twice)


def test_root_state_matches_initial_trajectory_pose_and_velocity_without_mutating_input():
    original = torch.full((2, 13), -9.0, dtype=torch.float64)
    position = torch.tensor([[1.0, 2.0, -8.0], [4.0, 5.0, -7.0]], dtype=torch.float64)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
    linear_velocity = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.1]], dtype=torch.float64)
    angular_velocity = torch.tensor([[0.0, 0.0, 0.2], [0.3, -0.1, 0.0]], dtype=torch.float64)

    aligned = guidance.root_state_at_tracking_target(
        original,
        position,
        quaternion,
        linear_velocity,
        angular_velocity,
    )

    assert torch.equal(original, torch.full_like(original, -9.0))
    assert torch.equal(aligned[:, :3], position)
    assert torch.equal(aligned[:, 3:7], quaternion)
    assert torch.equal(aligned[:, 7:10], linear_velocity)
    assert torch.equal(aligned[:, 10:13], angular_velocity)


def test_body_x_aligns_with_three_dimensional_velocity():
    velocity = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(len(velocity), 1)

    quaternion = guidance.quaternion_align_body_x_with_velocity(velocity, previous)
    body_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).repeat(len(velocity), 1)
    forward_w = _quat_apply_wxyz(quaternion, body_x)
    expected = velocity / torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)

    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
    assert torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-12)
    assert torch.allclose(forward_w, expected, atol=1.0e-12)


def test_near_zero_velocity_keeps_previous_attitude():
    previous = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
    velocity = torch.tensor([[1.0e-5, -2.0e-5, 1.0e-5]], dtype=torch.float64)

    quaternion = guidance.quaternion_align_body_x_with_velocity(velocity, previous, min_speed=1.0e-3)

    assert torch.equal(quaternion, previous)


def test_quaternion_sign_stays_continuous_across_yaw_wrap():
    epsilon = 1.0e-6
    velocity_before_wrap = torch.tensor([[-1.0, epsilon, 0.0]], dtype=torch.float64)
    velocity_after_wrap = torch.tensor([[-1.0, -epsilon, 0.0]], dtype=torch.float64)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    previous = guidance.quaternion_align_body_x_with_velocity(velocity_before_wrap, identity)

    current = guidance.quaternion_align_body_x_with_velocity(velocity_after_wrap, previous)

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


if __name__ == "__main__":
    test_body_x_aligns_with_three_dimensional_velocity()
    test_near_zero_velocity_keeps_previous_attitude()
    test_quaternion_sign_stays_continuous_across_yaw_wrap()
    test_quaternion_step_returns_shortest_body_angular_velocity()
    print("Trajectory guidance checks passed.")

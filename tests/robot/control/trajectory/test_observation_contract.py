"""Tests for deployable current and history observation assembly."""

import torch

from robot.control.trajectory.observation_contract import (
    BASE_OBSERVATION_DIM,
    OBSERVATION_FIELD_SLICES,
    normalized_history_observation,
)


def test_history_angular_velocity_uses_body_frame_tracking_error() -> None:
    observation = torch.zeros(2, BASE_OBSERVATION_DIM)
    observation[:, OBSERVATION_FIELD_SLICES["angular_velocity_b"]] = torch.tensor(
        [[0.25, -0.50, 0.75], [-0.25, 0.50, -0.75]]
    )
    observation[
        :, OBSERVATION_FIELD_SLICES["target_angular_velocity_b"]
    ] = torch.tensor([[1.0, 0.5, -0.5], [0.5, -1.0, 1.0]])

    history = normalized_history_observation(
        observation,
        ("angular_velocity_error_b",),
    )

    torch.testing.assert_close(
        history,
        torch.tensor([[0.75, 1.0, -1.25], [0.75, -1.5, 1.75]]),
    )

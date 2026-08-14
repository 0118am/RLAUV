"""Pure geometry checks for the non-instanced debug coordinate frames."""

from __future__ import annotations

import torch

from simulation.isaac.visualization_geometry import frame_line_data


def test_identity_frame_axes_and_arrowheads_are_finite():
    starts, ends, colors, thicknesses = frame_line_data(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        axis_length=0.5,
        alpha=1.0,
        shaft_thickness=4.0,
    )

    assert torch.allclose(
        torch.tensor(ends[:3]),
        torch.tensor([[1.5, 2.0, 3.0], [1.0, 2.5, 3.0], [1.0, 2.0, 3.5]]),
    )
    assert len(starts) == len(ends) == len(colors) == len(thicknesses) == 9
    assert torch.isfinite(torch.tensor(starts + ends)).all()


def test_frame_axes_follow_scalar_first_quaternion_rotation():
    half_angle = torch.tensor(torch.pi / 4.0, dtype=torch.float64)
    quaternion = torch.tensor(
        [[torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)]],
        dtype=torch.float64,
    )
    _, ends, _, _ = frame_line_data(
        torch.zeros(1, 3, dtype=torch.float64),
        quaternion,
        axis_length=1.0,
        alpha=1.0,
        shaft_thickness=4.0,
    )

    assert torch.allclose(
        torch.tensor(ends[:3], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        atol=1.0e-12,
    )

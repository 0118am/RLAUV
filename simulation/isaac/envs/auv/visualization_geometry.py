"""Pure Torch geometry for AUV debug coordinate frames."""

from __future__ import annotations

import torch


_AXIS_COLORS = (
    (1.0, 0.05, 0.05),  # +X: red, body forward
    (0.05, 1.0, 0.05),  # +Y: green, body left
    (0.05, 0.25, 1.0),  # +Z: blue, body up
)


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by scalar-first quaternions without Isaac imports."""
    vector_part = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(vector_part, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(vector_part, twice_cross, dim=-1)


def frame_line_data(
    origins_w: torch.Tensor,
    quaternions_wxyz: torch.Tensor,
    *,
    axis_length: float,
    alpha: float,
    shaft_thickness: float,
) -> tuple[list, list, list, list]:
    """Build colored XYZ shafts and arrowheads for batched world poses."""
    if origins_w.ndim != 2 or origins_w.shape[-1] != 3:
        raise ValueError(f"Expected origins with shape [N, 3], got {tuple(origins_w.shape)}.")
    if quaternions_wxyz.shape != (origins_w.shape[0], 4):
        raise ValueError(
            f"Expected quaternions with shape [{origins_w.shape[0]}, 4], "
            f"got {tuple(quaternions_wxyz.shape)}."
        )

    count = origins_w.shape[0]
    local_axes = torch.eye(3, dtype=origins_w.dtype, device=origins_w.device)
    local_axes = local_axes.unsqueeze(0).expand(count, -1, -1)
    frame_quaternions = quaternions_wxyz.unsqueeze(1).expand(-1, 3, -1)
    directions_w = _quat_apply_wxyz(
        frame_quaternions.reshape(-1, 4),
        local_axes.reshape(-1, 3),
    ).reshape(count, 3, 3)

    origins = origins_w.unsqueeze(1).expand(-1, 3, -1)
    tips = origins + float(axis_length) * directions_w
    # Use the following frame axis as an arrowhead width direction. Since the
    # frame is orthonormal, every arrowhead remains perpendicular to its shaft.
    perpendicular_w = torch.roll(directions_w, shifts=-1, dims=1)
    head_base = tips - (0.20 * float(axis_length)) * directions_w
    head_half_width = 0.075 * float(axis_length)
    head_side_positive = head_base + head_half_width * perpendicular_w
    head_side_negative = head_base - head_half_width * perpendicular_w

    starts = torch.cat((origins, tips, tips), dim=1).reshape(-1, 3)
    ends = torch.cat((tips, head_side_positive, head_side_negative), dim=1).reshape(-1, 3)
    axis_colors = [(*color, float(alpha)) for color in _AXIS_COLORS] * count
    colors = axis_colors * 3
    frame_thicknesses = [float(shaft_thickness)] * 3
    frame_thicknesses += [max(1.0, float(shaft_thickness) - 1.0)] * 6
    thicknesses = frame_thicknesses * count
    return (
        starts.detach().cpu().tolist(),
        ends.detach().cpu().tolist(),
        colors,
        thicknesses,
    )

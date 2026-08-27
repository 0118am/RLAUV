"""Robot guidance geometry for the T60 x-forward body convention."""

from __future__ import annotations

import torch

from common.tensor_math import (
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quaternion_to_rotation_vector,
)


def horizontal_heading_velocity(
    velocity_w: torch.Tensor,
    fixed_attitude: torch.Tensor,
) -> torch.Tensor:
    """Return the horizontal velocity used by the level-heading command.

    A reciprocating line has no continuous forward-facing attitude at its
    zero-speed turnaround: its tangent reverses by 180 degrees.  Those tasks
    therefore retain a fixed level heading instead of turning the vehicle
    around at every half-cycle. All other tasks use only the horizontal
    velocity projection; vertical motion is handled by heave while commanded
    roll and pitch remain zero.
    """

    horizontal = torch.stack(
        (velocity_w[..., 0], velocity_w[..., 1], torch.zeros_like(velocity_w[..., 2])),
        dim=-1,
    )
    return torch.where(
        fixed_attitude.unsqueeze(-1),
        torch.zeros_like(horizontal),
        horizontal,
    )


def root_state_at_tracking_target(
    root_state_w: torch.Tensor,
    target_position_w: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
    target_linear_velocity_w: torch.Tensor,
    target_angular_velocity_w: torch.Tensor,
) -> torch.Tensor:
    """Return a 13-D root state initialized exactly on a trajectory target.

    The shared rigid-body state convention stores world position, scalar-first
    quaternion, world linear velocity, and world angular velocity in that
    order. Copying all four target quantities prevents an initial discontinuity.
    """
    if root_state_w.shape[-1] != 13:
        raise ValueError(f"Expected a 13-D rigid-body root state, got shape {tuple(root_state_w.shape)}.")
    expected_shapes = (
        (target_position_w, 3, "target_position_w"),
        (target_quaternion_wxyz, 4, "target_quaternion_wxyz"),
        (target_linear_velocity_w, 3, "target_linear_velocity_w"),
        (target_angular_velocity_w, 3, "target_angular_velocity_w"),
    )
    for value, width, name in expected_shapes:
        if value.shape[:-1] != root_state_w.shape[:-1] or value.shape[-1] != width:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is incompatible with root state shape "
                f"{tuple(root_state_w.shape)}."
            )

    aligned_state = root_state_w.clone()
    aligned_state[..., :3] = target_position_w
    aligned_state[..., 3:7] = target_quaternion_wxyz
    aligned_state[..., 7:10] = target_linear_velocity_w
    aligned_state[..., 10:13] = target_angular_velocity_w
    return aligned_state


def quaternion_from_level_heading(
    horizontal_velocity_w: torch.Tensor,
    previous_quaternion_wxyz: torch.Tensor,
    min_horizontal_speed: float = 1.0e-3,
) -> torch.Tensor:
    """Return a yaw-only world quaternion from horizontal velocity.

    The quaternion uses Isaac's scalar-first ``(w, x, y, z)`` convention and
    always commands zero roll and pitch. At horizontal speeds where heading is
    undefined, the previous yaw is retained after explicitly projecting it
    onto a level attitude. Quaternion signs stay in the same hemisphere as the
    previous command, avoiding observation jumps at the ``+pi/-pi`` yaw
    boundary.
    """

    horizontal_speed = torch.linalg.vector_norm(horizontal_velocity_w[..., :2], dim=-1)

    previous_w, previous_x, previous_y, previous_z = previous_quaternion_wxyz.unbind(
        dim=-1
    )
    previous_yaw = torch.atan2(
        2.0 * (previous_w * previous_z + previous_x * previous_y),
        1.0 - 2.0 * (previous_y.square() + previous_z.square()),
    )
    previous_half_yaw = 0.5 * previous_yaw
    previous_level = torch.stack(
        (
            torch.cos(previous_half_yaw),
            torch.zeros_like(previous_half_yaw),
            torch.zeros_like(previous_half_yaw),
            torch.sin(previous_half_yaw),
        ),
        dim=-1,
    )
    previous_level = previous_level * torch.where(
        torch.sum(previous_level * previous_quaternion_wxyz, dim=-1, keepdim=True)
        < 0.0,
        -torch.ones_like(previous_level[..., :1]),
        torch.ones_like(previous_level[..., :1]),
    )

    yaw = torch.atan2(horizontal_velocity_w[..., 1], horizontal_velocity_w[..., 0])
    half_yaw = 0.5 * yaw
    cos_yaw = torch.cos(half_yaw)
    sin_yaw = torch.sin(half_yaw)
    candidate = torch.stack(
        (
            cos_yaw,
            torch.zeros_like(cos_yaw),
            torch.zeros_like(cos_yaw),
            sin_yaw,
        ),
        dim=-1,
    )
    same_hemisphere_sign = torch.where(
        torch.sum(candidate * previous_level, dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(candidate[..., :1]),
        torch.ones_like(candidate[..., :1]),
    )
    candidate = candidate * same_hemisphere_sign
    has_direction = horizontal_speed > min_horizontal_speed
    return torch.where(has_direction[..., None], candidate, previous_level)


def quaternion_step_angular_velocity_body(
    previous_quaternion_wxyz: torch.Tensor,
    current_quaternion_wxyz: torch.Tensor,
    dt_s: torch.Tensor | float,
) -> torch.Tensor:
    """Return the shortest-step body angular velocity between two orientations.

    ``previous_quaternion_wxyz`` and ``current_quaternion_wxyz`` map their
    respective body frames into the world frame.  The relative quaternion is
    therefore formed as ``conj(previous) * current`` so the returned rotation
    vector is expressed in the previous target-body frame.  Choosing the
    positive-real quaternion hemisphere removes the ``q``/``-q`` ambiguity.
    """

    relative = quat_multiply_wxyz(
        quat_conjugate_wxyz(previous_quaternion_wxyz),
        current_quaternion_wxyz,
    )
    rotation_vector = quaternion_to_rotation_vector(relative)
    dt = torch.as_tensor(dt_s, dtype=relative.dtype, device=relative.device)
    if dt.ndim == relative.ndim - 1:
        dt = dt.unsqueeze(-1)
    return rotation_vector / torch.clamp(dt, min=1.0e-8)

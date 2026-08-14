"""Guidance geometry for the AUV x-forward body convention."""

from __future__ import annotations

import torch


def root_state_at_tracking_target(
    root_state_w: torch.Tensor,
    target_position_w: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
    target_linear_velocity_w: torch.Tensor,
    target_angular_velocity_w: torch.Tensor,
) -> torch.Tensor:
    """Return a 13-D root state initialized exactly on a trajectory target.

    Isaac rigid-body root states store world position, scalar-first quaternion,
    world linear velocity, and world angular velocity in that order. Copying
    all four target quantities prevents a hidden initial position, attitude,
    or velocity discontinuity at the first policy step.
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


def quaternion_align_body_x_with_velocity(
    velocity_w: torch.Tensor,
    previous_quaternion_wxyz: torch.Tensor,
    min_speed: float = 1.0e-3,
) -> torch.Tensor:
    """Return a world quaternion whose body ``+X`` axis follows ``velocity_w``.

    The quaternion uses Isaac's scalar-first ``(w, x, y, z)`` convention. Roll
    is fixed to zero; yaw and pitch align the body-forward axis with the full
    three-dimensional velocity vector. At speeds where direction is undefined,
    the previous command is retained. Quaternion signs are chosen to stay in
    the same hemisphere as the previous command, avoiding observation jumps at
    the ``+pi/-pi`` yaw boundary.
    """

    horizontal_speed = torch.linalg.vector_norm(velocity_w[..., :2], dim=-1)
    speed = torch.linalg.vector_norm(velocity_w, dim=-1)
    yaw = torch.atan2(velocity_w[..., 1], velocity_w[..., 0])
    pitch = -torch.atan2(velocity_w[..., 2], horizontal_speed)

    half_yaw = 0.5 * yaw
    half_pitch = 0.5 * pitch
    cos_yaw = torch.cos(half_yaw)
    sin_yaw = torch.sin(half_yaw)
    cos_pitch = torch.cos(half_pitch)
    sin_pitch = torch.sin(half_pitch)

    # q = q_yaw * q_pitch, with roll fixed to zero.
    candidate = torch.stack(
        (
            cos_yaw * cos_pitch,
            -sin_yaw * sin_pitch,
            cos_yaw * sin_pitch,
            sin_yaw * cos_pitch,
        ),
        dim=-1,
    )

    same_hemisphere_sign = torch.where(
        torch.sum(candidate * previous_quaternion_wxyz, dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(candidate[..., :1]),
        torch.ones_like(candidate[..., :1]),
    )
    candidate = candidate * same_hemisphere_sign
    has_direction = speed > min_speed
    return torch.where(has_direction[..., None], candidate, previous_quaternion_wxyz)


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

    previous_conjugate = previous_quaternion_wxyz.clone()
    previous_conjugate[..., 1:] = -previous_conjugate[..., 1:]

    pw, px, py, pz = previous_conjugate.unbind(dim=-1)
    cw, cx, cy, cz = current_quaternion_wxyz.unbind(dim=-1)
    relative = torch.stack(
        (
            pw * cw - px * cx - py * cy - pz * cz,
            pw * cx + px * cw + py * cz - pz * cy,
            pw * cy - px * cz + py * cw + pz * cx,
            pw * cz + px * cy - py * cx + pz * cw,
        ),
        dim=-1,
    )
    relative = relative / torch.clamp(torch.linalg.vector_norm(relative, dim=-1, keepdim=True), min=1.0e-8)
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)

    vector = relative[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(relative[..., :1], min=0.0))
    rotation_vector = vector * (angle / torch.clamp(vector_norm, min=1.0e-8))
    dt = torch.as_tensor(dt_s, dtype=relative.dtype, device=relative.device)
    if dt.ndim == relative.ndim - 1:
        dt = dt.unsqueeze(-1)
    return rotation_vector / torch.clamp(dt, min=1.0e-8)

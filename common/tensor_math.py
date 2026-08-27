"""Shared scalar-first quaternion tensor operations."""

from __future__ import annotations

import torch


def quat_conjugate_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a ``(w, x, y, z)`` quaternion."""

    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply two ``(w, x, y, z)`` quaternions."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector with a ``(w, x, y, z)`` quaternion."""

    xyz = quaternion[..., 1:]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.cross(xyz, cross, dim=-1)


def quaternion_to_rotation_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the shortest rotation vector represented by a quaternion."""

    normalized = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    normalized = torch.where(normalized[..., :1] < 0.0, -normalized, normalized)
    vector = normalized[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, normalized[..., :1].clamp_min(0.0))
    return vector * (angle / vector_norm.clamp_min(1.0e-8))


@torch.jit.script
def quaternion_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Return the shortest angular distance between scalar-first quaternions."""

    q1_normalized = q1 / torch.clamp(torch.norm(q1, dim=-1, keepdim=True), min=1.0e-9)
    q2_normalized = q2 / torch.clamp(torch.norm(q2, dim=-1, keepdim=True), min=1.0e-9)
    q1_scalar = q1_normalized[..., :1]
    q1_vector = q1_normalized[..., 1:]
    q2_scalar = q2_normalized[..., :1]
    q2_vector = q2_normalized[..., 1:]
    relative_scalar = torch.abs(torch.sum(q1_normalized * q2_normalized, dim=-1))
    relative_vector = (
        q1_scalar * q2_vector
        - q2_scalar * q1_vector
        - torch.cross(q1_vector, q2_vector, dim=-1)
    )
    sine_half_angle = torch.linalg.vector_norm(relative_vector, dim=-1)
    return 2.0 * torch.atan2(sine_half_angle, relative_scalar)

"""Rigid-body property helpers shared by the AUV environment and tests."""

from __future__ import annotations

import torch


def inertia_matrix_tensor(
    values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize 3-value diagonal, 3x3, or flat 9-value inertia to 3x3."""

    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim == 1:
        if tensor.shape[0] == 3:
            return torch.diag(tensor)
        if tensor.shape[0] == 9:
            return tensor.reshape(3, 3)
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        return tensor
    raise ValueError(f"inertia_diag must be a 3-vector, 3x3 matrix, or flat 9-value matrix, got {tuple(tensor.shape)}.")


def inertia_diag_tensor(
    values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return inertia diagonal from any supported inertia tensor shape."""

    return torch.diagonal(inertia_matrix_tensor(values, device, dtype), dim1=-2, dim2=-1)


def principal_inertia_and_axes(
    values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    reconstruction_atol: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonalize a full body-frame inertia tensor for the PhysX COM API.

    Returns principal moments and ``R_body_from_principal`` such that
    ``I_body = R @ diag(moments) @ R.T``.  PhysX stores the moments and the
    principal-axis orientation separately in its COM pose, so this conversion
    preserves products of inertia instead of silently dropping them.
    """

    # Decompose in float64 so the invariant is checked at CAD precision even
    # when the runtime PhysX tensors themselves are float32.
    compute_dtype = torch.float64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else dtype
    inertia = inertia_matrix_tensor(values, device, compute_dtype)
    if not torch.allclose(inertia, inertia.T, atol=reconstruction_atol, rtol=0.0):
        raise ValueError("Inertia tensor must be symmetric.")

    principal_moments, axes = torch.linalg.eigh(inertia)
    if torch.any(principal_moments <= 0.0):
        raise ValueError("Inertia tensor must be positive definite.")
    # eigh may return an improper orthogonal basis.  Flip one eigenvector so
    # it is a valid rotation before converting it to a quaternion.
    if torch.linalg.det(axes) < 0.0:
        axes = axes.clone()
        axes[:, -1] *= -1.0

    reconstructed = axes @ torch.diag(principal_moments) @ axes.T
    if torch.max(torch.abs(reconstructed - inertia)) >= reconstruction_atol:
        raise ValueError("Principal-axis decomposition does not reconstruct the inertia tensor.")
    return principal_moments.to(dtype=dtype), axes.to(dtype=dtype)


def rotation_matrix_to_quat_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {tuple(rotation.shape)}.")
    if not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=rotation.dtype, device=rotation.device), atol=1.0e-6):
        raise ValueError("rotation must be orthonormal.")
    if torch.linalg.det(rotation) <= 0.0:
        raise ValueError("rotation must be right-handed.")

    # This branch formulation is numerically stable near 180 degree rotations.
    trace = torch.trace(rotation)
    if trace > 0.0:
        scale = 2.0 * torch.sqrt(trace + 1.0)
        quat = torch.stack((0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                            (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale))
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * torch.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        quat = torch.stack(((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                            (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale))
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * torch.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        quat = torch.stack(((rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                            0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale))
    else:
        scale = 2.0 * torch.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        quat = torch.stack(((rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale,
                            (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale))
    return quat / torch.linalg.norm(quat)


def physx_principal_inertia_and_com_quat_xyzw(
    values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PhysX principal moments and principal-axes quaternion in xyzw order."""

    moments, axes = principal_inertia_and_axes(values, device, dtype)
    quat_wxyz = rotation_matrix_to_quat_wxyz(axes)
    return moments, quat_wxyz[[1, 2, 3, 0]]


def rigid_body_mass_matrix(
    mass_kg: float,
    inertia_values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    center_of_mass_offset_m=None,
) -> torch.Tensor:
    """Construct the full 6x6 Fossen/Newton--Euler rigid-body mass matrix.

    The validation vehicle uses a COM-centred body origin, so its translation /
    rotation blocks are zero and the complete (non-diagonal) CAD inertia tensor
    forms the lower-right block.  A nonzero offset is rejected rather than
    quietly constructing a matrix with a convention different from the model.
    """

    if center_of_mass_offset_m is not None:
        offset = torch.as_tensor(center_of_mass_offset_m, dtype=dtype, device=device)
        if not torch.allclose(offset, torch.zeros(3, dtype=dtype, device=device), atol=1.0e-12, rtol=0.0):
            raise ValueError("rigid_body_mass_matrix requires a COM-centred body frame.")
    if float(mass_kg) <= 0.0:
        raise ValueError("mass_kg must be positive.")
    inertia = inertia_matrix_tensor(inertia_values, device, dtype)
    if torch.linalg.eigvalsh(inertia).min() <= 0.0:
        raise ValueError("Inertia tensor must be positive definite.")
    matrix = torch.zeros((6, 6), dtype=dtype, device=device)
    matrix[:3, :3] = torch.eye(3, dtype=dtype, device=device) * float(mass_kg)
    matrix[3:, 3:] = inertia
    return matrix

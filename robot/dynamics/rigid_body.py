"""Rigid-body property helpers shared by the AUV environment and tests."""

from __future__ import annotations

import numpy as np
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
    raise ValueError(
        "inertia_diag must be a 3-vector, 3x3 matrix, or flat 9-value matrix, "
        f"got {tuple(tensor.shape)}."
    )


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


def validate_inertia_tensor(values) -> None:
    """Validate a rigid-body inertia from its principal moments."""

    array = np.asarray(values, dtype=float)
    if array.shape == (3,):
        matrix = np.diag(array)
    elif array.shape == (9,):
        matrix = array.reshape(3, 3)
    elif array.shape == (3, 3):
        matrix = array
    else:
        raise ValueError("inertia must be a 3-vector, 3x3 matrix, or flat 9-value matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("inertia must contain only finite values.")
    if not np.allclose(matrix, matrix.T, atol=1.0e-8, rtol=0.0):
        raise ValueError("inertia must be symmetric.")
    principal = np.linalg.eigvalsh(matrix)
    if principal[0] <= 0.0:
        raise ValueError("inertia must be positive definite.")
    if any(moment > principal.sum() - moment + 1.0e-9 for moment in principal):
        raise ValueError("inertia must satisfy the rigid-body inertia triangle inequalities.")

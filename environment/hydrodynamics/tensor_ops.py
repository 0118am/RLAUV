"""Tensor and coefficient operations shared by hydrodynamic models."""

from __future__ import annotations

import torch


def skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    """Return S(vec), where S(a) b = a x b."""

    mat = torch.zeros((*vec.shape[:-1], 3, 3), dtype=vec.dtype, device=vec.device)
    mat[..., 0, 1] = -vec[..., 2]
    mat[..., 0, 2] = vec[..., 1]
    mat[..., 1, 0] = vec[..., 2]
    mat[..., 1, 2] = -vec[..., 0]
    mat[..., 2, 0] = -vec[..., 1]
    mat[..., 2, 1] = vec[..., 0]
    return mat


def expand_6d_matrix(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a batched 6x6 matrix from diagonal vectors or full matrices."""

    if values.ndim == 1:
        if values.shape[0] != 6:
            raise ValueError(f"Expected a 6-vector, got shape {tuple(values.shape)}.")
        return torch.diag_embed(values.reshape(1, 6).repeat(batch_size, 1))

    if values.ndim == 2:
        if values.shape == (6, 6):
            return values.reshape(1, 6, 6).repeat(batch_size, 1, 1)
        if values.shape[1] == 6:
            if values.shape[0] == 1:
                values = values.repeat(batch_size, 1)
            elif values.shape[0] != batch_size:
                raise ValueError(f"Expected batch size {batch_size}, got shape {tuple(values.shape)}.")
            return torch.diag_embed(values)

    if values.ndim == 3 and values.shape[1:] == (6, 6):
        if values.shape[0] == 1:
            return values.repeat(batch_size, 1, 1)
        if values.shape[0] == batch_size:
            return values

    raise ValueError(f"Expected a 6-vector, batched 6-vector, or 6x6 matrix, got shape {tuple(values.shape)}.")


def multiply_6d_matrix(values: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Multiply diagonal-vector or full-matrix 6D coefficients by a 6-vector."""

    matrix = expand_6d_matrix(values, vector.shape[0])
    return torch.bmm(matrix, vector.unsqueeze(-1)).squeeze(-1)


def mean_one_lognormal_scale(
    standard_normal: torch.Tensor,
    log_standard_deviation: torch.Tensor | float,
) -> torch.Tensor:
    """Map a normal latent variable to a positive scale with unit mean.

    ``exp(sigma*z - sigma^2/2)`` keeps every sampled scale positive while
    preserving ``E[scale] = 1`` for ``z ~ N(0, 1)``.  Sampling the latent
    variable at reset avoids non-physical white-noise wrenches.
    """

    sigma = torch.as_tensor(
        log_standard_deviation,
        dtype=standard_normal.dtype,
        device=standard_normal.device,
    )
    return torch.exp(sigma * standard_normal - 0.5 * sigma.square())


def scale_hydrodynamic_coefficients(
    coefficients: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Scale diagonal or full 6-DOF coefficients without breaking symmetry.

    For a full matrix this evaluates ``S M S`` with
    ``S = diag(sqrt(scale))``.  Positive scales therefore preserve symmetry
    and positive semidefiniteness of added-mass and passive damping matrices.
    """

    if coefficients.ndim in (1, 2) and coefficients.shape[-1] == 6:
        if coefficients.ndim == 2 and coefficients.shape == (6, 6) and scale.ndim == 1:
            root_scale = torch.sqrt(torch.clamp(scale, min=0.0))
            return coefficients * root_scale.unsqueeze(0) * root_scale.unsqueeze(1)
        return coefficients * scale
    if coefficients.ndim == 3 and coefficients.shape[1:] == (6, 6):
        if scale.ndim == 2 and scale.shape[1] == 6:
            matrix_scale = torch.sqrt(torch.clamp(scale.unsqueeze(1) * scale.unsqueeze(2), min=0.0))
            return coefficients * matrix_scale
        return coefficients * scale.reshape(scale.shape[0], 1, 1)
    raise ValueError(f"Expected 6-vector or 6x6 hydrodynamic coefficients, got {tuple(coefficients.shape)}.")


def calculate_speed_dependent_damping_scale(
    nu_r: torch.Tensor,
    speed_points: torch.Tensor | list[float] | tuple[float, ...],
    scale_points: torch.Tensor | list[float] | list[list[float]] | tuple,
    clamp: bool = True,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Interpolate per-DOF damping scales from ``|nu_r|``.

    ``scale_points`` may be shaped ``(num_speed_points,)`` for one shared curve
    or ``(num_speed_points, 6)`` for per-DOF curves.  The result has shape
    ``(num_envs, 6)`` and is intended to multiply linear or quadratic damping
    coefficients before evaluating the Fossen damping wrench.
    """

    if validate and (nu_r.ndim != 2 or nu_r.shape[1] != 6):
        raise ValueError(f"nu_r must have shape (N, 6), got {tuple(nu_r.shape)}.")

    speeds = torch.as_tensor(speed_points, dtype=nu_r.dtype, device=nu_r.device)
    scales = torch.as_tensor(scale_points, dtype=nu_r.dtype, device=nu_r.device)
    if validate and (speeds.ndim != 1 or speeds.numel() < 2):
        raise ValueError("speed_points must be a 1D sequence with at least two samples.")
    if validate and torch.any(speeds[1:] <= speeds[:-1]):
        raise ValueError("speed_points must be strictly increasing.")

    if scales.ndim == 1:
        if validate and scales.shape[0] != speeds.numel():
            raise ValueError("scale_points length must match speed_points.")
        scales = scales.reshape(-1, 1).repeat(1, 6)
    if validate and (scales.ndim != 2 or scales.shape != (speeds.numel(), 6)):
        raise ValueError(f"scale_points must have shape ({speeds.numel()},) or ({speeds.numel()}, 6).")

    query = torch.abs(nu_r)
    if clamp:
        query = torch.clamp(query, speeds[0], speeds[-1])

    high = torch.bucketize(query.contiguous(), speeds)
    high = torch.clamp(high, min=1, max=speeds.numel() - 1)
    low = high - 1

    x0 = speeds[low]
    x1 = speeds[high]
    blend = (query - x0) / torch.clamp(x1 - x0, min=1.0e-6)
    dof_indices = torch.arange(6, dtype=torch.long, device=nu_r.device).reshape(1, 6).repeat(nu_r.shape[0], 1)
    y0 = scales[low, dof_indices]
    y1 = scales[high, dof_indices]
    return y0 + blend * (y1 - y0)

"""Reusable linear algebra and energy calculations for hydrodynamic fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

def added_mass_coriolis_product(nu: np.ndarray, added_mass: np.ndarray) -> np.ndarray:
    """Return ``C_A(nu) nu`` using the simulator's full-matrix convention."""

    velocity = np.asarray(nu, dtype=float)
    matrix = np.asarray(added_mass, dtype=float)
    one_sample = velocity.ndim == 1
    velocity = np.atleast_2d(velocity)
    if velocity.shape[1] != 6 or matrix.shape != (6, 6):
        raise ValueError("nu must have shape (N,6) and added_mass must have shape (6,6)")
    momentum = velocity @ matrix.T
    linear_momentum, angular_momentum = momentum[:, :3], momentum[:, 3:]
    linear_velocity, omega = velocity[:, :3], velocity[:, 3:]
    top = -np.cross(linear_momentum, omega)
    bottom = -np.cross(linear_momentum, linear_velocity) - np.cross(angular_momentum, omega)
    result = np.concatenate((top, bottom), axis=1)
    return result[0] if one_sample else result


def project_symmetric_psd(matrix: np.ndarray, min_eigenvalue: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetrize a 6x6 matrix and clip its eigenvalues."""

    source = np.asarray(matrix, dtype=float)
    if source.shape != (6, 6):
        raise ValueError(f"matrix must have shape (6,6), got {source.shape}")
    symmetric = 0.5 * (source + source.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected_values = np.maximum(eigenvalues, float(min_eigenvalue))
    projected = (eigenvectors * projected_values) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    diagnostics = {
        "enabled": True,
        "raw_asymmetry_frobenius": float(np.linalg.norm(source - source.T)),
        "symmetric_eigenvalues": eigenvalues.tolist(),
        "projected_eigenvalues": projected_values.tolist(),
        "correction_frobenius": float(np.linalg.norm(projected - source)),
        "minimum_eigenvalue": float(min_eigenvalue),
    }
    return projected, diagnostics


def scaled_lstsq(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales <= np.finfo(float).tiny):
        raise ValueError(f"Unexcited regression column(s), norms={scales.tolist()}")
    scaled = design / scales
    coefficients_scaled, _, rank, singular = np.linalg.lstsq(scaled, target, rcond=None)
    coefficients = coefficients_scaled / scales[:, None]
    condition = float(np.inf if singular[-1] <= 0.0 else singular[0] / singular[-1])
    return coefficients, {
        "rank": int(rank),
        "condition_number_scaled": condition,
        "feature_norms": scales.tolist(),
        "singular_values_scaled": singular.tolist(),
    }


def damping_dissipated_power(
    nu: np.ndarray,
    linear_damping: np.ndarray,
    quadratic_damping: np.ndarray,
) -> np.ndarray:
    velocity = np.asarray(nu, dtype=float)
    linear_load = velocity @ np.asarray(linear_damping, dtype=float).T
    quadratic_load = (np.abs(velocity) * velocity) @ np.asarray(quadratic_damping, dtype=float).T
    return np.sum(velocity * (linear_load + quadratic_load), axis=1)


"""Scaled least squares for hydrodynamic fitting."""

from __future__ import annotations

from typing import Any

import numpy as np


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

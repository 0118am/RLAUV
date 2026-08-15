"""Small OpenFOAM text formatting primitives."""

from __future__ import annotations

import math
from typing import Any

def fmt(value: float) -> str:
    return f"{value:.12g}"


def foam_header(object_name: str, class_name: str = "dictionary") -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def _finite_vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector


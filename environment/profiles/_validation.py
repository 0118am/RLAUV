"""Shared validation helpers for simulator-independent profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


HydroCoefficients = Sequence[float] | Sequence[Sequence[float]]


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def validate_length(value: Sequence[Any], length: int, name: str) -> None:
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {len(value)}.")


def validate_nonnegative(value: float, name: str) -> None:
    if finite_float(value, name) < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def validate_positive(value: float, name: str) -> None:
    if finite_float(value, name) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def validate_range(value: Sequence[float], name: str, *, integer: bool = False) -> None:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a two-value sequence.")
    validate_length(value, 2, name)
    lower = finite_float(value[0], f"{name}[0]")
    upper = finite_float(value[1], f"{name}[1]")
    if upper < lower:
        raise ValueError(f"{name} upper bound must be >= lower bound.")
    if integer and (int(value[0]) != value[0] or int(value[1]) != value[1]):
        raise ValueError(f"{name} must contain integer values.")


def validate_nonnegative_sequence(value: Sequence[float], name: str) -> None:
    if not is_sequence(value) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty sequence.")
    for index, item in enumerate(value):
        if is_sequence(item):
            raise ValueError(f"{name}[{index}] must be a scalar.")
        validate_nonnegative(float(item), f"{name}[{index}]")


def validate_integer_sequence(
    value: Sequence[int],
    name: str,
    *,
    nonnegative: bool = False,
) -> None:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a sequence.")
    previous = None
    for index, item in enumerate(value):
        finite_float(item, f"{name}[{index}]")
        if int(item) != item:
            raise ValueError(f"{name}[{index}] must be an integer.")
        if nonnegative and int(item) < 0:
            raise ValueError(f"{name}[{index}] must be non-negative.")
        if previous is not None and int(item) <= previous:
            raise ValueError(f"{name} must be strictly increasing.")
        previous = int(item)


def validate_vector(value: Sequence[Any], length: int, name: str) -> None:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a sequence.")
    validate_length(value, length, name)
    for index, item in enumerate(value):
        if is_sequence(item):
            raise ValueError(f"{name}[{index}] must be a scalar.")
        finite_float(item, f"{name}[{index}]")


def count_current_vectors(value: Any, name: str) -> int:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a nested sequence of 3D current vectors.")
    if len(value) == 0:
        return 0
    if all(not is_sequence(item) for item in value):
        validate_vector(value, 3, name)
        return 1
    return sum(count_current_vectors(item, f"{name}[]") for item in value)


def validate_6_vector_or_matrix(value: HydroCoefficients, name: str) -> None:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a 6-vector or 6x6 matrix.")
    validate_length(value, 6, name)
    if is_sequence(value[0]):
        for row_index, row in enumerate(value):
            if not is_sequence(row):
                raise ValueError(f"{name}[{row_index}] must be a 6-value row.")
            validate_vector(row, 6, f"{name}[{row_index}]")
    else:
        validate_vector(value, 6, name)


def validate_increasing_axis(points: Sequence[float], name: str) -> None:
    if not is_sequence(points) or len(points) < 2:
        raise ValueError(f"{name} must contain at least two points.")
    previous = finite_float(points[0], f"{name}[0]")
    for index, point in enumerate(points[1:], start=1):
        point_value = finite_float(point, f"{name}[{index}]")
        if point_value <= previous:
            raise ValueError(f"{name} must be strictly increasing at index {index}.")
        previous = point_value


def validate_damping_speed_scale_curve(
    speed_points: Sequence[float],
    scale_points: Sequence[Any],
    name: str,
) -> None:
    validate_increasing_axis(speed_points, "hydrodynamics.damping_speed_points")
    if not is_sequence(scale_points) or len(scale_points) == 0:
        return
    if len(scale_points) != len(speed_points):
        raise ValueError(f"{name} must have one sample per damping_speed_points entry.")
    if is_sequence(scale_points[0]):
        for row_index, row in enumerate(scale_points):
            validate_vector(row, 6, f"{name}[{row_index}]")
            for col_index, item in enumerate(row):
                validate_nonnegative(float(item), f"{name}[{row_index}][{col_index}]")
    else:
        validate_vector(scale_points, len(speed_points), name)
        for index, item in enumerate(scale_points):
            validate_nonnegative(float(item), f"{name}[{index}]")


def validate_inertia_tensor(value: Any, name: str) -> None:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a 3-vector, 3x3 matrix, or flat 9-value matrix.")
    if len(value) == 3 and all(not is_sequence(item) for item in value):
        validate_vector(value, 3, name)
        moments = [float(item) for item in value]
        if any(item <= 0.0 for item in moments):
            raise ValueError(f"{name} diagonal entries must be positive.")
        if any(moment > sum(moments) - moment + 1.0e-9 for moment in moments):
            raise ValueError(f"{name} must satisfy the rigid-body inertia triangle inequalities.")
        return
    if len(value) == 9 and all(not is_sequence(item) for item in value):
        rows = [value[0:3], value[3:6], value[6:9]]
    elif len(value) == 3 and all(is_sequence(item) for item in value):
        rows = value
        for row_index, row in enumerate(rows):
            validate_vector(row, 3, f"{name}[{row_index}]")
    else:
        raise ValueError(f"{name} must be a 3-vector, 3x3 matrix, or flat 9-value matrix.")
    for index in range(3):
        if float(rows[index][index]) <= 0.0:
            raise ValueError(f"{name} diagonal entries must be positive.")
    for row in range(3):
        for col in range(row + 1, 3):
            if abs(float(rows[row][col]) - float(rows[col][row])) > 1.0e-6:
                raise ValueError(f"{name} must be symmetric.")
    a, b, c = (float(item) for item in rows[0])
    _, d, e = (float(item) for item in rows[1])
    _, _, f = (float(item) for item in rows[2])
    leading_minor_2 = a * d - b * b
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    if leading_minor_2 <= 0.0 or determinant <= 0.0:
        raise ValueError(f"{name} must be positive definite.")
    diagonal = (a, d, f)
    if any(moment > sum(diagonal) - moment + 1.0e-9 for moment in diagonal):
        raise ValueError(f"{name} must satisfy the rigid-body inertia triangle inequalities.")


def validate_payload_samples(value: Sequence[Mapping[str, Any]], name: str) -> None:
    if not is_sequence(value):
        raise ValueError(f"{name} must be a sequence of payload mappings.")
    allowed = {
        "name", "weight", "mass", "volume", "inertia", "center_of_mass_offset",
        "com_to_cob_offset", "linear_damping_scale", "quadratic_damping_scale", "added_mass_scale",
    }
    required = {"mass", "volume", "inertia", "center_of_mass_offset", "com_to_cob_offset"}
    names: set[str] = set()
    for index, sample in enumerate(value):
        sample_name = f"{name}[{index}]"
        if not isinstance(sample, Mapping):
            raise ValueError(f"{sample_name} must be a mapping.")
        unknown = sorted(set(sample) - allowed)
        missing = sorted(required - set(sample))
        if unknown:
            raise ValueError(f"{sample_name} contains unknown field(s): {', '.join(unknown)}.")
        if missing:
            raise ValueError(f"{sample_name} is missing field(s): {', '.join(missing)}.")
        label = str(sample.get("name", f"payload-{index}"))
        if not label.strip() or label in names:
            raise ValueError(f"{sample_name}.name must be non-empty and unique.")
        names.add(label)
        validate_positive(sample.get("weight", 1.0), f"{sample_name}.weight")
        validate_positive(sample["mass"], f"{sample_name}.mass")
        validate_positive(sample["volume"], f"{sample_name}.volume")
        validate_inertia_tensor(sample["inertia"], f"{sample_name}.inertia")
        validate_vector(sample["center_of_mass_offset"], 3, f"{sample_name}.center_of_mass_offset")
        validate_vector(sample["com_to_cob_offset"], 3, f"{sample_name}.com_to_cob_offset")
        for scale_name in ("linear_damping_scale", "quadratic_damping_scale", "added_mass_scale"):
            scale = sample.get(scale_name, 1.0)
            if is_sequence(scale):
                validate_length(scale, 6, f"{sample_name}.{scale_name}")
                validate_nonnegative_sequence(scale, f"{sample_name}.{scale_name}")
            else:
                validate_nonnegative(scale, f"{sample_name}.{scale_name}")

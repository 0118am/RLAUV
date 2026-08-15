"""Evaluation defaults, validation, and case naming."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


DEFAULT_EVALUATION_DURATION_S = 32.0
DEFAULT_RANDOM_CURVE_COUNT = 8
DEFAULT_CURRENT_TAU_S = 12.0
DEFAULT_DYNAMICS_SCALE = 1.0


def sanitize_evaluation_label(label: str) -> str:
    """Return the filesystem-safe representation used by every eval component."""

    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_")


def format_evaluation_token(value: float) -> str:
    """Format a finite scalar deterministically for a case-directory token."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Evaluation label values must be finite.")
    return f"{value:.3g}".replace("-", "m").replace(".", "p")


def validate_evaluation_parameters(
    *,
    duration_s: float,
    current_w: Sequence[float] | None = None,
    current_variation_std: float = 0.0,
    current_tau: float = 12.0,
    damping_scale: float = 1.0,
    thruster_scale: float = 1.0,
    thruster_tau_scale: float = 1.0,
    num_envs: int | None = None,
    random_curve_count: int = DEFAULT_RANDOM_CURVE_COUNT,
) -> None:
    """Reject non-finite or physically invalid evaluation requests early."""

    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("Evaluation duration must be finite and positive.")
    if current_w is not None:
        if len(current_w) != 3 or not all(math.isfinite(float(value)) for value in current_w):
            raise ValueError("eval_current must contain exactly three finite world-frame components.")
    variation = float(current_variation_std)
    if not math.isfinite(variation) or variation < 0.0:
        raise ValueError("eval_current_variation_std must be finite and non-negative.")
    positive_values = {
        "eval_current_tau": current_tau,
        "eval_damping_scale": damping_scale,
        "eval_thruster_scale": thruster_scale,
        "eval_thruster_tau_scale": thruster_tau_scale,
    }
    for name, raw_value in positive_values.items():
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if num_envs is not None and (int(num_envs) != num_envs or int(num_envs) <= 0):
        raise ValueError("num_envs must be a positive integer when provided.")
    if int(random_curve_count) != random_curve_count or int(random_curve_count) <= 0:
        raise ValueError("random_curve_count must be a positive integer.")


def resolve_positive_scalar_or_range(
    scalar: float | None,
    value_range: Sequence[float] | None,
    *,
    scalar_name: str,
    range_name: str,
) -> tuple[float, float] | None:
    """Resolve one positive scalar/range option without ambiguous overrides."""

    if scalar is not None and value_range is not None:
        raise ValueError(f"Specify only one of {scalar_name} or {range_name}.")
    if scalar is not None:
        value = float(scalar)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{scalar_name} must be finite and positive.")
        return value, value
    if value_range is None:
        return None
    if len(value_range) != 2:
        raise ValueError(f"{range_name} must contain exactly two values.")
    lower, upper = float(value_range[0]), float(value_range[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or upper < lower:
        raise ValueError(f"{range_name} must satisfy finite 0 < lower <= upper.")
    return lower, upper


def resolve_random_smooth_ranges(
    *,
    trajectory_amp_x: float | None,
    trajectory_amp_y: float | None,
    trajectory_amp_z: float | None,
    trajectory_period: float | None,
    trajectory_amp_x_range: Sequence[float] | None,
    trajectory_amp_y_range: Sequence[float] | None,
    trajectory_amp_z_range: Sequence[float] | None,
    trajectory_period_range: Sequence[float] | None,
) -> dict[str, tuple[float, float]]:
    """Return the mandatory non-static random-smooth evaluation envelope."""

    inputs = (
        ("trajectory_amp_x", trajectory_amp_x, "trajectory_amp_x_range", trajectory_amp_x_range),
        ("trajectory_amp_y", trajectory_amp_y, "trajectory_amp_y_range", trajectory_amp_y_range),
        ("trajectory_amp_z", trajectory_amp_z, "trajectory_amp_z_range", trajectory_amp_z_range),
        ("trajectory_period", trajectory_period, "trajectory_period_range", trajectory_period_range),
    )
    resolved = {
        range_name: resolve_positive_scalar_or_range(
            scalar,
            value_range,
            scalar_name=scalar_name,
            range_name=range_name,
        )
        for scalar_name, scalar, range_name, value_range in inputs
    }
    missing = [name for name, value in resolved.items() if value is None]
    if missing:
        raise ValueError(
            "random_smooth evaluation requires explicit positive amplitude and period ranges; missing "
            + ", ".join(missing)
            + "."
        )
    return {name: value for name, value in resolved.items() if value is not None}


def build_evaluation_case_label(
    *,
    evaluation_label: str = "",
    disturbance_name: str | None = None,
    sample_domain_randomization: bool = False,
    domain_randomization_name: str | None = None,
    seed: int = 0,
    current_w: Sequence[float] | None = None,
    smooth_current: bool = False,
    current_variation_std: float = 0.0,
    damping_scale: float = 1.0,
    thruster_scale: float = 1.0,
    thruster_tau_scale: float = 1.0,
) -> str:
    """Build the one canonical directory label for an evaluation request."""

    if evaluation_label:
        return sanitize_evaluation_label(evaluation_label)
    if disturbance_name:
        return sanitize_evaluation_label(disturbance_name)

    parts: list[str] = []
    if sample_domain_randomization:
        if not domain_randomization_name:
            raise ValueError("Sampled DR evaluation requires its resolved recipe name.")
        parts.append("dr_" + sanitize_evaluation_label(domain_randomization_name) + f"_seed{int(seed)}")
    if current_w is not None:
        parts.append("cur_" + "_".join(format_evaluation_token(value) for value in current_w))
    if smooth_current or float(current_variation_std) > 0.0:
        parts.append(f"smooth{format_evaluation_token(current_variation_std)}")
    scalar_tokens = (
        ("damp", damping_scale),
        ("thr", thruster_scale),
        ("tau", thruster_tau_scale),
    )
    for prefix, raw_value in scalar_tokens:
        value = float(raw_value)
        if abs(value - 1.0) > 1.0e-9:
            parts.append(f"{prefix}{format_evaluation_token(value)}")
    return sanitize_evaluation_label("_".join(parts))

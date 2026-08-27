"""Evaluation defaults, validation, and case naming."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from simulation.domain_randomization import disturbance_stage_count


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


def _scale_coefficients(values, scale: float) -> list:
    return [
        [float(item) * scale for item in row]
        if isinstance(row, (list, tuple))
        else float(row) * scale
        for row in values
    ]


def apply_evaluation_physics_overlay(cfg) -> None:
    """Apply CLI-selected evaluation changes after profile and DR resolution."""

    overlay = dict(cfg.evaluation_physics_overlay or {})
    damping_scale = float(overlay.get("damping_scale", 1.0))
    cfg.linear_damping = _scale_coefficients(cfg.linear_damping, damping_scale)
    cfg.quadratic_damping = _scale_coefficients(cfg.quadratic_damping, damping_scale)
    cfg.dyn_time_constant = float(cfg.dyn_time_constant) * float(
        overlay.get("thruster_tau_scale", 1.0)
    )

    current_override = "water_current_w" in overlay or bool(overlay.get("smooth_current", False))
    if current_override:
        cfg.water_current_w = [
            float(value) for value in overlay.get("water_current_w", cfg.water_current_w)
        ]
        variation_std = float(overlay.get("current_variation_std", 0.0))
        current_tau = float(overlay.get("current_tau", DEFAULT_CURRENT_TAU_S))
        cfg.evaluation_current_override = True
        cfg.evaluation_current_variation_std = variation_std
        cfg.evaluation_current_tau = current_tau
        if bool(overlay.get("smooth_current", False)) or variation_std > 0.0:
            cfg.domain_randomization.use_custom_randomization = True
            cfg.domain_randomization.water_current_smooth = True
            stage_count = disturbance_stage_count(cfg.domain_randomization)
            cfg.domain_randomization.water_current_variation_std_by_stage = [
                variation_std
            ] * stage_count
            horizontal_max = math.hypot(cfg.water_current_w[0], cfg.water_current_w[1])
            vertical_max = abs(cfg.water_current_w[2])
            cfg.domain_randomization.water_current_max_by_stage = [
                horizontal_max
            ] * stage_count
            cfg.domain_randomization.water_current_vertical_max_by_stage = [
                vertical_max
            ] * stage_count
            cfg.domain_randomization.water_current_tau_range = [current_tau, current_tau]
            cfg.eval_domain_randomization = True
            if bool(overlay.get("current_feature_only", False)):
                cfg.domain_randomization.enabled_features = ["current"]

    if "thruster_force_scale" in overlay:
        cfg.evaluation_thruster_force_scale_override = True
        cfg.evaluation_thruster_force_scale = float(overlay["thruster_force_scale"])


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

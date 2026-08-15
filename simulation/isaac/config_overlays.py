"""Evaluation-only configuration overlays applied after profile composition."""

from __future__ import annotations

import numpy as np

def _scale_evaluation_damping(values, scale: float, name: str) -> list:
    """Scale a diagonal 6-vector or full 6x6 damping matrix."""

    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape not in ((6,), (6, 6)):
        raise ValueError(f"{name} must be a 6-vector or 6x6 matrix, got shape {coefficients.shape}.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} must contain only finite values.")
    return (coefficients * scale).tolist()


def apply_evaluation_physics_overlay(cfg) -> None:
    """Apply the evaluation-only physics overlay after profile and DR setup."""

    overlay = dict(getattr(cfg, "evaluation_physics_overlay", {}) or {})
    allowed = {
        "damping_scale",
        "thruster_tau_scale",
        "water_current_w",
        "smooth_current",
        "current_variation_std",
        "current_tau",
        "current_feature_only",
        "thruster_force_scale",
    }
    unknown = set(overlay) - allowed
    if unknown:
        raise ValueError(f"Unknown evaluation physics overlay keys: {', '.join(sorted(unknown))}.")
    damping_scale = float(overlay.get("damping_scale", 1.0))
    tau_scale = float(overlay.get("thruster_tau_scale", 1.0))
    if not np.isfinite(damping_scale) or not np.isfinite(tau_scale) or damping_scale <= 0.0 or tau_scale <= 0.0:
        raise ValueError("Evaluation physics overlay scales must be finite and positive.")
    cfg.linear_damping = _scale_evaluation_damping(
        cfg.linear_damping,
        damping_scale,
        "linear_damping",
    )
    cfg.quadratic_damping = _scale_evaluation_damping(
        cfg.quadratic_damping,
        damping_scale,
        "quadratic_damping",
    )
    cfg.dyn_time_constant = float(cfg.dyn_time_constant) * tau_scale
    current_override = "water_current_w" in overlay or bool(overlay.get("smooth_current", False))
    if current_override:
        current = overlay.get("water_current_w", cfg.water_current_w)
        current_values = [float(value) for value in current]
        if len(current_values) != 3 or not all(np.isfinite(value) for value in current_values):
            raise ValueError("evaluation_physics_overlay.water_current_w must contain three finite values.")
        variation_std = float(overlay.get("current_variation_std", 0.0))
        current_tau = float(overlay.get("current_tau", 12.0))
        if not np.isfinite(variation_std) or variation_std < 0.0:
            raise ValueError("evaluation_physics_overlay.current_variation_std must be finite and non-negative.")
        if not np.isfinite(current_tau) or current_tau <= 0.0:
            raise ValueError("evaluation_physics_overlay.current_tau must be finite and positive.")

        cfg.water_current_w = current_values
        cfg.evaluation_current_override = True
        cfg.evaluation_current_variation_std = variation_std
        cfg.evaluation_current_tau = current_tau
        smooth_current = bool(overlay.get("smooth_current", False)) or variation_std > 0.0
        if smooth_current:
            cfg.domain_randomization.use_custom_randomization = True
            cfg.domain_randomization.water_current_smooth = True
            stage_count = max(1, len(cfg.domain_randomization.water_current_max_by_stage))
            cfg.domain_randomization.water_current_variation_std_by_stage = [variation_std] * stage_count
            cfg.domain_randomization.water_current_tau_range = [current_tau, current_tau]
            cfg.eval_domain_randomization = True
            if bool(overlay.get("current_feature_only", False)):
                cfg.domain_randomization.enabled_features = ["current"]

    if "thruster_force_scale" in overlay:
        force_scale = float(overlay["thruster_force_scale"])
        if not np.isfinite(force_scale) or force_scale <= 0.0:
            raise ValueError("evaluation_physics_overlay.thruster_force_scale must be finite and positive.")
        cfg.evaluation_thruster_force_scale_override = True
        cfg.evaluation_thruster_force_scale = force_scale

"""Fit diagnostics for cycle stability, passivity, and full-model residuals."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .cycles import OddProjectedCase, analysis_window
from .motion import CaseData, DOF_NAMES, WRENCH_NAMES
from .regression import added_mass_coriolis_product, damping_dissipated_power, scaled_lstsq
from .types import FitOptions

_COEFFICIENT_NAMES = ("added_mass_raw", "linear_damping", "quadratic_damping")


def _rms(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Return a float64 RMS without relying on the input dtype."""

    source = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(np.square(source), axis=axis))


def _last_two_coefficient_change(
    previous: float,
    latest: float,
    *,
    feature_rms: float,
    odd_load_rms: float,
) -> dict[str, Any]:
    """Describe a coefficient change without dividing by a numerical zero.

    The ordinary relative change uses the previous cycle as its denominator.
    If that coefficient is negligible compared with the coefficient scale
    capable of producing the observed odd load, the relative number is
    deliberately omitted.  The absolute change and its load-normalised
    contribution remain available, so a zero does not silently become a pass.
    """

    absolute_change = abs(latest - previous)
    tiny = np.finfo(float).tiny
    coefficient_scale = (
        odd_load_rms / feature_rms
        if feature_rms > tiny and odd_load_rms > tiny
        else None
    )
    near_zero_floor = (
        float(np.sqrt(np.finfo(float).eps) * coefficient_scale)
        if coefficient_scale is not None
        else float(64.0 * np.finfo(float).eps * max(1.0, abs(previous), abs(latest)))
    )
    previous_is_near_zero = abs(previous) <= near_zero_floor
    relative_change = (
        None
        if previous_is_near_zero
        else float(100.0 * absolute_change / abs(previous))
    )
    normalised_change = (
        None
        if coefficient_scale is None
        else float(100.0 * absolute_change / coefficient_scale)
    )
    return {
        "previous": float(previous),
        "latest": float(latest),
        "absolute_change": float(absolute_change),
        "absolute_relative_change_percent": relative_change,
        "relative_change_denominator": "absolute previous-cycle coefficient",
        "relative_change_status": (
            "near_zero_previous_use_absolute_and_load_normalized_change"
            if previous_is_near_zero
            else "reported"
        ),
        "near_zero_floor": near_zero_floor,
        "load_equivalent_coefficient_scale": (
            None if coefficient_scale is None else float(coefficient_scale)
        ),
        "absolute_change_normalized_by_odd_load_percent": normalised_change,
        "normalization_status": (
            "unavailable_near_zero_odd_load_or_feature"
            if coefficient_scale is None
            else "reported"
        ),
    }


def _last_two_waveform_change(
    previous: np.ndarray,
    latest: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Compare phase-aligned odd-load waveforms for every wrench component."""

    if previous.shape != latest.shape or previous.ndim != 2 or previous.shape[1] != 6:
        raise ValueError(
            "Last two odd-load cycles must have matching phase rows and six wrench columns"
        )
    previous_rms = _rms(previous, axis=0)
    latest_rms = _rms(latest, axis=0)
    difference_rms = _rms(latest - previous, axis=0)
    pooled_rms = _rms(np.concatenate((previous, latest), axis=0), axis=0)
    result: dict[str, dict[str, Any]] = {}
    for index, wrench_name in enumerate(WRENCH_NAMES):
        scale = float(pooled_rms[index])
        floor = float(
            64.0
            * np.finfo(float).eps
            * max(1.0, float(previous_rms[index]), float(latest_rms[index]))
        )
        previous_is_near_zero = float(previous_rms[index]) <= floor
        both_are_near_zero = scale <= floor
        result[wrench_name] = {
            "previous_waveform_rms": float(previous_rms[index]),
            "latest_waveform_rms": float(latest_rms[index]),
            "absolute_waveform_rms_change": float(
                abs(latest_rms[index] - previous_rms[index])
            ),
            "rms_of_phase_aligned_pointwise_change": float(difference_rms[index]),
            "absolute_relative_waveform_rms_change_percent": (
                None
                if previous_is_near_zero
                else float(
                    100.0
                    * abs(latest_rms[index] - previous_rms[index])
                    / previous_rms[index]
                )
            ),
            "relative_change_status": (
                "near_zero_previous_use_absolute_and_pooled_normalized_change"
                if previous_is_near_zero
                else "reported"
            ),
            "pooled_waveform_rms": scale,
            "phase_aligned_change_normalized_by_pooled_rms_percent": (
                None if both_are_near_zero else float(100.0 * difference_rms[index] / scale)
            ),
            "normalization_status": (
                "unavailable_both_waveforms_near_zero"
                if both_are_near_zero
                else "reported"
            ),
            "near_zero_floor": floor,
        }
    return result


def _cycle_convergence_diagnostic(item: OddProjectedCase) -> dict[str, Any]:
    """Fit each complete cycle and compare the final two phase-aligned cycles."""

    cycle_ids = np.unique(item.cycle_id)
    cycles: list[dict[str, Any]] = []
    rows_by_cycle: list[np.ndarray] = []
    coefficients_by_cycle: list[np.ndarray] = []
    for cycle_id in cycle_ids:
        rows = np.flatnonzero(item.cycle_id == cycle_id)
        design = item.design[rows]
        target = item.target[rows]
        coefficients, fit = scaled_lstsq(design, target)
        if fit["rank"] < 3:
            raise ValueError(
                f"{item.case_name}: per-cycle regression rank is below 3 for cycle {cycle_id}"
            )
        residual = target - design @ coefficients
        cycles.append(
            {
                "cycle_id": int(cycle_id),
                "odd_phase_row_count": int(rows.size),
                "coefficients_by_wrench": {
                    name: coefficients[index].tolist()
                    for index, name in enumerate(_COEFFICIENT_NAMES)
                },
                "odd_load_waveform_rms_by_wrench": _rms(target, axis=0).tolist(),
                "odd_model_residual_rms_by_wrench": _rms(residual, axis=0).tolist(),
                "fit": fit,
            }
        )
        rows_by_cycle.append(rows)
        coefficients_by_cycle.append(coefficients)

    result: dict[str, Any] = {
        "case_name": item.case_name,
        "dof": DOF_NAMES[item.dof_index],
        "dof_index": item.dof_index,
        "wrench_order": list(WRENCH_NAMES),
        "coefficient_order": list(_COEFFICIENT_NAMES),
        "weighting": "one identical uniform half-cycle phase grid per complete cycle",
        "cycles": cycles,
    }
    if cycle_ids.size < 2:
        result["last_two_cycle_comparison"] = {
            "available": False,
            "reason": "fewer_than_two_complete_cycles",
        }
        return result

    previous_rows = rows_by_cycle[-2]
    latest_rows = rows_by_cycle[-1]
    if previous_rows.size != latest_rows.size:
        result["last_two_cycle_comparison"] = {
            "available": False,
            "reason": "last_two_cycles_have_different_finite_phase_row_counts",
            "previous_cycle_id": int(cycle_ids[-2]),
            "latest_cycle_id": int(cycle_ids[-1]),
            "previous_row_count": int(previous_rows.size),
            "latest_row_count": int(latest_rows.size),
        }
        return result

    previous_coefficients = coefficients_by_cycle[-2]
    latest_coefficients = coefficients_by_cycle[-1]
    main_wrench = item.dof_index
    paired_design = np.concatenate(
        (item.design[previous_rows], item.design[latest_rows]), axis=0
    )
    paired_main_target = np.concatenate(
        (item.target[previous_rows, main_wrench], item.target[latest_rows, main_wrench])
    )
    feature_rms = _rms(paired_design, axis=0)
    odd_load_rms = float(_rms(paired_main_target))
    coefficient_changes = {
        name: _last_two_coefficient_change(
            previous_coefficients[index, main_wrench],
            latest_coefficients[index, main_wrench],
            feature_rms=float(feature_rms[index]),
            odd_load_rms=odd_load_rms,
        )
        for index, name in enumerate(_COEFFICIENT_NAMES)
    }
    result["last_two_cycle_comparison"] = {
        "available": True,
        "previous_cycle_id": int(cycle_ids[-2]),
        "latest_cycle_id": int(cycle_ids[-1]),
        "main_response_wrench": WRENCH_NAMES[main_wrench],
        "main_response_coefficient_changes": coefficient_changes,
        "odd_load_waveform_changes_by_wrench": _last_two_waveform_change(
            item.target[previous_rows], item.target[latest_rows]
        ),
    }
    return result


def _raw_intercept_diagnostic(case: CaseData) -> dict[str, Any]:
    mask = analysis_window(case)
    time = case.time_s[mask]
    target = case.wrench_body[mask]
    scalar_eta, scalar_nu, scalar_nudot = case.motion.kinematics(time)
    design = np.column_stack(
        (
            np.ones(time.size),
            -scalar_nudot,
            -scalar_nu,
            -np.abs(scalar_nu) * scalar_nu,
        )
    )
    finite = np.all(np.isfinite(design), axis=1) & np.all(np.isfinite(target), axis=1)
    coefficients, _, rank, singular = np.linalg.lstsq(design[finite], target[finite], rcond=None)
    residual = target[finite] - design[finite] @ coefficients
    return {
        "case_name": case.motion.case_name,
        "dof": case.motion.dof,
        "sample_count": int(np.count_nonzero(finite)),
        "rank": int(rank),
        "condition_number": float(np.inf if singular[-1] <= 0.0 else singular[0] / singular[-1]),
        "intercept_body": coefficients[0].tolist(),
        "residual_rms_by_wrench": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
    }


def _passivity_diagnostics(
    cases: Sequence[CaseData],
    linear: np.ndarray,
    quadratic: np.ndarray,
    options: FitOptions,
) -> dict[str, Any]:
    symmetric_linear = 0.5 * (linear + linear.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_linear)
    observed = np.concatenate([case.nu[analysis_window(case)] for case in cases], axis=0)
    observed_power = damping_dissipated_power(observed, linear, quadratic)
    result: dict[str, Any] = {
        "linear_symmetric_eigenvalues": eigenvalues.tolist(),
        "linear_minimum_symmetric_eigenvalue": float(eigenvalues[0]),
        "observed_minimum_dissipated_power": float(np.min(observed_power)),
        "observed_negative_fraction": float(np.mean(observed_power < -options.passivity_tolerance)),
        "tolerance": options.passivity_tolerance,
    }
    if options.passivity_samples > 0:
        envelope = np.max(np.abs(observed), axis=0)
        rng = np.random.default_rng(options.bootstrap_seed + 1)
        sampled_velocity = rng.uniform(-1.0, 1.0, (options.passivity_samples, 6)) * envelope
        power = damping_dissipated_power(sampled_velocity, linear, quadratic)
        worst = int(np.argmin(power))
        result.update(
            {
                "random_envelope_max_abs_velocity": envelope.tolist(),
                "random_sample_count": options.passivity_samples,
                "random_minimum_dissipated_power": float(power[worst]),
                "random_negative_fraction": float(np.mean(power < -options.passivity_tolerance)),
                "random_worst_velocity": sampled_velocity[worst].tolist(),
            }
        )
    return result


def _full_model_diagnostics(
    cases: Sequence[CaseData],
    added_mass: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate the complete model, including the even added-mass Coriolis term."""

    result: list[dict[str, Any]] = []
    for case in cases:
        mask = analysis_window(case)
        nu = case.nu[mask]
        nudot = case.nudot[mask]
        measured = case.wrench_body[mask]
        dynamic_prediction = (
            -nudot @ added_mass.T
            - added_mass_coriolis_product(nu, added_mass)
            - nu @ linear.T
            - (np.abs(nu) * nu) @ quadratic.T
        )
        intercept = np.mean(measured - dynamic_prediction, axis=0)
        residual = measured - (dynamic_prediction + intercept)
        result.append(
            {
                "case_name": case.motion.case_name,
                "dof": case.motion.dof,
                "fitted_intercept_body": intercept.tolist(),
                "residual_rms_by_wrench": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                "residual_peak_abs_by_wrench": np.max(np.abs(residual), axis=0).tolist(),
            }
        )
    return result


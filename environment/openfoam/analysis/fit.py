"""Identify full added-mass and damping matrices from single-DOF cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .motion import CaseData, DOF_NAMES, WRENCH_NAMES, load_case_data


@dataclass(frozen=True)
class FitOptions:
    """Numerical options for the coefficient fit."""

    project_added_mass_psd: bool = True
    min_added_mass_eigenvalue: float = 0.0
    bootstrap_samples: int = 200
    bootstrap_seed: int = 20260810
    passivity_samples: int = 10000
    passivity_tolerance: float = 1.0e-10
    minimum_samples_per_dof: int = 12
    phase_samples_per_cycle: int = 256
    diagonal_only: bool = False
    port_starboard_symmetry: bool = False
    include_rotation_attitude_term: bool = True
    include_roll_attitude_term: bool = True

    def __post_init__(self) -> None:
        if self.min_added_mass_eigenvalue < 0.0:
            raise ValueError("min_added_mass_eigenvalue must be non-negative")
        if self.bootstrap_samples < 0 or self.passivity_samples < 0:
            raise ValueError("bootstrap_samples and passivity_samples must be non-negative")
        if self.diagonal_only and self.port_starboard_symmetry:
            raise ValueError(
                "diagonal_only and port_starboard_symmetry are mutually exclusive"
            )
        if self.minimum_samples_per_dof < 3:
            raise ValueError("minimum_samples_per_dof must be at least 3")
        if (
            type(self.phase_samples_per_cycle) is not int
            or self.phase_samples_per_cycle < 8
            or self.phase_samples_per_cycle % 2
        ):
            raise ValueError("phase_samples_per_cycle must be an even integer of at least 8")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FitOptions":
        if not value:
            return cls()
        valid = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value[key] for key in valid if key in value})


@dataclass(frozen=True)
class _OddCase:
    case_name: str
    dof_index: int
    design: np.ndarray
    target: np.ndarray
    even_target: np.ndarray
    cycle_id: np.ndarray
    attitude_feature: np.ndarray | None = None


@dataclass
class HydroFitResult:
    added_mass_raw: np.ndarray
    added_mass: np.ndarray
    linear_damping: np.ndarray
    quadratic_damping: np.ndarray
    diagnostics: dict[str, Any]
    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    case_summaries: list[dict[str, Any]] = field(default_factory=list)
    options: FitOptions = field(default_factory=FitOptions)

    def config_updates(self) -> dict[str, list[list[float]]]:
        return {
            # The project keeps the historical key name while accepting 6x6.
            "added_mass_diag": self.added_mass.tolist(),
            "linear_damping": self.linear_damping.tolist(),
            "quadratic_damping": self.quadratic_damping.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "coordinate_convention": {
                "frame": "body FLU at moving COM",
                "dof_order": list(DOF_NAMES),
                "wrench_order": list(WRENCH_NAMES),
                "force_sign": "fluid-on-body",
                "model": (
                    "tau = -M_A*nudot - C_A(nu,M_A)*nu - D_L*nu "
                    "- D_Q*(abs(nu)*nu)"
                ),
            },
            "matrices": {
                "added_mass_raw": self.added_mass_raw.tolist(),
                "added_mass": self.added_mass.tolist(),
                "linear_damping": self.linear_damping.tolist(),
                "quadratic_damping": self.quadratic_damping.tolist(),
            },
            "config_updates": self.config_updates(),
            "diagnostics": self.diagnostics,
            "confidence_intervals": self.confidence_intervals,
            "cases": self.case_summaries,
            "options": asdict(self.options),
        }


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


def _analysis_window(case: CaseData) -> np.ndarray:
    period = case.motion.period_s
    start = case.motion.settle_cycles * period
    stop = np.inf
    if case.motion.sample_cycles is not None:
        stop = start + case.motion.sample_cycles * period
    tolerance = 1.0e-10 * max(1.0, period)
    return (case.time_s >= start - tolerance) & (case.time_s <= stop + tolerance)


def _odd_project(
    case: CaseData,
    phase_samples_per_cycle: int = 256,
    *,
    include_rotation_attitude_term: bool = True,
) -> _OddCase:
    """Odd-project complete cycles on a fixed phase grid.

    Adaptive time stepping changes the raw sample density over phase.  Using
    those rows directly would therefore turn solver time-step control into a
    regression weight.  Every complete cycle is instead represented by the
    same phase grid before half-period pairing.
    """

    if (
        type(phase_samples_per_cycle) is not int
        or phase_samples_per_cycle < 8
        or phase_samples_per_cycle % 2
    ):
        raise ValueError("phase_samples_per_cycle must be an even integer of at least 8")

    window = _analysis_window(case)
    window_time = case.time_s[window]
    time = case.time_s
    wrench = case.wrench_body
    if window_time.size < 4:
        raise ValueError(f"{case.motion.case_name}: too few samples after settling")
    period = case.motion.period_s
    half_period = 0.5 * period
    start = case.motion.settle_cycles * period
    tolerance = 1.0e-10 * max(1.0, period)
    if case.motion.sample_cycles is None:
        cycle_count = max(
            0,
            int(np.floor((window_time[-1] - start + tolerance) / period)),
        )
        require_all_cycles = False
    else:
        cycle_count = max(
            0,
            int(np.floor(case.motion.sample_cycles + 1.0e-10)),
        )
        require_all_cycles = True

    half_samples = phase_samples_per_cycle // 2
    first_half_phase = np.arange(half_samples, dtype=float) * (
        half_period / half_samples
    )
    first_times: list[np.ndarray] = []
    complete_cycles: list[int] = []
    incomplete_cycles: list[int] = []
    gap_failures: list[str] = []
    for cycle in range(cycle_count):
        cycle_start = start + cycle * period
        cycle_stop = cycle_start + period
        candidate = start + cycle * period + first_half_phase
        paired_candidate = candidate + half_period
        local_time = time[
            (time >= cycle_start - tolerance)
            & (time <= cycle_stop + tolerance)
        ]
        local_count = local_time.size
        if local_count >= 4:
            local_gaps = np.diff(local_time)
            nominal_delta = float(np.median(local_gaps))
            maximum_gap = float(np.max(local_gaps))
            phase_delta = period / phase_samples_per_cycle
            maximum_allowed_gap = max(2.0 * phase_delta, 4.0 * nominal_delta)
            if maximum_gap > maximum_allowed_gap * (1.0 + 1.0e-10):
                gap_failures.append(
                    f"cycle {cycle}: max {maximum_gap:.9g}s > "
                    f"limit {maximum_allowed_gap:.9g}s "
                    f"(median {nominal_delta:.9g}s, phase {phase_delta:.9g}s)"
                )
                incomplete_cycles.append(cycle)
                continue
        if (
            local_count < 4
            or time[0] > candidate[0] + tolerance
            or time[-1] < paired_candidate[-1] - tolerance
        ):
            incomplete_cycles.append(cycle)
            continue
        first_times.append(candidate)
        complete_cycles.append(cycle)

    if require_all_cycles and incomplete_cycles:
        if gap_failures:
            raise ValueError(
                f"{case.motion.case_name}: large raw time gap would be interpolated: "
                + "; ".join(gap_failures)
            )
        raise ValueError(
            f"{case.motion.case_name}: requested sample cycle(s) are incomplete: "
            f"{incomplete_cycles}"
        )
    if not first_times:
        raise ValueError(f"{case.motion.case_name}: no complete cycles after settling")
    pair_time = np.concatenate(first_times)
    pair_cycle = np.repeat(np.asarray(complete_cycles, dtype=int), half_samples)

    paired_wrench = np.column_stack(
        [np.interp(pair_time + half_period, time, wrench[:, component]) for component in range(6)]
    )
    first_wrench = np.column_stack(
        [np.interp(pair_time, time, wrench[:, component]) for component in range(6)]
    )
    odd = 0.5 * (first_wrench - paired_wrench)
    even = 0.5 * (first_wrench + paired_wrench)
    scalar_eta, scalar_nu, scalar_nudot = case.motion.kinematics(pair_time)
    background_fluid_dof = float(
        case.motion.background_fluid_velocity_body_m_s[
            case.motion.dof_index if case.motion.dof_index < 3 else case.motion.dof_index - 3
        ]
    ) if case.motion.dof_index < 3 else 0.0
    if case.motion.dof_index == 0 and abs(background_fluid_dof) > 0.0:
        # Surge damping is a function of the total relative velocity.  Pairing
        # half-period samples removes the steady towing bias while retaining
        # the exact odd part of |U+u|(U+u), rather than approximating it with
        # |u|u around a nonzero towing speed.
        background_relative_dof = -background_fluid_dof
        plus = background_relative_dof + scalar_nu
        minus = background_relative_dof - scalar_nu
        signed_square = 0.5 * (np.abs(plus) * plus - np.abs(minus) * minus)
    else:
        signed_square = np.abs(scalar_nu) * scalar_nu
    design = np.column_stack((-scalar_nudot, -scalar_nu, -signed_square))
    attitude_feature = (
        -scalar_eta
        if include_rotation_attitude_term and case.motion.motion_kind == "rotation"
        else None
    )
    finite = np.all(np.isfinite(design), axis=1) & np.all(np.isfinite(odd), axis=1)
    if np.count_nonzero(finite) < 3:
        raise ValueError(f"{case.motion.case_name}: too few finite odd-projected samples")
    return _OddCase(
        case.motion.case_name,
        case.motion.dof_index,
        design[finite],
        odd[finite],
        even[finite],
        pair_cycle[finite],
        None if attitude_feature is None else attitude_feature[finite],
    )


def _scaled_lstsq(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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


def _cycle_convergence_diagnostic(item: _OddCase) -> dict[str, Any]:
    """Fit each complete cycle and compare the final two phase-aligned cycles."""

    cycle_ids = np.unique(item.cycle_id)
    cycles: list[dict[str, Any]] = []
    rows_by_cycle: list[np.ndarray] = []
    coefficients_by_cycle: list[np.ndarray] = []
    for cycle_id in cycle_ids:
        rows = np.flatnonzero(item.cycle_id == cycle_id)
        design = item.design[rows]
        target = item.target[rows]
        coefficients, fit = _scaled_lstsq(design, target)
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
        "automatic_acceptance": "not_evaluated_no_threshold_configured",
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
    mask = _analysis_window(case)
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


def _fit_odd_groups(
    groups: Mapping[int, Sequence[_OddCase]],
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    added_mass = np.zeros((6, 6), dtype=float)
    linear = np.zeros((6, 6), dtype=float)
    quadratic = np.zeros((6, 6), dtype=float)
    diagnostics: dict[str, Any] = {}
    for dof_index, dof_name in enumerate(DOF_NAMES):
        datasets = list(groups.get(dof_index, ()))
        if not datasets:
            raise ValueError(f"No single-DOF cases supplied for {dof_name}")
        design = np.concatenate([item.design for item in datasets], axis=0)
        target = np.concatenate([item.target for item in datasets], axis=0)
        attitude = None
        if all(item.attitude_feature is not None for item in datasets):
            attitude = np.concatenate(
                [np.asarray(item.attitude_feature, dtype=float) for item in datasets]
            )
            design_fit = np.column_stack((design, attitude))
        else:
            design_fit = design
        if design.shape[0] < minimum_samples:
            raise ValueError(
                f"DOF {dof_name} has {design.shape[0]} odd samples; at least {minimum_samples} required"
            )
        coefficients_fit, dof_diagnostics = _scaled_lstsq(design_fit, target)
        required_rank = 4 if attitude is not None else 3
        if dof_diagnostics["rank"] < required_rank:
            raise ValueError(
                f"DOF {dof_name} regression rank is below {required_rank}; "
                "rotation attitude and added-mass terms require multiple frequencies"
            )
        coefficients = coefficients_fit[:3]
        added_mass[:, dof_index] = coefficients[0]
        linear[:, dof_index] = coefficients[1]
        quadratic[:, dof_index] = coefficients[2]
        residual = target - design_fit @ coefficients_fit
        even = np.concatenate([item.even_target for item in datasets], axis=0)
        dof_diagnostics.update(
            {
                "case_names": [item.case_name for item in datasets],
                "sample_count": int(design.shape[0]),
                "weighting": "equal uniform phase rows per complete cycle",
                "complete_cycles_by_case": {
                    item.case_name: int(np.unique(item.cycle_id).size)
                    for item in datasets
                },
                "odd_samples_by_case": {
                    item.case_name: int(item.design.shape[0]) for item in datasets
                },
                "odd_residual_rms_by_wrench": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                "even_component_rms_by_wrench": np.sqrt(np.mean(even**2, axis=0)).tolist(),
                "rotation_attitude_coefficient_by_wrench": (
                    None if attitude is None else coefficients_fit[3].tolist()
                ),
            }
        )
        diagnostics[dof_name] = dof_diagnostics
    return added_mass, linear, quadratic, diagnostics


def _bootstrap(
    groups: Mapping[int, Sequence[_OddCase]],
    options: FitOptions,
) -> dict[str, Any]:
    if options.bootstrap_samples <= 0:
        return {}
    rng = np.random.default_rng(options.bootstrap_seed)
    samples = np.empty((options.bootstrap_samples, 3, 6, 6), dtype=float)
    for sample_index in range(options.bootstrap_samples):
        resampled: dict[int, list[_OddCase]] = {}
        for dof_index, datasets in groups.items():
            resampled[dof_index] = []
            for item in datasets:
                cycles = np.unique(item.cycle_id)
                selected_cycles = rng.choice(cycles, size=cycles.size, replace=True)
                row_indices = np.concatenate([np.flatnonzero(item.cycle_id == cycle) for cycle in selected_cycles])
                resampled[dof_index].append(
                    _OddCase(
                        item.case_name,
                        item.dof_index,
                        item.design[row_indices],
                        item.target[row_indices],
                        item.even_target[row_indices],
                        item.cycle_id[row_indices],
                        (
                            None
                            if item.attitude_feature is None
                            else item.attitude_feature[row_indices]
                        ),
                    )
                )
        matrices = _fit_odd_groups(resampled, 3)[:3]
        samples[sample_index] = np.stack(matrices)
    lower, median, upper = np.percentile(samples, (2.5, 50.0, 97.5), axis=0)
    result: dict[str, Any] = {
        "method": "stratified bootstrap by complete cycle within case",
        "samples": options.bootstrap_samples,
        "seed": options.bootstrap_seed,
    }
    for index, name in enumerate(("added_mass_raw", "linear_damping", "quadratic_damping")):
        result[name] = {
            "lower_2p5": lower[index].tolist(),
            "median": median[index].tolist(),
            "upper_97p5": upper[index].tolist(),
        }
    return result


def damping_dissipated_power(
    nu: np.ndarray,
    linear_damping: np.ndarray,
    quadratic_damping: np.ndarray,
) -> np.ndarray:
    velocity = np.asarray(nu, dtype=float)
    linear_load = velocity @ np.asarray(linear_damping, dtype=float).T
    quadratic_load = (np.abs(velocity) * velocity) @ np.asarray(quadratic_damping, dtype=float).T
    return np.sum(velocity * (linear_load + quadratic_load), axis=1)


def _passivity_diagnostics(
    cases: Sequence[CaseData],
    linear: np.ndarray,
    quadratic: np.ndarray,
    options: FitOptions,
) -> dict[str, Any]:
    symmetric_linear = 0.5 * (linear + linear.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_linear)
    observed = np.concatenate([case.nu[_analysis_window(case)] for case in cases], axis=0)
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
        mask = _analysis_window(case)
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


def fit_case_data(cases: Sequence[CaseData], options: FitOptions | None = None) -> HydroFitResult:
    """Fit all 36 entries of each requested matrix from loaded cases."""

    if not cases:
        raise ValueError("At least one case is required")
    fit_options = options or FitOptions()
    odd_cases = [
        _odd_project(
            case,
            fit_options.phase_samples_per_cycle,
            include_rotation_attitude_term=(
                fit_options.include_rotation_attitude_term
                and (case.motion.dof != "p" or fit_options.include_roll_attitude_term)
            ),
        )
        for case in cases
    ]
    groups: dict[int, list[_OddCase]] = {}
    for item in odd_cases:
        groups.setdefault(item.dof_index, []).append(item)
    added_raw, linear, quadratic, fit_diagnostics = _fit_odd_groups(
        groups,
        fit_options.minimum_samples_per_dof,
    )
    if fit_options.diagonal_only:
        mask = np.eye(6, dtype=bool)
        added_raw = np.where(mask, added_raw, 0.0)
        linear = np.where(mask, linear, 0.0)
        quadratic = np.where(mask, quadratic, 0.0)
        matrix_structure = {
            "name": "diagonal",
            "assumption": "user-selected model reduction",
            "allowed_mask": mask.tolist(),
        }
    elif fit_options.port_starboard_symmetry:
        # Reflection in the body x-z plane.  In FLU, polar components
        # [u,v,w] and [X,Y,Z] have parity [+,-,+], while axial components
        # [p,q,r] and [K,M,N] have parity [-,+,-].  A coefficient can be
        # nonzero only when its wrench row and velocity column have equal
        # parity, yielding the two blocks [u,w,q] and [v,p,r].
        parity = np.asarray((1, -1, 1, -1, 1, -1), dtype=int)
        mask = parity[:, None] == parity[None, :]
        added_raw = np.where(mask, added_raw, 0.0)
        linear = np.where(mask, linear, 0.0)
        quadratic = np.where(mask, quadratic, 0.0)
        matrix_structure = {
            "name": "port_starboard_reflection_symmetric",
            "frame": "body FLU",
            "generalized_parity": parity.tolist(),
            "even_block": ["u", "w", "q"],
            "odd_block": ["v", "p", "r"],
            "allowed_mask": mask.tolist(),
        }
    else:
        mask = np.ones((6, 6), dtype=bool)
        matrix_structure = {
            "name": "full",
            "allowed_mask": mask.tolist(),
        }
    if fit_options.project_added_mass_psd:
        added_mass, projection = project_symmetric_psd(
            added_raw,
            fit_options.min_added_mass_eigenvalue,
        )
    else:
        added_mass = added_raw.copy()
        projection = {
            "enabled": False,
            "raw_asymmetry_frobenius": float(np.linalg.norm(added_raw - added_raw.T)),
        }
    # Numerical projection operates on the whole matrix.  Reapply the exact
    # structural mask so a future nonzero eigenvalue floor cannot populate
    # reflection-forbidden entries through roundoff or eigenvector mixing.
    added_mass = np.where(mask, added_mass, 0.0)
    raw_intercepts = [_raw_intercept_diagnostic(case) for case in cases]
    diagnostics = {
        "matrix_structure": matrix_structure,
        "fit_by_excited_dof": fit_diagnostics,
        "cycle_convergence_by_case": [
            _cycle_convergence_diagnostic(item) for item in odd_cases
        ],
        "raw_intercept_fits": raw_intercepts,
        "full_model_case_fits": _full_model_diagnostics(
            cases,
            added_mass,
            linear,
            quadratic,
        ),
        "added_mass_projection": projection,
        "passivity": _passivity_diagnostics(cases, linear, quadratic, fit_options),
    }
    case_summaries = [
        {
            "case_name": case.motion.case_name,
            "case_dir": case.case_dir,
            "dof": case.motion.dof,
            "amplitude_si": case.motion.amplitude_si,
            "omega_rad_s": case.motion.omega_rad_s,
            "settle_cycles": case.motion.settle_cycles,
            "sample_cycles": case.motion.sample_cycles,
            "force_files": list(case.force_series.source_files),
        }
        for case in cases
    ]
    return HydroFitResult(
        added_raw,
        added_mass,
        linear,
        quadratic,
        diagnostics,
        _bootstrap(groups, fit_options),
        case_summaries,
        fit_options,
    )


def _load_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Analysis config must contain a JSON object")
    return value


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wrench/dof", *DOF_NAMES))
        for name, row in zip(WRENCH_NAMES, matrix):
            writer.writerow((name, *(f"{float(value):.17g}" for value in row)))


def write_fit_outputs(result: HydroFitResult, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": destination / "hydrodynamic_fit.json",
        "config_updates": destination / "config_updates.json",
        "added_mass": destination / "added_mass.csv",
        "added_mass_raw": destination / "added_mass_raw.csv",
        "linear_damping": destination / "linear_damping.csv",
        "quadratic_damping": destination / "quadratic_damping.csv",
    }
    with paths["report"].open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with paths["config_updates"].open("w", encoding="utf-8") as stream:
        json.dump(result.config_updates(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _write_matrix_csv(paths["added_mass"], result.added_mass)
    _write_matrix_csv(paths["added_mass_raw"], result.added_mass_raw)
    _write_matrix_csv(paths["linear_damping"], result.linear_damping)
    _write_matrix_csv(paths["quadratic_damping"], result.quadratic_damping)
    return {name: str(path) for name, path in paths.items()}


def analyze_cases(
    case_dirs: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
) -> HydroFitResult:
    """Load generated cases, fit matrices and optionally write result files."""

    config_data = _load_config(config)
    analysis_config = config_data.get("analysis", config_data)
    options = FitOptions.from_mapping(analysis_config)
    overrides = {
        key: analysis_config[key]
        for key in ("settle_cycles", "sample_cycles")
        if key in analysis_config
    }
    cases: list[CaseData] = []
    for path in case_dirs:
        root = Path(path)
        with (root / "motion.json").open("r", encoding="utf-8") as stream:
            motion_metadata = json.load(stream)
        kind = str(motion_metadata.get("motion_kind", motion_metadata.get("kind", ""))).lower()
        dof = motion_metadata.get("dof")
        if (
            kind in {"baseline", "rest", "static"}
            or dof is None
            or motion_metadata.get("include_in_fit") is False
        ):
            continue
        cases.append(load_case_data(root, config_overrides=overrides))
    if not cases:
        raise ValueError("No oscillatory single-DOF cases were supplied (baseline cases are skipped)")
    result = fit_case_data(cases, options)
    if output_dir is not None:
        write_fit_outputs(result, output_dir)
    return result

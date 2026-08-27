"""Full-response estimators for the 24-case open-water CFD campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .cycles import OddProjectedCase
from .motion import DOF_NAMES, WRENCH_NAMES, SteadyCaseData
from .regression import scaled_lstsq
from .types import FitOptions


def _nrmse(target: np.ndarray, residual: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.square(target))))
    if scale <= np.finfo(float).tiny:
        return math.inf
    return float(np.sqrt(np.mean(np.square(residual))) / scale)


def _cross_axis_fraction(
    values: np.ndarray,
    excitation: int,
    reference_length_m: float,
) -> float:
    """Compare cross loads after converting moments to force-equivalent units."""

    if not math.isfinite(reference_length_m) or reference_length_m <= 0.0:
        raise ValueError("reference_length_m must be positive and finite")
    force_equivalent = np.asarray(values, dtype=float).copy()
    force_equivalent[:, 3:] /= reference_length_m
    rms = np.sqrt(np.mean(np.square(force_equivalent), axis=0))
    scale = max(float(rms[excitation]), np.finfo(float).tiny)
    return max(
        (float(value / scale) for index, value in enumerate(rms) if index != excitation),
        default=0.0,
    )


def _time_weighted_mean(
    time_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    stop_s: float,
) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    data = np.asarray(values, dtype=float)
    if time.ndim != 1 or data.shape != (time.size, 6) or np.any(np.diff(time) <= 0.0):
        raise ValueError("Steady wrench history must have increasing time and shape (N,6)")
    if time[0] > start_s or time[-1] < stop_s or stop_s <= start_s:
        raise ValueError(f"Steady history does not cover [{start_s}, {stop_s}]")
    inside = (time > start_s) & (time < stop_s)
    sample_time = np.concatenate(([start_s], time[inside], [stop_s]))
    sample_values = np.column_stack(
        [np.interp(sample_time, time, data[:, index]) for index in range(6)]
    )
    return np.trapz(sample_values, sample_time, axis=0) / (stop_s - start_s)


@dataclass(frozen=True)
class SteadySummary:
    case_name: str
    dof_index: int
    velocity: float
    mean_wrench: np.ndarray
    window_means: np.ndarray


def summarize_steady_cases(cases: Sequence[SteadyCaseData]) -> list[SteadySummary]:
    result: list[SteadySummary] = []
    for case in cases:
        edges = np.linspace(case.settle_end_s, case.end_time_s, 4)
        windows = np.stack(
            [
                _time_weighted_mean(
                    case.time_s, case.wrench_body, first, second
                )
                for first, second in zip(edges[:-1], edges[1:])
            ]
        )
        result.append(
            SteadySummary(
                case.case_name,
                case.dof_index,
                float(case.body_velocity_b_m_s[case.dof_index]),
                _time_weighted_mean(
                    case.time_s,
                    case.wrench_body,
                    case.settle_end_s,
                    case.end_time_s,
                ),
                windows,
            )
        )
    return result


def _steady_pairs(
    summaries: Sequence[SteadySummary], excitation: int
) -> list[tuple[SteadySummary, SteadySummary]]:
    selected = [item for item in summaries if item.dof_index == excitation]
    positive = sorted(
        (item for item in selected if item.velocity > 0.0),
        key=lambda item: item.velocity,
    )
    negative = [item for item in selected if item.velocity < 0.0]
    if len(positive) != 2 or len(negative) != 2:
        raise ValueError(
            f"{DOF_NAMES[excitation]} needs two positive and two negative steady cases"
        )
    pairs: list[tuple[SteadySummary, SteadySummary]] = []
    for item in positive:
        matches = [
            candidate
            for candidate in negative
            if math.isclose(
                item.velocity, -candidate.velocity, rel_tol=1.0e-12, abs_tol=1.0e-12
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"{DOF_NAMES[excitation]} has unmatched signed speeds")
        pairs.append((item, matches[0]))
    return pairs


def fit_translational_damping(
    summaries: Sequence[SteadySummary],
    reference_length_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    linear = np.zeros((6, 6))
    quadratic = np.zeros((6, 6))
    diagnostics: dict[str, Any] = {}
    for excitation in range(3):
        pairs = _steady_pairs(summaries, excitation)
        speeds = np.asarray([positive.velocity for positive, _ in pairs])
        positive = np.stack([item.mean_wrench for item, _ in pairs])
        negative = np.stack([item.mean_wrench for _, item in pairs])
        odd = 0.5 * (positive - negative)
        design = np.column_stack((-speeds, -np.square(speeds)))
        # A single-axis excitation identifies one complete matrix column: every
        # force/moment channel is a response to the same prescribed velocity.
        coefficients, fit = scaled_lstsq(design, odd)
        if fit["rank"] != 2:
            raise ValueError(f"{DOF_NAMES[excitation]} steady fit is rank deficient")
        prediction = design @ coefficients
        linear[:, excitation] = coefficients[0]
        quadratic[:, excitation] = coefficients[1]
        selected = [
            item for item in summaries if item.dof_index == excitation
        ]
        window_changes = []
        for item in selected:
            previous = float(item.window_means[-2, excitation])
            latest = float(item.window_means[-1, excitation])
            scale = max(abs(previous), abs(latest), np.finfo(float).tiny)
            window_changes.append(abs(latest - previous) / scale)
        diagnostics[DOF_NAMES[excitation]] = {
            "case_pairs": [
                {
                    "speed_m_s": positive_case.velocity,
                    "positive": positive_case.case_name,
                    "negative": negative_case.case_name,
                }
                for positive_case, negative_case in pairs
            ],
            "linear": float(coefficients[0, excitation]),
            "quadratic": float(coefficients[1, excitation]),
            "coefficients_by_wrench": {
                name: {
                    "linear": float(coefficients[0, index]),
                    "quadratic": float(coefficients[1, index]),
                }
                for index, name in enumerate(WRENCH_NAMES)
            },
            "nrmse": _nrmse(
                odd[:, excitation],
                odd[:, excitation] - prediction[:, excitation],
            ),
            "condition_number_scaled": fit["condition_number_scaled"],
            "maximum_steady_window_load_change": max(window_changes),
            "maximum_cross_axis_load_fraction": _cross_axis_fraction(
                odd, excitation, reference_length_m
            ),
        }
    return linear, quadratic, diagnostics


def fit_low_amplitude_added_mass(
    cases: Sequence[OddProjectedCase],
    reference_length_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.zeros((6, 6))
    diagnostics: dict[str, Any] = {}
    for excitation in range(6):
        selected = [item for item in cases if item.dof_index == excitation]
        if len(selected) != 1:
            raise ValueError(
                f"{DOF_NAMES[excitation]} needs exactly one added-mass case"
            )
        item = selected[0]
        target = item.target
        coefficients, fit = scaled_lstsq(item.design, target)
        if fit["rank"] != 3:
            raise ValueError(
                f"{DOF_NAMES[excitation]} added-mass fit is rank deficient"
            )
        prediction = item.design @ coefficients
        matrix[:, excitation] = coefficients[0]
        diagnostics[DOF_NAMES[excitation]] = {
            "case_name": item.case_name,
            "added_mass": float(coefficients[0, excitation]),
            "linear_nuisance": float(coefficients[1, excitation]),
            "quadratic_nuisance": float(coefficients[2, excitation]),
            "coefficients_by_wrench": {
                name: {
                    "added_mass": float(coefficients[0, index]),
                    "linear_nuisance": float(coefficients[1, index]),
                    "quadratic_nuisance": float(coefficients[2, index]),
                }
                for index, name in enumerate(WRENCH_NAMES)
            },
            "nrmse": _nrmse(
                target[:, excitation],
                target[:, excitation] - prediction[:, excitation],
            ),
            "condition_number_scaled": fit["condition_number_scaled"],
            "maximum_cross_axis_load_fraction": _cross_axis_fraction(
                item.target, excitation, reference_length_m
            ),
        }
    return matrix, diagnostics


def _last_cycle_change(item: OddProjectedCase, excitation: int) -> float:
    cycles = np.unique(item.cycle_id)
    if cycles.size < 2:
        raise ValueError(f"{item.case_name}: at least two sample cycles are required")
    previous = item.target[item.cycle_id == cycles[-2], excitation]
    latest = item.target[item.cycle_id == cycles[-1], excitation]
    if previous.shape != latest.shape:
        raise ValueError(f"{item.case_name}: final cycle grids differ")
    scale = max(
        float(np.sqrt(np.mean(np.square(np.concatenate((previous, latest)))))),
        np.finfo(float).tiny,
    )
    return float(np.sqrt(np.mean(np.square(latest - previous))) / scale)


def fit_rotational_coefficients(
    cases: Sequence[OddProjectedCase],
    reference_length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    added_mass = np.zeros((6, 6))
    linear = np.zeros((6, 6))
    quadratic = np.zeros((6, 6))
    diagnostics: dict[str, Any] = {}
    for excitation in range(3, 6):
        selected = [item for item in cases if item.dof_index == excitation]
        if len(selected) != 2:
            raise ValueError(
                f"{DOF_NAMES[excitation]} needs two rotational damping cases"
            )
        design = np.concatenate([item.design for item in selected])
        target = np.concatenate([item.target for item in selected])
        coefficients, fit = scaled_lstsq(design, target)
        if fit["rank"] != 3:
            raise ValueError(
                f"{DOF_NAMES[excitation]} rotational coefficient fit is rank deficient"
            )
        prediction = design @ coefficients
        residual = target - prediction
        damping_target = target[:, excitation] - design[:, 0] * coefficients[0, excitation]
        case_added_mass: dict[str, float] = {}
        for item in selected:
            case_coefficients, case_fit = scaled_lstsq(
                item.design, item.target[:, excitation : excitation + 1]
            )
            if case_fit["rank"] != 3:
                raise ValueError(
                    f"{item.case_name} rotational coefficient fit is rank deficient"
                )
            case_added_mass[item.case_name] = float(case_coefficients[0, 0])
        added_mass[:, excitation] = coefficients[0]
        linear[:, excitation] = coefficients[1]
        quadratic[:, excitation] = coefficients[2]
        diagnostics[DOF_NAMES[excitation]] = {
            "case_names": sorted(item.case_name for item in selected),
            "added_mass": float(coefficients[0, excitation]),
            "linear": float(coefficients[1, excitation]),
            "quadratic": float(coefficients[2, excitation]),
            "coefficients_by_wrench": {
                name: {
                    "added_mass": float(coefficients[0, index]),
                    "linear": float(coefficients[1, index]),
                    "quadratic": float(coefficients[2, index]),
                }
                for index, name in enumerate(WRENCH_NAMES)
            },
            "case_added_mass": case_added_mass,
            "nrmse": _nrmse(damping_target, residual[:, excitation]),
            "total_load_nrmse": _nrmse(
                target[:, excitation], residual[:, excitation]
            ),
            "nrmse_normalization": "RMS load after subtracting jointly fitted added mass",
            "condition_number_scaled": fit["condition_number_scaled"],
            "maximum_last_cycle_load_change": max(
                _last_cycle_change(item, excitation) for item in selected
            ),
            "maximum_cross_axis_load_fraction": max(
                _cross_axis_fraction(
                    item.target, excitation, reference_length_m
                )
                for item in selected
            ),
        }
    return added_mass, linear, quadratic, diagnostics

"""Complete-cycle selection and odd/even half-period projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion import CaseData

@dataclass(frozen=True)
class OddProjectedCase:
    case_name: str
    dof_index: int
    design: np.ndarray
    target: np.ndarray
    even_target: np.ndarray
    cycle_id: np.ndarray
    attitude_feature: np.ndarray | None = None


def analysis_window(case: CaseData) -> np.ndarray:
    period = case.motion.period_s
    start = case.motion.settle_cycles * period
    stop = np.inf
    if case.motion.sample_cycles is not None:
        stop = start + case.motion.sample_cycles * period
    tolerance = 1.0e-10 * max(1.0, period)
    return (case.time_s >= start - tolerance) & (case.time_s <= stop + tolerance)


def _validate_phase_samples(value: int) -> None:
    if type(value) is not int or value < 8 or value % 2:
        raise ValueError("phase_samples_per_cycle must be an even integer of at least 8")


def _requested_cycle_count(case: CaseData, window_time: np.ndarray) -> tuple[int, bool]:
    if case.motion.sample_cycles is not None:
        return max(0, int(np.floor(case.motion.sample_cycles + 1.0e-10))), True
    tolerance = 1.0e-10 * max(1.0, case.motion.period_s)
    start = case.motion.settle_cycles * case.motion.period_s
    return max(0, int(np.floor((window_time[-1] - start + tolerance) / case.motion.period_s))), False


def _cycle_gap_failure(
    local_time: np.ndarray,
    period: float,
    phase_samples_per_cycle: int,
    cycle: int,
) -> str | None:
    if local_time.size < 4:
        return None
    local_gaps = np.diff(local_time)
    nominal_delta = float(np.median(local_gaps))
    maximum_gap = float(np.max(local_gaps))
    phase_delta = period / phase_samples_per_cycle
    maximum_allowed_gap = max(2.0 * phase_delta, 4.0 * nominal_delta)
    if maximum_gap <= maximum_allowed_gap * (1.0 + 1.0e-10):
        return None
    return (
        f"cycle {cycle}: max {maximum_gap:.9g}s > limit {maximum_allowed_gap:.9g}s "
        f"(median {nominal_delta:.9g}s, phase {phase_delta:.9g}s)"
    )


def _complete_cycle_pairs(
    case: CaseData,
    phase_samples_per_cycle: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    window_time = case.time_s[analysis_window(case)]
    if window_time.size < 4:
        raise ValueError(f"{case.motion.case_name}: too few samples after settling")
    period = case.motion.period_s
    half_period = 0.5 * period
    start = case.motion.settle_cycles * period
    tolerance = 1.0e-10 * max(1.0, period)
    cycle_count, require_all_cycles = _requested_cycle_count(case, window_time)
    half_samples = phase_samples_per_cycle // 2
    first_half_phase = np.arange(half_samples, dtype=float) * (half_period / half_samples)
    first_times: list[np.ndarray] = []
    complete_cycles: list[int] = []
    incomplete_cycles: list[int] = []
    gap_failures: list[str] = []
    for cycle in range(cycle_count):
        cycle_start = start + cycle * period
        candidate = cycle_start + first_half_phase
        local_time = case.time_s[
            (case.time_s >= cycle_start - tolerance)
            & (case.time_s <= cycle_start + period + tolerance)
        ]
        gap_failure = _cycle_gap_failure(local_time, period, phase_samples_per_cycle, cycle)
        complete = (
            gap_failure is None
            and local_time.size >= 4
            and case.time_s[0] <= candidate[0] + tolerance
            and case.time_s[-1] >= candidate[-1] + half_period - tolerance
        )
        if not complete:
            incomplete_cycles.append(cycle)
            if gap_failure is not None:
                gap_failures.append(gap_failure)
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
            f"{case.motion.case_name}: requested sample cycle(s) are incomplete: {incomplete_cycles}"
        )
    if not first_times:
        raise ValueError(f"{case.motion.case_name}: no complete cycles after settling")
    return (
        np.concatenate(first_times),
        np.repeat(np.asarray(complete_cycles, dtype=int), half_samples),
        half_period,
    )


def _interpolated_wrench_components(
    case: CaseData,
    pair_time: np.ndarray,
    half_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.column_stack(
        [np.interp(pair_time, case.time_s, case.wrench_body[:, index]) for index in range(6)]
    )
    paired = np.column_stack(
        [
            np.interp(pair_time + half_period, case.time_s, case.wrench_body[:, index])
            for index in range(6)
        ]
    )
    return 0.5 * (first - paired), 0.5 * (first + paired)


def _signed_square_velocity(case: CaseData, scalar_nu: np.ndarray) -> np.ndarray:
    if case.motion.dof_index != 0:
        return np.abs(scalar_nu) * scalar_nu
    background_fluid_dof = float(case.motion.background_fluid_velocity_body_m_s[0])
    if abs(background_fluid_dof) == 0.0:
        return np.abs(scalar_nu) * scalar_nu
    background_relative_dof = -background_fluid_dof
    plus = background_relative_dof + scalar_nu
    minus = background_relative_dof - scalar_nu
    return 0.5 * (np.abs(plus) * plus - np.abs(minus) * minus)


def odd_project(
    case: CaseData,
    phase_samples_per_cycle: int = 256,
    *,
    include_rotation_attitude_term: bool = True,
) -> OddProjectedCase:
    """Odd-project complete cycles on a fixed phase grid.

    Adaptive time stepping changes the raw sample density over phase.  Using
    those rows directly would therefore turn solver time-step control into a
    regression weight.  Every complete cycle is instead represented by the
    same phase grid before half-period pairing.
    """

    _validate_phase_samples(phase_samples_per_cycle)
    pair_time, pair_cycle, half_period = _complete_cycle_pairs(case, phase_samples_per_cycle)
    odd, even = _interpolated_wrench_components(case, pair_time, half_period)
    scalar_eta, scalar_nu, scalar_nudot = case.motion.kinematics(pair_time)
    signed_square = _signed_square_velocity(case, scalar_nu)
    design = np.column_stack((-scalar_nudot, -scalar_nu, -signed_square))
    attitude_feature = (
        -scalar_eta
        if include_rotation_attitude_term and case.motion.motion_kind == "rotation"
        else None
    )
    finite = np.all(np.isfinite(design), axis=1) & np.all(np.isfinite(odd), axis=1)
    if np.count_nonzero(finite) < 3:
        raise ValueError(f"{case.motion.case_name}: too few finite odd-projected samples")
    return OddProjectedCase(
        case.motion.case_name,
        case.motion.dof_index,
        design[finite],
        odd[finite],
        even[finite],
        pair_cycle[finite],
        None if attitude_feature is None else attitude_feature[finite],
    )

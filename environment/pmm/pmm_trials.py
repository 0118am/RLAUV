"""Raw PMM trial loading, reconstruction, and harmonic projection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pmm_common import *
from .pmm_config import SixDofConfig, TrialPlan, _finite
from .pmm_kinematics import (
    FourierFit, fit_fourier, motion_origin_to_com_kinematics,
    rigid_body_wrench_at_com, rotate_twist_H_to_B, rotate_wrench_H_to_B,
    translate_wrench_origin_to_com,
)


def read_raw_pair(plan: TrialPlan) -> tuple[np.ndarray, np.ndarray]:
    gather = np.loadtxt(plan.gather_path)
    sensor = np.loadtxt(plan.sensor_path, delimiter=",")
    if gather.ndim != 2 or gather.shape[1] != 16:
        raise ValueError(f"{plan.gather_path}: expected 16 columns, got {gather.shape}")
    if sensor.ndim != 2 or sensor.shape[1] != 6:
        raise ValueError(f"{plan.sensor_path}: expected 6 columns, got {sensor.shape}")
    if not np.isfinite(gather).all() or not np.isfinite(sensor).all():
        raise ValueError(f"non-finite value in {plan.gather_path} or {plan.sensor_path}")
    return gather, sensor


def _fourier_sse(t: np.ndarray, y: np.ndarray, frequency_hz: float, harmonics: int) -> float:
    fit = fit_fourier(t, y, frequency_hz, harmonics)
    prediction = fit.evaluate(t)[0]
    residual = y - prediction
    return float(residual @ residual)


def estimate_trial_frequency(
    t: np.ndarray,
    motion: np.ndarray,
    nominal_frequency_hz: float,
    config: SixDofConfig,
) -> float:
    settings = config.frequency_estimation
    low = nominal_frequency_hz * float(settings["search_lower_fraction"])
    high = nominal_frequency_hz * float(settings["search_upper_fraction"])
    points = int(settings["grid_points"])
    harmonics = int(config.sampling["kinematic_harmonics"])
    best = nominal_frequency_hz
    for _ in range(int(settings["refinement_rounds"])):
        grid = np.linspace(low, high, points)
        scores = np.asarray([_fourier_sse(t, motion, float(f), harmonics) for f in grid])
        index = int(np.argmin(scores))
        best = float(grid[index])
        step = float(grid[1] - grid[0])
        low = max(nominal_frequency_hz * float(settings["search_lower_fraction"]), best - step)
        high = min(nominal_frequency_hz * float(settings["search_upper_fraction"]), best + step)
        if high <= low:
            break
    return best


def _block_average(sensor: np.ndarray, block: int, sensor_hz: float) -> tuple[np.ndarray, np.ndarray]:
    blocks = len(sensor) // block
    if blocks < 1:
        raise ValueError("sensor record is shorter than one averaging block")
    values = sensor[: blocks * block].reshape(blocks, block, sensor.shape[1]).mean(axis=1)
    centres = np.arange(blocks, dtype=float) * block + (block - 1.0) / 2.0
    return centres / sensor_hz, values


@dataclass
class SixDofTrial:
    plan: TrialPlan
    frequency_hz: float
    time: np.ndarray
    X: np.ndarray
    target: np.ndarray
    measured: np.ndarray
    u: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    diagnostics: dict[str, float]


def _body_to_H_rotation(plan: TrialPlan, config: SixDofConfig) -> np.ndarray:
    key = (
        "body_to_apparatus_rotation_vertical"
        if plan.timing_group == "vertical"
        else "body_to_apparatus_rotation_horizontal"
    )
    return np.asarray(config.apparatus[key], dtype=float)


def _fit_gather_motion(
    plan: TrialPlan,
    config: SixDofConfig,
    gather: np.ndarray,
) -> tuple[float, FourierFit, FourierFit, FourierFit | None]:
    sampling = config.sampling
    time = np.arange(len(gather), dtype=float) / float(sampling["gather_hz"])
    active = (time >= float(sampling["active_start_s"])) & (
        time <= float(sampling["active_end_s"])
    )
    if int(active.sum()) < 100:
        raise ValueError(f"{plan.gather_path}: insufficient motion samples in active interval")
    x = (gather[:, 1] + gather[:, 7]) / float(sampling["position_counts_per_m"])
    lateral = gather[:, 4] / float(sampling["position_counts_per_m"])
    angle = np.deg2rad(
        float(config.apparatus["yaw_encoder_sign_to_H_r"])
        * (gather[:, 10] - gather[0, 10])
        / float(sampling["angle_counts_per_degree"])
    )
    frequency_motion = lateral if DOF_RAW_KIND[plan.dof] == "sway" else angle
    frequency = estimate_trial_frequency(
        time[active],
        frequency_motion[active],
        plan.nominal_frequency_hz,
        config,
    )
    harmonics = int(sampling["kinematic_harmonics"])
    fit_x = fit_fourier(time[active], x[active], frequency, harmonics)
    fit_lateral = fit_fourier(time[active], lateral[active], frequency, harmonics)
    fit_angle = (
        fit_fourier(time[active], angle[active], frequency, harmonics)
        if DOF_RAW_KIND[plan.dof] == "yaw"
        else None
    )
    return frequency, fit_x, fit_lateral, fit_angle


def _paired_sensor_series(
    plan: TrialPlan,
    config: SixDofConfig,
    sensor: np.ndarray,
    shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sampling = config.sampling
    time, averaged = _block_average(
        sensor,
        int(sampling["sensor_decimation"]),
        float(sampling["sensor_hz"]),
    )
    actual_shift = config.shift_ms(plan.timing_group) if shift_ms is None else _finite(shift_ms, "shift_ms")
    active = (time >= float(sampling["active_start_s"])) & (
        time <= float(sampling["active_end_s"])
    )
    time, averaged = time[active], averaged[active]
    if len(time) < 100:
        raise ValueError(f"{plan.sensor_path}: insufficient paired samples in active interval")
    return time, averaged, time + actual_shift / 1000.0, actual_shift


def _evaluate_body_kinematics(
    plan: TrialPlan,
    config: SixDofConfig,
    time: np.ndarray,
    fit_x: FourierFit,
    fit_lateral: FourierFit,
    fit_angle: FourierFit | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    _, dx, ddx = fit_x.evaluate(time)
    _, dy, ddy = fit_lateral.evaluate(time)
    if fit_angle is None:
        psi, rate, rate_dot = np.zeros_like(time), np.zeros_like(time), np.zeros_like(time)
        angle_r2, angle_amplitude = 1.0, 0.0
    else:
        psi, rate, rate_dot = fit_angle.evaluate(time)
        angle_r2, angle_amplitude = fit_angle.r2, 0.5 * float(np.ptp(psi))
    cosine, sine = np.cos(psi), np.sin(psi)
    linear_velocity_H = np.column_stack(
        (cosine * dx + sine * dy, -sine * dx + cosine * dy, np.zeros_like(time))
    )
    linear_derivative_H = np.column_stack(
        (
            -sine * rate * dx + cosine * ddx + cosine * rate * dy + sine * ddy,
            -cosine * rate * dx - sine * ddx - sine * rate * dy + cosine * ddy,
            np.zeros_like(time),
        )
    )
    angular_velocity_H = np.column_stack((np.zeros_like(time), np.zeros_like(time), rate))
    angular_acceleration_H = np.column_stack((np.zeros_like(time), np.zeros_like(time), rate_dot))
    body_to_H = _body_to_H_rotation(plan, config)
    linear_velocity_origin_B, angular_velocity_B = rotate_twist_H_to_B(
        linear_velocity_H,
        angular_velocity_H,
        body_to_H,
    )
    linear_derivative_origin_B, angular_acceleration_B = rotate_twist_H_to_B(
        linear_derivative_H,
        angular_acceleration_H,
        body_to_H,
    )
    com_from_origin_B = np.asarray(config.model["com_from_motion_origin_flu_m"], dtype=float)
    linear_velocity_B, linear_derivative_B = motion_origin_to_com_kinematics(
        linear_velocity_origin_B,
        linear_derivative_origin_B,
        angular_velocity_B,
        angular_acceleration_B,
        com_from_origin_B,
    )
    return (
        linear_velocity_B,
        linear_derivative_B,
        angular_velocity_B,
        angular_acceleration_B,
        angle_r2,
        angle_amplitude,
        body_to_H,
        com_from_origin_B,
    )


def _regression_data(
    plan: TrialPlan,
    config: SixDofConfig,
    sensor: np.ndarray,
    kinematics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    body_to_H: np.ndarray,
    com_from_origin_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    linear_velocity_B, linear_derivative_B, angular_velocity_B, angular_acceleration_B = kinematics
    dof_index = DOF_INDEX[plan.dof]
    q = linear_velocity_B[:, dof_index] if dof_index < 3 else angular_velocity_B[:, dof_index - 3]
    qdot = (
        linear_derivative_B[:, dof_index]
        if dof_index < 3
        else angular_acceleration_B[:, dof_index - 3]
    )
    sensor_to_H = np.asarray(
        config.apparatus[f"sensor_to_H_wrench_matrix_{plan.timing_group}"],
        dtype=float,
    )
    wrench_B_at_origin = rotate_wrench_H_to_B(sensor @ sensor_to_H.T, body_to_H)
    wrench_B_at_com = translate_wrench_origin_to_com(wrench_B_at_origin, com_from_origin_B)
    rigid_wrench = rigid_body_wrench_at_com(
        float(config.model["mass_kg"]),
        np.asarray(config.model["inertia_at_com_flu_kg_m2"], dtype=float),
        linear_velocity_B,
        linear_derivative_B,
        angular_velocity_B,
        angular_acceleration_B,
    )
    measured = wrench_B_at_com[:, dof_index]
    target = measured - rigid_wrench[:, dof_index]
    u = linear_velocity_B[:, 0]
    design = np.column_stack([u * q, q * np.abs(q), qdot])
    return design, target, measured, u, q, qdot, wrench_B_at_com


def build_trial(
    plan: TrialPlan,
    config: SixDofConfig,
    *,
    shift_ms: float | None = None,
) -> SixDofTrial:
    gather, sensor = read_raw_pair(plan)
    frequency, fit_x, fit_lateral, fit_angle = _fit_gather_motion(plan, config, gather)
    ts, sensor_avg, motion_time, actual_shift_ms = _paired_sensor_series(
        plan,
        config,
        sensor,
        shift_ms,
    )
    body_state = _evaluate_body_kinematics(
        plan,
        config,
        motion_time,
        fit_x,
        fit_lateral,
        fit_angle,
    )
    linear_velocity_B, linear_derivative_B, angular_velocity_B, angular_acceleration_B = body_state[:4]
    angle_r2, angle_amplitude, body_to_H, com_from_origin_B = body_state[4:]
    X, target, measured, u, q, qdot, wrench_B_at_com = _regression_data(
        plan,
        config,
        sensor_avg,
        (linear_velocity_B, linear_derivative_B, angular_velocity_B, angular_acceleration_B),
        body_to_H,
        com_from_origin_B,
    )
    diagnostics = {
        "gather_rows": float(len(gather)),
        "sensor_rows": float(len(sensor)),
        "paired_rows": float(len(ts)),
        "x_fit_r2": fit_x.r2,
        "motion_fit_r2": fit_lateral.r2 if fit_angle is None else fit_angle.r2,
        "angle_fit_r2": angle_r2,
        "u_mean_m_s": float(np.mean(u)),
        "q_amplitude": 0.5 * float(np.ptp(q)),
        "angle_amplitude_deg": math.degrees(angle_amplitude),
        "surge_force_mean_n": float(np.mean(wrench_B_at_com[:, 0])),
        "surge_force_oscillatory_rms_n": float(np.std(wrench_B_at_com[:, 0])),
        "frequency_ratio_to_nominal": frequency / plan.nominal_frequency_hz,
        "condition_raw": float(np.linalg.cond(X / np.maximum(np.std(X, axis=0), 1e-15))),
        "sensor_time_shift_ms": actual_shift_ms,
        "body_to_H_det": float(np.linalg.det(body_to_H)),
        "com_translation_m": float(np.linalg.norm(com_from_origin_B)),
    }
    return SixDofTrial(plan, frequency, ts, X, target, measured, u, q, qdot, diagnostics)


def residualize_trial(trial: SixDofTrial) -> tuple[np.ndarray, np.ndarray]:
    tc = trial.time - float(np.mean(trial.time))
    nuisance = np.column_stack([np.ones_like(tc), tc])
    x = trial.X - nuisance @ np.linalg.lstsq(nuisance, trial.X, rcond=None)[0]
    y = trial.target - nuisance @ np.linalg.lstsq(nuisance, trial.target, rcond=None)[0]
    return x, y


def project_trial_harmonics(
    trial: SixDofTrial,
    harmonics: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project one residualized trial onto common sine/cosine harmonics.

    Regressing these harmonic coefficients, rather than all raw load samples,
    gives every trial six physical observations and rejects asynchronous sensor
    noise outside the commanded motion harmonics.
    """

    x, y = residualize_trial(trial)
    columns: list[np.ndarray] = []
    for harmonic in harmonics:
        phase = 2.0 * np.pi * int(harmonic) * trial.frequency_hz * trial.time
        columns.extend((np.sin(phase), np.cos(phase)))
    basis = np.column_stack(columns)
    x_harmonic = np.linalg.lstsq(basis, x, rcond=None)[0]
    y_harmonic = np.linalg.lstsq(basis, y, rcond=None)[0]
    return x_harmonic, y_harmonic

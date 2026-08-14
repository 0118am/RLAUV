#!/usr/bin/env python3
"""Identify frequency-resolved diagonal 6-DOF FLU models from PMM data.

The raw PMM apparatus is an FLU frame ``H`` (x forward, y left, z up).  The
horizontal records use ``R_HB=I``.  For the vertical records the user confirmed
that the vehicle body ``B`` was rolled 90 degrees about the forward axis; this
script uses the explicit canonical convention ``R_HB=Rx(+pi/2)`` and transforms
motion, wrench, COM offset and inertia with that matrix.  A diagonal derivative
is unchanged if the physical mounting used -90 degrees instead, because both
the generalized motion and its conjugate load reverse sign.

Only ``Y<-v``, ``Z<-w``, ``M<-q`` and ``N<-r`` are excited by the raw data.
The unexcited surge and roll diagonal entries are filled from the repository's
declared literature prior, so the final output is a 6x6 diagonal hybrid model.

Measured wrench is translated from the configured PMM motion origin to the
model COM before the Newton-Euler rigid-body load is subtracted.  Hydrodynamic
derivatives are estimated after projecting every trial onto its measured first
three harmonics.  Repeats at one nominal frequency are fitted together with a
Huber regression; different frequencies are never pooled.

All coefficients remain at the physical 0.562 m towing-model scale shared by
PMM and CFD.  This module performs no 0.7-to-1.0 Froude conversion.

Raw CSV files are read only.  Records which cannot cover the configured active
window are either rejected or explicitly excluded according to the checked-in
configuration; exclusions always remain visible in ``trial_diagnostics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DOFS = ("sway", "heave", "pitch", "yaw")
DOF_RESPONSE = {"sway": "Y", "heave": "Z", "pitch": "M", "yaw": "N"}
DOF_VARIABLE = {"sway": "v", "heave": "w", "pitch": "q", "yaw": "r"}
DOF_INDEX = {"sway": 1, "heave": 2, "pitch": 4, "yaw": 5}
DOF_FAMILY = {
    "sway": "pure_sway",
    "heave": "vertical_sway",
    "pitch": "vertical_yaw",
    "yaw": "pure_yaw",
}
DOF_RAW_KIND = {"sway": "sway", "heave": "sway", "pitch": "yaw", "yaw": "yaw"}
DOF_TIMING_GROUP = {"sway": "horizontal", "heave": "vertical", "pitch": "vertical", "yaw": "horizontal"}
SENSOR_COLUMNS = ("TX", "TY", "TZ", "FX", "FY", "FZ")
ROW_ORDER = ("X", "Y", "Z", "K", "M", "N")
COLUMN_ORDER = ("u", "v", "w", "p", "q", "r")

# The balance remains upright in both installations; only the vehicle is rolled
# about its forward axis.  The sensor reports the wrench exerted by the model on
# the balance, so the complete reaction wrench is negated to obtain the wrench
# on the model.  Raw columns are [TX, TY, TZ, FX, FY, FZ], while output rows are
# [X, Y, Z, K, M, N].
HORIZONTAL_SENSOR_TO_H_WRENCH = np.asarray(
    [
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)
VERTICAL_SENSOR_TO_H_WRENCH = HORIZONTAL_SENSOR_TO_H_WRENCH.copy()


@dataclass(frozen=True)
class FourierFit:
    """Constant/trend plus harmonic motion representation."""

    frequency_hz: float
    harmonics: int
    t_center: float
    coefficients: np.ndarray
    r2: float

    def evaluate(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = np.asarray(time_s, dtype=float)
        omega = 2.0 * np.pi * self.frequency_hz
        value = np.full_like(time, self.coefficients[0], dtype=float)
        velocity = np.full_like(time, self.coefficients[1], dtype=float)
        acceleration = np.zeros_like(time, dtype=float)
        value += self.coefficients[1] * (time - self.t_center)
        coefficient_index = 2
        for harmonic in range(1, self.harmonics + 1):
            sine = self.coefficients[coefficient_index]
            cosine = self.coefficients[coefficient_index + 1]
            harmonic_omega = harmonic * omega
            phase = harmonic_omega * time
            value += sine * np.sin(phase) + cosine * np.cos(phase)
            velocity += harmonic_omega * (
                sine * np.cos(phase) - cosine * np.sin(phase)
            )
            acceleration -= harmonic_omega**2 * (
                sine * np.sin(phase) + cosine * np.cos(phase)
            )
            coefficient_index += 2
        return value, velocity, acceleration


def fit_fourier(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
    harmonics: int,
) -> FourierFit:
    """Fit the motion representation used by the PMM identification."""

    time = np.asarray(time_s, dtype=float)
    response = np.asarray(values, dtype=float)
    centre = float(np.mean(time))
    columns = [np.ones_like(time), time - centre]
    omega = 2.0 * np.pi * frequency_hz
    for harmonic in range(1, harmonics + 1):
        phase = harmonic * omega * time
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    prediction = design @ coefficients
    return FourierFit(
        frequency_hz,
        harmonics,
        centre,
        coefficients,
        r2_score(response, prediction),
    )


def r2_score(values: np.ndarray, prediction: np.ndarray) -> float:
    """Return the ordinary coefficient of determination."""

    response = np.asarray(values, dtype=float)
    fitted = np.asarray(prediction, dtype=float)
    denominator = float(np.sum((response - np.mean(response)) ** 2))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.allclose(response, fitted) else float("-inf")
    return 1.0 - float(np.sum((response - fitted) ** 2)) / denominator


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def rotation_x(angle_rad: float) -> np.ndarray:
    """Return the active FLU rotation about +x, mapping body vectors to H."""

    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=float,
    )


def validate_rotation(matrix: np.ndarray, name: str) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be a proper rotation")


def skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotate_twist_H_to_B(
    linear_H: np.ndarray,
    angular_H: np.ndarray,
    body_to_H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate vector histories from apparatus FLU H into body FLU B."""

    return linear_H @ body_to_H, angular_H @ body_to_H


def rotate_wrench_H_to_B(wrench_H: np.ndarray, body_to_H: np.ndarray) -> np.ndarray:
    """Rotate ``[force, moment]`` histories from apparatus H into body B."""

    force_B = wrench_H[:, :3] @ body_to_H
    moment_B = wrench_H[:, 3:] @ body_to_H
    return np.column_stack((force_B, moment_B))


def translate_wrench_origin_to_com(
    wrench_at_origin_B: np.ndarray,
    com_from_origin_B: np.ndarray,
) -> np.ndarray:
    """Translate moment from PMM origin O to COM G.

    ``r_OG`` points from the PMM motion/wrench origin to COM.  Therefore
    ``M_G = M_O - r_OG x F``.
    """

    force = wrench_at_origin_B[:, :3]
    moment_com = wrench_at_origin_B[:, 3:] - np.cross(com_from_origin_B, force)
    return np.column_stack((force, moment_com))


def motion_origin_to_com_kinematics(
    linear_velocity_origin_B: np.ndarray,
    linear_derivative_origin_B: np.ndarray,
    angular_velocity_B: np.ndarray,
    angular_acceleration_B: np.ndarray,
    com_from_origin_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move body-frame linear kinematics from PMM origin O to COM G.

    With the constant body vector ``r_OG``, ``v_G=v_O+omega x r_OG`` and
    the body-coordinate derivative is
    ``v_G_dot=v_O_dot+omega_dot x r_OG``.
    """

    velocity_com = linear_velocity_origin_B + np.cross(angular_velocity_B, com_from_origin_B)
    derivative_com = linear_derivative_origin_B + np.cross(
        angular_acceleration_B, com_from_origin_B
    )
    return velocity_com, derivative_com


def rigid_body_wrench_at_com(
    mass_kg: float,
    inertia_at_com_B: np.ndarray,
    linear_velocity_B: np.ndarray,
    linear_acceleration_B: np.ndarray,
    angular_velocity_B: np.ndarray,
    angular_acceleration_B: np.ndarray,
) -> np.ndarray:
    """Return Newton-Euler inertial wrench at COM in rotating FLU axes."""

    force = mass_kg * (
        linear_acceleration_B + np.cross(angular_velocity_B, linear_velocity_B)
    )
    angular_momentum = angular_velocity_B @ inertia_at_com_B.T
    moment = angular_acceleration_B @ inertia_at_com_B.T + np.cross(
        angular_velocity_B, angular_momentum
    )
    return np.column_stack((force, moment))


@dataclass(frozen=True)
class SixDofConfig:
    source_path: Path
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        if int(self.raw.get("schema_version", -1)) != 1:
            raise ValueError("six-DOF config schema_version must be 1")
        if self.raw.get("analysis_scope") != "six_dof_diagonal_per_frequency_hybrid":
            raise ValueError(
                "analysis_scope must be 'six_dof_diagonal_per_frequency_hybrid'"
            )
        sampling = self.sampling
        frequency = self.frequency_estimation
        model = self.model
        apparatus = self.apparatus
        timing = self.timing
        quality = self.quality
        fit = self.fit
        for name in (
            "gather_hz", "sensor_hz", "sensor_decimation", "kinematic_harmonics",
            "position_counts_per_m", "angle_counts_per_degree",
        ):
            _positive(sampling.get(name), f"sampling.{name}")
        start = _finite(sampling.get("active_start_s"), "sampling.active_start_s")
        end = _finite(sampling.get("active_end_s"), "sampling.active_end_s")
        if start < 0.0 or end <= start:
            raise ValueError("sampling active window is invalid")
        lo = _positive(frequency.get("search_lower_fraction"), "frequency_estimation.search_lower_fraction")
        hi = _positive(frequency.get("search_upper_fraction"), "frequency_estimation.search_upper_fraction")
        _positive(
            frequency.get("nominal_frequency_step_hz"),
            "frequency_estimation.nominal_frequency_step_hz",
        )
        if not lo < 1.0 < hi:
            raise ValueError("frequency search bounds must bracket the nominal frequency")
        if int(frequency.get("refinement_rounds", 0)) < 1 or int(frequency.get("grid_points", 0)) < 3:
            raise ValueError("frequency refinement requires at least one round and three grid points")
        for name in ("mass_kg", "fluid_density_kg_m3", "wet_length_m"):
            _positive(model.get(name), f"model.{name}")
        inertia = np.asarray(model.get("inertia_at_com_flu_kg_m2"), dtype=float)
        if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
            raise ValueError("model.inertia_at_com_flu_kg_m2 must be a finite 3x3 matrix")
        if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1e-12):
            raise ValueError("model.inertia_at_com_flu_kg_m2 must be symmetric")
        if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
            raise ValueError("model.inertia_at_com_flu_kg_m2 must be positive definite")
        com = np.asarray(model.get("com_from_motion_origin_flu_m"), dtype=float)
        if com.shape != (3,) or not np.isfinite(com).all():
            raise ValueError("model.com_from_motion_origin_flu_m must be a finite 3-vector")
        expected_columns = list(SENSOR_COLUMNS)
        if list(apparatus.get("sensor_column_order", ())) != expected_columns:
            raise ValueError(f"apparatus.sensor_column_order must be {expected_columns}")
        if apparatus.get("sensor_mount") != "upright_not_rolled_same_apparatus_orientation":
            raise ValueError("apparatus.sensor_mount must describe the unchanged upright installation")
        if apparatus.get("sensor_mount_status") != "upright_not_rolled_and_not_yaw_rotated_user_confirmed":
            raise ValueError("apparatus.sensor_mount_status must record the user-confirmed fixed orientation")
        if "reaction" not in str(apparatus.get("sensor_axes_frame", "")):
            raise ValueError("apparatus.sensor_axes_frame must identify the raw reaction wrench")
        if _finite(
            apparatus.get("sensor_reaction_sign_to_model_wrench"),
            "apparatus.sensor_reaction_sign_to_model_wrench",
        ) != -1.0:
            raise ValueError("sensor reaction wrench must be negated to obtain model wrench")
        if _finite(apparatus.get("yaw_encoder_sign_to_H_r"), "apparatus.yaw_encoder_sign_to_H_r") not in (-1.0, 1.0):
            raise ValueError("apparatus.yaw_encoder_sign_to_H_r must be -1 or +1")
        for group, expected_map in (
            ("horizontal", HORIZONTAL_SENSOR_TO_H_WRENCH),
            ("vertical", VERTICAL_SENSOR_TO_H_WRENCH),
        ):
            wrench_map = np.asarray(apparatus.get(f"sensor_to_H_wrench_matrix_{group}"), dtype=float)
            if wrench_map.shape != (6, 6) or not np.isfinite(wrench_map).all():
                raise ValueError(f"apparatus sensor-to-H {group} matrix must be finite 6x6")
            if not np.array_equal(wrench_map, expected_map):
                raise ValueError(f"apparatus sensor-to-H {group} matrix disagrees with the resolved installation")
        horizontal = np.asarray(apparatus.get("body_to_apparatus_rotation_horizontal"), dtype=float)
        vertical = np.asarray(apparatus.get("body_to_apparatus_rotation_vertical"), dtype=float)
        validate_rotation(horizontal, "horizontal body-to-apparatus rotation")
        validate_rotation(vertical, "vertical body-to-apparatus rotation")
        expected_vertical = rotation_x(math.pi / 2.0)
        if not np.allclose(vertical, expected_vertical, rtol=0.0, atol=1e-12):
            raise ValueError(
                "vertical body-to-apparatus rotation must be FLU Rx(+pi/2) only"
            )
        _finite(
            apparatus.get("observed_vertical_encoder_initial_offset_deg"),
            "apparatus.observed_vertical_encoder_initial_offset_deg",
        )
        if apparatus.get("observed_vertical_encoder_offset_interpretation") != "encoder_zero_offset_not_body_yaw_user_confirmed":
            raise ValueError("the observed vertical encoder offset must not be used as a body yaw rotation")
        if bool(apparatus.get("pitch_hydrostatic_restoring_in_fit")):
            raise ValueError("rolled-model pitch data must not include normal-attitude hydrostatic restoring")
        for group in ("horizontal", "vertical"):
            _finite(timing.get(f"{group}_shift_ms"), f"timing.{group}_shift_ms")
            if not str(timing.get(f"{group}_status", "")):
                raise ValueError(f"timing.{group}_status must be non-empty")
        shifts = timing.get("sensitivity_shifts_ms")
        if not isinstance(shifts, list) or not shifts or not all(math.isfinite(float(item)) for item in shifts):
            raise ValueError("timing.sensitivity_shifts_ms must be a non-empty finite list")
        if quality.get("short_record_policy") not in {"error", "exclude"}:
            raise ValueError("quality.short_record_policy must be 'error' or 'exclude'")
        if int(quality.get("minimum_repeats_per_frequency", 0)) < 1:
            raise ValueError("minimum_repeats_per_frequency must be positive")
        _finite(quality.get("minimum_motion_fit_r2"), "quality.minimum_motion_fit_r2")
        load_r2 = _finite(
            quality.get("minimum_load_fit_r2_for_flag"),
            "quality.minimum_load_fit_r2_for_flag",
        )
        if not 0.0 <= load_r2 <= 1.0:
            raise ValueError("quality.minimum_load_fit_r2_for_flag must be within [0, 1]")
        if fit.get("method") != "trial_harmonic_projection_then_per_frequency_huber":
            raise ValueError("fit.method must select per-frequency harmonic projection")
        harmonics = fit.get("load_harmonics")
        if not isinstance(harmonics, list) or not harmonics or any(int(k) != k or int(k) < 1 for k in harmonics):
            raise ValueError("fit.load_harmonics must be a non-empty list of positive integers")
        _positive(fit.get("huber_k"), "fit.huber_k")
        completion = self.full_6d_completion
        for dof_name in ("surge_u", "roll_p"):
            values = _require_mapping(completion.get(dof_name), f"full_6d_completion.{dof_name}")
            for coefficient in ("added_mass", "linear_damping", "quadratic_damping"):
                value = _finite(values.get(coefficient), f"full_6d_completion.{dof_name}.{coefficient}")
                if value < 0.0:
                    raise ValueError(f"full_6d_completion.{dof_name}.{coefficient} must be non-negative")

    @property
    def sampling(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("sampling"), "sampling")

    @property
    def frequency_estimation(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("frequency_estimation"), "frequency_estimation")

    @property
    def model(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("model"), "model")

    @property
    def apparatus(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("apparatus"), "apparatus")

    @property
    def timing(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("timing"), "timing")

    @property
    def quality(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("quality"), "quality")

    @property
    def fit(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("fit"), "fit")

    @property
    def full_6d_completion(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("full_6d_completion"), "full_6d_completion")

    @property
    def model_inertia_flu(self) -> np.ndarray:
        return np.asarray(self.model["inertia_at_com_flu_kg_m2"], dtype=float)

    @property
    def model_com_from_motion_origin_flu(self) -> np.ndarray:
        return np.asarray(self.model["com_from_motion_origin_flu_m"], dtype=float)

    def shift_ms(self, timing_group: str) -> float:
        return float(self.timing[f"{timing_group}_shift_ms"])

    def resolved_dict(self) -> dict[str, Any]:
        result = json.loads(json.dumps(self.raw))
        result["config_path"] = str(self.source_path.resolve())
        return result


def load_config(path: Path) -> SixDofConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return SixDofConfig(path, _require_mapping(raw, "config"))


@dataclass(frozen=True)
class TrialPlan:
    dof: str
    repeat: int
    file_id: int
    nominal_frequency_hz: float
    gather_path: Path
    sensor_path: Path
    timing_group: str


@dataclass(frozen=True)
class AuditRow:
    plan: TrialPlan
    status: str
    reason: str
    gather_rows: int | None
    sensor_rows: int | None
    required_sensor_rows: int


@dataclass(frozen=True)
class PreflightResult:
    planned: tuple[TrialPlan, ...]
    included: tuple[TrialPlan, ...]
    audit: tuple[AuditRow, ...]


def expected_trials(root: Path, config: SixDofConfig) -> list[TrialPlan]:
    plans: list[TrialPlan] = []
    frequency_step = float(config.frequency_estimation["nominal_frequency_step_hz"])
    suffixes = {1: "ang0", 2: "ang60", 3: "ang120"}
    for dof in DOFS:
        kind = DOF_RAW_KIND[dof]
        file_ids = range(8, 15) if kind == "sway" else range(22, 29)
        base = 7 if kind == "sway" else 21
        for repeat in range(1, 4):
            directory = Path(root) / f"{DOF_FAMILY[dof]}{repeat}"
            for file_id in file_ids:
                if kind == "sway":
                    stem = str(file_id)
                else:
                    stem = f"{file_id}_{suffixes[repeat]}"
                plans.append(
                    TrialPlan(
                        dof=dof,
                        repeat=repeat,
                        file_id=file_id,
                        nominal_frequency_hz=frequency_step * (file_id - base),
                        gather_path=directory / f"gather_{stem}.csv",
                        sensor_path=directory / f"sensor_{stem}.csv",
                        timing_group=DOF_TIMING_GROUP[dof],
                    )
                )
    return plans


def _count_rows_and_width(path: Path, delimiter: str | None) -> tuple[int, set[int]]:
    rows = 0
    widths: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            widths.add(len(line.split() if delimiter is None else line.rstrip("\r\n").split(delimiter)))
    return rows, widths


def preflight(root: Path, config: SixDofConfig) -> PreflightResult:
    plans = expected_trials(root, config)
    audit: list[AuditRow] = []
    fatal: list[str] = []
    included: list[TrialPlan] = []
    sensor_hz = float(config.sampling["sensor_hz"])
    end = float(config.sampling["active_end_s"])
    # Positive motion shift pairs sensor(t) with motion(t+shift).  Sensor itself
    # must at least cover the unshifted time at which the configured active end
    # is requested.  Add one sample for zero-based indexing.
    for plan in plans:
        # The sensor window is fixed in its own clock. A configured shift only
        # changes which motion time is paired with sensor(t); it does not move
        # or shorten the measured-load window.
        required_sensor_rows = int(math.ceil(end * sensor_hz)) + 1
        missing = [str(path) for path in (plan.gather_path, plan.sensor_path) if not path.is_file()]
        if missing:
            reason = "missing_pair:" + ",".join(missing)
            fatal.append(reason)
            audit.append(AuditRow(plan, "fatal", reason, None, None, required_sensor_rows))
            continue
        gather_rows, gather_widths = _count_rows_and_width(plan.gather_path, None)
        sensor_rows, sensor_widths = _count_rows_and_width(plan.sensor_path, ",")
        shape_errors = []
        if gather_widths != {16}:
            shape_errors.append(f"gather_widths={sorted(gather_widths)}")
        if sensor_widths != {6}:
            shape_errors.append(f"sensor_widths={sorted(sensor_widths)}")
        if shape_errors:
            reason = "malformed:" + ";".join(shape_errors)
            fatal.append(f"{plan.gather_path}: {reason}")
            audit.append(AuditRow(plan, "fatal", reason, gather_rows, sensor_rows, required_sensor_rows))
            continue
        if sensor_rows < required_sensor_rows:
            reason = f"short_sensor_record:{sensor_rows}<{required_sensor_rows}"
            if config.quality["short_record_policy"] == "error":
                fatal.append(f"{plan.sensor_path}: {reason}")
                status = "fatal"
            else:
                status = "excluded"
            audit.append(AuditRow(plan, status, reason, gather_rows, sensor_rows, required_sensor_rows))
            continue
        audit.append(AuditRow(plan, "included", "", gather_rows, sensor_rows, required_sensor_rows))
        included.append(plan)
    if fatal:
        raise ValueError("preflight failed:\n" + "\n".join(fatal))
    minimum = int(config.quality["minimum_repeats_per_frequency"])
    for dof in DOFS:
        frequencies = sorted({plan.nominal_frequency_hz for plan in plans if plan.dof == dof})
        for frequency in frequencies:
            count = sum(
                plan.dof == dof and math.isclose(plan.nominal_frequency_hz, frequency)
                for plan in included
            )
            if count < minimum:
                raise ValueError(
                    f"{dof} nominal frequency {frequency:g} Hz has {count} included repeats; "
                    f"requires {minimum}"
                )
    return PreflightResult(tuple(plans), tuple(included), tuple(audit))


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


def build_trial(
    plan: TrialPlan,
    config: SixDofConfig,
    *,
    shift_ms: float | None = None,
) -> SixDofTrial:
    gather, sensor = read_raw_pair(plan)
    sampling = config.sampling
    gather_hz = float(sampling["gather_hz"])
    sensor_hz = float(sampling["sensor_hz"])
    start = float(sampling["active_start_s"])
    end = float(sampling["active_end_s"])
    harmonics = int(sampling["kinematic_harmonics"])
    tg = np.arange(len(gather), dtype=float) / gather_hz
    active_g = (tg >= start) & (tg <= end)
    if int(active_g.sum()) < 100:
        raise ValueError(f"{plan.gather_path}: insufficient motion samples in active interval")
    x = (gather[:, 1] + gather[:, 7]) / float(sampling["position_counts_per_m"])
    lateral = gather[:, 4] / float(sampling["position_counts_per_m"])
    angle = (
        float(config.apparatus["yaw_encoder_sign_to_H_r"])
        * (gather[:, 10] - gather[0, 10])
        / float(sampling["angle_counts_per_degree"])
    )
    angle = np.deg2rad(angle)
    frequency_motion = lateral if DOF_RAW_KIND[plan.dof] == "sway" else angle
    frequency = estimate_trial_frequency(
        tg[active_g], frequency_motion[active_g], plan.nominal_frequency_hz, config
    )
    fit_x = fit_fourier(tg[active_g], x[active_g], frequency, harmonics)
    fit_lateral = fit_fourier(tg[active_g], lateral[active_g], frequency, harmonics)
    fit_angle = (
        fit_fourier(tg[active_g], angle[active_g], frequency, harmonics)
        if DOF_RAW_KIND[plan.dof] == "yaw"
        else None
    )
    ts, sensor_avg = _block_average(sensor, int(sampling["sensor_decimation"]), sensor_hz)
    actual_shift_ms = config.shift_ms(plan.timing_group) if shift_ms is None else _finite(shift_ms, "shift_ms")
    active_s = (ts >= start) & (ts <= end)
    ts = ts[active_s]
    sensor_avg = sensor_avg[active_s]
    tm = ts + actual_shift_ms / 1000.0
    if len(ts) < 100:
        raise ValueError(f"{plan.sensor_path}: insufficient paired samples in active interval")
    _, dx, ddx = fit_x.evaluate(tm)
    _, dy, ddy = fit_lateral.evaluate(tm)
    if fit_angle is None:
        psi = np.zeros_like(tm)
        rate = np.zeros_like(tm)
        rate_dot = np.zeros_like(tm)
        angle_r2 = 1.0
        angle_amplitude = 0.0
    else:
        psi, rate, rate_dot = fit_angle.evaluate(tm)
        angle_r2 = fit_angle.r2
        angle_amplitude = 0.5 * float(np.ptp(psi))
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    linear_velocity_H = np.column_stack(
        (cpsi * dx + spsi * dy, -spsi * dx + cpsi * dy, np.zeros_like(tm))
    )
    linear_derivative_H = np.column_stack(
        (-spsi * rate * dx + cpsi * ddx + cpsi * rate * dy + spsi * ddy,
         -cpsi * rate * dx - spsi * ddx - spsi * rate * dy + cpsi * ddy,
         np.zeros_like(tm))
    )
    angular_velocity_H = np.column_stack((np.zeros_like(tm), np.zeros_like(tm), rate))
    angular_acceleration_H = np.column_stack((np.zeros_like(tm), np.zeros_like(tm), rate_dot))
    body_to_H = _body_to_H_rotation(plan, config)
    linear_velocity_origin_B, angular_velocity_B = rotate_twist_H_to_B(
        linear_velocity_H, angular_velocity_H, body_to_H
    )
    linear_derivative_origin_B, angular_acceleration_B = rotate_twist_H_to_B(
        linear_derivative_H, angular_acceleration_H, body_to_H
    )
    model = config.model
    com_from_origin_B = np.asarray(model["com_from_motion_origin_flu_m"], dtype=float)
    linear_velocity_B, linear_derivative_B = motion_origin_to_com_kinematics(
        linear_velocity_origin_B,
        linear_derivative_origin_B,
        angular_velocity_B,
        angular_acceleration_B,
        com_from_origin_B,
    )
    dof_index = DOF_INDEX[plan.dof]
    if dof_index < 3:
        q = linear_velocity_B[:, dof_index]
        qdot = linear_derivative_B[:, dof_index]
    else:
        q = angular_velocity_B[:, dof_index - 3]
        qdot = angular_acceleration_B[:, dof_index - 3]
    sensor_to_H = np.asarray(
        config.apparatus[f"sensor_to_H_wrench_matrix_{plan.timing_group}"],
        dtype=float,
    )
    # The configured map consumes the original sensor order
    # [TX, TY, TZ, FX, FY, FZ] and emits [X, Y, Z, K, M, N] in H.
    wrench_H_at_origin = sensor_avg @ sensor_to_H.T
    wrench_B_at_origin = rotate_wrench_H_to_B(wrench_H_at_origin, body_to_H)
    mass = float(model["mass_kg"])
    wrench_B_at_com = translate_wrench_origin_to_com(wrench_B_at_origin, com_from_origin_B)
    inertia_B = np.asarray(model["inertia_at_com_flu_kg_m2"], dtype=float)
    rigid_wrench_B_at_com = rigid_body_wrench_at_com(
        mass,
        inertia_B,
        linear_velocity_B,
        linear_derivative_B,
        angular_velocity_B,
        angular_acceleration_B,
    )
    measured = wrench_B_at_com[:, dof_index]
    target = measured - rigid_wrench_B_at_com[:, dof_index]
    u = linear_velocity_B[:, 0]
    X = np.column_stack([u * q, q * np.abs(q), qdot])
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


def robust_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 30,
    *,
    huber_k: float = 1.5,
) -> np.ndarray:
    """Huber IRLS with an explicit, reproducible tuning constant."""

    if not math.isfinite(huber_k) or huber_k <= 0.0:
        raise ValueError("huber_k must be finite and positive")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - X @ beta
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if scale <= 1e-15:
            break
        cutoff = huber_k * scale
        weights = np.ones_like(residual)
        large = np.abs(residual) > cutoff
        weights[large] = cutoff / np.abs(residual[large])
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(X * root_weights[:, None], y * root_weights, rcond=None)[0]
        if np.linalg.norm(updated - beta) <= 1e-10 * (1.0 + np.linalg.norm(beta)):
            beta = updated
            break
        beta = updated
    return beta


def _stack(
    trials: Sequence[SixDofTrial],
    harmonics: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not trials:
        raise ValueError("cannot fit an empty trial list")
    pairs = [
        residualize_trial(trial)
        if harmonics is None
        else project_trial_harmonics(trial, harmonics)
        for trial in trials
    ]
    return np.vstack([item[0] for item in pairs]), np.concatenate([item[1] for item in pairs])


def mount_sign_invariant_coefficients(X: np.ndarray, y: np.ndarray, sign: float) -> np.ndarray:
    """Return the diagonal fit after a rolled-mount sign transformation."""
    if sign not in (-1.0, 1.0):
        raise ValueError("mount sign must be -1 or +1")
    return robust_fit(sign * np.asarray(X), sign * np.asarray(y))


def fit_dof(
    trials: Sequence[SixDofTrial],
    config: SixDofConfig,
) -> dict[str, Any]:
    """Fit one diagonal channel for one nominal frequency."""

    harmonics = [int(value) for value in config.fit["load_harmonics"]]
    X, y = _stack(trials, harmonics)
    huber_k = float(config.fit["huber_k"])
    beta = robust_fit(X, y, huber_k=huber_k)
    prediction = X @ beta
    return {
        "beta": beta,
        "full_r2": r2_score(y, prediction),
        "fit_domain": "per_trial_harmonic_coefficients",
        "harmonics": harmonics,
        "condition": float(np.linalg.cond(X / np.maximum(np.std(X, axis=0), 1e-15))),
        "X": X,
        "y": y,
        "prediction": prediction,
    }


def _frequency_key(value: float) -> float:
    """Return a stable key for nominal frequencies generated by multiplication."""

    return round(float(value), 10)


def fit_by_frequency(
    trials_by_dof: Mapping[str, Sequence[SixDofTrial]],
    config: SixDofConfig,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Fit repeats together while keeping every nominal frequency independent."""

    results: dict[tuple[str, float], dict[str, Any]] = {}
    minimum = int(config.quality["minimum_repeats_per_frequency"])
    for dof in DOFS:
        grouped: dict[float, list[SixDofTrial]] = {}
        for trial in trials_by_dof[dof]:
            grouped.setdefault(_frequency_key(trial.plan.nominal_frequency_hz), []).append(trial)
        for nominal_frequency_hz, trials in sorted(grouped.items()):
            if len(trials) < minimum:
                raise ValueError(
                    f"{dof} nominal frequency {nominal_frequency_hz:g} Hz has "
                    f"{len(trials)} trials; requires {minimum}"
                )
            result = fit_dof(trials, config)
            estimated = np.asarray([trial.frequency_hz for trial in trials], dtype=float)
            result.update(
                {
                    "dof": dof,
                    "trials": trials,
                    "nominal_frequency_hz": nominal_frequency_hz,
                    "mean_estimated_frequency_hz": float(np.mean(estimated)),
                    "min_estimated_frequency_hz": float(np.min(estimated)),
                    "max_estimated_frequency_hz": float(np.max(estimated)),
                    "estimated_frequency_std_hz": float(np.std(estimated)),
                    "included_repeats": len(trials),
                }
            )
            results[(dof, nominal_frequency_hz)] = result
    return results


def coefficient_metadata(dof: str, config: SixDofConfig) -> tuple[list[str], list[str], np.ndarray]:
    if dof not in DOFS:
        raise ValueError(f"unknown DOF {dof!r}")
    response = DOF_RESPONSE[dof]
    variable = DOF_VARIABLE[dof]
    half_rho = 0.5 * float(config.model["fluid_density_kg_m3"])
    length = float(config.model["wet_length_m"])
    if dof in {"sway", "heave"}:
        units = ["kg/m", "kg/m", "kg"]
        divisors = np.asarray([half_rho * length**2, half_rho * length**2, half_rho * length**3])
    else:
        units = ["kg*m", "kg*m^2", "kg*m^2"]
        divisors = np.asarray([half_rho * length**4, half_rho * length**5, half_rho * length**5])
    return (
        [f"{response}_u{variable}", f"{response}_{variable}|{variable}|", f"{response}_{variable}dot"],
        units,
        divisors,
    )


def direct_term(dof: str) -> str:
    return f"{DOF_RESPONSE[dof]}_{DOF_VARIABLE[dof]}@Uref"


def direct_unit(dof: str) -> str:
    return "N*s/m" if dof in {"sway", "heave"} else "N*m*s"


def coefficient_rows(
    results: Mapping[tuple[str, float], Mapping[str, Any]],
    config: SixDofConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dof, nominal_frequency_hz), result in sorted(
        results.items(), key=lambda item: (DOFS.index(item[0][0]), item[0][1])
    ):
        trials = list(result["trials"])
        u_ref = float(np.mean([np.mean(trial.u) for trial in trials]))
        names, units, divisors = coefficient_metadata(dof, config)
        beta = np.asarray(result["beta"])
        common = {
            "dof": dof,
            "experiment": dof,
            "response": DOF_RESPONSE[dof],
            "variable": DOF_VARIABLE[dof],
            "nominal_frequency_hz": f"{nominal_frequency_hz:.10g}",
            "mean_estimated_frequency_hz": f"{float(result['mean_estimated_frequency_hz']):.10g}",
            "min_estimated_frequency_hz": f"{float(result['min_estimated_frequency_hz']):.10g}",
            "max_estimated_frequency_hz": f"{float(result['max_estimated_frequency_hz']):.10g}",
            "included_repeats": int(result["included_repeats"]),
            "reference_speed_m_s": f"{u_ref:.10g}",
            "full_r2": f"{float(result['full_r2']):.7g}",
            "fit_quality_flag": (
                "low_load_fit_r2"
                if float(result["full_r2"])
                < float(config.quality["minimum_load_fit_r2_for_flag"])
                else "ok"
            ),
            "standardized_condition_number": f"{float(result['condition']):.7g}",
        }
        for index, (name, unit) in enumerate(zip(names, units)):
            rows.append(
                {
                    **common,
                    "term": name,
                    "coefficient": f"{beta[index]:.10g}",
                    "unit": unit,
                    "nondimensional": f"{beta[index] / divisors[index]:.10g}",
                    "assumption_status": "0.562 m PMM scale; independent nominal-frequency fit; PLA wet rigid mass uses +15% CAD density correction then +2.5% water uptake; free water excluded; full sensor reaction wrench negated; timing and wrench origin remain systematic assumptions",
                }
            )
        rows.append(
            {
                **common,
                "term": direct_term(dof),
                "coefficient": f"{beta[0] * u_ref:.10g}",
                "unit": direct_unit(dof),
                "nondimensional": "",
                "assumption_status": "signed C1*Uref derivative at measured tow speed",
            }
        )
    return rows


COEFFICIENT_FIELDS = (
    "dof", "experiment", "response", "variable", "term", "coefficient", "unit",
    "nondimensional", "nominal_frequency_hz", "mean_estimated_frequency_hz",
    "min_estimated_frequency_hz", "max_estimated_frequency_hz", "included_repeats",
    "reference_speed_m_s", "full_r2", "fit_quality_flag",
    "standardized_condition_number", "assumption_status",
)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_rows(
    preflight_result: PreflightResult,
    built: Mapping[tuple[str, int, int], SixDofTrial],
    config: SixDofConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in preflight_result.audit:
        plan = audit.plan
        trial = built.get((plan.dof, plan.repeat, plan.file_id))
        row: dict[str, Any] = {
            "dof": plan.dof,
            "repeat": plan.repeat,
            "file_id": plan.file_id,
            "nominal_frequency_hz": f"{plan.nominal_frequency_hz:.10g}",
            "estimated_frequency_hz": "" if trial is None else f"{trial.frequency_hz:.10g}",
            "timing_group": plan.timing_group,
            "gather_file": str(plan.gather_path),
            "sensor_file": str(plan.sensor_path),
            "status": audit.status,
            "exclusion_reason": audit.reason,
            "gather_rows": "" if audit.gather_rows is None else audit.gather_rows,
            "sensor_rows": "" if audit.sensor_rows is None else audit.sensor_rows,
            "required_sensor_rows": audit.required_sensor_rows,
            "paired_rows": "",
            "motion_fit_r2": "",
            "u_mean_m_s": "",
            "surge_force_mean_n": "",
            "surge_force_oscillatory_rms_n": "",
            "q_amplitude": "",
            "condition_raw": "",
            "sensor_time_shift_ms": f"{config.shift_ms(plan.timing_group):.10g}",
            "quality_flag": audit.status if audit.status != "included" else "",
        }
        if trial is not None:
            d = trial.diagnostics
            for key in (
                "paired_rows", "motion_fit_r2", "u_mean_m_s", "surge_force_mean_n",
                "surge_force_oscillatory_rms_n", "q_amplitude", "condition_raw",
            ):
                row[key] = f"{d[key]:.10g}"
            flags: list[str] = []
            if d["motion_fit_r2"] < float(config.quality["minimum_motion_fit_r2"]):
                flags.append("motion_fit_low")
            ratio = d["frequency_ratio_to_nominal"]
            lo = float(config.frequency_estimation["search_lower_fraction"])
            hi = float(config.frequency_estimation["search_upper_fraction"])
            if math.isclose(ratio, lo, rel_tol=0.0, abs_tol=1e-6) or math.isclose(ratio, hi, rel_tol=0.0, abs_tol=1e-6):
                flags.append("frequency_search_boundary")
            row["quality_flag"] = ";".join(flags) if flags else "ok"
        rows.append(row)
    return rows


DIAGNOSTIC_FIELDS = (
    "dof", "repeat", "file_id", "nominal_frequency_hz", "estimated_frequency_hz",
    "timing_group", "gather_file", "sensor_file", "status", "exclusion_reason",
    "gather_rows", "sensor_rows", "required_sensor_rows", "paired_rows", "motion_fit_r2",
    "u_mean_m_s", "surge_force_mean_n", "surge_force_oscillatory_rms_n",
    "q_amplitude", "condition_raw", "sensor_time_shift_ms", "quality_flag",
)


def build_six_dof_diagonal(
    coefficient_rows_for_frequency: Sequence[Mapping[str, Any]],
    identification_metadata: Mapping[str, Any] | None = None,
    completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in coefficient_rows_for_frequency:
        key = (str(row["dof"]), str(row["term"]))
        if key in selected_rows:
            raise ValueError(f"duplicate coefficient row for {key}")
        selected_rows[key] = row
    added: list[float | None] = [None] * 6
    linear: list[float | None] = [None] * 6
    quadratic: list[float | None] = [None] * 6
    speeds: dict[str, float] = {}
    compact: dict[str, dict[str, float | str]] = {}
    for dof in DOFS:
        response = DOF_RESPONSE[dof]
        variable = DOF_VARIABLE[dof]
        names = [
            f"{response}_u{variable}",
            f"{response}_{variable}|{variable}|",
            f"{response}_{variable}dot",
        ]
        try:
            values = [-float(selected_rows[(dof, name)]["coefficient"]) for name in names]
            direct = selected_rows[(dof, direct_term(dof))]
        except KeyError as error:
            raise ValueError(f"missing coefficient row {error.args[0]}") from error
        index = DOF_INDEX[dof]
        added[index] = values[2]
        linear[index] = -float(direct["coefficient"])
        quadratic[index] = values[1]
        speed = float(direct.get("reference_speed_m_s", 0.0))
        speeds[variable] = speed
        compact[variable] = {
            "dof": dof,
            "response": response,
            "added_mass": added[index],
            "linear_damping_at_reference_speed": linear[index],
            "quadratic_damping": quadratic[index],
            "reference_speed_m_s": speed,
        }
    source_by_dof = ["unassigned", "PMM_experiment", "PMM_experiment", "unassigned", "PMM_experiment", "PMM_experiment"]
    if completion is not None:
        for index, name in ((0, "surge_u"), (3, "roll_p")):
            values = _require_mapping(completion.get(name), f"full_6d_completion.{name}")
            added[index] = float(values["added_mass"])
            linear[index] = float(values["linear_damping"])
            quadratic[index] = float(values["quadratic_damping"])
            source_by_dof[index] = "external_literature_prior"

    def matrix(diagonal: Sequence[float | None]) -> list[list[float | None]]:
        return [
            [diagonal[row] if row == column else 0.0 for column in range(6)]
            for row in range(6)
        ]

    return {
        "schema_version": 1,
        "result_scope": "one_nominal_frequency_complete_6x6_diagonal_hybrid_model_in_FLU_COM",
        "result_scale": "physical_0p562_m_PMM_towing_model",
        "frame": "FLU body axes at PMM towing-model COM: x forward, y left, z up",
        "row_order": list(ROW_ORDER),
        "column_order": list(COLUMN_ORDER),
        "experimentally_identified_mask": [False, True, True, False, True, True],
        "complete_diagonal_mask": [value is not None for value in added],
        "diagonal_source_by_dof": source_by_dof,
        "vertical_mapping": "explicit_FLU_R_HB_Rx_plus_90_about_forward_axis_after_full_sensor_reaction_wrench_negation",
        "fossen_definition": "tau_h=-M_A*nu_dot-D_L*nu-D_Q*(abs(nu)*nu)",
        "added_mass": matrix(added),
        "linear_damping_effective_at_reference_speed": matrix(linear),
        "quadratic_damping": matrix(quadratic),
        "diagonal_vectors": {
            "added_mass": added,
            "linear_damping_effective_at_reference_speed": linear,
            "quadratic_damping": quadratic,
        },
        "identified_by_velocity": compact,
        "reference_speed_m_s_by_velocity": speeds,
        "source_assumptions": dict(identification_metadata or {}),
        "physicality_warning": (
            "Signs are preserved from the source fit. Negative damping or added mass is reported "
            "without projection and remains provisional under timing and mount assumptions."
        ),
    }


def build_frequency_resolved_matrices(
    rows: Sequence[Mapping[str, Any]],
    identification_metadata: Mapping[str, Any],
    config: SixDofConfig,
) -> dict[str, Any]:
    """Build one model-scale diagonal matrix for each nominal PMM frequency."""

    frequencies = sorted({_frequency_key(float(row["nominal_frequency_hz"])) for row in rows})
    models: list[dict[str, Any]] = []
    for frequency in frequencies:
        selected = [
            row
            for row in rows
            if math.isclose(
                _frequency_key(float(row["nominal_frequency_hz"])),
                frequency,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ]
        model = build_six_dof_diagonal(
            selected,
            identification_metadata,
            config.full_6d_completion,
        )
        model["nominal_frequency_hz"] = frequency
        model["mean_estimated_frequency_hz_by_dof"] = {
            dof: float(
                next(row for row in selected if row["dof"] == dof)[
                    "mean_estimated_frequency_hz"
                ]
            )
            for dof in DOFS
        }
        models.append(model)
    return {
        "schema_version": 1,
        "result_scope": "frequency_resolved_complete_6x6_diagonal_hybrid_models",
        "result_scale": {
            "wet_length_m": float(config.model["wet_length_m"]),
            "geometry_scale_of_real_robot": float(config.model["geometry_scale_of_real_robot"]),
            "PMM_and_CFD_length_ratio": 1.0,
            "coefficient_scale_factor_applied": 1.0,
            "full_scale_conversion_performed": False,
        },
        "nominal_frequencies_hz": frequencies,
        "models": models,
    }


def _timing_fit_rows(
    plans: Sequence[TrialPlan], config: SixDofConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shift in sorted({float(item) for item in config.timing["sensitivity_shifts_ms"]}):
        for dof in DOFS:
            built = [build_trial(plan, config, shift_ms=shift) for plan in plans if plan.dof == dof]
            frequencies = sorted({_frequency_key(trial.plan.nominal_frequency_hz) for trial in built})
            for frequency in frequencies:
                trials = [
                    trial
                    for trial in built
                    if _frequency_key(trial.plan.nominal_frequency_hz) == frequency
                ]
                result = fit_dof(trials, config)
                names, _, _ = coefficient_metadata(dof, config)
                mean_estimated = float(np.mean([trial.frequency_hz for trial in trials]))
                for name, value in zip(names, result["beta"]):
                    rows.append(
                        {
                            "sensor_time_shift_ms": f"{shift:.10g}",
                            "timing_group": DOF_TIMING_GROUP[dof],
                            "dof": dof,
                            "nominal_frequency_hz": f"{frequency:.10g}",
                            "mean_estimated_frequency_hz": f"{mean_estimated:.10g}",
                            "term": name,
                            "coefficient": f"{value:.10g}",
                            "full_r2": f"{float(result['full_r2']):.10g}",
                        }
                    )
    return rows


TIMING_FIELDS = (
    "sensor_time_shift_ms", "timing_group", "dof", "nominal_frequency_hz",
    "mean_estimated_frequency_hz", "term", "coefficient", "full_r2",
)


def _make_plot(path: Path, panels: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3 if len(panels) > 4 else 2
    rows = int(math.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (label, result) in zip(axes_array, panels):
        y = np.asarray(result["y"])
        prediction = np.asarray(result["prediction"])
        stride = max(1, len(y) // 5000)
        axis.scatter(y[::stride], prediction[::stride], s=5, alpha=0.22, rasterized=True)
        lo = float(min(np.min(y), np.min(prediction)))
        hi = float(max(np.max(y), np.max(prediction)))
        axis.plot([lo, hi], [lo, hi], color="#b91c1c", lw=1.4)
        axis.set_title(f"{label}: R2={result['full_r2']:.3f}")
        axis.set_xlabel("detrended hydrodynamic load")
        axis.set_ylabel("prediction")
        axis.grid(alpha=0.25)
    for axis in axes_array[len(panels):]:
        axis.set_visible(False)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _metadata(
    config: SixDofConfig,
    preflight_result: PreflightResult,
    trials_by_dof: Mapping[str, Sequence[SixDofTrial]],
    results: Mapping[tuple[str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_scope": "six_dof_diagonal_per_frequency_hybrid",
        "result_scale": {
            "wet_length_m": float(config.model["wet_length_m"]),
            "geometry_scale_of_real_robot": float(config.model["geometry_scale_of_real_robot"]),
            "PMM_and_CFD_length_ratio": 1.0,
            "coefficient_scale_factor_applied": 1.0,
            "full_scale_conversion_performed": False,
        },
        "raw_data_policy": "read_only",
        "planned_trials": len(preflight_result.planned),
        "included_trials": len(preflight_result.included),
        "excluded_trials": sum(row.status == "excluded" for row in preflight_result.audit),
        "included_trials_by_dof": {dof: len(trials_by_dof[dof]) for dof in DOFS},
        "config": config.resolved_dict(),
        "coordinate_transform": {
            "apparatus_frame": "H FLU: x forward, y left, z up",
            "body_frame": "B FLU: x forward, y left, z up",
            "sensor_frame": "upright raw balance frame S; unchanged between horizontal and vertical tests",
            "sensor_mount_status": config.apparatus["sensor_mount_status"],
            "horizontal_R_HB": config.apparatus["body_to_apparatus_rotation_horizontal"],
            "vertical_R_HB": config.apparatus["body_to_apparatus_rotation_vertical"],
            "observed_vertical_encoder_initial_offset_deg": config.apparatus["observed_vertical_encoder_initial_offset_deg"],
            "observed_vertical_encoder_offset_interpretation": config.apparatus["observed_vertical_encoder_offset_interpretation"],
            "assumption": "vertical model is rolled +90 degrees about body FLU +x only; the encoder initial offset is not a body yaw rotation",
            "status": config.apparatus["vertical_mount_status"],
            "derivation": "twists and wrenches are explicitly rotated H to B; changing +90 to -90 reverses each diagonal generalized motion and conjugate load together",
            "scope": "diagonal terms only; no cross-coupling inference",
        },
        "pitch_restoring": {
            "included_in_fit": False,
            "reason": "the rolled-model pitch axis is parallel to gravity; normal-attitude hydrostatic restoring is separate",
        },
        "wrench_and_rigid_body": {
            "raw_sensor_order": list(SENSOR_COLUMNS),
            "sensor_mount": config.apparatus["sensor_mount"],
            "sensor_axes_frame": config.apparatus["sensor_axes_frame"],
            "sensor_to_H_wrench_matrix_horizontal": config.apparatus["sensor_to_H_wrench_matrix_horizontal"],
            "sensor_to_H_wrench_matrix_vertical": config.apparatus["sensor_to_H_wrench_matrix_vertical"],
            "wrench_mapping_scope": "unchanged upright balance: negate the complete raw reaction wrench so [X,Y,Z,K,M,N]=-[FX,FY,FZ,TX,TY,TZ], then apply the model-roll transform",
            "raw_wrench_reference": config.apparatus["wrench_reference"],
            "com_from_motion_origin_flu_m": config.model["com_from_motion_origin_flu_m"],
            "translation_formula": "M_COM=M_origin-r_origin_to_COM cross F using confirmed force channels",
            "rigid_force_formula": "m*(v_dot+omega cross v) at COM",
            "rigid_moment_formula": "I_COM*omega_dot+omega cross (I_COM*omega)",
            "hydrodynamic_target": "measured_wrench_at_COM-rigid_body_wrench_at_COM",
        },
        "time_alignment": {
            group: {
                "sensor_time_shift_ms": config.shift_ms(group),
                "status": config.timing[f"{group}_status"],
                "definition": "sensor(t) is paired with motion(t+shift)",
            }
            for group in ("horizontal", "vertical")
        },
        "frequency_estimation": {
            "method": "per-trial nonlinear grid refinement of constant+trend+three-harmonic motion fit",
            "estimated_frequency_hz_by_trial": {
                f"{trial.plan.dof}/r{trial.plan.repeat}/id{trial.plan.file_id}": trial.frequency_hz
                for dof in DOFS
                for trial in trials_by_dof[dof]
            },
        },
        "load_fit": {
            "method": config.fit["method"],
            "harmonics": config.fit["load_harmonics"],
            "regression": "one Huber fit over repeated trials at each nominal frequency; no cross-frequency pooling",
            "huber_k": config.fit["huber_k"],
            "frequency_group_results": {
                f"{dof}/{frequency:g}Hz": {
                    "included_repeats": int(result["included_repeats"]),
                    "mean_estimated_frequency_hz": float(result["mean_estimated_frequency_hz"]),
                    "estimated_frequency_std_hz": float(result["estimated_frequency_std_hz"]),
                    "full_r2": float(result["full_r2"]),
                    "fit_quality_flag": (
                        "low_load_fit_r2"
                        if float(result["full_r2"])
                        < float(config.quality["minimum_load_fit_r2_for_flag"])
                        else "ok"
                    ),
                    "standardized_condition_number": float(result["condition"]),
                }
                for (dof, frequency), result in sorted(
                    results.items(), key=lambda item: (DOFS.index(item[0][0]), item[0][1])
                )
            },
        },
    }


def _report(
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> str:
    lines = [
        "# PMM 单频率六自由度对角水动力结果",
        "",
        "## 范围",
        "",
        "结果保持在实物 PMM 模型的 `0.562 m` 尺度；PMM 与 CFD 都是真实机器人的 0.7 模型，因此系数缩放因子严格为 `1.0`，没有执行 0.7→1.0 换算。",
        "",
        "每个 DOF 的 0.1–0.7 Hz 七个名义频率分别拟合。每组只合并同一名义频率的重复试次，不跨频率联合回归。矩阵顺序为 `[u,v,w,p,q,r]`；`Y←v`、`Z←w`、`M←q`、`N←r` 来自 PMM，未激励的 surge/roll 使用既有先验补齐。",
        "",
        "六维传感器保持直立，原始值表示模型施加给传感器的反力/反力矩，因此完整取反为 `[X,Y,Z,K,M,N]=-[FX,FY,FZ,TX,TY,TZ]` 后才得到作用在模型上的FLU广义力。vertical仅把模型绕前向F轴旋转 `+90°`，即 `R_HB=Rx(+90°)`；传感器本身不随模型旋转。",
        "",
        f"计划 {metadata['planned_trials']} 条，纳入 {metadata['included_trials']} 条，显式排除 {metadata['excluded_trials']} 条。",
        "",
        "## 湿态质量假设",
        "",
        "材料已确认为PLA，且CAD材料密度低估15%。因此先将CAD质量 `6.4163 kg` 修正为干态 `7.378745 kg`，再采用FFF PLA长期浸水试验的平均增重 `2.5%`，得到湿态刚体质量 `7.563213625 kg`。其中密度修正增加 `0.962445 kg`，材料吸水增加 `0.184468625 kg`。",
        "",
        "约9 kg的体感重量仍比上述PLA湿态估计高约 `1.437 kg`，这部分按自由水/滞留水处理，不纳入刚体扣除；只有能够证明与模型机械锁定、无晃动和交换的水才应计入刚体质量。密度与吸水均暂按均匀分布，故惯量按原CAD张量乘 `1.15×1.025=1.17875`。资料：[PLA/PETG浸水研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC8036839/)、[PLA 28天浸水研究](https://pubs.rsc.org/en/content/articlehtml/2022/ma/d2ma00707j)、[ISO 62对多孔材料的适用性限制](https://www.iso.org/standard/41672.html)。",
        "",
        "## 系数",
        "",
        "| DOF | 名义频率/Hz | 实测均值/Hz | 重复数 | 项 | 系数 | 单位 | R² |",
        "|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in rows:
        term = str(row["term"]).replace("|", "\\|")
        lines.append(
            f"| {row['dof']} | {row['nominal_frequency_hz']} | "
            f"{row['mean_estimated_frequency_hz']} | {row['included_repeats']} | "
            f"{term} | {row['coefficient']} | {row['unit']} | {row['full_r2']} |"
        )
    excluded = [row for row in diagnostics if row["status"] == "excluded"]
    lines.extend(["", "## 数据质量", ""])
    if excluded:
        for row in excluded:
            lines.append(f"- 排除 `{row['sensor_file']}`：{row['exclusion_reason']}")
    else:
        lines.append("- 无排除记录。")
    low_fit_groups = sorted(
        {
            (str(row["dof"]), float(row["nominal_frequency_hz"]), float(row["full_r2"]))
            for row in rows
            if row["fit_quality_flag"] == "low_load_fit_r2"
        },
        key=lambda item: (DOFS.index(item[0]), item[1]),
    )
    if low_fit_groups:
        formatted = "、".join(
            f"{dof} {frequency:g} Hz (R²={score:.3f})"
            for dof, frequency, score in low_fit_groups
        )
        lines.append(f"- 低于载荷拟合提示阈值 R²=0.8：{formatted}。这些组保留但应降低权重。")
    else:
        lines.append("- 所有单频载荷拟合均达到 R²=0.8 提示阈值。")
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- horizontal和vertical都先将六维传感器完整反力取反；vertical仅把模型绕艇体前向F轴旋转+90°。",
            "- 每个试次都保留前进surge速度与X向力：u进入u·q项，完整FX/FY/FZ参与质心力矩平移。没有独立的纯surge振荡加速度，因此surge附加质量仍不能由这些记录单独识别；roll没有试验。",
            "- `u/p` 是文献先验，`v/w/q/r` 是 PMM 试验识别；完整 6×6 是混合来源矩阵。",
            "- 载荷先逐试次投影到各自实测频率的 1–3 次谐波，再按名义频率分组进行 Huber 拟合；不同频率之间没有共享系数。",
            "- PMM刚体扣除使用PLA密度+15%并叠加2.5%吸水后的湿质量7.563213625 kg，惯量为原CAD值的1.17875倍；约9 kg体感值中剩余的自由水不作刚体扣除。",
            "- PMM与CFD属于同一0.562 m、0.7几何模型，本目录所有结果均可直接与CFD模型尺度结果比较。",
            "- Fossen 输出保留原始拟合符号，不会把负阻尼静默取绝对值。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_analysis(
    root: Path,
    config: SixDofConfig,
    output: Path,
    *,
    write_timing: bool = True,
    write_plot: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    checked = preflight(root, config)
    trials_by_dof: dict[str, list[SixDofTrial]] = {dof: [] for dof in DOFS}
    built: dict[tuple[str, int, int], SixDofTrial] = {}
    for plan in checked.included:
        trial = build_trial(plan, config)
        trials_by_dof[plan.dof].append(trial)
        built[(plan.dof, plan.repeat, plan.file_id)] = trial
    results = fit_by_frequency(trials_by_dof, config)
    rows = coefficient_rows(results, config)
    diagnostics = diagnostic_rows(checked, built, config)
    metadata = _metadata(config, checked, trials_by_dof, results)
    matrices = build_frequency_resolved_matrices(rows, metadata, config)
    # All expensive/error-prone core work above completes before the first file
    # is opened. Individual files are then atomically replaced.
    output.mkdir(parents=True, exist_ok=True)
    coefficient_path = output / "hydrodynamic_coefficients.csv"
    diagnostic_path = output / "trial_diagnostics.csv"
    metadata_path = output / "identification_metadata.json"
    matrices_path = output / "fossen_6x6_by_frequency.json"
    report_path = output / "REPORT.md"
    _write_csv(coefficient_path, COEFFICIENT_FIELDS, rows)
    _write_csv(diagnostic_path, DIAGNOSTIC_FIELDS, diagnostics)
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(matrices_path, json.dumps(matrices, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(report_path, _report(rows, diagnostics, metadata))
    artifacts: dict[str, Path] = {
        "coefficients": coefficient_path,
        "diagnostics": diagnostic_path,
        "metadata": metadata_path,
        "fossen_by_frequency": matrices_path,
        "report": report_path,
    }
    if write_timing:
        timing_path = output / "timing_sensitivity.csv"
        _write_csv(timing_path, TIMING_FIELDS, _timing_fit_rows(checked.included, config))
        artifacts["timing_sensitivity"] = timing_path
    if write_plot:
        plot_path = output / "fit_diagnostics.png"
        panels = [
            (f"{dof} {frequency:g} Hz", result)
            for (dof, frequency), result in sorted(
                results.items(), key=lambda item: (DOFS.index(item[0][0]), item[0][1])
            )
        ]
        try:
            _make_plot(plot_path, panels)
        except (ImportError, AttributeError) as error:
            print(
                f"Warning: optional fit plot was skipped because Matplotlib could not load: {error}",
                file=sys.stderr,
            )
        else:
            artifacts["plot"] = plot_path
    return artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=script_root)
    parser.add_argument("--config", type=Path, default=script_root / "six_dof_config.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_root / "hydro_results" / "six_dof_diagonal" / "model_scale",
    )
    parser.add_argument("--skip-timing-sensitivity", action="store_true")
    parser.add_argument(
        "--write-plot",
        action="store_true",
        help="write the optional fit plot (requires a Matplotlib build compatible with NumPy)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    artifacts = run_analysis(
        args.root,
        config,
        args.output,
        write_timing=not args.skip_timing_sensitivity,
        write_plot=args.write_plot,
        overwrite=args.overwrite,
    )
    print(
        "Completed model-scale, per-frequency 6x6 diagonal identification; "
        f"wrote {len(artifacts)} artifacts."
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

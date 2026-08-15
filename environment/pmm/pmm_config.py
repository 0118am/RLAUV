"""Configuration contracts and raw-file preflight for PMM identification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .pmm_common import *
from .pmm_kinematics import rotation_x, validate_rotation


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


def _validate_sampling_and_frequency(
    sampling: Mapping[str, Any],
    frequency: Mapping[str, Any],
) -> None:
    for name in (
        "gather_hz",
        "sensor_hz",
        "sensor_decimation",
        "kinematic_harmonics",
        "position_counts_per_m",
        "angle_counts_per_degree",
    ):
        _positive(sampling.get(name), f"sampling.{name}")
    start = _finite(sampling.get("active_start_s"), "sampling.active_start_s")
    end = _finite(sampling.get("active_end_s"), "sampling.active_end_s")
    if start < 0.0 or end <= start:
        raise ValueError("sampling active window is invalid")
    lower = _positive(
        frequency.get("search_lower_fraction"),
        "frequency_estimation.search_lower_fraction",
    )
    upper = _positive(
        frequency.get("search_upper_fraction"),
        "frequency_estimation.search_upper_fraction",
    )
    _positive(
        frequency.get("nominal_frequency_step_hz"),
        "frequency_estimation.nominal_frequency_step_hz",
    )
    if not lower < 1.0 < upper:
        raise ValueError("frequency search bounds must bracket the nominal frequency")
    if int(frequency.get("refinement_rounds", 0)) < 1 or int(frequency.get("grid_points", 0)) < 3:
        raise ValueError("frequency refinement requires at least one round and three grid points")


def _validate_model(model: Mapping[str, Any]) -> None:
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


def _validate_apparatus(apparatus: Mapping[str, Any]) -> None:
    if list(apparatus.get("sensor_column_order", ())) != list(SENSOR_COLUMNS):
        raise ValueError(f"apparatus.sensor_column_order must be {list(SENSOR_COLUMNS)}")
    if apparatus.get("sensor_mount") != "upright_not_rolled_same_apparatus_orientation":
        raise ValueError("apparatus.sensor_mount must describe the unchanged upright installation")
    if apparatus.get("sensor_mount_status") != "upright_not_rolled_and_not_yaw_rotated_user_confirmed":
        raise ValueError("apparatus.sensor_mount_status must record the user-confirmed fixed orientation")
    if "reaction" not in str(apparatus.get("sensor_axes_frame", "")):
        raise ValueError("apparatus.sensor_axes_frame must identify the raw reaction wrench")
    if _finite(apparatus.get("sensor_reaction_sign_to_model_wrench"), "apparatus.sensor_reaction_sign_to_model_wrench") != -1.0:
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
    if not np.allclose(vertical, rotation_x(math.pi / 2.0), rtol=0.0, atol=1e-12):
        raise ValueError("vertical body-to-apparatus rotation must be FLU Rx(+pi/2) only")
    _finite(
        apparatus.get("observed_vertical_encoder_initial_offset_deg"),
        "apparatus.observed_vertical_encoder_initial_offset_deg",
    )
    if apparatus.get("observed_vertical_encoder_offset_interpretation") != "encoder_zero_offset_not_body_yaw_user_confirmed":
        raise ValueError("the observed vertical encoder offset must not be used as a body yaw rotation")
    if bool(apparatus.get("pitch_hydrostatic_restoring_in_fit")):
        raise ValueError("rolled-model pitch data must not include normal-attitude hydrostatic restoring")


def _validate_timing_quality_fit(
    timing: Mapping[str, Any],
    quality: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> None:
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
    load_r2 = _finite(quality.get("minimum_load_fit_r2_for_flag"), "quality.minimum_load_fit_r2_for_flag")
    if not 0.0 <= load_r2 <= 1.0:
        raise ValueError("quality.minimum_load_fit_r2_for_flag must be within [0, 1]")
    if fit.get("method") != "trial_harmonic_projection_then_per_frequency_huber":
        raise ValueError("fit.method must select per-frequency harmonic projection")
    harmonics = fit.get("load_harmonics")
    if not isinstance(harmonics, list) or not harmonics or any(int(k) != k or int(k) < 1 for k in harmonics):
        raise ValueError("fit.load_harmonics must be a non-empty list of positive integers")
    _positive(fit.get("huber_k"), "fit.huber_k")


def _validate_completion(completion: Mapping[str, Any]) -> None:
    for dof_name in ("surge_u", "roll_p"):
        values = _require_mapping(completion.get(dof_name), f"full_6d_completion.{dof_name}")
        for coefficient in ("added_mass", "linear_damping", "quadratic_damping"):
            value = _finite(values.get(coefficient), f"full_6d_completion.{dof_name}.{coefficient}")
            if value < 0.0:
                raise ValueError(f"full_6d_completion.{dof_name}.{coefficient} must be non-negative")


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
        _validate_sampling_and_frequency(self.sampling, self.frequency_estimation)
        _validate_model(self.model)
        _validate_apparatus(self.apparatus)
        _validate_timing_quality_fit(self.timing, self.quality, self.fit)
        _validate_completion(self.full_6d_completion)

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

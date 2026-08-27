"""Prescribed single-DOF motion metadata and body-frame wrench conversion."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .forces import ForceSeries, load_case_forces


DOF_NAMES = ("u", "v", "w", "p", "q", "r")
WRENCH_NAMES = ("X", "Y", "Z", "K", "M", "N")
_DOF_INDEX = {name: index for index, name in enumerate(DOF_NAMES)}


def _vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values, got {value!r}")
    return vector


@dataclass(frozen=True)
class MotionSpec:
    case_name: str
    case_family: str
    dof: str
    dof_index: int
    motion_kind: str
    axis: np.ndarray
    amplitude_si: float
    omega_rad_s: float
    phase_rad: float = 0.0
    ramp_duration_s: float = 0.0
    settle_cycles: float = 0.0
    sample_cycles: float | None = None
    cofr_global_m: np.ndarray | None = None
    com_initial_global_m: np.ndarray | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        axis = _vector3(self.axis, "axis")
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("axis must be nonzero")
        axis = axis / norm
        if self.dof not in _DOF_INDEX or self.dof_index != _DOF_INDEX[self.dof]:
            raise ValueError(f"Inconsistent dof/dof_index: {self.dof!r}, {self.dof_index!r}")
        expected_kind = "translation" if self.dof_index < 3 else "rotation"
        if self.motion_kind != expected_kind:
            raise ValueError(f"DOF {self.dof!r} requires motion_kind={expected_kind!r}")
        local_axis = self.dof_index if self.dof_index < 3 else self.dof_index - 3
        off_axis = np.delete(axis, local_axis)
        if abs(abs(float(axis[local_axis])) - 1.0) > 1.0e-8 or np.linalg.norm(off_axis) > 1.0e-8:
            raise ValueError(f"Single-DOF case {self.dof!r} has incompatible axis {axis.tolist()}")
        if not math.isfinite(self.amplitude_si) or self.amplitude_si <= 0.0:
            raise ValueError("amplitude_si must be finite and positive")
        if not math.isfinite(self.omega_rad_s) or self.omega_rad_s <= 0.0:
            raise ValueError("omega_rad_s must be finite and positive")
        if self.settle_cycles < 0.0 or (self.sample_cycles is not None and self.sample_cycles <= 0.0):
            raise ValueError("settle_cycles must be non-negative and sample_cycles must be positive")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(
            self,
            "cofr_global_m",
            _vector3(self.cofr_global_m if self.cofr_global_m is not None else (0, 0, 0), "cofr_global_m"),
        )
        object.__setattr__(
            self,
            "com_initial_global_m",
            _vector3(
                self.com_initial_global_m if self.com_initial_global_m is not None else (0, 0, 0),
                "com_initial_global_m",
            ),
        )

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.omega_rad_s

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source_path: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "MotionSpec":
        merged = dict(data)
        if overrides:
            for key in ("settle_cycles", "sample_cycles"):
                if key in overrides:
                    merged[key] = overrides[key]
        if merged["schema_version"] != 5:
            raise ValueError("Only preliminary case schema_version 5 is supported")
        dof = str(merged["dof"]).lower()
        index = int(merged["dof_index"])
        kind = str(merged["kind"]).lower()
        amplitude = float(
            merged["amplitude_m"] if kind == "translation" else merged["amplitude_rad"]
        )
        return cls(
            case_name=str(merged["case_name"]),
            case_family=str(merged["case_family"]),
            dof=dof,
            dof_index=index,
            motion_kind=kind,
            axis=merged["axis"],
            amplitude_si=amplitude,
            omega_rad_s=float(merged["omega_rad_s"]),
            phase_rad=float(merged["phase_rad"]),
            ramp_duration_s=float(merged["ramp_end_s"]),
            settle_cycles=float(merged["settle_cycles"]),
            sample_cycles=float(merged["sample_cycles"]),
            cofr_global_m=merged["centre_of_rotation_m"],
            com_initial_global_m=merged["com_initial_global_m"],
            source_path=source_path,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> "MotionSpec":
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, Mapping):
            raise ValueError(f"{source} must contain a JSON object")
        return cls.from_mapping(data, source_path=str(source), overrides=overrides)

    def kinematics(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return scalar generalized displacement, velocity and acceleration."""

        time = np.asarray(time_s, dtype=float)
        argument = self.omega_rad_s * time + self.phase_rad
        ramp = np.ones_like(time)
        ramp_rate = np.zeros_like(time)
        ramp_acceleration = np.zeros_like(time)
        if self.ramp_duration_s > 0.0:
            active = time < self.ramp_duration_s
            x = np.clip(time[active] / self.ramp_duration_s, 0.0, 1.0)
            ramp[active] = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
            ramp_rate[active] = (
                30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4
            ) / self.ramp_duration_s
            ramp_acceleration[active] = (
                60.0 * x - 180.0 * x**2 + 120.0 * x**3
            ) / self.ramp_duration_s**2
        direction = float(self.axis[self.dof_index if self.dof_index < 3 else self.dof_index - 3])
        sine = np.sin(argument)
        cosine = np.cos(argument)
        eta = direction * self.amplitude_si * ramp * sine
        nu = direction * self.amplitude_si * (
            ramp_rate * sine + ramp * self.omega_rad_s * cosine
        )
        nudot = direction * self.amplitude_si * (
            ramp_acceleration * sine
            + 2.0 * ramp_rate * self.omega_rad_s * cosine
            - ramp * self.omega_rad_s**2 * sine
        )
        return eta, nu, nudot


def axis_angle_rotation(axis: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    """Return body-to-global rotation matrices for a fixed axis-angle motion."""

    axis = _vector3(axis, "axis")
    axis = axis / np.linalg.norm(axis)
    angle = np.asarray(angle_rad, dtype=float).reshape(-1)
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    outer = np.outer(axis, axis)
    identity = np.eye(3)
    cosine = np.cos(angle)[:, None, None]
    sine = np.sin(angle)[:, None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


@dataclass(frozen=True)
class CaseData:
    case_dir: str
    motion: MotionSpec
    time_s: np.ndarray
    eta: np.ndarray
    nu: np.ndarray
    nudot: np.ndarray
    wrench_body: np.ndarray
    force_series: ForceSeries


@dataclass(frozen=True)
class SteadyCaseData:
    case_dir: str
    case_name: str
    case_family: str
    dof: str
    dof_index: int
    body_velocity_b_m_s: np.ndarray
    settle_end_s: float
    end_time_s: float
    time_s: np.ndarray
    wrench_body: np.ndarray
    force_series: ForceSeries


def transform_wrench_to_body(
    force_series: ForceSeries,
    motion: MotionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shift global moments to the moving COM and rotate the wrench to FLU."""

    time = force_series.time_s
    scalar_eta, scalar_nu, scalar_nudot = motion.kinematics(time)
    eta = np.zeros((time.size, 6), dtype=float)
    nu = np.zeros_like(eta)
    nudot = np.zeros_like(eta)
    eta[:, motion.dof_index] = scalar_eta
    nu[:, motion.dof_index] = scalar_nu
    nudot[:, motion.dof_index] = scalar_nudot

    if motion.motion_kind == "translation":
        rotations = np.broadcast_to(np.eye(3), (time.size, 3, 3))
        com = motion.com_initial_global_m + scalar_eta[:, None] * motion.axis
    else:
        rotations = axis_angle_rotation(motion.axis, scalar_eta)
        initial_lever = motion.com_initial_global_m - motion.cofr_global_m
        com = motion.cofr_global_m + np.einsum("nij,j->ni", rotations, initial_lever)

    lever = com - motion.cofr_global_m
    moment_com_global = force_series.moment_global - np.cross(lever, force_series.force_global)
    force_body = np.einsum("nji,nj->ni", rotations, force_series.force_global)
    moment_body = np.einsum("nji,nj->ni", rotations, moment_com_global)
    return eta, nu, nudot, np.concatenate((force_body, moment_body), axis=1)


def load_case_data(
    case_dir: str | Path,
    *,
    config_overrides: Mapping[str, Any] | None = None,
) -> CaseData:
    root = Path(case_dir)
    motion = MotionSpec.from_json(root / "case.json", overrides=config_overrides)
    forces = load_case_forces(root)
    eta, nu, nudot, wrench = transform_wrench_to_body(forces, motion)
    return CaseData(str(root), motion, forces.time_s, eta, nu, nudot, wrench, forces)


def load_steady_case_data(case_dir: str | Path) -> SteadyCaseData:
    root = Path(case_dir)
    source = root / "case.json"
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping) or data.get("schema_version") != 5:
        raise ValueError(f"{source} must use schema_version 5")
    if data.get("case_family") != "steady_damping" or data.get("kind") != "steady_translation":
        raise ValueError(f"{source} is not a steady translational damping case")
    dof = str(data["dof"])
    dof_index = int(data["dof_index"])
    if dof not in _DOF_INDEX or dof_index != _DOF_INDEX[dof] or dof_index >= 3:
        raise ValueError(f"{source}: invalid steady translation DOF")
    velocity = np.asarray(data["body_velocity_b_m_s"], dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError(f"{source}: body_velocity_b_m_s must contain three finite values")
    off_axis = np.delete(velocity, dof_index)
    if velocity[dof_index] == 0.0 or np.any(np.abs(off_axis) > 1.0e-12):
        raise ValueError(f"{source}: steady case must excite exactly its declared DOF")
    settle_end = float(data["settle_end_s"])
    end_time = float(data["end_time_s"])
    if not 0.0 < settle_end < end_time:
        raise ValueError(f"{source}: invalid steady analysis time window")
    forces = load_case_forces(root)
    wrench = np.concatenate((forces.force_global, forces.moment_global), axis=1)
    return SteadyCaseData(
        str(root),
        str(data["case_name"]),
        str(data["case_family"]),
        dof,
        dof_index,
        velocity,
        settle_end,
        end_time,
        forces.time_s,
        wrench,
        forces,
    )

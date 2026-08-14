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


def _first(mapping: Mapping[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class MotionSpec:
    case_name: str
    dof: str
    dof_index: int
    motion_kind: str
    axis: np.ndarray
    amplitude_si: float
    omega_rad_s: float
    phase_rad: float = 0.0
    settle_cycles: float = 0.0
    sample_cycles: float | None = None
    cofr_global_m: np.ndarray | None = None
    com_initial_global_m: np.ndarray | None = None
    background_fluid_velocity_body_m_s: np.ndarray | None = None
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
        object.__setattr__(
            self,
            "background_fluid_velocity_body_m_s",
            _vector3(
                self.background_fluid_velocity_body_m_s
                if self.background_fluid_velocity_body_m_s is not None
                else (0, 0, 0),
                "background_fluid_velocity_body_m_s",
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
        dof_value = _first(merged, ("dof", "dof_name"))
        index_value = _first(merged, ("dof_index", "axis_index"))
        if dof_value is None and index_value is None:
            raise ValueError("motion metadata requires dof or dof_index")
        if dof_value is None:
            index = int(index_value)
            if not 0 <= index < 6:
                raise ValueError(f"dof_index must be in [0, 5], got {index}")
            dof = DOF_NAMES[index]
        else:
            dof = str(dof_value).lower()
            if dof not in _DOF_INDEX:
                raise ValueError(f"Unknown DOF {dof!r}; expected one of {DOF_NAMES}")
            index = _DOF_INDEX[dof]
            if index_value is not None and int(index_value) != index:
                raise ValueError(f"dof={dof!r} conflicts with dof_index={index_value}")
        kind = str(_first(merged, ("motion_kind", "kind"), "translation" if index < 3 else "rotation"))

        amplitude_key = next(
            (
                key
                for key in (
                    "amplitude_si",
                    "amplitude_rad",
                    "amplitude_m",
                    "amplitude_deg",
                    "amplitude",
                    "displacement_amplitude",
                    "angle_amplitude",
                )
                if key in merged and merged[key] is not None
            ),
            None,
        )
        if amplitude_key is None:
            raise ValueError("motion metadata requires amplitude_si")
        amplitude = float(merged[amplitude_key])
        inferred_units = {
            "amplitude_m": "m",
            "amplitude_rad": "rad",
            "amplitude_deg": "deg",
        }.get(amplitude_key, "m" if index < 3 else "rad")
        units = str(merged.get("amplitude_units", inferred_units)).lower()
        if amplitude_key != "amplitude_si" and units in ("deg", "degree", "degrees"):
            amplitude = math.radians(amplitude)
        expected_units = ("m", "meter", "metre", "meters", "metres") if index < 3 else ("rad", "radian", "radians", "deg", "degree", "degrees")
        if units not in expected_units:
            raise ValueError(f"Unexpected amplitude_units={units!r} for {kind}")

        omega_value = _first(merged, ("omega_rad_s", "omega", "angular_frequency_rad_s"))
        if omega_value is None:
            frequency = _first(merged, ("frequency_hz", "frequency"))
            if frequency is None:
                raise ValueError("motion metadata requires omega_rad_s or frequency_hz")
            omega_value = 2.0 * math.pi * float(frequency)
        default_axis = np.eye(3)[index if index < 3 else index - 3]
        return cls(
            case_name=str(merged.get("case_name", Path(source_path).parent.name if source_path else dof)),
            dof=dof,
            dof_index=index,
            motion_kind=kind,
            axis=_first(merged, ("axis", "motion_axis"), default_axis),
            amplitude_si=amplitude,
            omega_rad_s=float(omega_value),
            phase_rad=float(_first(merged, ("phase_rad", "phase"), 0.0)),
            settle_cycles=float(merged.get("settle_cycles", 0.0)),
            sample_cycles=None if merged.get("sample_cycles") is None else float(merged["sample_cycles"]),
            cofr_global_m=_first(
                merged,
                ("cofr_global_m", "centre_of_rotation_m", "center_of_rotation_m", "cofr", "CofR"),
                (0, 0, 0),
            ),
            com_initial_global_m=_first(merged, ("com_initial_global_m", "com_initial", "origin"), (0, 0, 0)),
            background_fluid_velocity_body_m_s=_first(
                merged,
                (
                    "background_fluid_velocity_body_m_s",
                    "background_velocity_body_m_s",
                    "background_velocity_m_s",
                ),
                (0, 0, 0),
            ),
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
        direction = float(self.axis[self.dof_index if self.dof_index < 3 else self.dof_index - 3])
        eta = direction * self.amplitude_si * np.sin(argument)
        nu = direction * self.amplitude_si * self.omega_rad_s * np.cos(argument)
        nudot = -direction * self.amplitude_si * self.omega_rad_s**2 * np.sin(argument)
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
        physical_angle = motion.amplitude_si * np.sin(motion.omega_rad_s * time + motion.phase_rad)
        rotations = axis_angle_rotation(motion.axis, physical_angle)
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
    motion = MotionSpec.from_json(root / "motion.json", overrides=config_overrides)
    forces = load_case_forces(root)
    eta, nu, nudot, wrench = transform_wrench_to_body(forces, motion)
    return CaseData(str(root), motion, forces.time_s, eta, nu, nudot, wrench, forces)

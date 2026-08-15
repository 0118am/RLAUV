"""Configuration validation and case matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

OPENFOAM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = OPENFOAM_ROOT / "config.json"
DEFAULT_TEMPLATE = OPENFOAM_ROOT / "case_template"
DEFAULT_OUTPUT = OPENFOAM_ROOT / "cases"

DOFS = {
    "u": (0, "translation", (1, 0, 0)),
    "v": (1, "translation", (0, 1, 0)),
    "w": (2, "translation", (0, 0, 1)),
    "p": (3, "rotation", (1, 0, 0)),
    "q": (4, "rotation", (0, 1, 0)),
    "r": (5, "rotation", (0, 0, 1)),
}


def _config_finite_vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers.")
    if any(type(item) not in (int, float) for item in value):
        raise ValueError(f"{name} must contain exactly three finite numbers.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain exactly three finite numbers.")
    return result


@dataclass(frozen=True)
class CaseSpec:
    name: str
    dof: str | None
    dof_index: int | None
    kind: str
    axis: tuple[int, int, int]
    amplitude_m: float | None
    amplitude_deg: float | None
    amplitude_rad: float | None
    frequency_hz: float | None
    purpose: str = "identification"


def _require_base_fields(cfg: dict[str, Any]) -> None:
    required = (
        "openfoam_version",
        "solver",
        "geometry_filename",
        "rho_kg_m3",
        "nu_m2_s",
        "centre_of_rotation_m",
        "translation_amplitudes_m",
        "rotation_amplitudes_deg",
        "frequencies_hz",
        "settle_cycles",
        "sample_cycles",
        "steps_per_cycle",
        "initial_delta_t_fraction",
        "block_mesh",
        "snappy",
        "max_co",
        "wall_function_blending",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    if cfg["openfoam_version"] != "v2512" or cfg["solver"] != "pimpleFoam":
        raise ValueError("This generator targets OpenCFD OpenFOAM v2512 pimpleFoam only.")


def _validate_frequencies(cfg: dict[str, Any]) -> None:
    for key in ("translation_amplitudes_m", "rotation_amplitudes_deg", "frequencies_hz"):
        if not cfg[key] or any(float(value) <= 0 for value in cfg[key]):
            raise ValueError(f"{key} must contain positive values.")
    by_dof = cfg.get("frequencies_hz_by_dof")
    if by_dof is None:
        return
    if not isinstance(by_dof, dict):
        raise ValueError("frequencies_hz_by_dof must be a JSON object.")
    unknown = sorted(set(by_dof) - set(DOFS))
    missing = sorted(set(DOFS) - set(by_dof))
    if unknown or missing:
        raise ValueError(
            "frequencies_hz_by_dof must contain exactly u,v,w,p,q,r; "
            f"unknown={unknown}, missing={missing}"
        )
    for dof, values in by_dof.items():
        valid = (
            isinstance(values, list)
            and bool(values)
            and all(
                type(value) in (int, float) and math.isfinite(value) and value > 0.0
                for value in values
            )
        )
        if not valid:
            raise ValueError(f"frequencies_hz_by_dof.{dof} must contain positive finite numbers.")


def _validate_solver_controls(cfg: dict[str, Any]) -> None:
    for key in ("rho_kg_m3", "nu_m2_s"):
        if float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in (
        "steps_per_cycle",
        "writes_per_cycle",
        "purge_write",
        "gamg_update_interval",
        "pimple_outer_correctors",
        "force_execute_interval",
    ):
        value = cfg.get(key, 4) if key in {"writes_per_cycle", "purge_write"} else cfg[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    initial_fraction = cfg["initial_delta_t_fraction"]
    if (
        type(initial_fraction) not in (int, float)
        or not math.isfinite(initial_fraction)
        or not 0.0 < initial_fraction <= 1.0
    ):
        raise ValueError("initial_delta_t_fraction must be finite and lie in (0, 1].")
    max_co = cfg["max_co"]
    if type(max_co) not in (int, float) or not math.isfinite(max_co) or max_co <= 0:
        raise ValueError("max_co must be finite and positive.")
    if type(cfg["move_mesh_outer_correctors"]) is not bool:
        raise ValueError("move_mesh_outer_correctors must be a boolean.")
    if cfg.get("wall_function_blending") not in {"stepwise", "exponential"}:
        raise ValueError("wall_function_blending must be 'stepwise' or 'exponential'.")


def _validate_timeline(cfg: dict[str, Any]) -> None:
    cfg["background_velocity_m_s"] = list(
        _config_finite_vector(
            cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0)),
            "background_velocity_m_s",
        )
    )
    fixed_start = cfg.get("fixed_analysis_start_s")
    fixed_end = cfg.get("fixed_end_time_s")
    if (fixed_start is None) != (fixed_end is None):
        raise ValueError("fixed_analysis_start_s and fixed_end_time_s must be provided together.")
    if fixed_start is not None:
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (fixed_start, fixed_end)
        ):
            raise ValueError("fixed analysis times must be finite numbers.")
        if float(fixed_start) < 0.0 or float(fixed_end) <= float(fixed_start):
            raise ValueError("fixed times must satisfy 0 <= fixed_analysis_start_s < fixed_end_time_s.")

    convective_lengths = cfg.get("minimum_settle_convective_lengths", 0.0)
    if (
        type(convective_lengths) not in (int, float)
        or not math.isfinite(convective_lengths)
        or convective_lengths < 0.0
    ):
        raise ValueError("minimum_settle_convective_lengths must be finite and non-negative.")
    if convective_lengths <= 0.0:
        return
    characteristic_length = cfg.get("characteristic_length_m")
    if (
        type(characteristic_length) not in (int, float)
        or not math.isfinite(characteristic_length)
        or characteristic_length <= 0.0
    ):
        raise ValueError(
            "characteristic_length_m must be finite and positive when a convective settle time is requested."
        )
    if abs(float(cfg["background_velocity_m_s"][0])) <= 0.0:
        raise ValueError(
            "background_velocity_m_s x component must be nonzero when a convective settle time is requested."
        )


def _validate_block_mesh(cfg: dict[str, Any]) -> None:
    block_mesh = cfg["block_mesh"]
    if not isinstance(block_mesh, dict):
        raise ValueError("block_mesh must be a JSON object.")
    expected = {"domain_min", "domain_max", "base_cells"}
    missing = sorted(expected - set(block_mesh))
    unknown = sorted(set(block_mesh) - expected)
    if missing:
        raise ValueError(f"Missing block_mesh keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Unknown block_mesh keys: {', '.join(unknown)}")
    domain_min = _config_finite_vector(block_mesh["domain_min"], "block_mesh.domain_min")
    domain_max = _config_finite_vector(block_mesh["domain_max"], "block_mesh.domain_max")
    if any(lower >= upper for lower, upper in zip(domain_min, domain_max, strict=True)):
        raise ValueError("block_mesh.domain_min must be strictly below domain_max on every axis.")
    base_cells = block_mesh["base_cells"]
    if (
        not isinstance(base_cells, (list, tuple))
        or len(base_cells) != 3
        or any(type(value) is not int or value < 1 for value in base_cells)
    ):
        raise ValueError("block_mesh.base_cells must contain exactly three positive integers.")


def _validate_snappy(cfg: dict[str, Any]) -> None:
    snappy = cfg["snappy"]
    if not isinstance(snappy, dict):
        raise ValueError("snappy must be a JSON object.")
    expected = {
        "max_local_cells",
        "max_global_cells",
        "add_layers",
        "n_surface_layers",
        "relative_sizes",
        "expansion_ratio",
        "final_layer_thickness",
        "min_thickness",
        "n_grow",
        "n_buffer_cells_no_extrude",
    }
    missing = sorted(expected - set(snappy))
    unknown = sorted(set(snappy) - expected)
    if missing:
        raise ValueError(f"Missing snappy keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Unknown snappy keys: {', '.join(unknown)}")
    for key in ("max_local_cells", "max_global_cells"):
        if type(snappy[key]) is not int or snappy[key] < 1:
            raise ValueError(f"snappy.{key} must be a positive integer.")
    if snappy["max_global_cells"] < snappy["max_local_cells"]:
        raise ValueError("snappy.max_global_cells must be at least max_local_cells.")
    for key in ("add_layers", "relative_sizes"):
        if type(snappy[key]) is not bool:
            raise ValueError(f"snappy.{key} must be a boolean.")
    for key in ("n_surface_layers", "n_grow", "n_buffer_cells_no_extrude"):
        if type(snappy[key]) is not int or snappy[key] < 0:
            raise ValueError(f"snappy.{key} must be a non-negative integer.")
    if snappy["add_layers"] and snappy["n_surface_layers"] < 1:
        raise ValueError("snappy.n_surface_layers must be positive when layers are enabled.")
    for key in ("expansion_ratio", "final_layer_thickness", "min_thickness"):
        value = snappy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"snappy.{key} must be finite and positive.")
    if float(snappy["expansion_ratio"]) < 1.0:
        raise ValueError("snappy.expansion_ratio must be at least 1.")


def _validate_locked_rotor_mesh(cfg: dict[str, Any]) -> None:
    locked_mesh = cfg.get("locked_rotor_mesh", {})
    if not locked_mesh.get("enabled"):
        return
    for key in (
        "rotor_radius_m",
        "rotor_axial_length_m",
        "shaft_radius_m",
        "motor_radius_m",
        "motor_upstream_overlap_m",
        "near_field_radius_m",
    ):
        value = locked_mesh.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"locked_rotor_mesh.{key} must be finite and positive.")
    for key in ("rotor_level", "shaft_level", "motor_level", "near_field_level"):
        value = locked_mesh.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"locked_rotor_mesh.{key} must be a positive integer.")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    cfg.setdefault("move_mesh_outer_correctors", True)
    cfg.setdefault("gamg_update_interval", 1)
    cfg.setdefault("pimple_outer_correctors", 2)
    cfg.setdefault("force_execute_interval", 1)
    _require_base_fields(cfg)
    if len(cfg["centre_of_rotation_m"]) != 3:
        raise ValueError("centre_of_rotation_m must contain three values.")
    _validate_frequencies(cfg)
    _validate_solver_controls(cfg)
    _validate_timeline(cfg)
    _validate_block_mesh(cfg)
    _validate_snappy(cfg)
    _validate_locked_rotor_mesh(cfg)
    return cfg


def number_token(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def motion_specs(cfg: dict[str, Any]) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for dof, (index, kind, axis) in DOFS.items():
        amplitudes: Iterable[float]
        amplitudes = cfg["translation_amplitudes_m"] if kind == "translation" else cfg["rotation_amplitudes_deg"]
        frequencies = cfg.get("frequencies_hz_by_dof", {}).get(
            dof, cfg["frequencies_hz"]
        )
        for amplitude in amplitudes:
            for frequency in frequencies:
                amp = float(amplitude)
                freq = float(frequency)
                if kind == "translation":
                    name = f"{dof}_amp{number_token(amp, 3)}m_f{number_token(freq, 2)}hz"
                    amp_m, amp_deg, amp_rad = amp, None, None
                else:
                    name = f"{dof}_amp{number_token(amp, 1)}deg_f{number_token(freq, 2)}hz"
                    amp_m, amp_deg, amp_rad = None, amp, math.radians(amp)
                specs.append(CaseSpec(name, dof, index, kind, axis, amp_m, amp_deg, amp_rad, freq))
    return specs


def stationary_spec(name: str, purpose: str) -> CaseSpec:
    return CaseSpec(name, None, None, "baseline", (0, 0, 0), None, None, None, None, purpose)

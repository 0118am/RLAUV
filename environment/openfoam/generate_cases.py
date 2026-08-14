#!/usr/bin/env python3
"""Render OpenFOAM-v2512 forced-oscillation cases for all six body DOFs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_TEMPLATE = HERE / "case_template"
DEFAULT_OUTPUT = HERE / "cases"

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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--list", action="store_true", help="List case names without writing files.")
    p.add_argument("--dry-run", action="store_true", help="Print planned writes without changing files.")
    p.add_argument("--mesh-case-only", action="store_true", help="Render only one stationary mesh_case.")
    p.add_argument("--no-baseline", action="store_true", help="Omit the stationary baseline case.")
    p.add_argument("--force", action="store_true", help="Replace generated case directories that already exist.")
    p.add_argument("--geometry", type=Path, help="Override the metre-scaled STL source path.")
    p.add_argument("--geometry-mode", choices=("symlink", "copy", "none"), default="symlink")
    p.add_argument("--base-poly-mesh", type=Path, help="Existing constant/polyMesh shared by motion cases.")
    p.add_argument("--poly-mesh-mode", choices=("symlink", "copy", "none"), default="none")
    p.add_argument(
        "--repair-report",
        type=Path,
        help="STEP repair report containing measured locked-rotor axes and connector endpoints.",
    )
    return p


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    # Keep older reviewed configurations reproducible while allowing explicit
    # performance variants to opt into less frequent dynamic-mesh work.
    cfg.setdefault("move_mesh_outer_correctors", True)
    cfg.setdefault("gamg_update_interval", 1)
    cfg.setdefault("pimple_outer_correctors", 2)
    cfg.setdefault("force_execute_interval", 1)
    required = (
        "openfoam_version", "solver", "geometry_filename", "rho_kg_m3", "nu_m2_s",
        "centre_of_rotation_m", "translation_amplitudes_m", "rotation_amplitudes_deg",
        "frequencies_hz", "settle_cycles", "sample_cycles", "steps_per_cycle",
        "initial_delta_t_fraction",
        "block_mesh", "snappy", "max_co", "wall_function_blending",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    if cfg["openfoam_version"] != "v2512" or cfg["solver"] != "pimpleFoam":
        raise ValueError("This generator targets OpenCFD OpenFOAM v2512 pimpleFoam only.")
    if len(cfg["centre_of_rotation_m"]) != 3:
        raise ValueError("centre_of_rotation_m must contain three values.")
    for key in ("translation_amplitudes_m", "rotation_amplitudes_deg", "frequencies_hz"):
        if not cfg[key] or any(float(value) <= 0 for value in cfg[key]):
            raise ValueError(f"{key} must contain positive values.")
    frequencies_by_dof = cfg.get("frequencies_hz_by_dof")
    if frequencies_by_dof is not None:
        if not isinstance(frequencies_by_dof, dict):
            raise ValueError("frequencies_hz_by_dof must be a JSON object.")
        unknown_dofs = sorted(set(frequencies_by_dof) - set(DOFS))
        missing_dofs = sorted(set(DOFS) - set(frequencies_by_dof))
        if unknown_dofs or missing_dofs:
            raise ValueError(
                "frequencies_hz_by_dof must contain exactly u,v,w,p,q,r; "
                f"unknown={unknown_dofs}, missing={missing_dofs}"
            )
        for dof, values in frequencies_by_dof.items():
            if (
                not isinstance(values, list)
                or not values
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0.0
                    for value in values
                )
            ):
                raise ValueError(
                    f"frequencies_hz_by_dof.{dof} must contain positive finite numbers."
                )
    for key in ("rho_kg_m3", "nu_m2_s"):
        if float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if type(cfg["steps_per_cycle"]) is not int or cfg["steps_per_cycle"] < 1:
        raise ValueError("steps_per_cycle must be a positive integer.")
    initial_delta_t_fraction = cfg["initial_delta_t_fraction"]
    if (
        type(initial_delta_t_fraction) not in (int, float)
        or not math.isfinite(initial_delta_t_fraction)
        or not 0.0 < initial_delta_t_fraction <= 1.0
    ):
        raise ValueError("initial_delta_t_fraction must be finite and lie in (0, 1].")
    writes_per_cycle = cfg.get("writes_per_cycle", 4)
    if type(writes_per_cycle) is not int or writes_per_cycle < 1:
        raise ValueError("writes_per_cycle must be a positive integer.")
    purge_write = cfg.get("purge_write", 4)
    if type(purge_write) is not int or purge_write < 1:
        raise ValueError("purge_write must be a positive integer.")
    max_co = cfg["max_co"]
    if type(max_co) not in (int, float) or not math.isfinite(max_co) or max_co <= 0:
        raise ValueError("max_co must be finite and positive.")
    if type(cfg["move_mesh_outer_correctors"]) is not bool:
        raise ValueError("move_mesh_outer_correctors must be a boolean.")
    if type(cfg["gamg_update_interval"]) is not int or cfg["gamg_update_interval"] < 1:
        raise ValueError("gamg_update_interval must be a positive integer.")
    if type(cfg["pimple_outer_correctors"]) is not int or cfg["pimple_outer_correctors"] < 1:
        raise ValueError("pimple_outer_correctors must be a positive integer.")
    if type(cfg["force_execute_interval"]) is not int or cfg["force_execute_interval"] < 1:
        raise ValueError("force_execute_interval must be a positive integer.")
    background_velocity = _config_finite_vector(
        cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0)),
        "background_velocity_m_s",
    )
    cfg["background_velocity_m_s"] = list(background_velocity)
    fixed_start = cfg.get("fixed_analysis_start_s")
    fixed_end = cfg.get("fixed_end_time_s")
    if (fixed_start is None) != (fixed_end is None):
        raise ValueError(
            "fixed_analysis_start_s and fixed_end_time_s must be provided together."
        )
    if fixed_start is not None:
        for key, value in (
            ("fixed_analysis_start_s", fixed_start),
            ("fixed_end_time_s", fixed_end),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
        if float(fixed_start) < 0.0 or float(fixed_end) <= float(fixed_start):
            raise ValueError(
                "fixed times must satisfy 0 <= fixed_analysis_start_s < fixed_end_time_s."
            )
    convective_lengths = cfg.get("minimum_settle_convective_lengths", 0.0)
    if (
        type(convective_lengths) not in (int, float)
        or not math.isfinite(convective_lengths)
        or convective_lengths < 0.0
    ):
        raise ValueError("minimum_settle_convective_lengths must be finite and non-negative.")
    if convective_lengths > 0.0:
        characteristic_length = cfg.get("characteristic_length_m")
        if (
            type(characteristic_length) not in (int, float)
            or not math.isfinite(characteristic_length)
            or characteristic_length <= 0.0
        ):
            raise ValueError(
                "characteristic_length_m must be finite and positive when a convective settle time is requested."
            )
        towing_speed = abs(float(cfg["background_velocity_m_s"][0]))
        if towing_speed <= 0.0:
            raise ValueError(
                "background_velocity_m_s x component must be nonzero when a convective settle time is requested."
            )

    block_mesh = cfg["block_mesh"]
    if not isinstance(block_mesh, dict):
        raise ValueError("block_mesh must be a JSON object.")
    block_keys = {"domain_min", "domain_max", "base_cells"}
    block_missing = sorted(block_keys - set(block_mesh))
    if block_missing:
        raise ValueError(f"Missing block_mesh keys: {', '.join(block_missing)}")
    block_unknown = sorted(set(block_mesh) - block_keys)
    if block_unknown:
        raise ValueError(f"Unknown block_mesh keys: {', '.join(block_unknown)}")
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

    snappy = cfg["snappy"]
    if not isinstance(snappy, dict):
        raise ValueError("snappy must be a JSON object.")
    snappy_keys = {
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
    snappy_missing = sorted(snappy_keys - set(snappy))
    if snappy_missing:
        raise ValueError(f"Missing snappy keys: {', '.join(snappy_missing)}")
    snappy_unknown = sorted(set(snappy) - snappy_keys)
    if snappy_unknown:
        raise ValueError(f"Unknown snappy keys: {', '.join(snappy_unknown)}")
    for key in ("max_local_cells", "max_global_cells"):
        value = snappy[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"snappy.{key} must be a positive integer.")
    if snappy["max_global_cells"] < snappy["max_local_cells"]:
        raise ValueError("snappy.max_global_cells must be at least max_local_cells.")
    if type(snappy["add_layers"]) is not bool:
        raise ValueError("snappy.add_layers must be a boolean.")
    if type(snappy["relative_sizes"]) is not bool:
        raise ValueError("snappy.relative_sizes must be a boolean.")
    for key in ("n_surface_layers", "n_grow", "n_buffer_cells_no_extrude"):
        value = snappy[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"snappy.{key} must be a non-negative integer.")
    if snappy["add_layers"] and snappy["n_surface_layers"] < 1:
        raise ValueError("snappy.n_surface_layers must be positive when layers are enabled.")
    for key in ("expansion_ratio", "final_layer_thickness", "min_thickness"):
        value = snappy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"snappy.{key} must be finite and positive.")
    if float(snappy["expansion_ratio"]) < 1.0:
        raise ValueError("snappy.expansion_ratio must be at least 1.")

    blending = cfg.get("wall_function_blending")
    if blending not in {"stepwise", "exponential"}:
        raise ValueError(
            "wall_function_blending must be 'stepwise' or 'exponential'."
        )

    locked_mesh = cfg.get("locked_rotor_mesh", {})
    if locked_mesh.get("enabled"):
        positive = (
            "rotor_radius_m",
            "rotor_axial_length_m",
            "shaft_radius_m",
            "motor_radius_m",
            "motor_upstream_overlap_m",
            "near_field_radius_m",
        )
        levels = ("rotor_level", "shaft_level", "motor_level", "near_field_level")
        for key in positive:
            value = locked_mesh.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"locked_rotor_mesh.{key} must be finite and positive.")
        for key in levels:
            value = locked_mesh.get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f"locked_rotor_mesh.{key} must be a positive integer.")
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


def fmt(value: float) -> str:
    return f"{value:.12g}"


def foam_header(object_name: str, class_name: str = "dictionary") -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def _finite_vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector


def load_locked_rotor_report(path: Path, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the repair report used to derive local snappy refinement axes."""

    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("output_frame") != "body_flu_com":
        raise ValueError("repair report must use the body_flu_com output frame")
    if report.get("locked_rotor_condition") != "fully_assembled_static_locked":
        raise ValueError("repair report does not describe the fully assembled static-locked rotor")
    assemblies = report.get("locked_rotor_assemblies")
    if not isinstance(assemblies, list) or len(assemblies) != 8:
        raise ValueError("repair report must contain exactly eight locked rotor assemblies")
    expected_labels = {f"T{index}" for index in range(1, 9)}
    labels = {item.get("label") for item in assemblies if isinstance(item, dict)}
    if labels != expected_labels:
        raise ValueError("repair report locked rotors must contain unique T1--T8 labels")

    mesh = cfg["locked_rotor_mesh"]
    result: list[dict[str, Any]] = []
    for item in assemblies:
        label = str(item["label"])
        centre_mm = _finite_vector(
            item.get("propeller_centre_body_mm"), f"{label} propeller centre"
        )
        shaft_start_mm = _finite_vector(
            item.get("connector_axis_start_body_mm"), f"{label} connector start"
        )
        shaft_end_mm = _finite_vector(
            item.get("connector_axis_end_body_mm"), f"{label} connector end"
        )
        motor_end_mm = _finite_vector(
            item.get("motor_profile_axis_end_body_mm"), f"{label} motor profile end"
        )
        axis = _finite_vector(item.get("axis_direction_body"), f"{label} axis")
        magnitude = math.sqrt(sum(value * value for value in axis))
        if abs(magnitude - 1.0) > 1.0e-7:
            raise ValueError(f"{label} locked-rotor axis is not unit length")
        unit_axis = tuple(value / magnitude for value in axis)
        axial_projections: dict[str, float] = {}
        for name, first, second in (
            ("connector", shaft_start_mm, shaft_end_mm),
            ("motor profile", shaft_end_mm, motor_end_mm),
            ("propeller centre", shaft_end_mm, centre_mm),
        ):
            delta = tuple(b - a for a, b in zip(first, second, strict=True))
            projection = sum(value * direction for value, direction in zip(delta, unit_axis))
            axial_projections[name] = projection
            residual = math.sqrt(
                sum(
                    (value - projection * direction) ** 2
                    for value, direction in zip(delta, unit_axis)
                )
            )
            if residual > 1.0e-5:
                raise ValueError(f"{label} {name} is not collinear with its STEP-derived axis")
        if (
            axial_projections["connector"] <= 0.0
            or axial_projections["motor profile"] <= 0.0
            or axial_projections["propeller centre"] >= 0.0
        ):
            raise ValueError(f"{label} locked-rotor axial endpoint ordering is invalid")
        if item.get("representation") != "single_axisymmetric_smooth_motor_envelope":
            raise ValueError(f"{label} does not use the reviewed single-solid motor envelope")
        connector_radius_m = float(item.get("connector_radius_mm", math.nan)) * 0.001
        if not math.isfinite(connector_radius_m) or connector_radius_m <= 0.0:
            raise ValueError(f"{label} connector radius is invalid")
        if float(mesh["shaft_radius_m"]) <= connector_radius_m:
            raise ValueError(f"{label} shaft refinement radius must exceed connector radius")
        common = item.get("common_volume_mm3", {})
        minimum = item.get("minimum_common_volume_mm3", {})
        if set(common) != {"mount", "hub", "propeller"} or set(minimum) != set(common):
            raise ValueError(f"{label} motor-envelope common-volume evidence is incomplete")
        if any(float(common[name]) < float(minimum[name]) for name in common):
            raise ValueError(f"{label} motor envelope did not pass its common-volume gates")
        volume_error = float(item.get("source_volume_relative_error", math.nan))
        maximum_volume_error = float(
            item.get("maximum_source_volume_relative_error", math.nan)
        )
        if (
            not math.isfinite(volume_error)
            or not math.isfinite(maximum_volume_error)
            or volume_error < 0.0
            or volume_error > maximum_volume_error
        ):
            raise ValueError(f"{label} motor envelope did not pass its source-volume gate")
        result.append(
            {
                "label": label,
                "centre_m": tuple(value * 0.001 for value in centre_mm),
                "shaft_start_m": tuple(value * 0.001 for value in shaft_start_mm),
                "shaft_end_m": tuple(value * 0.001 for value in shaft_end_mm),
                "motor_end_m": tuple(value * 0.001 for value in motor_end_mm),
                "axis": unit_axis,
            }
        )
    return sorted(result, key=lambda item: int(item["label"][1:]))


def _offset(point: tuple[float, float, float], axis: tuple[float, float, float], distance: float):
    return tuple(value + distance * direction for value, direction in zip(point, axis, strict=True))


def _foam_vector(vector: tuple[float, float, float]) -> str:
    return "(" + " ".join(fmt(value) for value in vector) + ")"


def _replace_template_block(
    template: str,
    begin_marker: str,
    end_marker: str,
    replacement: str,
    description: str,
) -> str:
    if template.count(begin_marker) != 1 or template.count(end_marker) != 1:
        raise RuntimeError(f"{description} template markers changed")
    before, remainder = template.split(begin_marker, 1)
    _, after = remainder.split(end_marker, 1)
    return before + replacement + after


def render_block_mesh_dict(cfg: Mapping[str, Any]) -> str:
    """Render the outer-domain bounds and Cartesian base-cell counts."""

    template = (DEFAULT_TEMPLATE / "system" / "blockMeshDict").read_text(
        encoding="utf-8"
    )
    block_mesh = cfg["block_mesh"]
    lower = tuple(float(value) for value in block_mesh["domain_min"])
    upper = tuple(float(value) for value in block_mesh["domain_max"])
    vertices = (
        (lower[0], lower[1], lower[2]),
        (upper[0], lower[1], lower[2]),
        (upper[0], upper[1], lower[2]),
        (lower[0], upper[1], lower[2]),
        (lower[0], lower[1], upper[2]),
        (upper[0], lower[1], upper[2]),
        (upper[0], upper[1], upper[2]),
        (lower[0], upper[1], upper[2]),
    )
    template = _replace_template_block(
        template,
        "// __BLOCK_MESH_VERTICES_BEGIN__",
        "// __BLOCK_MESH_VERTICES_END__",
        "\n".join(f"    {_foam_vector(vertex)}" for vertex in vertices),
        "blockMeshDict vertices",
    )
    cells = " ".join(str(int(value)) for value in block_mesh["base_cells"])
    return _replace_template_block(
        template,
        "// __BLOCK_MESH_BLOCK_BEGIN__",
        "// __BLOCK_MESH_BLOCK_END__",
        f"    hex (0 1 2 3 4 5 6 7) ({cells}) simpleGrading (1 1 1)",
        "blockMeshDict block",
    )


def render_snappy_hex_mesh_dict(
    cfg: Mapping[str, Any], locked_rotors: list[dict[str, Any]] | None
) -> str:
    """Inject STEP-derived locked-rotor and isotropic near-field refinements."""

    template = (DEFAULT_TEMPLATE / "system" / "snappyHexMeshDict").read_text(
        encoding="utf-8"
    )
    snappy = cfg["snappy"]
    template = _replace_template_block(
        template,
        "// __SNAPPY_ADD_LAYERS_BEGIN__",
        "// __SNAPPY_ADD_LAYERS_END__",
        f"addLayers       {'true' if snappy['add_layers'] else 'false'};",
        "snappyHexMeshDict addLayers switch",
    )
    template = _replace_template_block(
        template,
        "// __SNAPPY_CELL_LIMITS_BEGIN__",
        "// __SNAPPY_CELL_LIMITS_END__",
        "    maxLocalCells       "
        f"{int(snappy['max_local_cells'])};\n"
        "    maxGlobalCells      "
        f"{int(snappy['max_global_cells'])};",
        "snappyHexMeshDict cell limits",
    )
    template = _replace_template_block(
        template,
        "    // __SNAPPY_LAYER_CONTROLS_BEGIN__",
        "    // __SNAPPY_LAYER_CONTROLS_END__",
        "    relativeSizes           "
        f"{'true' if snappy['relative_sizes'] else 'false'};\n"
        "    layers\n"
        "    {\n"
        "        auv { nSurfaceLayers "
        f"{int(snappy['n_surface_layers'])}; }}\n"
        "    }\n"
        f"    expansionRatio          {fmt(float(snappy['expansion_ratio']))};\n"
        f"    finalLayerThickness     {fmt(float(snappy['final_layer_thickness']))};\n"
        f"    minThickness            {fmt(float(snappy['min_thickness']))};\n"
        f"    nGrow                   {int(snappy['n_grow'])};\n"
        "    nBufferCellsNoExtrude   "
        f"{int(snappy['n_buffer_cells_no_extrude'])};",
        "snappyHexMeshDict layer controls",
    )
    geometry_marker = "// __LOCKED_ROTOR_GEOMETRY__"
    refinement_marker = "// __LOCKED_ROTOR_REFINEMENT__"
    if template.count(geometry_marker) != 1 or template.count(refinement_marker) != 1:
        raise RuntimeError("snappyHexMeshDict locked-rotor insertion markers changed")
    mesh = cfg.get("locked_rotor_mesh", {})
    if not mesh.get("enabled"):
        return template.replace(geometry_marker, "").replace(refinement_marker, "")
    if not locked_rotors:
        raise ValueError("locked rotor mesh refinement requires a STEP repair report")

    geometry_blocks: list[str] = []
    refinement_blocks: list[str] = []
    for item in locked_rotors:
        label = item["label"]
        axis = item["axis"]
        centre = item["centre_m"]
        rotor_half = 0.5 * float(mesh["rotor_axial_length_m"])
        definitions = (
            (
                f"rotor{label}",
                _offset(centre, axis, -rotor_half),
                _offset(centre, axis, rotor_half),
                float(mesh["rotor_radius_m"]),
                int(mesh["rotor_level"]),
            ),
            (
                f"shaft{label}",
                item["shaft_start_m"],
                item["shaft_end_m"],
                float(mesh["shaft_radius_m"]),
                int(mesh["shaft_level"]),
            ),
            (
                f"motor{label}",
                _offset(
                    item["shaft_end_m"],
                    axis,
                    -float(mesh["motor_upstream_overlap_m"]),
                ),
                item["motor_end_m"],
                float(mesh["motor_radius_m"]),
                int(mesh["motor_level"]),
            ),
        )
        for name, point1, point2, radius, level in definitions:
            geometry_blocks.append(
                f"    {name}\n"
                "    {\n"
                "        type    searchableCylinder;\n"
                f"        point1  {_foam_vector(point1)};\n"
                f"        point2  {_foam_vector(point2)};\n"
                f"        radius  {fmt(radius)};\n"
                "    }"
            )
            refinement_blocks.append(
                f"        {name}\n"
                "        {\n"
                "            mode inside;\n"
                f"            levels ((1e15 {level}));\n"
                "        }"
            )
        near_field_name = f"nearField{label}"
        geometry_blocks.append(
            f"    {near_field_name}\n"
            "    {\n"
            "        type    searchableSphere;\n"
            f"        centre  {_foam_vector(centre)};\n"
            f"        radius  {fmt(float(mesh['near_field_radius_m']))};\n"
            "    }"
        )
        refinement_blocks.append(
            f"        {near_field_name}\n"
            "        {\n"
            "            mode inside;\n"
            f"            levels ((1e15 {int(mesh['near_field_level'])}));\n"
            "        }"
        )
    return template.replace(geometry_marker, "\n".join(geometry_blocks)).replace(
        refinement_marker, "\n".join(refinement_blocks)
    )


def render_wall_function_field(name: str, cfg: Mapping[str, Any]) -> str:
    """Render a turbulence field with an explicit low/high-Re blender."""

    if name not in {"nut", "omega"}:
        raise ValueError(f"Unsupported wall-function field: {name}")
    template = (DEFAULT_TEMPLATE / "0" / name).read_text(encoding="utf-8")
    marker = "        // __WALL_FUNCTION_BLENDING__"
    if template.count(marker) != 1:
        raise RuntimeError(f"{name} wall-function blending marker changed")
    return template.replace(
        marker,
        f"        blending        {cfg['wall_function_blending']};",
    )


def render_velocity_field(cfg: Mapping[str, Any]) -> str:
    """Render a uniform towing-stream velocity without rotating the fixed far field."""

    velocity = _finite_vector(
        cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0)),
        "background_velocity_m_s",
    )
    foam_velocity = _foam_vector(velocity)
    return foam_header("U", "volVectorField") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform {foam_velocity};

boundaryField
{{
    auv
    {{
        type            movingWallVelocity;
        value           uniform (0 0 0);
    }}
    farField
    {{
        type            freestream;
        freestreamValue uniform {foam_velocity};
        value           uniform {foam_velocity};
    }}
}}
"""


def render_fv_solution(cfg: Mapping[str, Any]) -> str:
    """Render dynamic-mesh and GAMG performance controls explicitly."""

    template = (DEFAULT_TEMPLATE / "system" / "fvSolution").read_text(
        encoding="utf-8"
    )
    gamg_marker = "        // __GAMG_UPDATE_INTERVAL__"
    motion_marker = "    // __MOVE_MESH_OUTER_CORRECTORS__"
    outer_marker = "__PIMPLE_OUTER_CORRECTORS__"
    if template.count(gamg_marker) != 2:
        raise RuntimeError("fvSolution GAMG update markers changed")
    if template.count(motion_marker) != 1:
        raise RuntimeError("fvSolution dynamic-mesh marker changed")
    if template.count(outer_marker) != 1:
        raise RuntimeError("fvSolution PIMPLE outer-corrector marker changed")
    return template.replace(
        gamg_marker,
        f"        updateInterval  {int(cfg['gamg_update_interval'])};",
    ).replace(
        motion_marker,
        "    moveMeshOuterCorrectors "
        f"{'yes' if cfg['move_mesh_outer_correctors'] else 'no'};",
    ).replace(
        outer_marker,
        str(int(cfg["pimple_outer_correctors"])),
    )


def timeline(spec: CaseSpec, cfg: dict[str, Any]) -> dict[str, float]:
    if spec.frequency_hz is None:
        max_frequency = max(float(v) for v in cfg["frequencies_hz"])
        max_delta_t = 1.0 / (max_frequency * int(cfg["steps_per_cycle"]))
        return {
            "period_s": 0.0,
            "omega_rad_s": 0.0,
            "settle_end_s": float(cfg.get("baseline_settle_s", 0.5)),
            "end_time_s": float(cfg.get("baseline_duration_s", 2.0)),
            "delta_t_s": max_delta_t,
            "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
            "write_interval_s": 1.0 / max_frequency / int(cfg.get("writes_per_cycle", 4)),
        }
    period = 1.0 / spec.frequency_hz
    max_delta_t = period / int(cfg["steps_per_cycle"])
    fixed_start = cfg.get("fixed_analysis_start_s")
    fixed_end = cfg.get("fixed_end_time_s")
    if fixed_start is not None and fixed_end is not None:
        settle_end = float(fixed_start)
        end_time = float(fixed_end)
    else:
        settle_end = float(cfg["settle_cycles"]) * period
        convective_lengths = float(cfg.get("minimum_settle_convective_lengths", 0.0))
        if convective_lengths > 0.0:
            towing_speed = abs(float(cfg["background_velocity_m_s"][0]))
            convective_time = (
                convective_lengths * float(cfg["characteristic_length_m"]) / towing_speed
            )
            settle_end = max(settle_end, convective_time)
        end_time = settle_end + float(cfg["sample_cycles"]) * period
    return {
        "period_s": period,
        "omega_rad_s": 2.0 * math.pi * spec.frequency_hz,
        "settle_end_s": settle_end,
        "end_time_s": end_time,
        "delta_t_s": max_delta_t,
        "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
        "write_interval_s": period / int(cfg.get("writes_per_cycle", 4)),
    }


def render_point_displacement(spec: CaseSpec, cfg: dict[str, Any]) -> str:
    time = timeline(spec, cfg)
    if spec.kind == "translation":
        vector = tuple(spec.amplitude_m * component for component in spec.axis)  # type: ignore[operator]
        body = f"""        type            oscillatingDisplacement;
        amplitude       ({' '.join(fmt(v) for v in vector)});
        omega           {fmt(time['omega_rad_s'])};
        value           uniform (0 0 0);"""
    elif spec.kind == "rotation":
        origin = " ".join(fmt(float(v)) for v in cfg["centre_of_rotation_m"])
        body = f"""        type            angularOscillatingDisplacement;
        axis            ({' '.join(str(v) for v in spec.axis)});
        origin          ({origin});
        angle0          0;
        amplitude       {fmt(spec.amplitude_rad or 0.0)};
        omega           {fmt(time['omega_rad_s'])};
        value           uniform (0 0 0);"""
    else:
        body = """        type            fixedValue;
        value           uniform (0 0 0);"""
    return foam_header("pointDisplacement", "pointVectorField") + f"""
dimensions      [0 1 0 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    auv
    {{
{body}
    }}
    farField
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
}}
"""


def render_control_dict(spec: CaseSpec, cfg: dict[str, Any]) -> str:
    time = timeline(spec, cfg)
    origin = " ".join(fmt(float(v)) for v in cfg["centre_of_rotation_m"])
    purge_write = int(cfg.get("purge_write", 4))
    max_co = float(cfg["max_co"])
    return foam_header("controlDict") + f"""
application         pimpleFoam;
startFrom           startTime;
startTime           0;
stopAt              endTime;
endTime             {fmt(time['end_time_s'])};
deltaT              {fmt(time['initial_delta_t_s'])};
adjustTimeStep      yes;
maxCo               {fmt(max_co)};
maxDeltaT           {fmt(time['delta_t_s'])};
writeControl        adjustable;
writeInterval       {fmt(time['write_interval_s'])};
purgeWrite          {purge_write};
writeFormat         binary;
writePrecision      10;
writeCompression    off;
timeFormat          general;
timePrecision       10;
runTimeModifiable   true;

functions
{{
    forces
    {{
        type            forces;
        libs            (forces);
        executeControl  timeStep;
        executeInterval {int(cfg.get('force_execute_interval', 1))};
        writeControl    timeStep;
        writeInterval   {int(cfg.get('force_execute_interval', 1))};
        // Record through the settling interval as well.  The fitter uses the
        // samples bracketing settle_end_s to interpolate an exact full-cycle
        // phase grid; starting output at settle_end_s would lose that bracket
        // under adaptive time stepping.
        timeStart       0;
        log             true;
        patches         ({cfg.get('force_patch', 'auv')});
        rho             rhoInf;
        rhoInf          {fmt(float(cfg['rho_kg_m3']))};
        CofR            ({origin});
    }}
    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        executeControl  writeTime;
        writeControl    writeTime;
    }}
}}
"""


def render_transport_properties(cfg: dict[str, Any]) -> str:
    return foam_header("transportProperties") + f"""
transportModel  Newtonian;
nu              {fmt(float(cfg['nu_m2_s']))};
"""


def metadata(spec: CaseSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    time = timeline(spec, cfg)
    if spec.frequency_hz:
        settle_cycles = time["settle_end_s"] / time["period_s"]
        sample_cycles = (time["end_time_s"] - time["settle_end_s"]) / time["period_s"]
    else:
        settle_cycles = 0.0
        sample_cycles = 0.0
    return {
        "schema_version": 1,
        "openfoam_version": cfg["openfoam_version"],
        "solver": cfg["solver"],
        "case_name": spec.name,
        "dof": spec.dof,
        "dof_index": spec.dof_index,
        "kind": spec.kind,
        "axis": list(spec.axis),
        "amplitude_m": spec.amplitude_m,
        "amplitude_deg": spec.amplitude_deg,
        "amplitude_rad": spec.amplitude_rad,
        "frequency_hz": spec.frequency_hz,
        "omega_rad_s": time["omega_rad_s"],
        "period_s": time["period_s"],
        "settle_cycles": settle_cycles,
        "sample_cycles": sample_cycles,
        "settle_end_s": time["settle_end_s"],
        "end_time_s": time["end_time_s"],
        "delta_t_s": time["delta_t_s"],
        "initial_delta_t_s": time["initial_delta_t_s"],
        "max_co": float(cfg["max_co"]),
        "rho_kg_m3": float(cfg["rho_kg_m3"]),
        "nu_m2_s": float(cfg["nu_m2_s"]),
        "wall_function_blending": cfg["wall_function_blending"],
        "move_mesh_outer_correctors": bool(cfg["move_mesh_outer_correctors"]),
        "gamg_update_interval": int(cfg["gamg_update_interval"]),
        "pimple_outer_correctors": int(cfg.get("pimple_outer_correctors", 2)),
        "force_execute_interval": int(cfg.get("force_execute_interval", 1)),
        "centre_of_rotation_m": [float(v) for v in cfg["centre_of_rotation_m"]],
        "force_patch": cfg.get("force_patch", "auv"),
        "background_velocity_m_s": [
            float(value) for value in cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0))
        ],
        "background_fluid_velocity_body_m_s": [
            float(value) for value in cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0))
        ],
        "purpose": spec.purpose,
        "include_in_fit": spec.purpose == "identification",
    }


def attach(source: Path, destination: Path, mode: str) -> None:
    if mode == "none":
        return
    if not source.exists():
        raise FileNotFoundError(f"Required source does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()), target_is_directory=source.is_dir())


def render_case(
    spec: CaseSpec,
    cfg: dict[str, Any],
    output: Path,
    geometry: Path,
    args: argparse.Namespace,
    locked_rotors: list[dict[str, Any]] | None = None,
) -> None:
    case_dir = output / spec.name
    if case_dir.exists():
        if not args.force:
            raise FileExistsError(f"Case already exists (use --force): {case_dir}")
        shutil.rmtree(case_dir)
    shutil.copytree(DEFAULT_TEMPLATE, case_dir)
    if spec.purpose == "shared_mesh":
        (case_dir / "system" / "blockMeshDict").write_text(
            render_block_mesh_dict(cfg), encoding="utf-8"
        )
        (case_dir / "system" / "snappyHexMeshDict").write_text(
            render_snappy_hex_mesh_dict(cfg, locked_rotors), encoding="utf-8"
        )
    (case_dir / "0" / "pointDisplacement").write_text(render_point_displacement(spec, cfg), encoding="utf-8")
    (case_dir / "0" / "U").write_text(render_velocity_field(cfg), encoding="utf-8")
    for field_name in ("nut", "omega"):
        (case_dir / "0" / field_name).write_text(
            render_wall_function_field(field_name, cfg), encoding="utf-8"
        )
    (case_dir / "system" / "controlDict").write_text(render_control_dict(spec, cfg), encoding="utf-8")
    (case_dir / "system" / "fvSolution").write_text(
        render_fv_solution(cfg), encoding="utf-8"
    )
    (case_dir / "constant" / "transportProperties").write_text(render_transport_properties(cfg), encoding="utf-8")
    (case_dir / "motion.json").write_text(json.dumps(metadata(spec, cfg), indent=2) + "\n", encoding="utf-8")
    attach(geometry, case_dir / "constant" / "triSurface" / cfg["geometry_filename"], args.geometry_mode)
    if args.poly_mesh_mode != "none":
        if args.base_poly_mesh is None:
            raise ValueError("--base-poly-mesh is required when --poly-mesh-mode is not 'none'.")
        attach(args.base_poly_mesh, case_dir / "constant" / "polyMesh", args.poly_mesh_mode)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cfg = load_config(args.config.resolve())
    if args.mesh_case_only:
        specs = [stationary_spec("mesh_case", "shared_mesh")]
    else:
        specs = motion_specs(cfg)
        if not args.no_baseline:
            specs.append(stationary_spec("baseline", "stationary_tare"))
    if args.list:
        for spec in specs:
            print(spec.name)
        return 0
    locked_rotors = None
    if args.repair_report is not None:
        locked_rotors = load_locked_rotor_report(args.repair_report.resolve(), cfg)
    elif args.mesh_case_only and cfg.get("locked_rotor_mesh", {}).get("enabled"):
        raise ValueError(
            "--repair-report is required to derive locked-rotor mesh refinement axes"
        )
    output = args.output.resolve()
    geometry = args.geometry.resolve() if args.geometry else (HERE / cfg.get("geometry_path", "geometry/auv_visual_m.stl")).resolve()
    if args.dry_run:
        print(f"output={output}")
        print(f"geometry={geometry} mode={args.geometry_mode}")
        if args.base_poly_mesh:
            print(f"base_poly_mesh={args.base_poly_mesh.resolve()} mode={args.poly_mesh_mode}")
        for spec in specs:
            print(f"would generate {output / spec.name}")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        render_case(spec, cfg, output, geometry, args, locked_rotors)
        print(f"generated {output / spec.name}")
    manifest = {"schema_version": 1, "case_count": len(specs), "cases": [metadata(s, cfg) for s in specs]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

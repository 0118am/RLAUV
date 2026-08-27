"""Render the single-wall, no-prism-layer shared mesh."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from environment.openfoam.case_generation.config import DEFAULT_TEMPLATE
from environment.openfoam.case_generation.formatting import (
    _finite_vector,
    foam_vector,
    fmt,
)


def load_locked_rotor_report(path: Path) -> list[dict[str, Any]]:
    """Load the eight repaired rotor axes used only for local refinement."""

    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("output_frame") != "body_flu_com":
        raise ValueError("repair report must use the body_flu_com output frame")
    assemblies = report.get("locked_rotor_assemblies")
    if not isinstance(assemblies, list) or len(assemblies) != 8:
        raise ValueError("repair report must contain exactly eight locked rotors")
    expected = {f"T{index}" for index in range(1, 9)}
    labels = {
        item.get("label") for item in assemblies if isinstance(item, dict)
    }
    if labels != expected:
        raise ValueError("repair report must contain unique T1--T8 labels")

    result: list[dict[str, Any]] = []
    for item in assemblies:
        label = str(item["label"])
        centre = _finite_vector(
            item.get("propeller_centre_body_mm"), f"{label} propeller centre"
        )
        shaft_start = _finite_vector(
            item.get("connector_axis_start_body_mm"), f"{label} connector start"
        )
        shaft_end = _finite_vector(
            item.get("connector_axis_end_body_mm"), f"{label} connector end"
        )
        motor_end = _finite_vector(
            item.get("motor_profile_axis_end_body_mm"), f"{label} motor profile end"
        )
        axis = _finite_vector(item.get("axis_direction_body"), f"{label} axis")
        norm = math.sqrt(sum(value * value for value in axis))
        if abs(norm - 1.0) > 1.0e-7:
            raise ValueError(f"{label} axis is not unit length")
        result.append(
            {
                "label": label,
                "centre_m": tuple(value * 0.001 for value in centre),
                "shaft_start_m": tuple(value * 0.001 for value in shaft_start),
                "shaft_end_m": tuple(value * 0.001 for value in shaft_end),
                "motor_end_m": tuple(value * 0.001 for value in motor_end),
                "axis": tuple(value / norm for value in axis),
            }
        )
    return sorted(result, key=lambda item: int(item["label"][1:]))


def _offset(
    point: tuple[float, float, float],
    axis: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float]:
    return tuple(
        value + distance * direction
        for value, direction in zip(point, axis)
    )


def _replace_block(
    template: str,
    begin: str,
    end: str,
    replacement: str,
    description: str,
) -> str:
    if template.count(begin) != 1 or template.count(end) != 1:
        raise RuntimeError(f"{description} template markers changed")
    before, remainder = template.split(begin, 1)
    _, after = remainder.split(end, 1)
    return before + replacement + after


def render_block_mesh_dict(cfg: Mapping[str, Any]) -> str:
    template = (DEFAULT_TEMPLATE / "system" / "blockMeshDict").read_text(
        encoding="utf-8"
    )
    block = cfg["block_mesh"]
    lower = tuple(float(value) for value in block["domain_min"])
    upper = tuple(float(value) for value in block["domain_max"])
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
    template = _replace_block(
        template,
        "// __BLOCK_MESH_VERTICES_BEGIN__",
        "// __BLOCK_MESH_VERTICES_END__",
        "\n".join(f"    {foam_vector(vertex)}" for vertex in vertices),
        "blockMesh vertices",
    )
    cells = " ".join(str(int(value)) for value in block["base_cells"])
    return _replace_block(
        template,
        "// __BLOCK_MESH_BLOCK_BEGIN__",
        "// __BLOCK_MESH_BLOCK_END__",
        f"    hex (0 1 2 3 4 5 6 7) ({cells}) simpleGrading (1 1 1)",
        "blockMesh block",
    )


def _cylinder_blocks(
    name: str,
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
    radius: float,
    level: int,
) -> tuple[str, str]:
    geometry = f"""    {name}
    {{
        type    searchableCylinder;
        point1  {foam_vector(point1)};
        point2  {foam_vector(point2)};
        radius  {fmt(radius)};
    }}"""
    refinement = f"""        {name}
        {{
            mode inside;
            levels ((1e15 {level}));
        }}"""
    return geometry, refinement


def _locked_rotor_blocks(
    locked_rotors: list[dict[str, Any]],
    mesh: Mapping[str, Any],
) -> tuple[str, str]:
    geometry: list[str] = []
    refinement: list[str] = []
    rotor_half = 0.5 * float(mesh["rotor_axial_length_m"])
    for item in locked_rotors:
        label = item["label"]
        axis = item["axis"]
        definitions = (
            (
                f"rotor{label}",
                _offset(item["centre_m"], axis, -rotor_half),
                _offset(item["centre_m"], axis, rotor_half),
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
        for definition in definitions:
            geometry_block, refinement_block = _cylinder_blocks(*definition)
            geometry.append(geometry_block)
            refinement.append(refinement_block)
        near_field = f"nearField{label}"
        geometry.append(
            f"""    {near_field}
    {{
        type    searchableSphere;
        centre  {foam_vector(item['centre_m'])};
        radius  {fmt(float(mesh['near_field_radius_m']))};
    }}"""
        )
        refinement.append(
            f"""        {near_field}
        {{
            mode inside;
            levels ((1e15 {int(mesh['near_field_level'])}));
        }}"""
        )
    return "\n".join(geometry), "\n".join(refinement)


def render_snappy_hex_mesh_dict(
    cfg: Mapping[str, Any],
    locked_rotors: list[dict[str, Any]] | None,
) -> str:
    """Render one castellated/snap pass; prism layers are intentionally absent."""

    template = (DEFAULT_TEMPLATE / "system" / "snappyHexMeshDict").read_text(
        encoding="utf-8"
    )
    replacements = {
        "__GEOMETRY_FILENAME__": str(cfg["geometry_filename"]),
        "__MAX_LOCAL_CELLS__": str(int(cfg["snappy"]["max_local_cells"])),
        "__MAX_GLOBAL_CELLS__": str(int(cfg["snappy"]["max_global_cells"])),
        "__SURFACE_LEVELS__": " ".join(
            [str(int(cfg["snappy"]["surface_level"]))] * 2
        ),
        "__NEAR_BODY_LEVEL__": str(int(cfg["snappy"]["near_body_level"])),
    }
    lower = tuple(float(value) for value in cfg["block_mesh"]["domain_min"])
    upper = tuple(float(value) for value in cfg["block_mesh"]["domain_max"])
    # This point lies far from the centred vehicle and inside the fluid domain.
    location = tuple(
        minimum + 0.837 * (maximum - minimum)
        for minimum, maximum in zip(lower, upper)
    )
    replacements["__LOCATION_IN_MESH__"] = foam_vector(location)
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise RuntimeError(f"snappy marker changed: {marker}")
        template = template.replace(marker, value)

    geometry_marker = "// __LOCKED_ROTOR_GEOMETRY__"
    refinement_marker = "// __LOCKED_ROTOR_REFINEMENT__"
    if template.count(geometry_marker) != 1 or template.count(refinement_marker) != 1:
        raise RuntimeError("locked-rotor snappy markers changed")
    mesh = cfg["locked_rotor_mesh"]
    if not mesh["enabled"]:
        return template.replace(geometry_marker, "").replace(refinement_marker, "")
    if not locked_rotors:
        raise ValueError("locked-rotor refinement requires the STEP repair report")
    geometry, refinement = _locked_rotor_blocks(locked_rotors, mesh)
    return template.replace(geometry_marker, geometry).replace(
        refinement_marker, refinement
    )

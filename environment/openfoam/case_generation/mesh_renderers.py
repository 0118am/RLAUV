"""Geometry report loading and mesh dictionary rendering."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from environment.openfoam.case_generation.config import DEFAULT_TEMPLATE
from environment.openfoam.case_generation.formatting import _finite_vector, fmt

def load_locked_rotor_report(path: Path) -> list[dict[str, Any]]:
    """Load the repaired rotor axes used for local snappy refinement."""

    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("output_frame") != "body_flu_com":
        raise ValueError("repair report must use the body_flu_com output frame")
    assemblies = report.get("locked_rotor_assemblies")
    if not isinstance(assemblies, list) or len(assemblies) != 8:
        raise ValueError("repair report must contain exactly eight locked rotor assemblies")
    expected_labels = {f"T{index}" for index in range(1, 9)}
    labels = {item.get("label") for item in assemblies if isinstance(item, dict)}
    if labels != expected_labels:
        raise ValueError("repair report locked rotors must contain unique T1--T8 labels")

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


def _render_snappy_controls(template: str, snappy: Mapping[str, Any]) -> str:
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
        f"    maxLocalCells       {int(snappy['max_local_cells'])};\n"
        f"    maxGlobalCells      {int(snappy['max_global_cells'])};",
        "snappyHexMeshDict cell limits",
    )
    controls = (
        f"    relativeSizes           {'true' if snappy['relative_sizes'] else 'false'};\n"
        "    layers\n"
        "    {\n"
        f"        auv {{ nSurfaceLayers {int(snappy['n_surface_layers'])}; }}\n"
        "    }\n"
        f"    expansionRatio          {fmt(float(snappy['expansion_ratio']))};\n"
        f"    finalLayerThickness     {fmt(float(snappy['final_layer_thickness']))};\n"
        f"    minThickness            {fmt(float(snappy['min_thickness']))};\n"
        f"    nGrow                   {int(snappy['n_grow'])};\n"
        f"    nBufferCellsNoExtrude   {int(snappy['n_buffer_cells_no_extrude'])};"
    )
    return _replace_template_block(
        template,
        "    // __SNAPPY_LAYER_CONTROLS_BEGIN__",
        "    // __SNAPPY_LAYER_CONTROLS_END__",
        controls,
        "snappyHexMeshDict layer controls",
    )


def _cylinder_blocks(
    name: str,
    point1: tuple[float, float, float],
    point2: tuple[float, float, float],
    radius: float,
    level: int,
) -> tuple[str, str]:
    geometry = (
        f"    {name}\n"
        "    {\n"
        "        type    searchableCylinder;\n"
        f"        point1  {_foam_vector(point1)};\n"
        f"        point2  {_foam_vector(point2)};\n"
        f"        radius  {fmt(radius)};\n"
        "    }"
    )
    refinement = (
        f"        {name}\n"
        "        {\n"
        "            mode inside;\n"
        f"            levels ((1e15 {level}));\n"
        "        }"
    )
    return geometry, refinement


def _rotor_cylinders(item: Mapping[str, Any], mesh: Mapping[str, Any]):
    label = item["label"]
    axis = item["axis"]
    centre = item["centre_m"]
    rotor_half = 0.5 * float(mesh["rotor_axial_length_m"])
    return (
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
            _offset(item["shaft_end_m"], axis, -float(mesh["motor_upstream_overlap_m"])),
            item["motor_end_m"],
            float(mesh["motor_radius_m"]),
            int(mesh["motor_level"]),
        ),
    )


def _locked_rotor_blocks(
    locked_rotors: list[dict[str, Any]],
    mesh: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    geometry_blocks: list[str] = []
    refinement_blocks: list[str] = []
    for item in locked_rotors:
        for definition in _rotor_cylinders(item, mesh):
            geometry, refinement = _cylinder_blocks(*definition)
            geometry_blocks.append(geometry)
            refinement_blocks.append(refinement)
        near_field = f"nearField{item['label']}"
        geometry_blocks.append(
            f"    {near_field}\n"
            "    {\n"
            "        type    searchableSphere;\n"
            f"        centre  {_foam_vector(item['centre_m'])};\n"
            f"        radius  {fmt(float(mesh['near_field_radius_m']))};\n"
            "    }"
        )
        refinement_blocks.append(
            f"        {near_field}\n"
            "        {\n"
            "            mode inside;\n"
            f"            levels ((1e15 {int(mesh['near_field_level'])}));\n"
            "        }"
        )
    return geometry_blocks, refinement_blocks


def render_snappy_hex_mesh_dict(
    cfg: Mapping[str, Any], locked_rotors: list[dict[str, Any]] | None
) -> str:
    """Inject STEP-derived locked-rotor and isotropic near-field refinements."""

    template = (DEFAULT_TEMPLATE / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    template = _render_snappy_controls(template, cfg["snappy"])
    geometry_marker = "// __LOCKED_ROTOR_GEOMETRY__"
    refinement_marker = "// __LOCKED_ROTOR_REFINEMENT__"
    if template.count(geometry_marker) != 1 or template.count(refinement_marker) != 1:
        raise RuntimeError("snappyHexMeshDict locked-rotor insertion markers changed")
    mesh = cfg.get("locked_rotor_mesh", {})
    if not mesh.get("enabled"):
        return template.replace(geometry_marker, "").replace(refinement_marker, "")
    if not locked_rotors:
        raise ValueError("locked rotor mesh refinement requires a STEP repair report")
    geometry_blocks, refinement_blocks = _locked_rotor_blocks(locked_rotors, mesh)
    return template.replace(geometry_marker, "\n".join(geometry_blocks)).replace(
        refinement_marker,
        "\n".join(refinement_blocks),
    )

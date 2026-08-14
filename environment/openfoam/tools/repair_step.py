#!/usr/bin/env python3
"""Prepare the verification-assembly STEP for a CFD surface wrap.

The source is a flattened STEP surface model without an assembly tree.  This
tool therefore applies a source-specific reviewed shell selection.  Propellers
and hubs are retained; detailed motors and known sealed-hull internal frame
parts are removed.  Each detailed motor is replaced by a reviewed smooth
axisymmetric casing/nose/locked-shaft envelope.
The externally wetted main pressure hull and annular end fittings are retained.
Two thin disks derived from the measured tube mouths represent the waterproof
end-seal boundary during the outside-flood voxel wrap without filling external
fitting recesses.
All other closed shells are retained, including holes represented by inner
face loops.  The result is an intersecting multi-solid STL intended for the
subsequent voxel-wrap step, not a final CFD surface by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path


OPENFOAM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = OPENFOAM_ROOT / "geometry" / "verification_assembly_repair.json"
DEFAULT_OCP_SITE = (
    OPENFOAM_ROOT / ".runtime" / "cadquery-ocp" / "site-packages"
)
EXPECTED_OCP_VERSION = "7.9.3.1"


def _load_ocp() -> str:
    if not DEFAULT_OCP_SITE.is_dir():
        raise SystemExit(
            "Pinned OCP runtime is unavailable. Run environment/openfoam/install_cad_tools.sh first."
        )
    sys.path.insert(0, str(DEFAULT_OCP_SITE))
    try:
        import OCP
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OCP is unavailable. Install the pinned CAD runtime with "
            "environment/openfoam/install_cad_tools.sh first."
        ) from exc
    actual_version = getattr(OCP, "__version__", "unknown")
    module_path = Path(OCP.__file__).resolve()
    if actual_version != EXPECTED_OCP_VERSION or not module_path.is_relative_to(
        DEFAULT_OCP_SITE.resolve()
    ):
        raise SystemExit(
            "Wrong OCP runtime loaded: "
            f"version={actual_version}, path={module_path}; expected "
            f"{EXPECTED_OCP_VERSION} below {DEFAULT_OCP_SITE.resolve()}"
        )
    return actual_version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_subshapes(shape, kind) -> int:
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    mapped = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, mapped)
    return mapped.Extent()


def _shell_map(shape):
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    mapped = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_ShapeEnum.TopAbs_SHELL, mapped)
    return mapped


def _read_step(path: Path):
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP reader failed with status {status}")
    transferred = reader.TransferRoots()
    if transferred < 1:
        raise RuntimeError("STEP contains no transferable roots")
    return reader.OneShape()


def _closed_solid(shell):
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Solid

    if not BRep_Tool.IsClosed_s(shell):
        return None
    solid = ShapeFix_Solid().SolidFromShell(shell)
    if solid.IsNull() or not BRepCheck_Analyzer(solid, True).IsValid():
        raise RuntimeError("A retained closed shell could not be made into a valid solid")
    return solid


def _dominant_motor_cylinder(shell, mount_extension_mm: float, mount_side: str):
    """Return a smooth cylinder matching the largest cylindrical motor face."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    candidates = []
    explorer = TopExp_Explorer(shell, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face, True)
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            candidates.append(
                {
                    "radius": cylinder.Radius(),
                    "axis": cylinder.Axis(),
                    "vmin": surface.FirstVParameter(),
                    "vmax": surface.LastVParameter(),
                }
            )
        explorer.Next()
    if not candidates:
        raise RuntimeError("Reviewed motor shell has no cylindrical face")

    radius = max(item["radius"] for item in candidates)
    dominant = [item for item in candidates if abs(item["radius"] - radius) < 1e-6]
    reference = dominant[0]
    axis = reference["axis"]
    origin = axis.Location()
    direction = axis.Direction()
    aligned = []
    for item in dominant:
        other = item["axis"]
        dot = direction.Dot(other.Direction())
        if abs(abs(dot) - 1.0) > 1e-7:
            continue
        delta = other.Location().XYZ() - origin.XYZ()
        if delta.Crossed(direction.XYZ()).Modulus() > 1e-5:
            continue
        offset = delta.Dot(direction.XYZ())
        if dot > 0:
            aligned.extend([offset + item["vmin"], offset + item["vmax"]])
        else:
            aligned.extend([offset - item["vmax"], offset - item["vmin"]])
    if not aligned:
        raise RuntimeError("Unable to consolidate dominant motor-cylinder faces")
    vmin, vmax = min(aligned), max(aligned)
    measured_height = vmax - vmin
    if not (15.0 <= radius <= 18.0 and 20.0 <= measured_height <= 40.0):
        raise RuntimeError(
            "Motor signature changed unexpectedly: "
            f"radius={radius:g}, height={measured_height:g}"
        )
    if not (0.0 <= mount_extension_mm <= 10.0):
        raise RuntimeError("Motor mount extension must be between 0 and 10 mm")
    if mount_side not in {"vmin", "vmax"}:
        raise RuntimeError(f"Unknown reviewed motor mount side: {mount_side!r}")
    height = measured_height + mount_extension_mm
    start_parameter = vmin - mount_extension_mm if mount_side == "vmin" else vmin
    start = origin.XYZ() + direction.XYZ() * start_parameter
    axis2 = gp_Ax2(
        gp_Pnt(start.X(), start.Y(), start.Z()),
        gp_Dir(direction.X(), direction.Y(), direction.Z()),
    )
    solid = BRepPrimAPI_MakeCylinder(axis2, radius, height).Solid()
    return solid, {
        "radius_mm": radius,
        "measured_main_casing_height_mm": measured_height,
        "height_mm": height,
        "mount_extension_mm": mount_extension_mm,
        "reviewed_mount_side": mount_side,
        "start_step_mm": [start.X(), start.Y(), start.Z()],
        "direction_step": [direction.X(), direction.Y(), direction.Z()],
    }


def _make_compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def _common_volume(first, second) -> float:
    """Return the solid intersection volume, failing closed on Boolean errors."""

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    operation = BRepAlgoAPI_Common(first, second)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError("OpenCascade failed to validate motor/mount intersection")
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(operation.Shape(), properties, True, False, False)
    return abs(float(properties.Mass()))


def _solid_volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, True, False, False)
    return abs(float(properties.Mass()))


EXPECTED_SOURCE_COM_BODY_MM = (-1.306, 0.061, 2.385)
EXPECTED_BODY_TRANSLATION_MM = tuple(-value for value in EXPECTED_SOURCE_COM_BODY_MM)


def _to_body_flu(shape, translation_body_mm):
    """Apply the STEP-axis permutation, then translate the measured COM to zero."""

    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    tx, ty, tz = (float(value) for value in translation_body_mm)
    transform = gp_Trsf()
    transform.SetValues(
        0.0, 0.0, 1.0, tx,
        1.0, 0.0, 0.0, ty,
        0.0, 1.0, 0.0, tz,
    )
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _bbox(shape) -> list[float]:
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return list(box.Get())


def _validate_buoyancy_material(shape, config: dict) -> dict:
    """Measure the preserved buoyancy solid without adding its volume separately."""

    expected_volume = float(config["expected_closed_solid_volume_mm3"])
    actual_volume = _solid_volume(shape)
    relative_error = abs(actual_volume - expected_volume) / expected_volume
    volume_tolerance = float(config["volume_relative_tolerance"])
    if relative_error > volume_tolerance:
        raise RuntimeError(
            "Main buoyancy-material volume changed: "
            f"expected {expected_volume:.12g} mm^3, got {actual_volume:.12g} mm^3 "
            f"(relative error {relative_error:.6%} > {volume_tolerance:.6%})"
        )

    expected_bbox = [float(value) for value in config["expected_bbox_step_mm"]]
    actual_bbox = _bbox(shape)
    bbox_errors = [
        abs(actual - expected)
        for actual, expected in zip(actual_bbox, expected_bbox, strict=True)
    ]
    bbox_tolerance = float(config["bbox_absolute_tolerance_mm"])
    if max(bbox_errors) > bbox_tolerance:
        raise RuntimeError(
            "Main buoyancy-material STEP placement changed: "
            f"maximum bbox error {max(bbox_errors):.12g} mm > "
            f"{bbox_tolerance:.12g} mm"
        )

    return {
        "source_shell_index": int(config["shell"]),
        "role": config["role"],
        "condition": config["condition"],
        "identification_status": config["identification_status"],
        "preserved": True,
        "actual_closed_solid_volume_mm3": actual_volume,
        "expected_closed_solid_volume_mm3": expected_volume,
        "volume_relative_error": relative_error,
        "volume_relative_tolerance": volume_tolerance,
        "actual_bbox_step_mm": actual_bbox,
        "expected_bbox_step_mm": expected_bbox,
        "bbox_max_absolute_error_mm": max(bbox_errors),
        "bbox_absolute_tolerance_mm": bbox_tolerance,
        "hydrodynamic_accounting": config["hydrodynamic_accounting"],
        "double_count_prevention": (
            "This preserved solid is already part of the output geometry union; "
            "its measured volume is not added numerically to the displacement target."
        ),
    }


def _body_point_from_step(
    point: list[float] | tuple[float, ...],
    translation_body_mm: list[float] | tuple[float, ...],
) -> list[float]:
    """Map a STEP point to the measured-COM-centred body-FLU frame."""

    mapped = [float(point[2]), float(point[0]), float(point[1])]
    return [
        value + float(offset)
        for value, offset in zip(mapped, translation_body_mm, strict=True)
    ]


def _body_direction_from_step(direction: list[float] | tuple[float, ...]) -> list[float]:
    """Apply only the cyclic STEP-to-body rotation to a direction vector."""

    mapped = [float(direction[2]), float(direction[0]), float(direction[1])]
    magnitude = math.sqrt(sum(value * value for value in mapped))
    if magnitude <= 0.0:
        raise RuntimeError("Cannot transform a zero motor-axis direction")
    return [value / magnitude for value in mapped]


def _coaxial_cylinder_records(shape, axis_point, axis_direction) -> list[dict]:
    """Return cylindrical-face radii and axial intervals on one reference axis."""

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Dir, gp_Pnt

    reference_point = gp_Pnt(*(float(value) for value in axis_point)).XYZ()
    reference_direction = gp_Dir(
        *(float(value) for value in axis_direction)
    ).XYZ()
    records: list[dict] = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face, True)
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            other = cylinder.Axis()
            dot = reference_direction.Dot(other.Direction().XYZ())
            delta = other.Location().XYZ() - reference_point
            radial_offset = delta.Crossed(reference_direction).Modulus()
            if abs(abs(dot) - 1.0) <= 1.0e-7 and radial_offset <= 1.0e-5:
                offset = delta.Dot(reference_direction)
                first = float(surface.FirstVParameter())
                last = float(surface.LastVParameter())
                if dot > 0.0:
                    bounds = (offset + first, offset + last)
                else:
                    bounds = (offset - last, offset - first)
                records.append(
                    {
                        "radius_mm": float(cylinder.Radius()),
                        "start_mm": float(min(bounds)),
                        "end_mm": float(max(bounds)),
                    }
                )
        explorer.Next()
    return records


def _longest_radius_interval(
    records: list[dict], radius_mm: float, tolerance_mm: float, description: str
) -> dict:
    matches = [
        item
        for item in records
        if abs(float(item["radius_mm"]) - radius_mm) <= tolerance_mm
    ]
    if not matches:
        raise RuntimeError(f"Missing reviewed {description} radius {radius_mm:g} mm")
    return max(matches, key=lambda item: float(item["end_mm"]) - float(item["start_mm"]))


def _sealed_pressure_boundary(
    hull_shell,
    hull_solid,
    endcap_solids: dict[int, object],
    config: dict,
    translation_body_mm,
) -> tuple[object, dict]:
    """Derive two thin waterproof patches at the measured pressure-tube mouths."""

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    cylinders = []
    explorer = TopExp_Explorer(hull_shell, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face, True)
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinders.append(surface.Cylinder())
        explorer.Next()
    if len(cylinders) < 2:
        raise RuntimeError("Reviewed pressure hull has fewer than two cylindrical faces")
    reference = max(cylinders, key=lambda cylinder: float(cylinder.Radius()))
    axis = reference.Axis()
    axis_point = [axis.Location().X(), axis.Location().Y(), axis.Location().Z()]
    axis_direction = [
        axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z()
    ]
    tolerance = float(config["signature_tolerance_mm"])
    records = _coaxial_cylinder_records(hull_shell, axis_point, axis_direction)
    inner = _longest_radius_interval(
        records,
        float(config["expected_inner_radius_mm"]),
        tolerance,
        "pressure-hull inner cylinder",
    )
    outer = _longest_radius_interval(
        records,
        float(config["expected_outer_radius_mm"]),
        tolerance,
        "pressure-hull outer cylinder",
    )
    inner_radius = float(inner["radius_mm"])
    outer_radius = float(outer["radius_mm"])
    hull_start = max(float(inner["start_mm"]), float(outer["start_mm"]))
    hull_end = min(float(inner["end_mm"]), float(outer["end_mm"]))
    hull_length = hull_end - hull_start
    expected_length = float(config["expected_hull_length_mm"])
    if abs(hull_length - expected_length) > tolerance:
        raise RuntimeError(
            "Reviewed pressure-hull length changed: "
            f"expected {expected_length:g} mm, got {hull_length:.12g} mm"
        )

    radial_overlap = float(config["radial_wall_overlap_mm"])
    half_thickness = float(config["disk_half_thickness_mm"])
    disk_radius = inner_radius + radial_overlap
    if not inner_radius < disk_radius < outer_radius:
        raise RuntimeError(
            "Derived pressure-seal disks must overlap but remain inside the tube wall"
        )
    direction_xyz = gp_Dir(*axis_direction).XYZ()
    origin_xyz = gp_Pnt(*axis_point).XYZ()
    disk_solids = {}
    disk_metadata = []
    common = {}
    endcap_by_end = {
        side: int(index) for side, index in config["endcap_by_hull_end"].items()
    }
    for side, centre_parameter in (("start", hull_start), ("end", hull_end)):
        disk_start_parameter = centre_parameter - half_thickness
        disk_end_parameter = centre_parameter + half_thickness
        disk_start_xyz = origin_xyz + direction_xyz * disk_start_parameter
        disk_end_xyz = origin_xyz + direction_xyz * disk_end_parameter
        disk_centre_xyz = origin_xyz + direction_xyz * centre_parameter
        disk = BRepPrimAPI_MakeCylinder(
            gp_Ax2(
                gp_Pnt(
                    disk_start_xyz.X(), disk_start_xyz.Y(), disk_start_xyz.Z()
                ),
                gp_Dir(*axis_direction),
            ),
            disk_radius,
            2.0 * half_thickness,
        ).Solid()
        disk_solids[side] = disk
        endcap_index = endcap_by_end[side]
        common[f"{side}_disk_hull"] = _common_volume(disk, hull_solid)
        common[f"{side}_disk_endcap"] = _common_volume(
            disk, endcap_solids[endcap_index]
        )
        disk_metadata.append(
            {
                "hull_end": side,
                "reviewed_endcap_shell": endcap_index,
                "centre_parameter_from_axis_origin_mm": centre_parameter,
                "centre_step_mm": [
                    disk_centre_xyz.X(),
                    disk_centre_xyz.Y(),
                    disk_centre_xyz.Z(),
                ],
                "centre_body_mm": _body_point_from_step(
                    [
                        disk_centre_xyz.X(),
                        disk_centre_xyz.Y(),
                        disk_centre_xyz.Z(),
                    ],
                    translation_body_mm,
                ),
                "axis_start_step_mm": [
                    disk_start_xyz.X(),
                    disk_start_xyz.Y(),
                    disk_start_xyz.Z(),
                ],
                "axis_end_step_mm": [
                    disk_end_xyz.X(), disk_end_xyz.Y(), disk_end_xyz.Z()
                ],
                "radius_mm": disk_radius,
                "thickness_mm": 2.0 * half_thickness,
                "solid_volume_mm3": _solid_volume(disk),
                "common_with_hull_mm3": common[f"{side}_disk_hull"],
                "common_with_reviewed_endcap_mm3": common[
                    f"{side}_disk_endcap"
                ],
            }
        )

    boundary = _make_compound(disk_solids.values())
    hull_endcap_evidence = {}
    for index, endcap in sorted(endcap_solids.items()):
        distance = BRepExtrema_DistShapeShape(hull_solid, endcap)
        distance.Perform()
        hull_endcap_evidence[str(index)] = {
            "distance_to_hull_mm": float(distance.Value()) if distance.IsDone() else None,
            "common_with_hull_mm3": _common_volume(hull_solid, endcap),
            "common_with_start_disk_mm3": _common_volume(
                disk_solids["start"], endcap
            ),
            "common_with_end_disk_mm3": _common_volume(
                disk_solids["end"], endcap
            ),
        }
    minimum_common = {
        name: float(value)
        for name, value in config["minimum_common_volume_mm3"].items()
    }
    failures = [
        f"{name} {common.get(name, -math.inf):.6g} < {minimum:.6g} mm^3"
        for name, minimum in minimum_common.items()
        if common.get(name, -math.inf) < minimum
    ]
    if failures:
        raise RuntimeError(
            "Sealed pressure-boundary overlap gate failed: " + "; ".join(failures)
        )

    return boundary, {
        "condition": config["condition"],
        "representation": config["representation"],
        "source_hull_shell": int(config["hull_shell"]),
        "source_endcap_shells": sorted(int(value) for value in config["endcap_shells"]),
        "measured_inner_radius_mm": inner_radius,
        "measured_outer_radius_mm": outer_radius,
        "measured_hull_length_mm": hull_length,
        "measured_hull_interval_from_axis_origin_mm": [hull_start, hull_end],
        "radial_wall_overlap_mm": radial_overlap,
        "disk_half_thickness_mm": half_thickness,
        "derived_disk_radius_mm": disk_radius,
        "derived_disk_thickness_mm": 2.0 * half_thickness,
        "derived_disks": disk_metadata,
        "axis_direction_step": axis_direction,
        "axis_direction_body": _body_direction_from_step(axis_direction),
        "nominal_cavity_volume_mm3": math.pi * inner_radius**2 * hull_length,
        "derived_boundary_patch_solid_volume_mm3": _solid_volume(boundary),
        "common_volume_mm3": common,
        "minimum_common_volume_mm3": minimum_common,
        "hull_endcap_original_contact": hull_endcap_evidence,
        "hydrodynamic_accounting": (
            "The disks close the waterproof boundary only; external annular-fitting "
            "recesses remain floodable and no cavity volume is numerically added."
        ),
    }


def _axisymmetric_profile_solid(axis_point, axis_direction, profile_points):
    """Revolve one closed axial/radial profile into a single valid solid."""

    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt

    origin = gp_Pnt(*(float(value) for value in axis_point))
    direction = gp_Dir(*(float(value) for value in axis_direction))
    axial = direction.XYZ()
    radial = gp_Ax2(origin, direction).XDirection().XYZ()
    polygon = BRepBuilderAPI_MakePolygon()
    for axial_mm, radius_mm in profile_points:
        xyz = origin.XYZ() + axial * float(axial_mm) + radial * float(radius_mm)
        polygon.Add(gp_Pnt(xyz.X(), xyz.Y(), xyz.Z()))
    polygon.Close()
    if not polygon.IsDone():
        raise RuntimeError("OpenCascade could not close the reviewed motor profile")
    face = BRepBuilderAPI_MakeFace(polygon.Wire())
    if not face.IsDone():
        raise RuntimeError("OpenCascade could not make the reviewed motor-profile face")
    revolved = BRepPrimAPI_MakeRevol(
        face.Face(), gp_Ax1(origin, direction), 2.0 * math.pi, True
    )
    revolved.Build()
    if not revolved.IsDone():
        raise RuntimeError("OpenCascade could not revolve the reviewed motor profile")
    solid = revolved.Shape()
    if (
        solid.IsNull()
        or solid.ShapeType() != TopAbs_ShapeEnum.TopAbs_SOLID
        or not BRepCheck_Analyzer(solid, True).IsValid()
    ):
        raise RuntimeError("Reviewed motor profile did not produce one valid solid")
    return solid


def _locked_rotor_envelope(
    motor_shell,
    mount_reference,
    hub_solid,
    propeller_solid,
    motor_metadata: dict,
    config: dict,
    translation_body_mm,
) -> tuple[object, dict]:
    """Build and validate one smooth casing/nose/locked-shaft motor solid."""

    from OCP.gp import gp_Dir, gp_Pnt

    axis_point = [float(value) for value in motor_metadata["start_step_mm"]]
    axis_direction = [float(value) for value in motor_metadata["direction_step"]]
    tolerance = float(config["signature_tolerance_mm"])
    nominal_radius = float(config["nominal_motor_shaft_radius_mm"])
    motor_records = _coaxial_cylinder_records(
        motor_shell, axis_point, axis_direction
    )
    hub_records = _coaxial_cylinder_records(
        hub_solid, axis_point, axis_direction
    )
    propeller_records = _coaxial_cylinder_records(
        propeller_solid, axis_point, axis_direction
    )
    motor_shaft = _longest_radius_interval(
        motor_records, nominal_radius, tolerance, "motor shaft"
    )
    propeller_bore = _longest_radius_interval(
        propeller_records, nominal_radius, tolerance, "propeller bore"
    )
    if not hub_records:
        raise RuntimeError("Reviewed locked-propeller hub has no coaxial cylinders")

    motor_shaft_length = motor_shaft["end_mm"] - motor_shaft["start_mm"]
    propeller_hub_length = propeller_bore["end_mm"] - propeller_bore["start_mm"]
    hub_start = min(item["start_mm"] for item in hub_records)
    hub_end = max(item["end_mm"] for item in hub_records)
    hub_length = hub_end - hub_start
    expected_signatures = (
        (motor_shaft_length, float(config["expected_motor_shaft_length_mm"]), "motor shaft"),
        (
            propeller_hub_length,
            float(config["expected_propeller_hub_length_mm"]),
            "propeller hub",
        ),
        (hub_length, float(config["expected_separate_hub_length_mm"]), "separate hub"),
    )
    for measured, expected, description in expected_signatures:
        if abs(measured - expected) > tolerance:
            raise RuntimeError(
                f"Reviewed {description} length changed: expected {expected:g} mm, "
                f"got {measured:.12g} mm"
            )
    if abs(hub_end - propeller_bore["start_mm"]) > tolerance:
        raise RuntimeError("Reviewed hub and propeller no longer meet at one axial plane")

    profile_points = [
        (float(point[0]), float(point[1]))
        for point in config["axisymmetric_profile_mm"]
    ]
    profile_start = profile_points[0][0]
    profile_end = profile_points[-1][0]
    expected_start = hub_start - float(config["shaft_tip_extension_mm"])
    if abs(profile_start - expected_start) > tolerance:
        raise RuntimeError(
            "Reviewed motor profile no longer starts at the expected hub-tip overlap"
        )
    if abs(profile_end - float(motor_metadata["height_mm"])) > tolerance:
        raise RuntimeError(
            "Reviewed motor profile no longer ends at the mount-side casing extension"
        )
    envelope = _axisymmetric_profile_solid(
        axis_point, axis_direction, profile_points
    )

    common = {
        "mount": _common_volume(envelope, mount_reference),
        "hub": _common_volume(envelope, hub_solid),
        "propeller": _common_volume(envelope, propeller_solid),
    }
    minimum_common = {
        name: float(value)
        for name, value in config["minimum_common_volume_mm3"].items()
    }
    failures = [
        f"{name} {common[name]:.6g} < {minimum:.6g} mm^3"
        for name, minimum in minimum_common.items()
        if common.get(name, -math.inf) < minimum
    ]
    if failures:
        raise RuntimeError(
            "Locked motor-envelope overlap gate failed: " + "; ".join(failures)
        )

    source_motor = _closed_solid(motor_shell)
    if source_motor is None:
        raise RuntimeError("Reviewed detailed motor shell is no longer a closed solid")
    source_volume = _solid_volume(source_motor)
    envelope_volume = _solid_volume(envelope)
    relative_volume_error = abs(envelope_volume - source_volume) / source_volume
    maximum_volume_error = float(config["maximum_source_volume_relative_error"])
    if relative_volume_error > maximum_volume_error:
        raise RuntimeError(
            "Smooth motor-envelope volume changed too much relative to the source motor: "
            f"{relative_volume_error:.6%} > {maximum_volume_error:.6%}"
        )

    direction_xyz = gp_Dir(*axis_direction).XYZ()
    shaft_start_parameter = profile_start
    shaft_end_parameter = float(config["shaft_refinement_end_mm"])
    axis_origin_xyz = gp_Pnt(*axis_point).XYZ()
    start_xyz = axis_origin_xyz + direction_xyz * shaft_start_parameter
    end_xyz = axis_origin_xyz + direction_xyz * shaft_end_parameter
    profile_end_xyz = axis_origin_xyz + direction_xyz * profile_end
    propeller_centre_parameter = 0.5 * (
        propeller_bore["start_mm"] + propeller_bore["end_mm"]
    )
    propeller_centre_xyz = (
        gp_Pnt(*axis_point).XYZ() + direction_xyz * propeller_centre_parameter
    )
    start_step = [start_xyz.X(), start_xyz.Y(), start_xyz.Z()]
    end_step = [end_xyz.X(), end_xyz.Y(), end_xyz.Z()]
    profile_end_step = [profile_end_xyz.X(), profile_end_xyz.Y(), profile_end_xyz.Z()]
    propeller_centre_step = [
        propeller_centre_xyz.X(),
        propeller_centre_xyz.Y(),
        propeller_centre_xyz.Z(),
    ]
    return envelope, {
        "condition": "static_locked",
        "representation": "single_axisymmetric_smooth_motor_envelope",
        "nominal_motor_shaft_radius_mm": nominal_radius,
        "connector_radius_mm": float(config["connector_radius_mm"]),
        "connector_length_mm": shaft_end_parameter - shaft_start_parameter,
        "connector_axis_start_step_mm": start_step,
        "connector_axis_end_step_mm": end_step,
        "connector_axis_start_body_mm": _body_point_from_step(
            start_step, translation_body_mm
        ),
        "connector_axis_end_body_mm": _body_point_from_step(
            end_step, translation_body_mm
        ),
        "motor_profile_axis_end_step_mm": profile_end_step,
        "motor_profile_axis_end_body_mm": _body_point_from_step(
            profile_end_step, translation_body_mm
        ),
        "axis_direction_step": axis_direction,
        "axis_direction_body": _body_direction_from_step(axis_direction),
        "propeller_centre_step_mm": propeller_centre_step,
        "propeller_centre_body_mm": _body_point_from_step(
            propeller_centre_step, translation_body_mm
        ),
        "measured_motor_shaft_interval_from_casing_start_mm": [
            motor_shaft["start_mm"],
            motor_shaft["end_mm"],
        ],
        "measured_propeller_bore_interval_from_casing_start_mm": [
            propeller_bore["start_mm"],
            propeller_bore["end_mm"],
        ],
        "measured_hub_interval_from_casing_start_mm": [hub_start, hub_end],
        "axisymmetric_profile_mm": [list(point) for point in profile_points],
        "source_detailed_motor_volume_mm3": source_volume,
        "smooth_envelope_volume_mm3": envelope_volume,
        "source_volume_relative_error": relative_volume_error,
        "maximum_source_volume_relative_error": maximum_volume_error,
        "common_volume_mm3": common,
        "minimum_common_volume_mm3": minimum_common,
    }


def _validate_body_frame(shells, validation: dict, translation_body_mm) -> dict:
    """Fingerprint the COM-centred body mapping with labelled thrusters."""

    from OCP.TopoDS import TopoDS

    landmarks = validation["propeller_landmarks"]
    if not landmarks:
        raise RuntimeError("Frame validation requires labelled propeller landmarks")
    if any(len(item["expected_body_mm"]) != 3 for item in landmarks):
        raise RuntimeError("Every expected thruster centre must contain three coordinates")

    matches = []
    for item in landmarks:
        index = int(item["shell_index"])
        bounds = _bbox(TopoDS.Shell_s(shells.FindKey(index)))
        centre_step = [
            (bounds[axis] + bounds[axis + 3]) / 2.0 for axis in range(3)
        ]
        measured = _body_point_from_step(centre_step, translation_body_mm)
        expected = [float(value) for value in item["expected_body_mm"]]
        matches.append(
            {
                "source_shell_index": index,
                "measured_body_mm": measured,
                "expected_label": item["label"],
                "expected_body_mm": expected,
                "error_mm": math.dist(measured, expected),
            }
        )

    maximum_error = max(item["error_mm"] for item in matches)
    tolerance = float(validation["max_nearest_error_mm"])
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise RuntimeError("Frame-validation tolerance must be finite and positive")
    if maximum_error > tolerance:
        raise RuntimeError(
            "STEP-to-body frame validation failed: "
            f"maximum thruster-centre error {maximum_error:.6g} mm exceeds "
            f"{tolerance:.6g} mm"
        )

    return {
        "method": "fixed labelled propeller-centre landmarks",
        "translation_body_mm": [float(value) for value in translation_body_mm],
        "tolerance_mm": tolerance,
        "maximum_error_mm": maximum_error,
        "matches": matches,
    }


def _write_stl(shape, path: Path, deflection: float, angular_deflection: float) -> dict:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS

    mesher = BRepMesh_IncrementalMesh(
        shape, deflection, False, angular_deflection, True
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade triangulation did not complete")
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_ShapeEnum.TopAbs_FACE, faces)
    null_faces = []
    face_triangle_count = 0
    for index in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(index))
        triangulation = BRep_Tool.Triangulation_s(face, TopLoc_Location())
        if triangulation is None:
            null_faces.append(index)
        else:
            face_triangle_count += triangulation.NbTriangles()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    if not writer.Write(shape, str(path)):
        raise RuntimeError(f"Failed to write STL: {path}")
    return {
        "open_cascade_face_count": faces.Extent(),
        "null_triangulation_face_indices": null_faces,
        "open_cascade_face_triangle_count": face_triangle_count,
    }


def _audit_binary_stl(path: Path) -> dict:
    """Report finite/degenerate records without building a global edge graph."""

    import numpy as np

    record_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
        align=False,
    )
    with path.open("rb") as stream:
        header = stream.read(80)
        raw_count = stream.read(4)
    if len(header) != 80 or len(raw_count) != 4:
        raise RuntimeError("OpenCascade wrote a truncated STL header")
    triangle_count = struct.unpack("<I", raw_count)[0]
    expected_size = 84 + triangle_count * record_dtype.itemsize
    if record_dtype.itemsize != 50 or path.stat().st_size != expected_size:
        raise RuntimeError("OpenCascade output is not an exact binary STL")
    records = np.memmap(path, dtype=record_dtype, mode="r", offset=84)
    vertices = np.asarray(records["vertices"])
    normals = np.asarray(records["normal"])
    finite = bool(np.isfinite(vertices).all() and np.isfinite(normals).all())
    repeated = (
        np.all(vertices[:, 0] == vertices[:, 1], axis=1)
        | np.all(vertices[:, 1] == vertices[:, 2], axis=1)
        | np.all(vertices[:, 2] == vertices[:, 0], axis=1)
    )
    cross = np.cross(
        vertices[:, 1].astype(np.float64) - vertices[:, 0],
        vertices[:, 2].astype(np.float64) - vertices[:, 0],
    )
    zero_area = np.einsum("ij,ij->i", cross, cross) <= 1e-24
    report = {
        "binary_layout_valid": True,
        "binary_triangle_count": triangle_count,
        "all_values_finite": finite,
        "repeated_vertex_triangle_count": int(np.count_nonzero(repeated)),
        "zero_area_triangle_count": int(np.count_nonzero(zero_area)),
    }
    del records
    if not finite:
        raise RuntimeError("Intermediate STL contains non-finite coordinates or normals")
    return report


def _preflight_paths(
    source: Path, config: Path, output: Path, report: Path, force: bool
) -> tuple[Path, Path, Path, Path]:
    resolved = tuple(path.resolve() for path in (source, config, output, report))
    if len(set(resolved)) != len(resolved):
        raise ValueError("input, config, output and report paths must all be distinct")
    source, config, output, report = resolved
    if not source.is_file():
        raise FileNotFoundError(source)
    if not config.is_file():
        raise FileNotFoundError(config)
    existing = [path for path in (output, report) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite " + ", ".join(map(str, existing)) + "; pass --force"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return source, config, output, report


def _validate_config(config: dict, source: Path) -> None:
    """Reject semantic drift before loading the expensive CAD model."""

    if not isinstance(config, dict):
        raise ValueError("Repair configuration root must be a JSON object")
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported repair configuration schema")
    source_config = config.get("source", {})
    if source_config.get("basename") != source.name:
        raise ValueError("Repair configuration basename does not match the input STEP")
    if source_config.get("units") != "mm":
        raise ValueError("This repair workflow requires a millimetre STEP source")
    digest = source_config.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        value not in "0123456789abcdef" for value in digest.lower()
    ):
        raise ValueError("Source SHA-256 must contain exactly 64 hexadecimal digits")
    expected_count = source_config.get("expected_shell_count")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("Expected shell count must be a positive integer")

    selection = config.get("selection", {})
    if selection.get("default_closed_shell_action") != "keep":
        raise ValueError("Closed-shell default must remain 'keep'")
    if selection.get("open_shell_action") != "remove":
        raise ValueError("Open-shell action must remain 'remove'")
    if selection.get("replace_groups") != {
        "thruster_motor_with_cable": "smooth_axisymmetric_envelope"
    }:
        raise ValueError("Motor replacement contract changed unexpectedly")
    if not selection.get("hole_policy"):
        raise ValueError("Retained-hole policy must be explicit")

    output_frame = config.get("output_frame", {})
    if output_frame.get("name") != "body_flu_com":
        raise ValueError("Output frame must remain body_flu_com")
    if output_frame.get("mapping") != {
        "x_body": "z_step",
        "y_body": "x_step",
        "z_body": "y_step",
    }:
        raise ValueError("Output-frame mapping must remain (z_step,x_step,y_step)")
    if output_frame.get("source_com_body_mm") != list(EXPECTED_SOURCE_COM_BODY_MM):
        raise ValueError("Source COM must match the reviewed SolidWorks mass report")
    if output_frame.get("translation_mm") != list(EXPECTED_BODY_TRANSLATION_MM):
        raise ValueError("Output translation must move the reviewed source COM to zero")
    if not output_frame.get("reference_assumption"):
        raise ValueError("STEP/Coordinate System1 origin assumption must be explicit")

    groups = config.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Repair configuration contains no reviewed shell groups")
    seen: dict[int, str] = {}
    for name, values in groups.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Reviewed shell group {name!r} must be non-empty")
        for index in values:
            if not isinstance(index, int) or not 1 <= index <= expected_count:
                raise ValueError(f"Invalid shell index {index!r} in group {name!r}")
            if index in seen:
                raise ValueError(
                    f"Shell {index} occurs in both {seen[index]!r} and {name!r}"
                )
            seen[index] = name

    remove_names = selection.get("remove_groups", [])
    preserve_names = selection.get("preserve_groups", [])
    if len(remove_names) != len(set(remove_names)) or len(preserve_names) != len(
        set(preserve_names)
    ):
        raise ValueError("Remove/preserve group lists must not contain duplicates")
    unknown = (set(remove_names) | set(preserve_names)) - set(groups)
    if unknown:
        raise ValueError("Unknown reviewed shell group(s): " + ", ".join(sorted(unknown)))
    remove_indices = {index for name in remove_names for index in groups[name]}
    preserve_indices = {index for name in preserve_names for index in groups[name]}
    overlap = remove_indices & preserve_indices
    if overlap:
        raise ValueError(f"Shells cannot be both removed and preserved: {sorted(overlap)}")
    pressure_hull = set(groups.get("main_pressure_hull", []))
    if pressure_hull != {30} or not pressure_hull <= preserve_indices:
        raise ValueError("Reviewed main pressure hull shell 30 must be explicitly preserved")
    pressure_endcaps = set(groups.get("pressure_hull_endcaps", []))
    if pressure_endcaps != {42, 43} or not pressure_endcaps <= preserve_indices:
        raise ValueError("Reviewed pressure-hull endcap shells 42/43 must be explicitly preserved")
    buoyancy_material = set(groups.get("main_closed_cell_buoyancy_material", []))
    if buoyancy_material != {257} or not buoyancy_material <= preserve_indices:
        raise ValueError("Reviewed closed-cell buoyancy-material shell 257 must be preserved")
    motors = set(groups.get("thruster_motor_with_cable", []))
    if not motors or not motors <= remove_indices:
        raise ValueError("Every detailed motor must be removed before replacement")
    propellers = set(groups.get("propeller_3blade", [])) | set(
        groups.get("propeller_4blade", [])
    )
    hubs = set(groups.get("propeller_hub_or_nut", []))
    if len(propellers) != 8 or len(hubs) != 8:
        raise ValueError("Locked-rotor geometry requires eight propellers and eight hubs")
    if not propellers <= preserve_indices or not hubs <= preserve_indices:
        raise ValueError("Every locked propeller and hub must be explicitly preserved")

    replacement = config.get("motor_replacement", {})
    if replacement.get("shape") != "smooth_axisymmetric_envelope":
        raise ValueError("Only the reviewed smooth axisymmetric motor envelope is supported")
    side_keys = {int(value) for value in replacement.get("mount_side_by_shell", {})}
    mount_keys = {int(value) for value in replacement.get("mount_shells_by_motor", {})}
    if side_keys != motors or mount_keys != motors:
        raise ValueError("Motor side/mount maps must exactly cover the detailed motors")
    if any(
        side not in {"vmin", "vmax"}
        for side in replacement["mount_side_by_shell"].values()
    ):
        raise ValueError("Motor mount sides must be vmin or vmax")
    if any(
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) for value in values)
        for values in replacement["mount_shells_by_motor"].values()
    ):
        raise ValueError("Every motor mount validation set must be a non-empty integer list")
    mount_targets = {
        int(index)
        for values in replacement["mount_shells_by_motor"].values()
        for index in values
    }
    if not mount_targets <= preserve_indices:
        raise ValueError("Every motor validation mount must be explicitly preserved")
    extension = replacement.get("mount_extension_mm")
    minimum_common = replacement.get("minimum_common_volume_mm3")
    if not isinstance(extension, (int, float)) or not math.isfinite(extension) or not (
        0.0 < extension <= 10.0
    ):
        raise ValueError("Motor mount extension must be finite and in (0, 10] mm")
    if (
        not isinstance(minimum_common, (int, float))
        or not math.isfinite(minimum_common)
        or minimum_common <= 0.0
    ):
        raise ValueError("Minimum motor/mount common volume must be finite and positive")

    locked = config.get("locked_propeller", {})
    if locked.get("condition") != "fully_assembled_static_locked":
        raise ValueError("Production rotor condition must remain fully assembled and static locked")
    assemblies = locked.get("assemblies_by_motor", {})
    try:
        assembly_motor_indices = {int(index) for index in assemblies}
    except (TypeError, ValueError) as exc:
        raise ValueError("Locked-rotor motor keys must be integer strings") from exc
    if assembly_motor_indices != motors:
        raise ValueError("Locked-rotor assemblies must exactly cover all replacement motors")
    assembly_propellers = {int(item.get("propeller_shell", -1)) for item in assemblies.values()}
    assembly_hubs = {int(item.get("hub_shell", -1)) for item in assemblies.values()}
    assembly_labels = [item.get("label") for item in assemblies.values()]
    if assembly_propellers != propellers or assembly_hubs != hubs:
        raise ValueError("Locked-rotor assemblies must exactly cover reviewed propellers and hubs")
    if set(assembly_labels) != {f"T{index}" for index in range(1, 9)}:
        raise ValueError("Locked-rotor assemblies must contain unique T1--T8 labels")
    locked_numeric = (
        "nominal_motor_shaft_radius_mm",
        "connector_radius_mm",
        "shaft_tip_extension_mm",
        "expected_motor_shaft_length_mm",
        "expected_propeller_hub_length_mm",
        "expected_separate_hub_length_mm",
        "signature_tolerance_mm",
    )
    for name in locked_numeric:
        value = locked.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Locked-rotor {name} must be finite and positive")
    nominal_shaft_radius = float(locked["nominal_motor_shaft_radius_mm"])
    connector_radius = float(locked["connector_radius_mm"])
    if not nominal_shaft_radius < connector_radius <= nominal_shaft_radius + 0.1:
        raise ValueError("Locked-rotor connector radius must use only the reviewed bore overlap")
    refinement_end = locked.get("shaft_refinement_end_mm")
    if not isinstance(refinement_end, (int, float)) or not math.isfinite(refinement_end):
        raise ValueError("Locked-rotor shaft refinement end must be finite")
    maximum_volume_error = locked.get("maximum_source_volume_relative_error")
    if (
        not isinstance(maximum_volume_error, (int, float))
        or not math.isfinite(maximum_volume_error)
        or not 0.0 < maximum_volume_error <= 0.05
    ):
        raise ValueError("Locked motor-envelope source-volume tolerance must lie in (0, 0.05]")
    profile = locked.get("axisymmetric_profile_mm")
    if (
        not isinstance(profile, list)
        or len(profile) < 4
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in point
            )
            for point in profile
        )
    ):
        raise ValueError("Locked motor axisymmetric profile must contain finite [axial,radius] points")
    if profile[0][1] != 0.0 or profile[-1][1] != 0.0:
        raise ValueError("Locked motor profile must start and end on its axis")
    if any(float(point[1]) < 0.0 for point in profile):
        raise ValueError("Locked motor profile radii cannot be negative")
    if any(
        float(second[0]) < float(first[0])
        for first, second in zip(profile, profile[1:])
    ):
        raise ValueError("Locked motor profile axial coordinates must be nondecreasing")
    if not float(profile[0][0]) < float(refinement_end) < float(profile[-1][0]):
        raise ValueError("Shaft refinement endpoint must lie inside the motor profile")
    locked_common = locked.get("minimum_common_volume_mm3", {})
    if set(locked_common) != {"mount", "hub", "propeller"} or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in locked_common.values()
    ):
        raise ValueError("Locked-rotor common-volume gates are incomplete or invalid")

    sealed = config.get("sealed_pressure_boundary", {})
    if sealed.get("condition") != "waterproof_assembled_vehicle":
        raise ValueError(
            "Pressure boundary requires the waterproof assembled-vehicle condition"
        )
    if sealed.get("representation") != "two_tube_opening_sealing_disks":
        raise ValueError("Pressure boundary must use two tube-opening sealing disks")
    if int(sealed.get("hull_shell", -1)) not in pressure_hull:
        raise ValueError("Pressure boundary must derive from the reviewed hull shell")
    if set(sealed.get("endcap_shells", [])) != pressure_endcaps:
        raise ValueError("Pressure boundary must reference both reviewed end fittings")
    endcap_by_end = sealed.get("endcap_by_hull_end", {})
    if set(endcap_by_end) != {"start", "end"} or set(endcap_by_end.values()) != pressure_endcaps:
        raise ValueError(
            "Pressure boundary must map one reviewed end fitting to each hull end"
        )
    sealed_numeric = (
        "expected_inner_radius_mm",
        "expected_outer_radius_mm",
        "expected_hull_length_mm",
        "radial_wall_overlap_mm",
        "disk_half_thickness_mm",
        "signature_tolerance_mm",
    )
    for name in sealed_numeric:
        value = sealed.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Sealed pressure-boundary {name} must be finite and positive")
    inner_radius = float(sealed["expected_inner_radius_mm"])
    outer_radius = float(sealed["expected_outer_radius_mm"])
    radial_overlap = float(sealed["radial_wall_overlap_mm"])
    if not inner_radius < inner_radius + radial_overlap < outer_radius:
        raise ValueError(
            "Sealed pressure-boundary radial overlap must remain inside the hull wall"
        )
    if 2.0 * float(sealed["disk_half_thickness_mm"]) > 2.0:
        raise ValueError("Pressure-boundary disks must remain no thicker than 2 mm")
    expected_sealed_common = {
        "start_disk_hull",
        "start_disk_endcap",
        "end_disk_hull",
        "end_disk_endcap",
    }
    sealed_common = sealed.get("minimum_common_volume_mm3", {})
    if set(sealed_common) != expected_sealed_common or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in sealed_common.values()
    ):
        raise ValueError(
            "Sealed pressure-boundary common-volume gates are incomplete or invalid"
        )

    buoyancy = config.get("buoyancy_material_validation", {})
    if int(buoyancy.get("shell", -1)) not in buoyancy_material:
        raise ValueError("Buoyancy-material validation must reference shell 257")
    if buoyancy.get("role") != "waterproof closed-cell main buoyancy material":
        raise ValueError("Shell 257 must remain identified as the main buoyancy material")
    if buoyancy.get("condition") != "waterproof_closed_cell":
        raise ValueError("Main buoyancy material must remain waterproof and closed-cell")
    if buoyancy.get("identification_status") != "high-confidence geometry/placement inference":
        raise ValueError("Buoyancy-material identification confidence must remain explicit")
    expected_buoyancy_volume = buoyancy.get("expected_closed_solid_volume_mm3")
    buoyancy_volume_tolerance = buoyancy.get("volume_relative_tolerance")
    buoyancy_bbox_tolerance = buoyancy.get("bbox_absolute_tolerance_mm")
    if (
        not isinstance(expected_buoyancy_volume, (int, float))
        or not math.isfinite(expected_buoyancy_volume)
        or expected_buoyancy_volume <= 0.0
        or not isinstance(buoyancy_volume_tolerance, (int, float))
        or not math.isfinite(buoyancy_volume_tolerance)
        or not 0.0 < buoyancy_volume_tolerance < 0.05
        or not isinstance(buoyancy_bbox_tolerance, (int, float))
        or not math.isfinite(buoyancy_bbox_tolerance)
        or buoyancy_bbox_tolerance <= 0.0
    ):
        raise ValueError("Buoyancy-material volume gate is invalid")
    buoyancy_bbox = buoyancy.get("expected_bbox_step_mm")
    if (
        not isinstance(buoyancy_bbox, list)
        or len(buoyancy_bbox) != 6
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in buoyancy_bbox
        )
    ):
        raise ValueError("Buoyancy-material expected STEP bbox must contain six finite values")
    if not buoyancy.get("hydrodynamic_accounting"):
        raise ValueError("Buoyancy-material anti-double-counting policy must be explicit")

    volume_validation = config.get("volume_validation", {})
    target_volume = volume_validation.get("target_displaced_volume_mm3")
    wrapped_tolerance = volume_validation.get("wrapped_surface_relative_tolerance")
    mesh_tolerance = volume_validation.get("snappy_excluded_volume_relative_tolerance")
    if (
        not isinstance(target_volume, (int, float))
        or not math.isfinite(target_volume)
        or target_volume <= 0.0
    ):
        raise ValueError("Target displaced volume must be a positive finite mm^3 value")
    for name, value in (
        ("wrapped surface", wrapped_tolerance),
        ("snappy excluded volume", mesh_tolerance),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
        ):
            raise ValueError(f"{name} relative tolerance must lie in (0, 1)")

    frame_validation = output_frame.get("validation", {})
    landmarks = frame_validation.get(
        "propeller_landmarks", []
    )
    landmark_indices = {int(item["shell_index"]) for item in landmarks}
    propeller_indices = propellers
    if landmark_indices != propeller_indices or len(landmarks) != len(
        landmark_indices
    ):
        raise ValueError("Labelled frame landmarks must exactly cover all propellers")
    labels = [item.get("label") for item in landmarks]
    if set(labels) != {f"T{index}" for index in range(1, 9)}:
        raise ValueError("Frame landmarks must contain unique T1--T8 labels")
    for item in landmarks:
        point = item.get("expected_body_mm")
        if (
            not isinstance(point, list)
            or len(point) != 3
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in point
            )
        ):
            raise ValueError("Frame landmark coordinates must be finite XYZ triples")
    frame_tolerance = frame_validation.get("max_nearest_error_mm")
    if (
        not isinstance(frame_tolerance, (int, float))
        or not math.isfinite(frame_tolerance)
        or frame_tolerance <= 0.0
    ):
        raise ValueError("Frame landmark tolerance must be finite and positive")

    triangulation = config.get("triangulation", {})
    for name in ("linear_deflection_mm", "angular_deflection_rad"):
        value = triangulation.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"Configured {name} must be finite and positive")
    gate = config.get("triangulation_reference", {})
    expected_gate_types = {
        "null_triangulation_face_indices": list,
        "binary_triangle_count": int,
        "repeated_vertex_triangle_count": int,
        "zero_area_triangle_count": int,
    }
    for name, expected_type in expected_gate_types.items():
        value = gate.get(name)
        if type(value) is not expected_type:
            raise ValueError(f"Triangulation reference {name!r} has the wrong type")
        if expected_type is int and value < 0:
            raise ValueError(f"Triangulation reference {name!r} cannot be negative")
    if any(
        not isinstance(value, int) or value < 1
        for value in gate["null_triangulation_face_indices"]
    ):
        raise ValueError("Null-triangulation face indices must be positive integers")

    crosscheck = config.get("entity_crosscheck", {})
    for name in ("main_outer_fairing", "thruster_support_complex", "propellers", "motors"):
        values = crosscheck.get(name)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, int) or value < 1 for value in values
        ):
            raise ValueError(f"Missing reviewed STEP entity cross-check group {name!r}")


def prepare(args: argparse.Namespace) -> dict:
    source, config_path, output, report_path = _preflight_paths(
        args.input, args.config, args.output, args.report, args.force
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, source)
    actual_sha = _sha256(source)
    expected_sha = config["source"]["sha256"]
    source_fingerprint_match = actual_sha == expected_sha
    ocp_version = _load_ocp()
    from OCP.TopoDS import TopoDS

    root = _read_step(source)
    shells = _shell_map(root)
    expected_count = config["source"]["expected_shell_count"]
    if shells.Extent() != expected_count:
        raise RuntimeError(
            f"Shell count changed: expected {expected_count}, got {shells.Extent()}"
        )

    frame_config = config["output_frame"]
    expected_mapping = {
        "x_body": "z_step",
        "y_body": "x_step",
        "z_body": "y_step",
    }
    if frame_config.get("mapping") != expected_mapping:
        raise RuntimeError("Configuration axis mapping disagrees with the implemented transform")
    translation_body_mm = [float(value) for value in frame_config["translation_mm"]]
    if translation_body_mm != list(EXPECTED_BODY_TRANSLATION_MM):
        raise RuntimeError("Configured body translation does not centre the reviewed COM")
    frame_validation = _validate_body_frame(
        shells, frame_config["validation"], translation_body_mm
    )
    mesh_config = config["triangulation"]
    deflection_mm = float(mesh_config["linear_deflection_mm"])
    angular_deflection_rad = float(mesh_config["angular_deflection_rad"])
    if not math.isfinite(deflection_mm) or deflection_mm <= 0:
        raise RuntimeError("Configured linear triangulation deflection must be positive")
    if not math.isfinite(angular_deflection_rad) or angular_deflection_rad <= 0:
        raise RuntimeError("Configured angular triangulation deflection must be positive")

    groups = config["groups"]
    removal_names = set(config["selection"]["remove_groups"])
    remove_indices = {
        index
        for name in removal_names
        for index in groups.get(name, [])
    }
    motor_indices = groups["thruster_motor_with_cable"]
    motor_extension_mm = float(config["motor_replacement"]["mount_extension_mm"])
    motor_mount_sides = {
        int(index): side
        for index, side in config["motor_replacement"]["mount_side_by_shell"].items()
    }
    if set(motor_indices) != set(motor_mount_sides):
        raise RuntimeError("Every reviewed motor shell must have exactly one mount-side decision")
    motor_mount_shells = {
        int(index): [int(value) for value in shell_indices]
        for index, shell_indices in config["motor_replacement"][
            "mount_shells_by_motor"
        ].items()
    }
    if set(motor_indices) != set(motor_mount_shells):
        raise RuntimeError("Every reviewed motor shell must have a mount-shell validation set")
    locked_config = config["locked_propeller"]
    locked_assemblies = {
        int(index): {
            "label": item["label"],
            "propeller_shell": int(item["propeller_shell"]),
            "hub_shell": int(item["hub_shell"]),
        }
        for index, item in locked_config["assemblies_by_motor"].items()
    }
    preserve_indices = {
        index
        for name in config["selection"]["preserve_groups"]
        for index in groups[name]
    }

    kept = []
    open_removed = []
    explicit_removed = []
    motor_replacements = []
    locked_rotor_assemblies = []
    sealed_pressure_boundary_metadata = None
    buoyancy_material_metadata = None
    for index in range(1, shells.Extent() + 1):
        shell = TopoDS.Shell_s(shells.FindKey(index))
        if index in remove_indices:
            explicit_removed.append(index)
            continue
        solid = _closed_solid(shell)
        if solid is None:
            open_removed.append(index)
            continue
        kept.append((index, solid))

    kept_by_index = dict(kept)

    buoyancy_config = config["buoyancy_material_validation"]
    buoyancy_shell_index = int(buoyancy_config["shell"])
    if buoyancy_shell_index not in kept_by_index:
        raise RuntimeError(
            f"Main buoyancy-material shell {buoyancy_shell_index} was not retained"
        )
    buoyancy_material_metadata = _validate_buoyancy_material(
        kept_by_index[buoyancy_shell_index], buoyancy_config
    )

    for index in motor_indices:
        shell = TopoDS.Shell_s(shells.FindKey(index))
        _, metadata = _dominant_motor_cylinder(
            shell, motor_extension_mm, motor_mount_sides[index]
        )
        missing_mounts = [
            mount for mount in motor_mount_shells[index] if mount not in kept_by_index
        ]
        if missing_mounts:
            raise RuntimeError(
                f"Motor {index} validation mounts were not retained: {missing_mounts}"
            )
        mount_reference = _make_compound(
            [kept_by_index[mount] for mount in motor_mount_shells[index]]
        )
        metadata["source_shell_index"] = index
        metadata["validated_mount_shell_indices"] = motor_mount_shells[index]
        assembly = locked_assemblies[index]
        propeller_index = assembly["propeller_shell"]
        hub_index = assembly["hub_shell"]
        missing_rotor_shells = [
            shell_index
            for shell_index in (propeller_index, hub_index)
            if shell_index not in kept_by_index
        ]
        if missing_rotor_shells:
            raise RuntimeError(
                f"Locked rotor {assembly['label']} shells were not retained: "
                f"{missing_rotor_shells}"
            )
        envelope, envelope_metadata = _locked_rotor_envelope(
            shell,
            mount_reference,
            kept_by_index[hub_index],
            kept_by_index[propeller_index],
            metadata,
            locked_config,
            translation_body_mm,
        )
        envelope_metadata.update(
            {
                "label": assembly["label"],
                "source_motor_shell_index": index,
                "source_propeller_shell_index": propeller_index,
                "source_hub_shell_index": hub_index,
            }
        )
        metadata["locked_rotor_envelope"] = envelope_metadata
        motor_replacements.append(metadata)
        locked_rotor_assemblies.append(envelope_metadata)
        kept.append((f"motor_envelope_{index}", envelope))

    sealed_config = config["sealed_pressure_boundary"]
    sealed_hull_index = int(sealed_config["hull_shell"])
    sealed_endcap_indices = [int(value) for value in sealed_config["endcap_shells"]]
    missing_sealed_parts = [
        index
        for index in [sealed_hull_index, *sealed_endcap_indices]
        if index not in kept_by_index
    ]
    if missing_sealed_parts:
        raise RuntimeError(
            "Sealed pressure-boundary source shells were not retained: "
            f"{missing_sealed_parts}"
        )
    sealed_pressure_boundary, sealed_pressure_boundary_metadata = _sealed_pressure_boundary(
        TopoDS.Shell_s(shells.FindKey(sealed_hull_index)),
        kept_by_index[sealed_hull_index],
        {index: kept_by_index[index] for index in sealed_endcap_indices},
        sealed_config,
        translation_body_mm,
    )
    kept.append(("sealed_pressure_boundary", sealed_pressure_boundary))

    kept_indices = {index for index, _ in kept if isinstance(index, int)}
    missing_preserved = sorted(preserve_indices - kept_indices)
    if missing_preserved:
        raise RuntimeError(f"Required preserved shells were not retained: {missing_preserved}")

    compound_step = _make_compound([shape for _, shape in kept])
    compound_body = _to_body_flu(compound_step, translation_body_mm)
    descriptor, temporary_output_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".stl"
    )
    os.close(descriptor)
    temporary_output = Path(temporary_output_name)
    descriptor, temporary_report_name = tempfile.mkstemp(
        dir=report_path.parent, prefix=f".{report_path.name}.", suffix=".json"
    )
    os.close(descriptor)
    temporary_report = Path(temporary_report_name)
    try:
        triangulation = _write_stl(
            compound_body,
            temporary_output,
            deflection_mm,
            angular_deflection_rad,
        )
        intermediate_audit = _audit_binary_stl(temporary_output)
        expected_audit = config["triangulation_reference"]
        actual_locked = {
            "null_triangulation_face_indices": triangulation[
                "null_triangulation_face_indices"
            ],
            "binary_triangle_count": intermediate_audit["binary_triangle_count"],
            "repeated_vertex_triangle_count": intermediate_audit[
                "repeated_vertex_triangle_count"
            ],
            "zero_area_triangle_count": intermediate_audit["zero_area_triangle_count"],
        }
        mismatches = [
            f"{name}: expected {expected_audit[name]}, got {actual}"
            for name, actual in actual_locked.items()
            if actual != expected_audit[name]
        ]

        report = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": actual_sha,
        "source_fingerprint_match": source_fingerprint_match,
        "cad_backend": {
            "name": "cadquery-ocp-novtk",
            "ocp_version": ocp_version,
            "module_path": str(DEFAULT_OCP_SITE.resolve()),
        },
        "selection_config": str(config_path),
        "selection_config_sha256": _sha256(config_path),
        "repair_script_sha256": _sha256(Path(__file__).resolve()),
        "invocation": list(sys.argv),
        "source_units": "mm",
        "output_units": "mm",
        "output_frame": "body_flu_com",
        "axis_mapping": ["x_body=z_step", "y_body=x_step", "z_body=y_step"],
        "source_com_body_mm": frame_config["source_com_body_mm"],
        "translation_body_mm": translation_body_mm,
        "reference_assumption": frame_config["reference_assumption"],
        "body_frame_validation": frame_validation,
        "volume_validation": config["volume_validation"],
        "reviewed_entity_crosscheck": config["entity_crosscheck"],
        "shell_count": shells.Extent(),
        "closed_shells_kept": len(kept_indices),
        "explicit_shells_removed": sorted(explicit_removed),
        "open_shells_removed": sorted(open_removed),
        "smooth_motor_replacements": motor_replacements,
        "locked_rotor_condition": "fully_assembled_static_locked",
        "locked_rotor_assemblies": sorted(
            locked_rotor_assemblies, key=lambda item: item["label"]
        ),
        "main_buoyancy_material": buoyancy_material_metadata,
        "sealed_pressure_boundary": sealed_pressure_boundary_metadata,
        "output": str(output),
        "output_sha256": _sha256(temporary_output),
        "output_size_bytes": temporary_output.stat().st_size,
        "output_bbox_body_mm": _bbox(compound_body),
        "triangulation": {
            "linear_deflection_mm": deflection_mm,
            "angular_deflection_rad": angular_deflection_rad,
            **triangulation,
            **intermediate_audit,
        },
        "triangulation_reference_comparison": {
            "expected": {name: expected_audit[name] for name in actual_locked},
            "actual": actual_locked,
            "expected_match": not mismatches,
            "mismatches": mismatches,
            "gating": False,
        },
        "warning": (
            "This STL contains intersecting retained solids and is an intermediate "
            "input to voxel_wrap.py, not the final CFD wetted surface."
        ),
        }
        temporary_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_output, output)
        os.replace(temporary_report, report_path)
        return report
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source STEP (SHA recorded for provenance)")
    parser.add_argument("output", type=Path, help="Intermediate body-FLU STL in mm")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = prepare(args)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Buoyancy material and sealed pressure-boundary reconstruction."""

from __future__ import annotations

import math

from environment.openfoam.step_repair.core import (
    _bbox,
    _common_volume,
    _make_compound,
    _solid_volume,
)

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


def _pressure_tube_signature(hull_shell, config: dict):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    cylinders = []
    explorer = TopExp_Explorer(hull_shell, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        surface = BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current()), True)
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinders.append(surface.Cylinder())
        explorer.Next()
    if len(cylinders) < 2:
        raise RuntimeError("Reviewed pressure hull has fewer than two cylindrical faces")

    reference = max(cylinders, key=lambda cylinder: float(cylinder.Radius()))
    axis = reference.Axis()
    axis_point = [axis.Location().X(), axis.Location().Y(), axis.Location().Z()]
    axis_direction = [axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z()]
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
    hull_start = max(float(inner["start_mm"]), float(outer["start_mm"]))
    hull_end = min(float(inner["end_mm"]), float(outer["end_mm"]))
    hull_length = hull_end - hull_start
    expected_length = float(config["expected_hull_length_mm"])
    if abs(hull_length - expected_length) > tolerance:
        raise RuntimeError(
            f"Reviewed pressure-hull length changed: expected {expected_length:g} mm, got {hull_length:.12g} mm"
        )
    return (
        axis_point,
        axis_direction,
        float(inner["radius_mm"]),
        float(outer["radius_mm"]),
        hull_start,
        hull_end,
    )


def _make_seal_disk(axis_point, axis_direction, centre_parameter: float, radius: float, half_thickness: float):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    direction = gp_Dir(*axis_direction).XYZ()
    origin = gp_Pnt(*axis_point).XYZ()
    start = origin + direction * (centre_parameter - half_thickness)
    end = origin + direction * (centre_parameter + half_thickness)
    centre = origin + direction * centre_parameter
    disk = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(start.X(), start.Y(), start.Z()), gp_Dir(*axis_direction)),
        radius,
        2.0 * half_thickness,
    ).Solid()
    return disk, start, end, centre


def _build_seal_disks(
    *,
    axis_point,
    axis_direction,
    hull_start: float,
    hull_end: float,
    disk_radius: float,
    half_thickness: float,
    hull_solid,
    endcap_solids: dict[int, object],
    config: dict,
    translation_body_mm,
):
    disks = {}
    metadata = []
    common = {}
    endcap_by_end = {side: int(index) for side, index in config["endcap_by_hull_end"].items()}
    for side, parameter in (("start", hull_start), ("end", hull_end)):
        disk, start, end, centre = _make_seal_disk(
            axis_point,
            axis_direction,
            parameter,
            disk_radius,
            half_thickness,
        )
        disks[side] = disk
        endcap_index = endcap_by_end[side]
        common[f"{side}_disk_hull"] = _common_volume(disk, hull_solid)
        common[f"{side}_disk_endcap"] = _common_volume(disk, endcap_solids[endcap_index])
        centre_step = [centre.X(), centre.Y(), centre.Z()]
        metadata.append(
            {
                "hull_end": side,
                "reviewed_endcap_shell": endcap_index,
                "centre_parameter_from_axis_origin_mm": parameter,
                "centre_step_mm": centre_step,
                "centre_body_mm": _body_point_from_step(centre_step, translation_body_mm),
                "axis_start_step_mm": [start.X(), start.Y(), start.Z()],
                "axis_end_step_mm": [end.X(), end.Y(), end.Z()],
                "radius_mm": disk_radius,
                "thickness_mm": 2.0 * half_thickness,
                "solid_volume_mm3": _solid_volume(disk),
                "common_with_hull_mm3": common[f"{side}_disk_hull"],
                "common_with_reviewed_endcap_mm3": common[f"{side}_disk_endcap"],
            }
        )
    return disks, metadata, common


def _endcap_contact_evidence(hull_solid, endcap_solids: dict[int, object], disks: dict) -> dict:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    evidence = {}
    for index, endcap in sorted(endcap_solids.items()):
        distance = BRepExtrema_DistShapeShape(hull_solid, endcap)
        distance.Perform()
        evidence[str(index)] = {
            "distance_to_hull_mm": float(distance.Value()) if distance.IsDone() else None,
            "common_with_hull_mm3": _common_volume(hull_solid, endcap),
            "common_with_start_disk_mm3": _common_volume(disks["start"], endcap),
            "common_with_end_disk_mm3": _common_volume(disks["end"], endcap),
        }
    return evidence


def _validate_seal_overlap(common: dict[str, float], config: dict) -> dict[str, float]:
    minimum = {name: float(value) for name, value in config["minimum_common_volume_mm3"].items()}
    failures = [
        f"{name} {common.get(name, -math.inf):.6g} < {required:.6g} mm^3"
        for name, required in minimum.items()
        if common.get(name, -math.inf) < required
    ]
    if failures:
        raise RuntimeError("Sealed pressure-boundary overlap failed: " + "; ".join(failures))
    return minimum


def _sealed_pressure_boundary(
    hull_shell,
    hull_solid,
    endcap_solids: dict[int, object],
    config: dict,
    translation_body_mm,
) -> tuple[object, dict]:
    """Derive two thin waterproof patches at the measured pressure-tube mouths."""

    axis_point, axis_direction, inner_radius, outer_radius, hull_start, hull_end = (
        _pressure_tube_signature(hull_shell, config)
    )
    radial_overlap = float(config["radial_wall_overlap_mm"])
    half_thickness = float(config["disk_half_thickness_mm"])
    disk_radius = inner_radius + radial_overlap
    if not inner_radius < disk_radius < outer_radius:
        raise RuntimeError("Derived pressure-seal disks must overlap but remain inside the tube wall")
    disks, disk_metadata, common = _build_seal_disks(
        axis_point=axis_point,
        axis_direction=axis_direction,
        hull_start=hull_start,
        hull_end=hull_end,
        disk_radius=disk_radius,
        half_thickness=half_thickness,
        hull_solid=hull_solid,
        endcap_solids=endcap_solids,
        config=config,
        translation_body_mm=translation_body_mm,
    )
    boundary = _make_compound(disks.values())
    minimum_common = _validate_seal_overlap(common, config)
    hull_length = hull_end - hull_start
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
        "hull_endcap_original_contact": _endcap_contact_evidence(
            hull_solid,
            endcap_solids,
            disks,
        ),
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

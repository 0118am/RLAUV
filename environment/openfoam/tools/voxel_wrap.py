#!/usr/bin/env python3
"""Rebuild a triangle-soup STL as one watertight shared-vertex OBJ exterior.

The implementation is deliberately conservative:

* input triangles are rasterized as barrier voxels;
* a six-neighbour flood fill identifies fluid reachable from the padded box;
* optional binary closing seals only the configured voxel-scale gaps;
* Lewiner's topologically guaranteed Marching Cubes 33 implementation extracts
  the binary half-voxel isosurface without smoothing;
* disconnected contour debris is removed only when its aggregate volume and area
  remain below explicit sub-voxel limits;
* the exact serialized metre-scale OBJ is read back by OpenFOAM and uniformly
  normalized for a scale-conditioned OpenFOAM ``surfaceCheck`` audit, without
  geometry-changing fallback repair; and
* topology, self-intersection and bidirectional surface-distance evidence is
  written to JSON.
Coordinates and ``--voxel-size`` use the input STL's native unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

try:
    from .voxel_backend import (
        VoxelWrapError,
        backend_status,
        connected_regions as _connected_regions,
        distribution as _distribution,
        extract_surface as _extract_surface,
        feature_edge_count as _feature_edge_count,
        geometry_metrics as _geometry_metrics,
        largest_connected_region as _largest_connected_region,
        orient_normals as _orient_normals,
        point_distances as _point_distances,
        read_surface as _read_surface,
        reflect_about_y_plane as _reflect_about_y_plane,
        require_backend as _require_backend,
        scale_uniform as _scale_uniform,
        topology as _topology,
        write_stl as _write_stl,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from voxel_backend import (
        VoxelWrapError,
        backend_status,
        connected_regions as _connected_regions,
        distribution as _distribution,
        extract_surface as _extract_surface,
        feature_edge_count as _feature_edge_count,
        geometry_metrics as _geometry_metrics,
        largest_connected_region as _largest_connected_region,
        orient_normals as _orient_normals,
        point_distances as _point_distances,
        read_surface as _read_surface,
        reflect_about_y_plane as _reflect_about_y_plane,
        require_backend as _require_backend,
        scale_uniform as _scale_uniform,
        topology as _topology,
        write_stl as _write_stl,
    )


REPORT_SCHEMA_VERSION = 6
DEFAULT_CLOSING_ITERATIONS = 1
DEFAULT_PAD_VOXELS = 4
DEFAULT_MAX_OUTPUT_TO_INPUT_DISTANCE_VOXELS = 4.0
DEFAULT_MAX_OUTPUT_TO_INPUT_P99_DISTANCE_VOXELS = 2.0
DEFAULT_MAX_VOXELS = 100_000_000
DEFAULT_VOLUME_RELATIVE_TOLERANCE = 0.02
MAX_DISTANCE_SAMPLE_POINTS = 500_000
DEFAULT_OPENFOAM_LAUNCHER = Path(__file__).resolve().parents[1] / "launch_openfoam.sh"


def _subvoxel_component_cleanup_allowed(
    discarded_volume: float,
    discarded_area: float,
    spacing: float,
) -> bool:
    """Only classify disconnected extraction debris below one voxel as numerical."""

    return (
        discarded_volume >= 0.0
        and discarded_area >= 0.0
        and discarded_volume <= spacing**3 * (1.0 + 1.0e-9)
        and discarded_area <= 8.0 * spacing**2 * (1.0 + 1.0e-9)
    )


def _remove_subvoxel_component_artifacts(
    surface: Any,
    spacing: float,
    vtk: Any,
) -> tuple[Any, dict[str, Any]]:
    """Drop disconnected contour fragments only when their aggregate is sub-voxel."""

    regions_before = _connected_regions(surface, vtk)
    report: dict[str, Any] = {
        "regions_before": regions_before,
        "regions_after": regions_before,
        "removed": False,
        "criterion": (
            "aggregate discarded volume <= one voxel and aggregate discarded "
            "area <= eight voxel-face areas"
        ),
        "maximum_discarded_volume_source_units_cubed": spacing**3,
        "maximum_discarded_area_source_units_squared": 8.0 * spacing**2,
    }
    if regions_before <= 1:
        report["reason"] = "surface already has one connected region"
        return surface, report

    largest = _largest_connected_region(surface, vtk)
    total_geometry = _geometry_metrics(surface, vtk)
    largest_geometry = _geometry_metrics(largest, vtk)
    discarded_volume = max(
        0.0,
        float(total_geometry["enclosed_volume_source_units_cubed"])
        - float(largest_geometry["enclosed_volume_source_units_cubed"]),
    )
    discarded_area = max(
        0.0,
        float(total_geometry["surface_area_source_units_squared"])
        - float(largest_geometry["surface_area_source_units_squared"]),
    )
    report.update(
        {
            "discarded_volume_source_units_cubed": discarded_volume,
            "discarded_area_source_units_squared": discarded_area,
            "discarded_triangles": int(
                surface.GetNumberOfCells() - largest.GetNumberOfCells()
            ),
        }
    )
    if not _subvoxel_component_cleanup_allowed(
        discarded_volume,
        discarded_area,
        spacing,
    ):
        report["reason"] = "disconnected geometry exceeds the sub-voxel artifact limits"
        return surface, report

    report.update(
        {
            "regions_after": _connected_regions(largest, vtk),
            "removed": True,
            "reason": "removed disconnected contour debris below the voxel resolution",
        }
    )
    return largest, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _six_neighbour_structure(ndimage: Any) -> Any:
    return ndimage.generate_binary_structure(3, 1)


def _outside_flood(solid_barrier: Any, ndimage: Any) -> Any:
    """Return all voxels reachable from the padded boundary through six-neighbour free space."""

    free = ~solid_barrier
    seed = free.copy()
    seed[1:-1, 1:-1, 1:-1] = False
    return ndimage.binary_propagation(
        seed,
        structure=_six_neighbour_structure(ndimage),
        mask=free,
    )


def _solid_from_barrier(barrier: Any, ndimage: Any, closing_iterations: int) -> Any:
    import numpy as np

    solid = ~_outside_flood(barrier, ndimage)
    if closing_iterations:
        solid = ndimage.binary_closing(
            solid,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=closing_iterations,
        )
        # Closing may seal previously fluid-reachable cavities. Re-flood so
        # the output contains only the post-closing wetted exterior.
        solid = ~_outside_flood(solid, ndimage)
    return solid


def _voxelize_surface(
    surface: Any,
    voxel_size: float,
    pad_voxels: int,
    max_voxels: int,
    grid_anchor: Sequence[float],
    vtk: Any,
    vtk_to_numpy: Any,
) -> tuple[Any, tuple[float, float, float], tuple[int, int, int], dict[str, Any]]:
    bounds = surface.GetBounds()
    origin = tuple(
        grid_anchor[axis]
        + (
            math.floor((bounds[2 * axis] - grid_anchor[axis]) / voxel_size)
            - pad_voxels
        )
        * voxel_size
        for axis in range(3)
    )
    upper = tuple(
        grid_anchor[axis]
        + (
            math.ceil((bounds[2 * axis + 1] - grid_anchor[axis]) / voxel_size)
            + pad_voxels
        )
        * voxel_size
        for axis in range(3)
    )
    dimensions = tuple(
        int(round((upper[axis] - origin[axis]) / voxel_size)) + 1
        for axis in range(3)
    )
    voxel_count = math.prod(dimensions)
    if max_voxels and voxel_count > max_voxels:
        raise VoxelWrapError(
            f"requested grid has {voxel_count:,} voxels, exceeding --max-voxels "
            f"{max_voxels:,}; increase voxel size or explicitly raise the limit"
        )
    model_upper = tuple(
        origin[axis] + voxel_size * (dimensions[axis] - 1)
        for axis in range(3)
    )
    model_bounds = (
        origin[0],
        model_upper[0],
        origin[1],
        model_upper[1],
        origin[2],
        model_upper[2],
    )
    diagonal = math.sqrt(
        sum((bounds[2 * axis + 1] - bounds[2 * axis]) ** 2 for axis in range(3))
    )

    voxels = vtk.vtkVoxelModeller()
    voxels.SetInputData(surface)
    voxels.SetSampleDimensions(*dimensions)
    voxels.SetModelBounds(*model_bounds)
    voxels.SetScalarTypeToUnsignedChar()
    voxels.SetForegroundValue(1)
    voxels.SetBackgroundValue(0)
    if diagonal > 0.0:
        voxels.SetMaximumDistance(min(1.0, 2.5 * voxel_size / diagonal))
    voxels.Update()
    flat = vtk_to_numpy(voxels.GetOutput().GetPointData().GetScalars())
    barrier = flat.reshape((dimensions[2], dimensions[1], dimensions[0])).astype(bool)
    details = {
        "dimensions_xyz": list(dimensions),
        "voxel_count": int(voxel_count),
        "origin": list(origin),
        "grid_anchor_source_units": list(grid_anchor),
        "bounds": list(model_bounds),
        "surface_barrier_voxels": int(barrier.sum()),
    }
    return barrier, origin, dimensions, details


def _parse_surface_check_result(output: str) -> int:
    """Return the reported self-intersection count from OpenFOAM surfaceCheck."""

    if "Surface is not self-intersecting" in output:
        return 0
    match = re.search(r"Surface is self-intersecting at\s+(\d+)\s+locations?", output)
    if match is None:
        raise VoxelWrapError(
            "surfaceCheck did not report a recognizable self-intersection result"
        )
    return int(match.group(1))


def _read_obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if not fields or fields[0] == "#":
            continue
        if len(fields) != 4 or fields[0] != "v":
            raise VoxelWrapError(
                f"unexpected selfInterPoints.obj record at line {line_number}: {raw_line!r}"
            )
        point = tuple(float(value) for value in fields[1:])
        if not all(math.isfinite(value) for value in point):
            raise VoxelWrapError(
                f"non-finite self-intersection point at line {line_number}"
            )
        points.append(point)
    return points


def _audit_self_intersections(
    surface: Any,
    vtk: Any,
    openfoam_launcher: Path,
    output_scale: float,
) -> dict[str, Any]:
    """Uniformly scale a surface for a conditioned OpenFOAM intersection audit."""

    if not openfoam_launcher.is_file():
        raise VoxelWrapError(f"OpenFOAM launcher does not exist: {openfoam_launcher}")
    with tempfile.TemporaryDirectory(prefix="auv-surface-check-") as temporary_directory:
        directory = Path(temporary_directory)
        candidate = directory / "candidate.stl"
        scaled = _scale_uniform(surface, output_scale, vtk)
        _write_stl(scaled, candidate, vtk)
        completed = subprocess.run(
            [
                str(openfoam_launcher),
                "surfaceCheck",
                "-checkSelfIntersection",
                str(candidate),
            ],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise VoxelWrapError(
                "OpenFOAM surfaceCheck failed while auditing the rebuilt surface: "
                f"exit={completed.returncode}\n{output[-4000:]}"
            )
        count = _parse_surface_check_result(output)
        points_path = directory / "selfInterPoints.obj"
        if count:
            if not points_path.is_file():
                raise VoxelWrapError(
                    "surfaceCheck reported self-intersections but did not write "
                    "selfInterPoints.obj"
                )
            output_points = _read_obj_vertices(points_path)
            if len(output_points) != count:
                raise VoxelWrapError(
                    "surfaceCheck self-intersection count disagrees with its point file: "
                    f"reported={count}, points={len(output_points)}"
                )
            points = [
                tuple(value / output_scale for value in point)
                for point in output_points
            ]
        else:
            points = []
        return {
            "intersection_count": count,
            "points_source_units": [list(point) for point in points],
            "audited_output_scale": output_scale,
            "surface_check_log_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }


def _transform_surface_file(
    source: Path,
    output: Path,
    scale: float,
    openfoam_launcher: Path,
) -> str:
    completed = subprocess.run(
        [
            str(openfoam_launcher),
            "surfaceTransformPoints",
            "-write-scale",
            f"{scale:.17g}",
            str(source),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    log = completed.stdout + completed.stderr
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise VoxelWrapError(
            "OpenFOAM surfaceTransformPoints failed while serializing the surface: "
            f"exit={completed.returncode}\n{log[-4000:]}"
        )
    return log


def _audit_serialized_surface(
    surface_path: Path,
    openfoam_launcher: Path,
    source_scale: float,
) -> dict[str, Any]:
    """Audit an exact output file after reversible conditioning to source scale."""

    with tempfile.TemporaryDirectory(prefix="auv-serialized-surface-audit-") as name:
        directory = Path(name)
        normalized = directory / "exact_output_source_scale.stl"
        transform_log = _transform_surface_file(
            surface_path,
            normalized,
            source_scale,
            openfoam_launcher,
        )
        completed = subprocess.run(
            [
                str(openfoam_launcher),
                "surfaceCheck",
                "-checkSelfIntersection",
                str(normalized),
            ],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
        )
        log = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise VoxelWrapError(
                "OpenFOAM surfaceCheck failed while auditing the serialized surface: "
                f"exit={completed.returncode}\n{log[-4000:]}"
            )
        count = _parse_surface_check_result(log)
        points_path = directory / "selfInterPoints.obj"
        if count:
            if not points_path.is_file():
                raise VoxelWrapError(
                    "surfaceCheck reported serialized-surface intersections but did "
                    "not write selfInterPoints.obj"
                )
            points = _read_obj_vertices(points_path)
            if len(points) != count:
                raise VoxelWrapError(
                    "serialized-surface intersection count disagrees with its point file"
                )
        else:
            points = []
        return {
            "intersection_count": count,
            "points_source_units": [list(point) for point in points],
            "audited_output_scale": source_scale,
            "surface_transform_log_sha256": hashlib.sha256(
                transform_log.encode("utf-8")
            ).hexdigest(),
            "surface_check_log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
        }


def _volume_validation(
    actual_volume: float,
    expected_volume: float | None,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Return a unit-agnostic displaced-volume validation record."""

    if not math.isfinite(actual_volume) or actual_volume <= 0.0:
        raise VoxelWrapError("rebuilt surface has a non-positive or non-finite volume")
    if not math.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise VoxelWrapError("volume relative tolerance must lie in (0, 1)")
    if expected_volume is None:
        return {
            "enabled": False,
            "actual_volume_source_units_cubed": float(actual_volume),
        }
    if not math.isfinite(expected_volume) or expected_volume <= 0.0:
        raise VoxelWrapError("expected volume must be positive and finite")
    relative_error = abs(float(actual_volume) - float(expected_volume)) / float(expected_volume)
    return {
        "enabled": True,
        "expected_volume_source_units_cubed": float(expected_volume),
        "actual_volume_source_units_cubed": float(actual_volume),
        "relative_error": float(relative_error),
        "relative_tolerance": float(relative_tolerance),
        "passed": bool(relative_error <= relative_tolerance),
    }


def _temporary_path(parent: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=".voxel-wrap-", suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _resolve_wrap_paths(input_path, output_path, report_path, force: bool):
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    report = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output.with_suffix(output.suffix + ".json")
    )
    if not source.is_file():
        raise VoxelWrapError(f"input STL does not exist: {source}")
    if source.suffix.lower() != ".stl":
        raise VoxelWrapError(f"input must be an STL: {source}")
    if output.suffix.lower() != ".obj":
        raise VoxelWrapError(f"output must end in .obj: {output}")
    if source == output:
        raise VoxelWrapError("input and output surface paths must differ")
    if output == report:
        raise VoxelWrapError("surface and JSON output paths must differ")
    existing = [path for path in (output, report) if path.exists()]
    if existing and not force:
        raise VoxelWrapError(
            "refusing to overwrite existing output(s): " + ", ".join(str(path) for path in existing)
        )
    return source, output, report


def _validate_wrap_parameters(
    *,
    voxel_size: float,
    closing_iterations: int,
    pad_voxels: int,
    grid_anchor: Sequence[float],
    max_voxels: int,
    expected_volume: float | None,
    volume_relative_tolerance: float,
    maximum_output_to_input_distance_voxels: float,
    maximum_output_to_input_p99_distance_voxels: float,
) -> None:
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise VoxelWrapError("voxel size must be finite and greater than zero")
    if closing_iterations < 0:
        raise VoxelWrapError("closing iterations cannot be negative")
    if pad_voxels < 2:
        raise VoxelWrapError("at least two padding voxels are required")
    if len(grid_anchor) != 3 or not all(
        math.isfinite(float(value)) for value in grid_anchor
    ):
        raise VoxelWrapError("grid anchor must contain three finite coordinates")
    if max_voxels < 0:
        raise VoxelWrapError("max voxels cannot be negative (use 0 for no limit)")
    if expected_volume is not None and (
        not math.isfinite(expected_volume) or expected_volume <= 0.0
    ):
        raise VoxelWrapError("expected volume must be positive and finite")
    if not math.isfinite(volume_relative_tolerance) or not 0.0 < volume_relative_tolerance < 1.0:
        raise VoxelWrapError("volume relative tolerance must lie in (0, 1)")
    if (
        not math.isfinite(maximum_output_to_input_distance_voxels)
        or maximum_output_to_input_distance_voxels <= 0.0
        or not math.isfinite(maximum_output_to_input_p99_distance_voxels)
        or maximum_output_to_input_p99_distance_voxels <= 0.0
    ):
        raise VoxelWrapError("surface-distance limits must be finite and positive")


def _rebuild_voxel_surface(
    source: Path,
    *,
    voxel_size: float,
    closing_iterations: int,
    pad_voxels: int,
    grid_anchor: Sequence[float],
    max_voxels: int,
) -> dict[str, Any]:
    np, ndimage, vtk, numpy_to_vtk, vtk_to_numpy = _require_backend()
    input_surface = _read_surface(source, vtk)
    input_topology = _topology(input_surface, vtk)
    barrier, origin, _dimensions, voxel_report = _voxelize_surface(
        input_surface,
        voxel_size,
        pad_voxels,
        max_voxels,
        grid_anchor,
        vtk,
        vtk_to_numpy,
    )
    solid = _solid_from_barrier(barrier, ndimage, closing_iterations)
    solid_voxels = int(solid.sum())
    labels, solid_components = ndimage.label(solid, structure=_six_neighbour_structure(ndimage))
    del labels
    rebuilt = _extract_surface(solid, origin, voxel_size, vtk, numpy_to_vtk)
    rebuilt, component_cleanup = _remove_subvoxel_component_artifacts(
        rebuilt,
        voxel_size,
        vtk,
    )
    rebuilt = _orient_normals(rebuilt, vtk)
    return {
        "np": np,
        "ndimage": ndimage,
        "vtk": vtk,
        "numpy_to_vtk": numpy_to_vtk,
        "vtk_to_numpy": vtk_to_numpy,
        "origin": origin,
        "input_surface": input_surface,
        "input_topology": input_topology,
        "rebuilt": rebuilt,
        "solid": solid,
        "solid_components": int(solid_components),
        "solid_voxels": solid_voxels,
        "voxel_report": voxel_report,
        "component_cleanup_history": [component_cleanup],
    }


def _validate_rebuilt_surface(
    rebuilt: Any,
    vtk: Any,
    *,
    require_single_component: bool,
    expected_volume: float | None,
    volume_relative_tolerance: float,
):
    topology = _topology(rebuilt, vtk)
    if not topology["watertight_manifold"] or (
        require_single_component and not topology["single_component"]
    ):
        raise VoxelWrapError(
            "rebuilt surface failed topology validation: "
            f"boundary={topology['boundary_edges']}, "
            f"non-manifold={topology['non_manifold_edges']}, "
            f"components={topology['connected_regions']}"
        )
    geometry = _geometry_metrics(rebuilt, vtk)
    volume = _volume_validation(
        geometry["enclosed_volume_source_units_cubed"],
        expected_volume,
        volume_relative_tolerance,
    )
    if volume.get("enabled") and not volume["passed"]:
        raise VoxelWrapError(
            "rebuilt surface displaced volume failed: "
            f"actual={volume['actual_volume_source_units_cubed']:.12g}, "
            f"expected={volume['expected_volume_source_units_cubed']:.12g}, "
            f"relative_error={volume['relative_error']:.6g} exceeds "
            f"tolerance={volume['relative_tolerance']:.6g}"
        )
    return topology, geometry, volume


def _deterministic_point_sample(
    surface: Any, vtk: Any, maximum_points: int
) -> tuple[Any, dict[str, Any]]:
    source_count = int(surface.GetNumberOfPoints())
    stride = max(1, int(math.ceil(source_count / maximum_points)))
    if stride == 1:
        return surface, {
            "source_point_count": source_count,
            "sample_point_count": source_count,
            "stride": 1,
            "method": "all points",
        }
    mask = vtk.vtkMaskPoints()
    mask.SetInputData(surface)
    mask.RandomModeOff()
    mask.SetOnRatio(stride)
    mask.SetOffset(0)
    mask.GenerateVerticesOn()
    mask.Update()
    sampled = vtk.vtkPolyData()
    sampled.ShallowCopy(mask.GetOutput())
    return sampled, {
        "source_point_count": source_count,
        "sample_point_count": int(sampled.GetNumberOfPoints()),
        "stride": stride,
        "method": "deterministic VTK point-index stride from offset zero",
    }


def _sampled_distance_distribution(
    source: Any, target: Any, state: dict[str, Any]
) -> dict[str, Any]:
    sampled, sampling = _deterministic_point_sample(
        source, state["vtk"], MAX_DISTANCE_SAMPLE_POINTS
    )
    return {
        **_distribution(
            _point_distances(
                sampled,
                target,
                state["vtk"],
                state["vtk_to_numpy"],
            ),
            state["np"],
        ),
        "sampling": sampling,
    }


def _full_distance_distribution(
    source: Any, target: Any, state: dict[str, Any]
) -> dict[str, Any]:
    """Measure every source vertex for a geometry acceptance gate."""

    values = _point_distances(
        source,
        target,
        state["vtk"],
        state["vtk_to_numpy"],
    )
    return {
        **_distribution(values, state["np"]),
        "sampling": {
            "source_point_count": int(source.GetNumberOfPoints()),
            "sample_point_count": int(source.GetNumberOfPoints()),
            "stride": 1,
            "method": "all output points (hard acceptance gate)",
        },
    }


def _surface_distance_report(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_points_to_output_surface": _sampled_distance_distribution(
            state["input_surface"], state["rebuilt"], state
        ),
        "output_points_to_input_surface": _full_distance_distribution(
            state["rebuilt"], state["input_surface"], state
        ),
        "unit": "source_coordinate_unit",
        "purpose": (
            "input-to-output is a sampled diagnostic because the selected STEP "
            "surface contains discarded internal detail; output-to-input uses every "
            "output point and is a hard local-shape gate"
        ),
    }


def _shape_deviation_validation(
    distances: Mapping[str, Any],
    voxel_size: float,
    maximum_distance_voxels: float,
    maximum_p99_distance_voxels: float,
) -> dict[str, Any]:
    output_to_input = distances["output_points_to_input_surface"]
    maximum = float(output_to_input["max"])
    p99 = float(output_to_input["p99"])
    maximum_limit = voxel_size * maximum_distance_voxels
    p99_limit = voxel_size * maximum_p99_distance_voxels
    passed = maximum <= maximum_limit and p99 <= p99_limit
    return {
        "method": "unsigned distance from every output vertex to the selected STEP-derived surface",
        "maximum_distance_source_units": maximum,
        "maximum_distance_limit_source_units": maximum_limit,
        "maximum_distance_limit_voxels": maximum_distance_voxels,
        "p99_distance_source_units": p99,
        "p99_distance_limit_source_units": p99_limit,
        "p99_distance_limit_voxels": maximum_p99_distance_voxels,
        "passed": passed,
    }


def _port_starboard_symmetry_report(
    state: dict[str, Any], plane_y: float
) -> dict[str, Any]:
    reflected = _reflect_about_y_plane(state["rebuilt"], plane_y, state["vtk"])
    return {
        "reflection_plane_y_source_units": plane_y,
        "output_points_to_reflected_surface": _sampled_distance_distribution(
            state["rebuilt"], reflected, state
        ),
        "reflected_points_to_output_surface": _sampled_distance_distribution(
            reflected, state["rebuilt"], state
        ),
        "unit": "source_coordinate_unit",
        "sampling_note": (
            "deterministic point subsampling is a diagnostic; forbidden CFD load "
            "fractions remain the coefficient-identification acceptance gate"
        ),
    }


def _wrap_payload(
    *,
    source: Path,
    output: Path,
    temporary_surface: Path,
    output_surface: Any,
    state: dict[str, Any],
    output_topology: dict[str, Any],
    output_geometry: dict[str, Any],
    volume_validation: dict[str, Any],
    distances: dict[str, Any],
    symmetry: dict[str, Any],
    shape_deviation_validation: dict[str, Any],
    parameters: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    input_surface = state["input_surface"]
    rebuilt = state["rebuilt"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "backend": backend_status(),
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
            "points": int(input_surface.GetNumberOfPoints()),
            "triangles": int(input_surface.GetNumberOfCells()),
            "geometry": _geometry_metrics(input_surface, state["vtk"]),
            "topology": state["input_topology"],
        },
        "parameters": parameters,
        "voxel_grid": {
            **state["voxel_report"],
            "solid_components": state["solid_components"],
            "solid_voxels": state["solid_voxels"],
        },
        "component_cleanup_history": state["component_cleanup_history"],
        "source_scale_self_intersection_audit": state[
            "source_scale_self_intersection_audit"
        ],
        "self_intersection_audit": state["self_intersection_audit"],
        "output": {
            "path": str(output),
            "sha256": _sha256(temporary_surface),
            "size_bytes": temporary_surface.stat().st_size,
            "points": int(output_surface.GetNumberOfPoints()),
            "triangles": int(output_surface.GetNumberOfCells()),
            "geometry": output_geometry,
            "topology": output_topology,
            "normals": {"consistent_and_auto_oriented": True},
            "self_intersection": {
                "audited": True,
                "value": False,
                "method": state["self_intersection_audit"]["method"],
                "final_log_sha256": state["self_intersection_audit"][
                    "surface_check_log_sha256"
                ],
            },
        },
        "volume_validation": volume_validation,
        "distance": distances,
        "shape_deviation_validation": shape_deviation_validation,
        "port_starboard_symmetry": symmetry,
        "elapsed_seconds": elapsed_seconds,
    }


def wrap_surface(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    voxel_size: float,
    report_path: str | os.PathLike[str] | None = None,
    closing_iterations: int = DEFAULT_CLOSING_ITERATIONS,
    pad_voxels: int = DEFAULT_PAD_VOXELS,
    grid_anchor: Sequence[float] = (0.0, 0.0, 0.0),
    openfoam_launcher: str | os.PathLike[str] = DEFAULT_OPENFOAM_LAUNCHER,
    output_scale: float = 1.0,
    symmetry_plane_y: float = 0.0,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    require_single_component: bool = True,
    expected_volume: float | None = None,
    volume_relative_tolerance: float = DEFAULT_VOLUME_RELATIVE_TOLERANCE,
    maximum_output_to_input_distance_voxels: float = (
        DEFAULT_MAX_OUTPUT_TO_INPUT_DISTANCE_VOXELS
    ),
    maximum_output_to_input_p99_distance_voxels: float = (
        DEFAULT_MAX_OUTPUT_TO_INPUT_P99_DISTANCE_VOXELS
    ),
    force: bool = False,
) -> dict[str, Any]:
    """Create a watertight metre-scale OBJ and an atomic JSON audit report."""

    start = time.monotonic()
    source, output, report = _resolve_wrap_paths(input_path, output_path, report_path, force)
    _validate_wrap_parameters(
        voxel_size=voxel_size,
        closing_iterations=closing_iterations,
        pad_voxels=pad_voxels,
        grid_anchor=grid_anchor,
        max_voxels=max_voxels,
        expected_volume=expected_volume,
        volume_relative_tolerance=volume_relative_tolerance,
        maximum_output_to_input_distance_voxels=(
            maximum_output_to_input_distance_voxels
        ),
        maximum_output_to_input_p99_distance_voxels=(
            maximum_output_to_input_p99_distance_voxels
        ),
    )
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise VoxelWrapError("output_scale must be finite and positive")
    state = _rebuild_voxel_surface(
        source,
        voxel_size=voxel_size,
        closing_iterations=closing_iterations,
        pad_voxels=pad_voxels,
        grid_anchor=grid_anchor,
        max_voxels=max_voxels,
    )
    if not math.isfinite(symmetry_plane_y):
        raise VoxelWrapError("symmetry plane y coordinate must be finite")
    source_topology, source_geometry, volume_validation = _validate_rebuilt_surface(
        state["rebuilt"],
        state["vtk"],
        require_single_component=require_single_component,
        expected_volume=expected_volume,
        volume_relative_tolerance=volume_relative_tolerance,
    )
    print(
        "surface extraction "
        f"triangles={state['rebuilt'].GetNumberOfCells()} "
        f"non_manifold_edges={source_topology['non_manifold_edges']} "
        f"components={source_topology['connected_regions']}",
        flush=True,
    )
    launcher = Path(openfoam_launcher).expanduser().resolve()
    state["source_scale_self_intersection_audit"] = _audit_self_intersections(
        state["rebuilt"], state["vtk"], launcher, 1.0
    )
    print(
        "source-scale self-intersection audit "
        f"count={state['source_scale_self_intersection_audit']['intersection_count']}",
        flush=True,
    )
    if state["source_scale_self_intersection_audit"]["intersection_count"] != 0:
        raise VoxelWrapError(
            "Lewiner marching-cubes output is self-intersecting at "
            f"{state['source_scale_self_intersection_audit']['intersection_count']} "
            "location(s) in source coordinates"
        )
    print("surface distance and symmetry audits", flush=True)
    distances = _surface_distance_report(state)
    shape_deviation_validation = _shape_deviation_validation(
        distances,
        voxel_size,
        maximum_output_to_input_distance_voxels,
        maximum_output_to_input_p99_distance_voxels,
    )
    if not shape_deviation_validation["passed"]:
        raise VoxelWrapError(
            "rebuilt surface local shape deviation failed: "
            f"max={shape_deviation_validation['maximum_distance_source_units']:.12g} "
            f"> {shape_deviation_validation['maximum_distance_limit_source_units']:.12g} "
            "or "
            f"p99={shape_deviation_validation['p99_distance_source_units']:.12g} "
            f"> {shape_deviation_validation['p99_distance_limit_source_units']:.12g}"
        )
    symmetry = _port_starboard_symmetry_report(state, symmetry_plane_y)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary_source_stl = _temporary_path(output.parent, ".stl")
    temporary_surface = _temporary_path(output.parent, ".obj")
    temporary_json = _temporary_path(report.parent, ".json")
    try:
        _write_stl(state["rebuilt"], temporary_source_stl, state["vtk"])
        serialization_log = _transform_surface_file(
            temporary_source_stl,
            temporary_surface,
            output_scale,
            launcher,
        )
        output_surface = _read_surface(temporary_surface, state["vtk"])
        output_topology = _topology(output_surface, state["vtk"])
        output_geometry = _geometry_metrics(output_surface, state["vtk"])
        if output_topology != source_topology:
            raise VoxelWrapError(
                "metre-OBJ serialization changed the audited surface topology"
            )
        state["self_intersection_audit"] = _audit_serialized_surface(
            temporary_surface,
            launcher,
            1.0 / output_scale,
        )
        print(
            "scale-normalized exact-file self-intersection audit "
            f"count={state['self_intersection_audit']['intersection_count']}",
            flush=True,
        )
        state["self_intersection_audit"].update(
            {
                "serialized_metre_file_sha256": _sha256(temporary_surface),
                "serialization_log_sha256": hashlib.sha256(
                    serialization_log.encode("utf-8")
                ).hexdigest(),
                "method": (
                    "read exact shared-vertex metre OBJ with OpenFOAM, uniformly "
                    "normalize to source-unit scale, then surfaceCheck "
                    "-checkSelfIntersection"
                ),
                "reason_for_normalization": (
                    "surfaceCheck intersection tolerances are not scale invariant; "
                    "uniform scaling preserves geometric intersection topology"
                ),
            }
        )
        if state["self_intersection_audit"]["intersection_count"] != 0:
            raise VoxelWrapError(
                "serialized metre OBJ is self-intersecting at "
                f"{state['self_intersection_audit']['intersection_count']} location(s) "
                "after scale-normalized audit"
            )
        parameters = {
            "voxel_size_source_units": voxel_size,
            "source_to_output_coordinate_scale": output_scale,
            "output_surface_format": "OpenFOAM shared-vertex OBJ",
            "outside_connectivity": 6,
            "closing_iterations": closing_iterations,
            "closing_kernel": [3, 3, 3],
            "pad_voxels": pad_voxels,
            "grid_anchor_source_units": [float(value) for value in grid_anchor],
            "openfoam_launcher": str(Path(openfoam_launcher).expanduser().resolve()),
            "symmetry_plane_y_source_units": symmetry_plane_y,
            "max_voxels": max_voxels,
            "require_single_component": require_single_component,
            "expected_volume_source_units_cubed": expected_volume,
            "volume_relative_tolerance": volume_relative_tolerance,
            "maximum_output_to_input_distance_voxels": (
                maximum_output_to_input_distance_voxels
            ),
            "maximum_output_to_input_p99_distance_voxels": (
                maximum_output_to_input_p99_distance_voxels
            ),
            "surface_extraction": {
                "method": (
                    "scikit-image Lewiner/Chernyaev Marching Cubes 33 at binary level 0.5"
                ),
                "vertex_smoothing": False,
                "reason": (
                    "the Lewiner case table resolves trilinear-interpolant ambiguities "
                    "with topological guarantees; the result is not smoothed or changed "
                    "to repair a failed topology audit"
                ),
            },
        }
        payload = _wrap_payload(
            source=source,
            output=output,
            temporary_surface=temporary_surface,
            output_surface=output_surface,
            state=state,
            output_topology=output_topology,
            output_geometry=output_geometry,
            volume_validation=volume_validation,
            distances=distances,
            symmetry=symmetry,
            shape_deviation_validation=shape_deviation_validation,
            parameters=parameters,
            elapsed_seconds=float(time.monotonic() - start),
        )
        temporary_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_surface, output)
        os.replace(temporary_json, report)
        return payload
    finally:
        temporary_source_stl.unlink(missing_ok=True)
        temporary_surface.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a triangle-soup STL as one unsmoothed watertight metre OBJ."
    )
    parser.add_argument("input", help="input triangle STL")
    parser.add_argument("output", help="output shared-vertex OBJ")
    parser.add_argument(
        "--voxel-size",
        type=float,
        required=True,
        help="isotropic voxel size in the input STL coordinate unit",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        required=True,
        help="uniform source-to-output coordinate scale (0.001 for mm to m)",
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="diagnostic mode: allow multiple closed output components",
    )
    parser.add_argument(
        "--json",
        dest="report_path",
        help="JSON report path (default: OUTPUT.obj.json)",
    )
    parser.add_argument(
        "--closing",
        type=int,
        default=DEFAULT_CLOSING_ITERATIONS,
        help=f"3x3x3 binary closing iterations (default: {DEFAULT_CLOSING_ITERATIONS})",
    )
    parser.add_argument(
        "--pad-voxels",
        type=int,
        default=DEFAULT_PAD_VOXELS,
        help=f"padding around the source bounds (default: {DEFAULT_PAD_VOXELS})",
    )
    parser.add_argument(
        "--grid-anchor",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="source-coordinate lattice anchor (default: body COM origin 0 0 0)",
    )
    parser.add_argument(
        "--openfoam-launcher",
        default=str(DEFAULT_OPENFOAM_LAUNCHER),
        help="launcher that supplies the pinned OpenFOAM surfaceCheck environment",
    )
    parser.add_argument(
        "--symmetry-plane-y",
        type=float,
        default=0.0,
        help="body-frame port/starboard reflection plane in input coordinates (default: 0)",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=DEFAULT_MAX_VOXELS,
        help=f"grid safety limit; 0 disables it (default: {DEFAULT_MAX_VOXELS})",
    )
    parser.add_argument(
        "--expected-volume",
        type=float,
        required=True,
        help="required enclosed volume in the input STL unit cubed",
    )
    parser.add_argument(
        "--volume-relative-tolerance",
        type=float,
        default=DEFAULT_VOLUME_RELATIVE_TOLERANCE,
        help=(
            "maximum relative error from --expected-volume "
            f"(default: {DEFAULT_VOLUME_RELATIVE_TOLERANCE:g})"
        ),
    )
    parser.add_argument("--force", action="store_true", help="replace existing OBJ and JSON outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = wrap_surface(
            arguments.input,
            arguments.output,
            voxel_size=arguments.voxel_size,
            report_path=arguments.report_path,
            closing_iterations=arguments.closing,
            pad_voxels=arguments.pad_voxels,
            grid_anchor=arguments.grid_anchor,
            openfoam_launcher=arguments.openfoam_launcher,
            output_scale=arguments.output_scale,
            symmetry_plane_y=arguments.symmetry_plane_y,
            max_voxels=arguments.max_voxels,
            require_single_component=not arguments.allow_multiple,
            expected_volume=arguments.expected_volume,
            volume_relative_tolerance=arguments.volume_relative_tolerance,
            force=arguments.force,
        )
    except VoxelWrapError as exc:
        print(f"voxel_wrap: error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "output": result["output"]["path"],
        "report": str(
            Path(arguments.report_path).expanduser().resolve()
            if arguments.report_path
            else Path(arguments.output).expanduser().resolve().with_suffix(
                Path(arguments.output).suffix + ".json"
            )
        ),
        "triangles": result["output"]["triangles"],
        "topology": result["output"]["topology"],
        "volume_validation": result["volume_validation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

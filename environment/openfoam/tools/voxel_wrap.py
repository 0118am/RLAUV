#!/usr/bin/env python3
"""Rebuild a triangle-soup STL as one watertight voxel exterior.

The implementation is deliberately conservative:

* input triangles are rasterized as barrier voxels;
* a six-neighbour flood fill identifies fluid reachable from the padded box;
* optional binary closing seals only the configured voxel-scale gaps;
* ``vtkSurfaceNets3D`` extracts a discrete, unsmoothed boundary;
* remaining digital non-manifold configurations are repaired locally; and
* topology and bidirectional surface-distance metrics are written to JSON.

The JSON report does not claim to audit triangle self-intersection.  The final
CFD gate remains ``surfaceCheck -checkSelfIntersection`` from OpenFOAM.
Coordinates and ``--voxel-size`` use the STL's native unit.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence


REPORT_SCHEMA_VERSION = 1
DEFAULT_CLOSING_ITERATIONS = 1
DEFAULT_PAD_VOXELS = 4
DEFAULT_REPAIR_ITERATIONS = 8
DEFAULT_REPAIR_RADIUS = 1
DEFAULT_MAX_VOXELS = 100_000_000
DEFAULT_VOLUME_RELATIVE_TOLERANCE = 0.02


class VoxelWrapError(RuntimeError):
    """Raised when a safe voxel wrap cannot be produced."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backend_status() -> dict[str, Any]:
    """Return availability details without making import-time VTK mandatory."""

    status: dict[str, Any] = {"ready": False}
    try:
        import numpy as np  # noqa: F401

        status["numpy_version"] = str(np.__version__)
    except Exception as exc:  # pragma: no cover - depends on local Python
        status["reason"] = f"NumPy unavailable: {exc}"
        return status
    try:
        # Some hosts have an incompatible binary SciPy beside NumPy. Suppress
        # its import-time ABI diagnostic and return it as structured status.
        with contextlib.redirect_stderr(io.StringIO()):
            from scipy import ndimage  # noqa: F401
            import scipy

        status["scipy_version"] = str(scipy.__version__)
    except Exception as exc:  # pragma: no cover - depends on local Python
        status["reason"] = f"SciPy ndimage unavailable or incompatible: {exc}"
        return status
    try:
        import vtk

        status["vtk_version"] = str(vtk.vtkVersion.GetVTKVersion())
        if not hasattr(vtk, "vtkSurfaceNets3D"):
            status["reason"] = "VTK lacks vtkSurfaceNets3D (VTK 9.3 or newer is required)"
            return status
    except Exception as exc:  # pragma: no cover - depends on local Python
        status["reason"] = f"VTK unavailable: {exc}"
        return status
    status["ready"] = True
    return status


def _require_backend() -> tuple[Any, Any, Any, Any, Any]:
    status = backend_status()
    if not status["ready"]:
        raise VoxelWrapError(str(status.get("reason", "voxel backend unavailable")))
    import numpy as np
    from scipy import ndimage
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    return np, ndimage, vtk, numpy_to_vtk, vtk_to_numpy


def _read_surface(path: Path, vtk: Any) -> Any:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.MergingOn()
    reader.Update()
    if reader.GetOutput().GetNumberOfPolys() <= 0:
        raise VoxelWrapError(f"VTK read no triangles from {path}")

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.PointMergingOn()
    clean.Update()
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(clean.GetOutput())
    if surface.GetNumberOfCells() <= 0:
        raise VoxelWrapError(f"cleaned input contains no triangles: {path}")
    return surface


def _connected_regions(surface: Any, vtk: Any) -> int:
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    return int(connectivity.GetNumberOfExtractedRegions())


def _feature_edge_count(
    surface: Any,
    vtk: Any,
    *,
    boundary: bool = False,
    non_manifold: bool = False,
) -> int:
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(surface)
    edges.BoundaryEdgesOff()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    if boundary:
        edges.BoundaryEdgesOn()
    if non_manifold:
        edges.NonManifoldEdgesOn()
    edges.Update()
    return int(edges.GetOutput().GetNumberOfCells())


def _non_manifold_edges(surface: Any, vtk: Any) -> Any:
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(surface)
    edges.BoundaryEdgesOff()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOn()
    edges.ColoringOff()
    edges.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(edges.GetOutput())
    return output


def _image_from_mask(
    mask: Any,
    origin: Sequence[float],
    spacing: float,
    vtk: Any,
    numpy_to_vtk: Any,
) -> Any:
    nz, ny, nx = mask.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetOrigin(*origin)
    image.SetSpacing(spacing, spacing, spacing)
    scalars = numpy_to_vtk(
        mask.astype("uint8").ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    scalars.SetName("solid")
    image.GetPointData().SetScalars(scalars)
    return image


def _extract_surface(
    mask: Any,
    origin: Sequence[float],
    spacing: float,
    vtk: Any,
    numpy_to_vtk: Any,
) -> Any:
    image = _image_from_mask(mask, origin, spacing, vtk, numpy_to_vtk)
    surface_nets = vtk.vtkSurfaceNets3D()
    surface_nets.SetInputData(image)
    surface_nets.SetValue(0, 1)
    surface_nets.SetBackgroundLabel(0)
    surface_nets.SetOutputStyleToBoundary()
    surface_nets.SetOutputMeshTypeToTriangles()
    # Smoothing created self-intersections in the representative AUV mesh.
    # Keep this topology-preserving reconstruction intentionally unsmoothed.
    surface_nets.SmoothingOff()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(surface_nets.GetOutputPort())
    clean.PointMergingOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(triangles.GetOutput())
    return output


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
    vtk: Any,
    vtk_to_numpy: Any,
) -> tuple[Any, tuple[float, float, float], tuple[int, int, int], dict[str, Any]]:
    bounds = surface.GetBounds()
    pad = pad_voxels * voxel_size
    origin = (bounds[0] - pad, bounds[2] - pad, bounds[4] - pad)
    upper = (bounds[1] + pad, bounds[3] + pad, bounds[5] + pad)
    dimensions = tuple(
        int(math.ceil((upper[axis] - origin[axis]) / voxel_size)) + 1
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
        "bounds": list(model_bounds),
        "surface_barrier_voxels": int(barrier.sum()),
    }
    return barrier, origin, dimensions, details


def _repair_non_manifold(
    solid: Any,
    origin: Sequence[float],
    spacing: float,
    max_iterations: int,
    radius: int,
    ndimage: Any,
    vtk: Any,
    numpy_to_vtk: Any,
) -> tuple[Any, Any, list[dict[str, int]]]:
    history: list[dict[str, int]] = []
    for iteration in range(max_iterations + 1):
        surface = _extract_surface(solid, origin, spacing, vtk, numpy_to_vtk)
        bad = _non_manifold_edges(surface, vtk)
        count = int(bad.GetNumberOfCells())
        entry = {
            "iteration": iteration,
            "non_manifold_edges": count,
            "solid_voxels": int(solid.sum()),
            "triangles": int(surface.GetNumberOfCells()),
            "new_voxels": 0,
        }
        history.append(entry)
        if count == 0:
            return solid, surface, history
        if iteration == max_iterations:
            break

        centres: list[Any] = []
        ids = vtk.vtkIdList()
        for cell_index in range(count):
            bad.GetCellPoints(cell_index, ids)
            points = [bad.GetPoint(ids.GetId(index)) for index in range(ids.GetNumberOfIds())]
            centres.append(
                tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
            )

        nz, ny, nx = solid.shape
        changed = 0
        for xyz in centres:
            ix, iy, iz = (
                int(math.floor((xyz[axis] - origin[axis]) / spacing))
                for axis in range(3)
            )
            z0, z1 = max(0, iz - radius), min(nz, iz + radius + 1)
            y0, y1 = max(0, iy - radius), min(ny, iy + radius + 1)
            x0, x1 = max(0, ix - radius), min(nx, ix + radius + 1)
            patch = solid[z0:z1, y0:y1, x0:x1]
            changed += int(patch.size - patch.sum())
            patch[...] = True
        entry["new_voxels"] = changed
        if changed == 0:
            break
        solid = ~_outside_flood(solid, ndimage)

    remaining = history[-1]["non_manifold_edges"]
    raise VoxelWrapError(
        f"local topology repair did not converge: {remaining} non-manifold edges remain "
        f"after {history[-1]['iteration']} iteration(s)"
    )


def _orient_normals(surface: Any, vtk: Any) -> Any:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(normals.GetOutput())
    return output


def _distribution(values: Any, np: Any) -> dict[str, float | int]:
    if values.size == 0:
        raise VoxelWrapError("distance filter produced an empty array")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _point_distances(source: Any, target: Any, vtk: Any, vtk_to_numpy: Any) -> Any:
    distances = vtk.vtkDistancePolyDataFilter()
    distances.SetInputData(0, source)
    distances.SetInputData(1, target)
    distances.SignedDistanceOff()
    distances.ComputeSecondDistanceOff()
    distances.ComputeCellCenterDistanceOff()
    distances.Update()
    array = distances.GetOutput().GetPointData().GetArray("Distance")
    if array is None:
        raise VoxelWrapError("VTK failed to compute point-to-surface distances")
    return vtk_to_numpy(array)


def _bounds(surface: Any) -> dict[str, list[float]]:
    raw = surface.GetBounds()
    minimum = [float(raw[0]), float(raw[2]), float(raw[4])]
    maximum = [float(raw[1]), float(raw[3]), float(raw[5])]
    return {
        "min": minimum,
        "max": maximum,
        "size": [maximum[index] - minimum[index] for index in range(3)],
    }


def _topology(surface: Any, vtk: Any) -> dict[str, Any]:
    boundary_edges = _feature_edge_count(surface, vtk, boundary=True)
    non_manifold_edges = _feature_edge_count(surface, vtk, non_manifold=True)
    connected_regions = _connected_regions(surface, vtk)
    extract_edges = vtk.vtkExtractEdges()
    extract_edges.SetInputData(surface)
    extract_edges.Update()
    vertices = int(surface.GetNumberOfPoints())
    edges = int(extract_edges.GetOutput().GetNumberOfCells())
    faces = int(surface.GetNumberOfCells())
    euler = vertices - edges + faces
    genus: int | None = None
    if boundary_edges == 0 and non_manifold_edges == 0 and connected_regions == 1:
        candidate = (2 - euler) / 2
        if candidate >= 0 and candidate.is_integer():
            genus = int(candidate)
    return {
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "connected_regions": connected_regions,
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "euler_characteristic": euler,
        "genus": genus,
        "watertight_manifold": boundary_edges == 0 and non_manifold_edges == 0,
        "single_component": connected_regions == 1,
    }


def _geometry_metrics(surface: Any, vtk: Any) -> dict[str, Any]:
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()
    return {
        "bounds": _bounds(surface),
        "surface_area_source_units_squared": float(mass.GetSurfaceArea()),
        "enclosed_volume_source_units_cubed": float(mass.GetVolume()),
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


def _write_stl(surface: Any, path: Path, vtk: Any) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if int(writer.Write()) != 1 or not path.is_file() or path.stat().st_size <= 84:
        raise VoxelWrapError(f"VTK failed to write {path}")


def wrap_surface(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    voxel_size: float,
    report_path: str | os.PathLike[str] | None = None,
    closing_iterations: int = DEFAULT_CLOSING_ITERATIONS,
    pad_voxels: int = DEFAULT_PAD_VOXELS,
    repair_iterations: int = DEFAULT_REPAIR_ITERATIONS,
    repair_radius: int = DEFAULT_REPAIR_RADIUS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    require_single_component: bool = True,
    expected_volume: float | None = None,
    volume_relative_tolerance: float = DEFAULT_VOLUME_RELATIVE_TOLERANCE,
    force: bool = False,
    invocation: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a watertight exterior STL and an atomic JSON audit report."""

    start = time.monotonic()
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
    if output.suffix.lower() != ".stl":
        raise VoxelWrapError(f"output must end in .stl: {output}")
    if source == output:
        raise VoxelWrapError("input and output STL paths must differ")
    if output == report:
        raise VoxelWrapError("STL and JSON output paths must differ")
    existing = [path for path in (output, report) if path.exists()]
    if existing and not force:
        raise VoxelWrapError(
            "refusing to overwrite existing output(s): " + ", ".join(str(path) for path in existing)
        )
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise VoxelWrapError("voxel size must be finite and greater than zero")
    if closing_iterations < 0:
        raise VoxelWrapError("closing iterations cannot be negative")
    if pad_voxels < 2:
        raise VoxelWrapError("at least two padding voxels are required")
    if repair_iterations < 0 or repair_radius < 0:
        raise VoxelWrapError("repair iterations and radius cannot be negative")
    if max_voxels < 0:
        raise VoxelWrapError("max voxels cannot be negative (use 0 for no limit)")
    if expected_volume is not None and (
        not math.isfinite(expected_volume) or expected_volume <= 0.0
    ):
        raise VoxelWrapError("expected volume must be positive and finite")
    if not math.isfinite(volume_relative_tolerance) or not 0.0 < volume_relative_tolerance < 1.0:
        raise VoxelWrapError("volume relative tolerance must lie in (0, 1)")

    np, ndimage, vtk, numpy_to_vtk, vtk_to_numpy = _require_backend()
    input_surface = _read_surface(source, vtk)
    input_topology = _topology(input_surface, vtk)
    barrier, origin, _dimensions, voxel_report = _voxelize_surface(
        input_surface,
        voxel_size,
        pad_voxels,
        max_voxels,
        vtk,
        vtk_to_numpy,
    )
    solid = _solid_from_barrier(barrier, ndimage, closing_iterations)
    solid_before_repair = int(solid.sum())
    labels, solid_components = ndimage.label(solid, structure=_six_neighbour_structure(ndimage))
    del labels
    solid, rebuilt, repair_history = _repair_non_manifold(
        solid,
        origin,
        voxel_size,
        repair_iterations,
        repair_radius,
        ndimage,
        vtk,
        numpy_to_vtk,
    )
    rebuilt = _orient_normals(rebuilt, vtk)
    output_topology = _topology(rebuilt, vtk)
    topology_failed = not output_topology["watertight_manifold"]
    component_failed = require_single_component and not output_topology["single_component"]
    if topology_failed or component_failed:
        raise VoxelWrapError(
            "rebuilt surface failed the VTK topology gate: "
            f"boundary={output_topology['boundary_edges']}, "
            f"non-manifold={output_topology['non_manifold_edges']}, "
            f"components={output_topology['connected_regions']}"
        )
    output_geometry = _geometry_metrics(rebuilt, vtk)
    volume_validation = _volume_validation(
        output_geometry["enclosed_volume_source_units_cubed"],
        expected_volume,
        volume_relative_tolerance,
    )
    if volume_validation.get("enabled") and not volume_validation["passed"]:
        raise VoxelWrapError(
            "rebuilt surface displaced-volume gate failed: "
            f"actual={volume_validation['actual_volume_source_units_cubed']:.12g}, "
            f"expected={volume_validation['expected_volume_source_units_cubed']:.12g}, "
            f"relative_error={volume_validation['relative_error']:.6g} exceeds "
            f"tolerance={volume_validation['relative_tolerance']:.6g}"
        )

    distances = {
        "input_points_to_output_surface": _distribution(
            _point_distances(input_surface, rebuilt, vtk, vtk_to_numpy), np
        ),
        "output_points_to_input_surface": _distribution(
            _point_distances(rebuilt, input_surface, vtk, vtk_to_numpy), np
        ),
        "unit": "source_coordinate_unit",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary_stl = _temporary_path(output.parent, ".stl")
    temporary_json = _temporary_path(report.parent, ".json")
    try:
        _write_stl(rebuilt, temporary_stl, vtk)
        payload: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at_utc": _utc_timestamp(),
            "invocation": list(invocation) if invocation is not None else None,
            "backend": backend_status(),
            "input": {
                "path": str(source),
                "sha256": _sha256_file(source),
                "size_bytes": source.stat().st_size,
                "points": int(input_surface.GetNumberOfPoints()),
                "triangles": int(input_surface.GetNumberOfCells()),
                "geometry": _geometry_metrics(input_surface, vtk),
                "topology": input_topology,
            },
            "parameters": {
                "voxel_size_source_units": voxel_size,
                "outside_connectivity": 6,
                "closing_iterations": closing_iterations,
                "closing_kernel": [3, 3, 3],
                "pad_voxels": pad_voxels,
                "repair_iterations": repair_iterations,
                "repair_radius_voxels": repair_radius,
                "max_voxels": max_voxels,
                "require_single_component": require_single_component,
                "expected_volume_source_units_cubed": expected_volume,
                "volume_relative_tolerance": volume_relative_tolerance,
                "surface_smoothing": False,
            },
            "voxel_grid": {
                **voxel_report,
                "solid_components_before_repair": int(solid_components),
                "solid_voxels_before_repair": solid_before_repair,
                "solid_voxels_after_repair": int(solid.sum()),
            },
            "repair_history": repair_history,
            "output": {
                "path": str(output),
                "sha256": _sha256_file(temporary_stl),
                "size_bytes": temporary_stl.stat().st_size,
                "points": int(rebuilt.GetNumberOfPoints()),
                "triangles": int(rebuilt.GetNumberOfCells()),
                "geometry": output_geometry,
                "topology": output_topology,
                "normals": {"consistent_and_auto_oriented": True},
                "self_intersection": {
                    "audited": False,
                    "value": None,
                    "required_gate": "surfaceCheck -checkSelfIntersection",
                },
            },
            "volume_validation": volume_validation,
            "distance": distances,
            "elapsed_seconds": float(time.monotonic() - start),
        }
        temporary_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_stl, output)
        os.replace(temporary_json, report)
        return payload
    finally:
        temporary_stl.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a triangle-soup STL as one unsmoothed watertight voxel exterior."
    )
    parser.add_argument("input", help="input triangle STL")
    parser.add_argument("output", help="output binary STL")
    parser.add_argument(
        "--voxel-size",
        type=float,
        required=True,
        help="isotropic voxel size in the input STL coordinate unit",
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="diagnostic mode: allow multiple closed output components",
    )
    parser.add_argument(
        "--json",
        dest="report_path",
        help="JSON report path (default: OUTPUT.stl.json)",
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
        "--repair-iterations",
        type=int,
        default=DEFAULT_REPAIR_ITERATIONS,
        help=f"maximum local non-manifold repair passes (default: {DEFAULT_REPAIR_ITERATIONS})",
    )
    parser.add_argument(
        "--repair-radius",
        type=int,
        default=DEFAULT_REPAIR_RADIUS,
        help=f"local repair radius in voxels (default: {DEFAULT_REPAIR_RADIUS})",
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
    parser.add_argument("--force", action="store_true", help="replace existing STL and JSON outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    invocation = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    try:
        result = wrap_surface(
            arguments.input,
            arguments.output,
            voxel_size=arguments.voxel_size,
            report_path=arguments.report_path,
            closing_iterations=arguments.closing,
            pad_voxels=arguments.pad_voxels,
            repair_iterations=arguments.repair_iterations,
            repair_radius=arguments.repair_radius,
            max_voxels=arguments.max_voxels,
            require_single_component=not arguments.allow_multiple,
            expected_volume=arguments.expected_volume,
            volume_relative_tolerance=arguments.volume_relative_tolerance,
            force=arguments.force,
            invocation=invocation,
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

#!/usr/bin/env python3
"""Rebuild a triangle-soup STL as one watertight voxel exterior.

The implementation is deliberately conservative:

* input triangles are rasterized as barrier voxels;
* a six-neighbour flood fill identifies fluid reachable from the padded box;
* optional binary closing seals only the configured voxel-scale gaps;
* ``vtkSurfaceNets3D`` extracts a discrete, unsmoothed boundary;
* remaining digital non-manifold configurations are repaired locally; and
* topology and bidirectional surface-distance metrics are written to JSON.

The JSON report does not claim to audit triangle self-intersection; use
``surfaceCheck -checkSelfIntersection`` from OpenFOAM for that check.
Coordinates and ``--voxel-size`` use the STL's native unit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

try:
    from .voxel_backend import (
        VoxelWrapError,
        backend_status,
        connected_regions as _connected_regions,
        distribution as _distribution,
        extract_surface as _extract_surface,
        feature_edge_count as _feature_edge_count,
        geometry_metrics as _geometry_metrics,
        non_manifold_edges as _non_manifold_edges,
        orient_normals as _orient_normals,
        point_distances as _point_distances,
        read_surface as _read_surface,
        require_backend as _require_backend,
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
        non_manifold_edges as _non_manifold_edges,
        orient_normals as _orient_normals,
        point_distances as _point_distances,
        read_surface as _read_surface,
        require_backend as _require_backend,
        topology as _topology,
        write_stl as _write_stl,
    )


REPORT_SCHEMA_VERSION = 1
DEFAULT_CLOSING_ITERATIONS = 1
DEFAULT_PAD_VOXELS = 4
DEFAULT_REPAIR_ITERATIONS = 8
DEFAULT_REPAIR_RADIUS = 1
DEFAULT_MAX_VOXELS = 100_000_000
DEFAULT_VOLUME_RELATIVE_TOLERANCE = 0.02


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
    return source, output, report


def _validate_wrap_parameters(
    *,
    voxel_size: float,
    closing_iterations: int,
    pad_voxels: int,
    repair_iterations: int,
    repair_radius: int,
    max_voxels: int,
    expected_volume: float | None,
    volume_relative_tolerance: float,
) -> None:
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


def _rebuild_voxel_surface(
    source: Path,
    *,
    voxel_size: float,
    closing_iterations: int,
    pad_voxels: int,
    repair_iterations: int,
    repair_radius: int,
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
    return {
        "np": np,
        "vtk": vtk,
        "vtk_to_numpy": vtk_to_numpy,
        "input_surface": input_surface,
        "input_topology": input_topology,
        "rebuilt": rebuilt,
        "solid": solid,
        "solid_components": int(solid_components),
        "solid_before_repair": solid_before_repair,
        "voxel_report": voxel_report,
        "repair_history": repair_history,
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


def _surface_distance_report(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_points_to_output_surface": _distribution(
            _point_distances(
                state["input_surface"],
                state["rebuilt"],
                state["vtk"],
                state["vtk_to_numpy"],
            ),
            state["np"],
        ),
        "output_points_to_input_surface": _distribution(
            _point_distances(
                state["rebuilt"],
                state["input_surface"],
                state["vtk"],
                state["vtk_to_numpy"],
            ),
            state["np"],
        ),
        "unit": "source_coordinate_unit",
    }


def _wrap_payload(
    *,
    source: Path,
    output: Path,
    temporary_stl: Path,
    state: dict[str, Any],
    output_topology: dict[str, Any],
    output_geometry: dict[str, Any],
    volume_validation: dict[str, Any],
    distances: dict[str, Any],
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
            "size_bytes": source.stat().st_size,
            "points": int(input_surface.GetNumberOfPoints()),
            "triangles": int(input_surface.GetNumberOfCells()),
            "geometry": _geometry_metrics(input_surface, state["vtk"]),
            "topology": state["input_topology"],
        },
        "parameters": parameters,
        "voxel_grid": {
            **state["voxel_report"],
            "solid_components_before_repair": state["solid_components"],
            "solid_voxels_before_repair": state["solid_before_repair"],
            "solid_voxels_after_repair": int(state["solid"].sum()),
        },
        "repair_history": state["repair_history"],
        "output": {
            "path": str(output),
            "size_bytes": temporary_stl.stat().st_size,
            "points": int(rebuilt.GetNumberOfPoints()),
            "triangles": int(rebuilt.GetNumberOfCells()),
            "geometry": output_geometry,
            "topology": output_topology,
            "normals": {"consistent_and_auto_oriented": True},
            "self_intersection": {
                "audited": False,
                "value": None,
                "recommended_check": "surfaceCheck -checkSelfIntersection",
            },
        },
        "volume_validation": volume_validation,
        "distance": distances,
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
    repair_iterations: int = DEFAULT_REPAIR_ITERATIONS,
    repair_radius: int = DEFAULT_REPAIR_RADIUS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    require_single_component: bool = True,
    expected_volume: float | None = None,
    volume_relative_tolerance: float = DEFAULT_VOLUME_RELATIVE_TOLERANCE,
    force: bool = False,
) -> dict[str, Any]:
    """Create a watertight exterior STL and an atomic JSON audit report."""

    start = time.monotonic()
    source, output, report = _resolve_wrap_paths(input_path, output_path, report_path, force)
    _validate_wrap_parameters(
        voxel_size=voxel_size,
        closing_iterations=closing_iterations,
        pad_voxels=pad_voxels,
        repair_iterations=repair_iterations,
        repair_radius=repair_radius,
        max_voxels=max_voxels,
        expected_volume=expected_volume,
        volume_relative_tolerance=volume_relative_tolerance,
    )
    state = _rebuild_voxel_surface(
        source,
        voxel_size=voxel_size,
        closing_iterations=closing_iterations,
        pad_voxels=pad_voxels,
        repair_iterations=repair_iterations,
        repair_radius=repair_radius,
        max_voxels=max_voxels,
    )
    output_topology, output_geometry, volume_validation = _validate_rebuilt_surface(
        state["rebuilt"],
        state["vtk"],
        require_single_component=require_single_component,
        expected_volume=expected_volume,
        volume_relative_tolerance=volume_relative_tolerance,
    )
    distances = _surface_distance_report(state)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary_stl = _temporary_path(output.parent, ".stl")
    temporary_json = _temporary_path(report.parent, ".json")
    try:
        _write_stl(state["rebuilt"], temporary_stl, state["vtk"])
        parameters = {
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
        }
        payload = _wrap_payload(
            source=source,
            output=output,
            temporary_stl=temporary_stl,
            state=state,
            output_topology=output_topology,
            output_geometry=output_geometry,
            volume_validation=volume_validation,
            distances=distances,
            parameters=parameters,
            elapsed_seconds=float(time.monotonic() - start),
        )
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

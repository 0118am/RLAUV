"""Optional VTK/SciPy backend and surface measurements for voxel wrapping."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Sequence


class VoxelWrapError(RuntimeError):
    """Raised when a safe voxel wrap cannot be produced."""


def backend_status() -> dict[str, Any]:
    status: dict[str, Any] = {"ready": False}
    try:
        import numpy as np

        status["numpy_version"] = str(np.__version__)
    except Exception as exc:
        status["reason"] = f"NumPy unavailable: {exc}"
        return status
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from scipy import ndimage  # noqa: F401
            import scipy

        status["scipy_version"] = str(scipy.__version__)
    except Exception as exc:
        status["reason"] = f"SciPy ndimage unavailable or incompatible: {exc}"
        return status
    try:
        import vtk

        status["vtk_version"] = str(vtk.vtkVersion.GetVTKVersion())
        if not hasattr(vtk, "vtkSurfaceNets3D"):
            status["reason"] = "VTK lacks vtkSurfaceNets3D (VTK 9.3 or newer is required)"
            return status
    except Exception as exc:
        status["reason"] = f"VTK unavailable: {exc}"
        return status
    status["ready"] = True
    return status


def require_backend() -> tuple[Any, Any, Any, Any, Any]:
    status = backend_status()
    if not status["ready"]:
        raise VoxelWrapError(str(status.get("reason", "voxel backend unavailable")))
    import numpy as np
    from scipy import ndimage
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    return np, ndimage, vtk, numpy_to_vtk, vtk_to_numpy


def read_surface(path: Path, vtk: Any) -> Any:
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


def connected_regions(surface: Any, vtk: Any) -> int:
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    return int(connectivity.GetNumberOfExtractedRegions())


def feature_edge_count(
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


def non_manifold_edges(surface: Any, vtk: Any) -> Any:
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


def image_from_mask(
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


def extract_surface(
    mask: Any,
    origin: Sequence[float],
    spacing: float,
    vtk: Any,
    numpy_to_vtk: Any,
) -> Any:
    image = image_from_mask(mask, origin, spacing, vtk, numpy_to_vtk)
    surface_nets = vtk.vtkSurfaceNets3D()
    surface_nets.SetInputData(image)
    surface_nets.SetValue(0, 1)
    surface_nets.SetBackgroundLabel(0)
    surface_nets.SetOutputStyleToBoundary()
    surface_nets.SetOutputMeshTypeToTriangles()
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


def orient_normals(surface: Any, vtk: Any) -> Any:
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


def distribution(values: Any, np: Any) -> dict[str, float | int]:
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


def point_distances(source: Any, target: Any, vtk: Any, vtk_to_numpy: Any) -> Any:
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


def bounds(surface: Any) -> dict[str, list[float]]:
    raw = surface.GetBounds()
    minimum = [float(raw[0]), float(raw[2]), float(raw[4])]
    maximum = [float(raw[1]), float(raw[3]), float(raw[5])]
    return {
        "min": minimum,
        "max": maximum,
        "size": [maximum[index] - minimum[index] for index in range(3)],
    }


def topology(surface: Any, vtk: Any) -> dict[str, Any]:
    boundary_edges = feature_edge_count(surface, vtk, boundary=True)
    non_manifold = feature_edge_count(surface, vtk, non_manifold=True)
    regions = connected_regions(surface, vtk)
    extract_edges = vtk.vtkExtractEdges()
    extract_edges.SetInputData(surface)
    extract_edges.Update()
    vertices = int(surface.GetNumberOfPoints())
    edges = int(extract_edges.GetOutput().GetNumberOfCells())
    faces = int(surface.GetNumberOfCells())
    euler = vertices - edges + faces
    genus: int | None = None
    if boundary_edges == 0 and non_manifold == 0 and regions == 1:
        candidate = (2 - euler) / 2
        if candidate >= 0 and candidate.is_integer():
            genus = int(candidate)
    return {
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold,
        "connected_regions": regions,
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "euler_characteristic": euler,
        "genus": genus,
        "watertight_manifold": boundary_edges == 0 and non_manifold == 0,
        "single_component": regions == 1,
    }


def geometry_metrics(surface: Any, vtk: Any) -> dict[str, Any]:
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()
    return {
        "bounds": bounds(surface),
        "surface_area_source_units_squared": float(mass.GetSurfaceArea()),
        "enclosed_volume_source_units_cubed": float(mass.GetVolume()),
    }


def write_stl(surface: Any, path: Path, vtk: Any) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if int(writer.Write()) != 1 or not path.is_file() or path.stat().st_size <= 84:
        raise VoxelWrapError(f"VTK failed to write {path}")

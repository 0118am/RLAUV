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
        from skimage.measure import marching_cubes  # noqa: F401
        import skimage

        status["scikit_image_version"] = str(skimage.__version__)
    except Exception as exc:
        status["reason"] = f"scikit-image Lewiner marching cubes unavailable: {exc}"
        return status
    try:
        import vtk

        status["vtk_version"] = str(vtk.vtkVersion.GetVTKVersion())
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
    suffix = path.suffix.lower()
    if suffix == ".stl":
        reader = vtk.vtkSTLReader()
        reader.MergingOn()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise VoxelWrapError(f"unsupported surface format: {path}")
    reader.SetFileName(str(path))
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


def largest_connected_region(surface: Any, vtk: Any) -> Any:
    """Return a compact triangle mesh containing only the largest region."""

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToLargestRegion()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(connectivity.GetOutputPort())
    clean.PointMergingOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(triangles.GetOutput())
    return output


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


def extract_surface(
    mask: Any,
    origin: Sequence[float],
    spacing: float,
    vtk: Any,
    numpy_to_vtk: Any,
) -> Any:
    import numpy as np
    from skimage.measure import marching_cubes
    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

    vertices_zyx, faces, _normals, _values = marching_cubes(
        mask,
        level=0.5,
        spacing=(spacing, spacing, spacing),
        method="lewiner",
        allow_degenerate=False,
    )
    vertices_xyz = np.ascontiguousarray(vertices_zyx[:, ::-1], dtype=np.float64)
    vertices_xyz += np.asarray(origin, dtype=np.float64)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices_xyz, deep=True))

    vtk_cells = np.empty((faces.shape[0], 4), dtype=np.int64)
    vtk_cells[:, 0] = 3
    vtk_cells[:, 1:] = faces
    cells = vtk.vtkCellArray()
    cells.SetCells(
        int(faces.shape[0]),
        numpy_to_vtkIdTypeArray(vtk_cells.ravel(), deep=True),
    )
    output = vtk.vtkPolyData()
    output.SetPoints(points)
    output.SetPolys(cells)
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


def reflect_about_y_plane(surface: Any, plane_y: float, vtk: Any) -> Any:
    """Return the geometric reflection ``(x, y, z) -> (x, 2a-y, z)``."""

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(1, 1, -1.0)
    matrix.SetElement(1, 3, 2.0 * plane_y)
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    apply_transform = vtk.vtkTransformPolyDataFilter()
    apply_transform.SetInputData(surface)
    apply_transform.SetTransform(transform)
    apply_transform.Update()
    reflected = vtk.vtkPolyData()
    reflected.ShallowCopy(apply_transform.GetOutput())
    return reflected


def scale_uniform(surface: Any, factor: float, vtk: Any) -> Any:
    """Scale a polydata point field without an intermediate STL quantization."""

    transform = vtk.vtkTransform()
    transform.Scale(factor, factor, factor)
    apply_transform = vtk.vtkTransformPolyDataFilter()
    apply_transform.SetInputData(surface)
    apply_transform.SetTransform(transform)
    apply_transform.Update()
    scaled = vtk.vtkPolyData()
    scaled.ShallowCopy(apply_transform.GetOutput())
    return scaled


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

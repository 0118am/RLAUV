"""Affine transform parsing and STL backend implementations."""

from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence


DEFAULT_SCALE = 0.001
DEFAULT_AXIS_MAP = ("x", "y", "z")
DEFAULT_TRANSLATE_AFTER_MAP = (0.0, 0.0, 0.0)
_AXIS_TOKEN_PATTERN = re.compile(r"^([+-]?)([xyz])$", re.IGNORECASE)


class GeometryPreparationError(RuntimeError):
    """Raised when a verified geometry preparation cannot be completed."""


def vtk_available() -> bool:
    try:
        import vtk  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def select_backend(requested: str, surface_transform_points: str | None) -> tuple[str, str | None]:
    executable = surface_transform_points or shutil.which("surfaceTransformPoints")
    if requested == "vtk":
        if not vtk_available():
            raise GeometryPreparationError("VTK backend requested but Python cannot import vtk")
        return "vtk", None
    if requested == "openfoam":
        if not executable:
            raise GeometryPreparationError(
                "OpenFOAM backend requested but surfaceTransformPoints is not on PATH"
            )
        return "openfoam", str(Path(executable).expanduser().resolve())
    if requested != "auto":
        raise GeometryPreparationError(f"unsupported geometry backend: {requested}")
    if vtk_available():
        return "vtk", None
    if executable:
        return "openfoam", str(Path(executable).expanduser().resolve())
    raise GeometryPreparationError(
        "no scaling backend available: install Python VTK or load OpenFOAM surfaceTransformPoints"
    )


def parse_axis_map(axis_map: str | Sequence[str]) -> tuple[str, str, str]:
    raw_tokens = axis_map.split(",") if isinstance(axis_map, str) else list(axis_map)
    if len(raw_tokens) != 3:
        raise GeometryPreparationError(
            "axis map must contain exactly three comma-separated axes, for example z,x,y"
        )
    canonical: list[str] = []
    input_axes: list[str] = []
    for raw_token in raw_tokens:
        token = str(raw_token).strip().lower()
        match = _AXIS_TOKEN_PATTERN.fullmatch(token)
        if match is None:
            raise GeometryPreparationError(
                f"invalid axis-map token {raw_token!r}; expected x, y, z or a signed form such as -x"
            )
        sign, input_axis = match.groups()
        canonical.append(("-" if sign == "-" else "") + input_axis)
        input_axes.append(input_axis)
    if set(input_axes) != {"x", "y", "z"}:
        raise GeometryPreparationError(
            "axis map must use each input axis exactly once (a signed permutation of x,y,z)"
        )
    return canonical[0], canonical[1], canonical[2]


def axis_map_matrix(axis_map: Sequence[str]) -> list[list[float]]:
    axes = {"x": 0, "y": 1, "z": 2}
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for output_axis, token in enumerate(axis_map):
        matrix[output_axis][axes[token[-1]]] = -1.0 if token.startswith("-") else 1.0
    return matrix


def normalize_translation(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise GeometryPreparationError("translate-after-map must contain exactly three values")
    translation = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in translation):
        raise GeometryPreparationError("translate-after-map values must all be finite")
    return translation[0], translation[1], translation[2]


def affine_matrix(
    scale: float,
    axis_mapping: Sequence[Sequence[float]],
    translate_after_map: Sequence[float],
) -> list[list[float]]:
    matrix = [
        [scale * value for value in row] + [scale * translate_after_map[index]]
        for index, row in enumerate(axis_mapping)
    ]
    matrix.append([0.0, 0.0, 0.0, 1.0])
    return matrix


def is_default_frame_transform(
    axis_map: Sequence[str],
    translate_after_map: Sequence[float],
) -> bool:
    return tuple(axis_map) == DEFAULT_AXIS_MAP and all(value == 0.0 for value in translate_after_map)


def transform_with_vtk(
    source: Path,
    destination: Path,
    transform_matrix: Sequence[Sequence[float]],
    *,
    reverse_winding: bool,
) -> dict[str, Any]:
    import vtk  # type: ignore[import-not-found]

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(source))
    reader.MergingOff()
    reader.Update()
    if reader.GetOutput().GetNumberOfPolys() <= 0:
        raise GeometryPreparationError("VTK read no triangles from the source STL")
    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, transform_matrix[row][column])
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    transformed = vtk.vtkTransformPolyDataFilter()
    transformed.SetInputConnection(reader.GetOutputPort())
    transformed.SetTransform(transform)
    transformed.Update()
    output_connection = transformed.GetOutputPort()
    if reverse_winding:
        reverse = vtk.vtkReverseSense()
        reverse.SetInputConnection(output_connection)
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.Update()
        output_connection = reverse.GetOutputPort()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(destination))
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(output_connection)
    if int(writer.Write()) != 1 or not destination.is_file():
        raise GeometryPreparationError("VTK failed to write the processed STL")
    return {
        "name": "vtk",
        "version": str(vtk.vtkVersion.GetVTKVersion()),
        "winding_reversed_for_reflection": reverse_winding,
    }


def scale_with_openfoam(
    source: Path,
    destination: Path,
    scale: float,
    executable: str,
) -> dict[str, Any]:
    scale_option = "-write-scale"
    scale_vector = f"({scale:.17g} {scale:.17g} {scale:.17g})"
    command = [executable, scale_option, scale_vector, str(source), str(destination)]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GeometryPreparationError(f"surfaceTransformPoints could not run: {exc}") from exc
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise GeometryPreparationError(
            f"surfaceTransformPoints failed with exit {completed.returncode}: {detail}"
        )
    return {
        "name": "openfoam",
        "executable": executable,
        "scale_option": scale_option,
        "arguments": [scale_option, scale_vector],
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def affine_bounds_match(
    source_report: dict[str, Any],
    output_report: dict[str, Any],
    transform_matrix: Sequence[Sequence[float]],
    *,
    default_frame_transform: bool,
) -> tuple[bool, dict[str, Any]]:
    source_min, source_max = source_report["bbox"]["min"], source_report["bbox"]["max"]
    corners = [
        (x, y, z)
        for x in (source_min[0], source_max[0])
        for y in (source_min[1], source_max[1])
        for z in (source_min[2], source_max[2])
    ]
    transformed = [
        [
            sum(transform_matrix[row][column] * point[column] for column in range(3))
            + transform_matrix[row][3]
            for row in range(3)
        ]
        for point in corners
    ]
    expected_min = [min(point[axis] for point in transformed) for axis in range(3)]
    expected_max = [max(point[axis] for point in transformed) for axis in range(3)]
    actual_min, actual_max = output_report["bbox"]["min"], output_report["bbox"]["max"]
    comparisons = []
    for expected, actual in zip(expected_min + expected_max, actual_min + actual_max):
        tolerance = max(1.0e-9, 5.0e-7 * max(1.0, abs(expected)))
        comparisons.append(math.isclose(expected, actual, rel_tol=5.0e-7, abs_tol=tolerance))
    detail = {
        "expected_min": expected_min,
        "expected_max": expected_max,
        "actual_min": actual_min,
        "actual_max": actual_max,
        "affine_matrix": [list(row) for row in transform_matrix],
        "affine_transform_match": all(comparisons),
        "uniform_scale_about_origin": default_frame_transform and all(comparisons),
    }
    return all(comparisons), detail

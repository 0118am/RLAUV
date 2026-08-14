#!/usr/bin/env python3
"""Verify and apply an explicit, provenance-recorded affine transform to an STL.

The source digest is mandatory.  The default transform remains the CAD
millimetre to SI metre scale ``0.001`` about ``(0, 0, 0)``.  Callers may also
map signed input axes into body ``(x, y, z)`` coordinates and translate in the
mapped input units before scaling.  Dirty or unaudited topology is rejected
unless the caller makes the explicit, provenance-recorded ``--allow-dirty``
choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence

try:  # Support both direct execution and namespace-package imports.
    from .inspect_stl import inspect_stl, is_confirmed_watertight, write_json_report
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from inspect_stl import inspect_stl, is_confirmed_watertight, write_json_report


PROVENANCE_SCHEMA_VERSION = 2
DEFAULT_SCALE = 0.001
DEFAULT_AXIS_MAP = ("x", "y", "z")
DEFAULT_TRANSLATE_AFTER_MAP = (0.0, 0.0, 0.0)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_AXIS_TOKEN_PATTERN = re.compile(r"^([+-]?)([xyz])$", re.IGNORECASE)


class GeometryPreparationError(RuntimeError):
    """Raised when a safe, verified geometry preparation cannot be completed."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _vtk_available() -> bool:
    try:
        import vtk  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _select_backend(requested: str, surface_transform_points: str | None) -> tuple[str, str | None]:
    executable = surface_transform_points or shutil.which("surfaceTransformPoints")
    if requested == "vtk":
        if not _vtk_available():
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
    if _vtk_available():
        return "vtk", None
    if executable:
        return "openfoam", str(Path(executable).expanduser().resolve())
    raise GeometryPreparationError(
        "no scaling backend available: install Python VTK or load OpenFOAM surfaceTransformPoints"
    )


def _parse_axis_map(axis_map: str | Sequence[str]) -> tuple[str, str, str]:
    """Return a canonical signed permutation mapping input axes to body axes."""

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


def _axis_map_matrix(axis_map: Sequence[str]) -> list[list[float]]:
    axes = {"x": 0, "y": 1, "z": 2}
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for output_axis, token in enumerate(axis_map):
        sign = -1.0 if token.startswith("-") else 1.0
        matrix[output_axis][axes[token[-1]]] = sign
    return matrix


def _normalize_translation(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise GeometryPreparationError("translate-after-map must contain exactly three values")
    translation = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in translation):
        raise GeometryPreparationError("translate-after-map values must all be finite")
    return translation[0], translation[1], translation[2]


def _affine_matrix(
    scale: float,
    axis_map_matrix: Sequence[Sequence[float]],
    translate_after_map: Sequence[float],
) -> list[list[float]]:
    """Build the homogeneous input-to-output matrix for ``scale * (P*p + t)``."""

    matrix = [
        [scale * value for value in row] + [scale * translate_after_map[index]]
        for index, row in enumerate(axis_map_matrix)
    ]
    matrix.append([0.0, 0.0, 0.0, 1.0])
    return matrix


def _is_default_frame_transform(
    axis_map: Sequence[str], translate_after_map: Sequence[float]
) -> bool:
    return tuple(axis_map) == DEFAULT_AXIS_MAP and all(value == 0.0 for value in translate_after_map)


def _transform_with_vtk(
    source: Path,
    destination: Path,
    affine_matrix: Sequence[Sequence[float]],
    *,
    reverse_winding: bool,
) -> dict[str, Any]:
    import vtk  # type: ignore[import-not-found]

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(source))
    # Scaling does not require point welding.  Preserve the input facets here;
    # the separate topology audit deliberately performs exact point merging.
    reader.MergingOff()
    reader.Update()
    if reader.GetOutput().GetNumberOfPolys() <= 0:
        raise GeometryPreparationError("VTK read no triangles from the source STL")

    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, affine_matrix[row][column])
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)

    transformed = vtk.vtkTransformPolyDataFilter()
    transformed.SetInputConnection(reader.GetOutputPort())
    transformed.SetTransform(transform)
    transformed.Update()

    output_connection = transformed.GetOutputPort()
    reverse = None
    if reverse_winding:
        # A negative-determinant coordinate map mirrors cell winding.  Reverse
        # the cells so outward-facing STL orientation remains outward-facing.
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
    status = int(writer.Write())
    if status != 1 or not destination.is_file():
        raise GeometryPreparationError("VTK failed to write the processed STL")
    return {
        "name": "vtk",
        "version": str(vtk.vtkVersion.GetVTKVersion()),
        "winding_reversed_for_reflection": reverse_winding,
    }


def _scale_with_openfoam(
    source: Path,
    destination: Path,
    scale: float,
    executable: str,
) -> dict[str, Any]:
    scale_vector = f"({scale:.17g} {scale:.17g} {scale:.17g})"
    command = [executable, "-write-scale", scale_vector, str(source), str(destination)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
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


def _determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _affine_bounds_match(
    source_report: dict[str, Any],
    output_report: dict[str, Any],
    affine_matrix: Sequence[Sequence[float]],
    *,
    default_frame_transform: bool,
) -> tuple[bool, dict[str, Any]]:
    source_min = source_report["bbox"]["min"]
    source_max = source_report["bbox"]["max"]
    corners = [
        (x, y, z)
        for x in (source_min[0], source_max[0])
        for y in (source_min[1], source_max[1])
        for z in (source_min[2], source_max[2])
    ]
    transformed_corners = [
        [
            sum(affine_matrix[row][column] * point[column] for column in range(3))
            + affine_matrix[row][3]
            for row in range(3)
        ]
        for point in corners
    ]
    expected_min = [min(point[axis] for point in transformed_corners) for axis in range(3)]
    expected_max = [max(point[axis] for point in transformed_corners) for axis in range(3)]
    actual_min = output_report["bbox"]["min"]
    actual_max = output_report["bbox"]["max"]

    comparisons = []
    for expected, actual in zip(expected_min + expected_max, actual_min + actual_max):
        tolerance = max(1.0e-9, 5.0e-7 * max(1.0, abs(expected)))
        comparisons.append(math.isclose(expected, actual, rel_tol=5.0e-7, abs_tol=tolerance))
    detail = {
        "expected_min": expected_min,
        "expected_max": expected_max,
        "actual_min": actual_min,
        "actual_max": actual_max,
        "affine_matrix": [list(row) for row in affine_matrix],
        "affine_transform_match": all(comparisons),
        "uniform_scale_about_origin": default_frame_transform and all(comparisons),
    }
    return all(comparisons), detail


def _new_temporary_path(target: Path, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def prepare_geometry(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    scale: float = DEFAULT_SCALE,
    axis_map: str | Sequence[str] = DEFAULT_AXIS_MAP,
    translate_after_map: Sequence[float] = DEFAULT_TRANSLATE_AFTER_MAP,
    backend: str = "auto",
    allow_dirty: bool = False,
    force: bool = False,
    provenance_path: str | os.PathLike[str] | None = None,
    surface_transform_points: str | None = None,
    invocation: list[str] | None = None,
) -> dict[str, Any]:
    """Verify, transform, validate, and atomically publish an STL plus provenance."""

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    provenance = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else output.with_suffix(output.suffix + ".provenance.json")
    )

    if source == output:
        raise GeometryPreparationError("input and output paths must differ; in-place transforms are forbidden")
    if output.suffix.lower() != ".stl":
        raise GeometryPreparationError("output path must use the .stl suffix")
    if provenance in (source, output):
        raise GeometryPreparationError("provenance path must differ from input and output paths")
    if expected_sha256 is not None and not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise GeometryPreparationError("--expected-sha256 must contain exactly 64 hexadecimal characters")
    if not math.isfinite(scale) or scale <= 0.0:
        raise GeometryPreparationError("scale must be finite and greater than zero")
    parsed_axis_map = _parse_axis_map(axis_map)
    parsed_translation = _normalize_translation(translate_after_map)
    axis_matrix = _axis_map_matrix(parsed_axis_map)
    affine_matrix = _affine_matrix(scale, axis_matrix, parsed_translation)
    default_frame_transform = _is_default_frame_transform(parsed_axis_map, parsed_translation)
    if not force:
        existing = [str(path) for path in (output, provenance) if path.exists()]
        if existing:
            raise GeometryPreparationError(
                "refusing to overwrite existing output; use --force explicitly: " + ", ".join(existing)
            )

    source_report = inspect_stl(source, topology=True)
    expected_digest = expected_sha256.lower() if expected_sha256 is not None else None
    if expected_digest is not None and source_report["sha256"] != expected_digest:
        raise GeometryPreparationError(
            "source SHA-256 mismatch: "
            f"expected={expected_digest}, actual={source_report['sha256']}"
        )
    source_is_watertight = is_confirmed_watertight(source_report)
    if not source_is_watertight and not allow_dirty:
        topology = source_report["topology"]
        raise GeometryPreparationError(
            "source topology is not confirmed watertight; repair it first or explicitly use "
            f"--allow-dirty (audit={json.dumps(topology, sort_keys=True)})"
        )

    selected_backend, executable = _select_backend(backend, surface_transform_points)
    if not default_frame_transform and selected_backend != "vtk":
        raise GeometryPreparationError(
            "axis mapping and translation require the VTK backend; use --backend vtk or install VTK"
        )
    temporary_output = _new_temporary_path(output, ".stl")
    temporary_provenance = _new_temporary_path(provenance, ".json")
    try:
        if selected_backend == "vtk":
            backend_report = _transform_with_vtk(
                source,
                temporary_output,
                affine_matrix,
                reverse_winding=_determinant_3x3(axis_matrix) < 0.0,
            )
        else:
            assert executable is not None
            backend_report = _scale_with_openfoam(source, temporary_output, scale, executable)

        output_report = inspect_stl(temporary_output, topology=True)
        if output_report["triangle_count"] != source_report["triangle_count"]:
            raise GeometryPreparationError(
                "triangle count changed during scaling: "
                f"source={source_report['triangle_count']}, output={output_report['triangle_count']}"
            )
        bounds_match, bounds_validation = _affine_bounds_match(
            source_report,
            output_report,
            affine_matrix,
            default_frame_transform=default_frame_transform,
        )
        if not bounds_match:
            raise GeometryPreparationError(
                "processed bounds do not match the requested affine transform; refusing output"
            )
        output_is_watertight = is_confirmed_watertight(output_report)
        if not output_is_watertight and not allow_dirty:
            raise GeometryPreparationError("processed STL topology is not confirmed watertight")

        # Temporary paths are implementation details, not durable provenance.
        output_report["path"] = str(output)
        provenance_report: dict[str, Any] = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "created_utc": _utc_timestamp(),
            "operation": (
                "uniform_scale_about_origin"
                if default_frame_transform
                else "axis_map_translate_then_uniform_scale"
            ),
            "invocation": invocation,
            "source": source_report,
            "output": output_report,
            "transform": {
                "input_units": "mm" if math.isclose(scale, 0.001) else "unspecified",
                "output_units": "m" if math.isclose(scale, 0.001) else "unspecified",
                "uniform_scale": scale,
                "origin": [0.0, 0.0, 0.0],
                "axis_map": list(parsed_axis_map),
                "axis_map_matrix": axis_matrix,
                "translate_after_map": list(parsed_translation),
                "translate_after_map_units": "input_units",
                "translation": [scale * value for value in parsed_translation],
                "affine_matrix": affine_matrix,
                "auto_center": False,
            },
            "backend": backend_report,
            "validation": {
                "expected_source_sha256": expected_digest,
                "source_sha256_match": True if expected_digest is not None else None,
                "source_watertight": source_is_watertight,
                "output_watertight": output_is_watertight,
                "dirty_override_used": allow_dirty and (
                    not source_is_watertight or not output_is_watertight
                ),
                "triangle_count_preserved": True,
                "bounds": bounds_validation,
            },
        }
        write_json_report(provenance_report, temporary_provenance)
        output.parent.mkdir(parents=True, exist_ok=True)
        provenance.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_output, output)
        os.replace(temporary_provenance, provenance)
        return provenance_report
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source STL in CAD coordinates")
    parser.add_argument("output", help="processed STL path; must not equal input")
    parser.add_argument(
        "--expected-sha256",
        help="optional digest check; actual source/output digests are always recorded",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="uniform scale applied after axis mapping and translation (default: 0.001, mm to m)",
    )
    parser.add_argument(
        "--axis-map",
        default=",".join(DEFAULT_AXIS_MAP),
        metavar="SIGNED_AXES",
        help=(
            "comma-separated input axes supplying body x,y,z; signs are allowed "
            "(default: x,y,z; example: z,-x,y)"
        ),
    )
    parser.add_argument(
        "--translate-after-map",
        type=float,
        nargs=3,
        default=DEFAULT_TRANSLATE_AFTER_MAP,
        metavar=("TX", "TY", "TZ"),
        help="translation in mapped input units, applied before --scale (default: 0 0 0)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "vtk", "openfoam"),
        default="auto",
        help="scaling backend (default: auto, preferring VTK)",
    )
    parser.add_argument(
        "--surface-transform-points",
        metavar="PATH",
        help="explicit OpenFOAM surfaceTransformPoints executable",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="explicitly permit dirty or unaudited topology and record the override",
    )
    parser.add_argument("--force", action="store_true", help="explicitly replace existing output files")
    parser.add_argument(
        "--provenance",
        metavar="PATH",
        help="provenance JSON path (default: OUTPUT.provenance.json)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = build_argument_parser().parse_args(arguments)
    try:
        report = prepare_geometry(
            args.input,
            args.output,
            expected_sha256=args.expected_sha256,
            scale=args.scale,
            axis_map=args.axis_map,
            translate_after_map=args.translate_after_map,
            backend=args.backend,
            allow_dirty=args.allow_dirty,
            force=args.force,
            provenance_path=args.provenance,
            surface_transform_points=args.surface_transform_points,
            invocation=[sys.argv[0], *arguments],
        )
    except (OSError, ValueError, GeometryPreparationError) as exc:
        sys.stderr.write(json.dumps({"error": str(exc)}, sort_keys=True) + "\n")
        return 1

    summary = {
        "output": report["output"]["path"],
        "output_sha256": report["output"]["sha256"],
        "provenance": str(
            Path(args.provenance).expanduser().resolve()
            if args.provenance
            else Path(args.output).expanduser().resolve().with_suffix(Path(args.output).suffix + ".provenance.json")
        ),
        "dirty_override_used": report["validation"]["dirty_override_used"],
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

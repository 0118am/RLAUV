#!/usr/bin/env python3
"""Validate and apply an explicit affine transform to an STL.

The default transform remains the CAD millimetre to SI metre scale ``0.001``
about ``(0, 0, 0)``. Callers may also map signed input axes into body
``(x, y, z)`` coordinates and translate in the mapped input units before
scaling. Dirty or unaudited topology is rejected unless ``--allow-dirty`` is
set. A compact transform report records geometry facts needed by later steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

try:  # Support both direct execution and namespace-package imports.
    from .inspect_stl import inspect_stl, is_confirmed_watertight, write_json_report
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from inspect_stl import inspect_stl, is_confirmed_watertight, write_json_report

try:
    from .geometry_transform import (
        DEFAULT_AXIS_MAP,
        DEFAULT_SCALE,
        DEFAULT_TRANSLATE_AFTER_MAP,
        GeometryPreparationError,
        affine_bounds_match as _affine_bounds_match,
        affine_matrix as _affine_matrix,
        axis_map_matrix as _axis_map_matrix,
        determinant_3x3 as _determinant_3x3,
        is_default_frame_transform as _is_default_frame_transform,
        normalize_translation as _normalize_translation,
        parse_axis_map as _parse_axis_map,
        scale_with_openfoam as _scale_with_openfoam,
        select_backend as _select_backend,
        transform_with_vtk as _transform_with_vtk,
        vtk_available as _vtk_available,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from geometry_transform import (
        DEFAULT_AXIS_MAP,
        DEFAULT_SCALE,
        DEFAULT_TRANSLATE_AFTER_MAP,
        GeometryPreparationError,
        affine_bounds_match as _affine_bounds_match,
        affine_matrix as _affine_matrix,
        axis_map_matrix as _axis_map_matrix,
        determinant_3x3 as _determinant_3x3,
        is_default_frame_transform as _is_default_frame_transform,
        normalize_translation as _normalize_translation,
        parse_axis_map as _parse_axis_map,
        scale_with_openfoam as _scale_with_openfoam,
        select_backend as _select_backend,
        transform_with_vtk as _transform_with_vtk,
        vtk_available as _vtk_available,
    )


TRANSFORM_REPORT_SCHEMA_VERSION = 1
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


def _resolve_preparation_request(
    input_path,
    output_path,
    report_path,
    scale: float,
    axis_map,
    translate_after_map,
    force: bool,
):
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    report = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output.with_suffix(output.suffix + ".transform.json")
    )
    if source == output:
        raise GeometryPreparationError("input and output paths must differ; in-place transforms are forbidden")
    if output.suffix.lower() != ".stl":
        raise GeometryPreparationError("output path must use the .stl suffix")
    if report in (source, output):
        raise GeometryPreparationError("report path must differ from input and output paths")
    if not math.isfinite(scale) or scale <= 0.0:
        raise GeometryPreparationError("scale must be finite and greater than zero")
    parsed_axis_map = _parse_axis_map(axis_map)
    translation = _normalize_translation(translate_after_map)
    axis_matrix = _axis_map_matrix(parsed_axis_map)
    affine_matrix = _affine_matrix(scale, axis_matrix, translation)
    if not force:
        existing = [str(path) for path in (output, report) if path.exists()]
        if existing:
            raise GeometryPreparationError(
                "refusing to overwrite existing output; use --force explicitly: " + ", ".join(existing)
            )
    return source, output, report, parsed_axis_map, translation, axis_matrix, affine_matrix


def _validate_source_topology(source: Path, allow_dirty: bool) -> tuple[dict[str, Any], bool]:
    source_report = inspect_stl(source, topology=True)
    watertight = is_confirmed_watertight(source_report)
    if not watertight and not allow_dirty:
        raise GeometryPreparationError(
            "source topology is not confirmed watertight; repair it first or use --allow-dirty "
            f"(audit={json.dumps(source_report['topology'], sort_keys=True)})"
        )
    return source_report, watertight


def _execute_transform(
    source: Path,
    temporary_output: Path,
    *,
    backend: str,
    executable: str | None,
    affine_matrix: Sequence[Sequence[float]],
    axis_matrix: Sequence[Sequence[float]],
    scale: float,
) -> dict[str, Any]:
    if backend == "vtk":
        return _transform_with_vtk(
            source,
            temporary_output,
            affine_matrix,
            reverse_winding=_determinant_3x3(axis_matrix) < 0.0,
        )
    assert executable is not None
    return _scale_with_openfoam(source, temporary_output, scale, executable)


def _validate_transformed_geometry(
    temporary_output: Path,
    source_report: dict[str, Any],
    affine_matrix: Sequence[Sequence[float]],
    *,
    default_frame_transform: bool,
    allow_dirty: bool,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
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
        raise GeometryPreparationError("processed bounds do not match the requested affine transform")
    watertight = is_confirmed_watertight(output_report)
    if not watertight and not allow_dirty:
        raise GeometryPreparationError("processed STL topology is not confirmed watertight")
    return output_report, watertight, bounds_validation


def _build_transform_report(
    *,
    source_report: dict[str, Any],
    output_report: dict[str, Any],
    output: Path,
    scale: float,
    parsed_axis_map: Sequence[str],
    axis_matrix: Sequence[Sequence[float]],
    translation: Sequence[float],
    affine_matrix: Sequence[Sequence[float]],
    default_frame_transform: bool,
    backend_report: dict[str, Any],
    source_watertight: bool,
    output_watertight: bool,
    allow_dirty: bool,
    bounds_validation: dict[str, Any],
) -> dict[str, Any]:
    output_report["path"] = str(output)
    return {
        "schema_version": TRANSFORM_REPORT_SCHEMA_VERSION,
        "operation": (
            "uniform_scale_about_origin"
            if default_frame_transform
            else "axis_map_translate_then_uniform_scale"
        ),
        "source": source_report,
        "output": output_report,
        "transform": {
            "input_units": "mm" if math.isclose(scale, 0.001) else "unspecified",
            "output_units": "m" if math.isclose(scale, 0.001) else "unspecified",
            "uniform_scale": scale,
            "origin": [0.0, 0.0, 0.0],
            "axis_map": list(parsed_axis_map),
            "axis_map_matrix": axis_matrix,
            "translate_after_map": list(translation),
            "translate_after_map_units": "input_units",
            "translation": [scale * value for value in translation],
            "affine_matrix": affine_matrix,
            "auto_center": False,
        },
        "backend": backend_report,
        "validation": {
            "source_watertight": source_watertight,
            "output_watertight": output_watertight,
            "dirty_override_used": allow_dirty and (
                not source_watertight or not output_watertight
            ),
            "triangle_count_preserved": True,
            "bounds": bounds_validation,
        },
    }


def prepare_geometry(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    scale: float = DEFAULT_SCALE,
    axis_map: str | Sequence[str] = DEFAULT_AXIS_MAP,
    translate_after_map: Sequence[float] = DEFAULT_TRANSLATE_AFTER_MAP,
    backend: str = "auto",
    allow_dirty: bool = False,
    force: bool = False,
    report_path: str | os.PathLike[str] | None = None,
    surface_transform_points: str | None = None,
) -> dict[str, Any]:
    """Validate, transform, and atomically publish an STL plus transform report."""

    source, output, report, parsed_map, translation, axis_matrix, affine_matrix = (
        _resolve_preparation_request(
            input_path,
            output_path,
            report_path,
            scale,
            axis_map,
            translate_after_map,
            force,
        )
    )
    source_report, source_watertight = _validate_source_topology(source, allow_dirty)
    selected_backend, executable = _select_backend(backend, surface_transform_points)
    default_transform = _is_default_frame_transform(parsed_map, translation)
    if not default_transform and selected_backend != "vtk":
        raise GeometryPreparationError("axis mapping and translation require the VTK backend")

    temporary_output = _new_temporary_path(output, ".stl")
    temporary_report = _new_temporary_path(report, ".json")
    try:
        backend_report = _execute_transform(
            source,
            temporary_output,
            backend=selected_backend,
            executable=executable,
            affine_matrix=affine_matrix,
            axis_matrix=axis_matrix,
            scale=scale,
        )
        output_report, output_watertight, bounds = _validate_transformed_geometry(
            temporary_output,
            source_report,
            affine_matrix,
            default_frame_transform=default_transform,
            allow_dirty=allow_dirty,
        )
        payload = _build_transform_report(
            source_report=source_report,
            output_report=output_report,
            output=output,
            scale=scale,
            parsed_axis_map=parsed_map,
            axis_matrix=axis_matrix,
            translation=translation,
            affine_matrix=affine_matrix,
            default_frame_transform=default_transform,
            backend_report=backend_report,
            source_watertight=source_watertight,
            output_watertight=output_watertight,
            allow_dirty=allow_dirty,
            bounds_validation=bounds,
        )
        write_json_report(payload, temporary_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_output, output)
        os.replace(temporary_report, report)
        return payload
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source STL in CAD coordinates")
    parser.add_argument("output", help="processed STL path; must not equal input")
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
        "--report",
        metavar="PATH",
        help="transform report path (default: OUTPUT.transform.json)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = build_argument_parser().parse_args(arguments)
    try:
        report = prepare_geometry(
            args.input,
            args.output,
            scale=args.scale,
            axis_map=args.axis_map,
            translate_after_map=args.translate_after_map,
            backend=args.backend,
            allow_dirty=args.allow_dirty,
            force=args.force,
            report_path=args.report,
            surface_transform_points=args.surface_transform_points,
        )
    except (OSError, ValueError, GeometryPreparationError) as exc:
        sys.stderr.write(json.dumps({"error": str(exc)}, sort_keys=True) + "\n")
        return 1

    summary = {
        "output": report["output"]["path"],
        "transform_report": str(
            Path(args.report).expanduser().resolve()
            if args.report
            else Path(args.output).expanduser().resolve().with_suffix(Path(args.output).suffix + ".transform.json")
        ),
        "dirty_override_used": report["validation"]["dirty_override_used"],
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

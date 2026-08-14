#!/usr/bin/env python3
"""Inspect STL geometry without requiring OpenFOAM.

The metadata parser uses only the Python standard library.  When VTK is
available, the report also includes a merged-edge topology audit.  A strict
inspection succeeds only when VTK positively confirms a closed two-manifold
surface; an unavailable topology backend is deliberately not treated as a
pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Iterable, Sequence


REPORT_SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STRICT_FAILURE = 2
_BINARY_FACET = struct.Struct("<12fH")
_UINT32 = struct.Struct("<I")


class STLInspectionError(ValueError):
    """Raised when a file is not a structurally valid STL."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_bounds() -> tuple[list[float], list[float]]:
    return [math.inf, math.inf, math.inf], [-math.inf, -math.inf, -math.inf]


def _extend_bounds(minimum: list[float], maximum: list[float], point: Sequence[float]) -> None:
    for axis in range(3):
        value = float(point[axis])
        if not math.isfinite(value):
            raise STLInspectionError("STL contains a non-finite vertex coordinate")
        minimum[axis] = min(minimum[axis], value)
        maximum[axis] = max(maximum[axis], value)


def _triangle_is_degenerate(vertices: Sequence[Sequence[float]]) -> bool:
    a, b, c = vertices
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return cross == (0.0, 0.0, 0.0)


def _bounds_report(minimum: Sequence[float], maximum: Sequence[float]) -> dict[str, list[float]]:
    if any(not math.isfinite(value) for value in (*minimum, *maximum)):
        raise STLInspectionError("STL contains no vertices")
    size = [maximum[i] - minimum[i] for i in range(3)]
    center = [(maximum[i] + minimum[i]) / 2.0 for i in range(3)]
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "size": size,
        "center": center,
    }


def _binary_layout(path: Path) -> tuple[bool, int | None, int | None]:
    """Return ``(exact_binary_layout, count, expected_size)``."""

    size = path.stat().st_size
    if size < 84:
        return False, None, None
    with path.open("rb") as stream:
        stream.seek(80)
        raw_count = stream.read(4)
    if len(raw_count) != 4:
        return False, None, None
    count = _UINT32.unpack(raw_count)[0]
    expected_size = 84 + count * _BINARY_FACET.size
    return size == expected_size, count, expected_size


def _detect_format(path: Path) -> tuple[str, int | None, int | None]:
    exact_binary, count, expected_size = _binary_layout(path)
    if exact_binary:
        return "binary", count, expected_size

    with path.open("rb") as stream:
        prefix = stream.read(512).lstrip()
    prefix_lower = prefix.lower()
    if prefix_lower.startswith(b"solid") or prefix_lower.startswith(b"facet"):
        return "ascii", None, None

    if count is not None:
        raise STLInspectionError(
            "binary STL length does not match its declared triangle count: "
            f"actual={path.stat().st_size}, expected={expected_size}, triangles={count}"
        )
    raise STLInspectionError("file is neither a complete binary STL nor a recognizable ASCII STL")


def _inspect_binary(path: Path, declared_count: int) -> dict[str, Any]:
    minimum, maximum = _empty_bounds()
    degenerate = 0
    nonzero_attribute_words = 0

    with path.open("rb") as stream:
        header = stream.read(80)
        raw_count = stream.read(4)
        if len(header) != 80 or len(raw_count) != 4:
            raise STLInspectionError("binary STL header is truncated")
        count = _UINT32.unpack(raw_count)[0]
        if count != declared_count:
            raise STLInspectionError("binary STL triangle count changed during inspection")

        for triangle_index in range(count):
            record = stream.read(_BINARY_FACET.size)
            if len(record) != _BINARY_FACET.size:
                raise STLInspectionError(f"binary STL is truncated at triangle {triangle_index}")
            values = _BINARY_FACET.unpack(record)
            vertices = (
                values[3:6],
                values[6:9],
                values[9:12],
            )
            for vertex in vertices:
                _extend_bounds(minimum, maximum, vertex)
            if _triangle_is_degenerate(vertices):
                degenerate += 1
            if values[12] != 0:
                nonzero_attribute_words += 1

        if stream.read(1):
            raise STLInspectionError("binary STL has unexpected trailing bytes")

    return {
        "triangle_count": count,
        "bbox": _bounds_report(minimum, maximum),
        "degenerate_triangles": degenerate,
        "binary": {
            "header_ascii": header.rstrip(b"\x00 ").decode("ascii", errors="replace"),
            "declared_triangle_count": count,
            "expected_size_bytes": 84 + count * _BINARY_FACET.size,
            "size_matches_declared_count": True,
            "nonzero_attribute_words": nonzero_attribute_words,
        },
    }


def _parse_ascii_vertex(parts: Sequence[str], line_number: int) -> tuple[float, float, float]:
    if len(parts) != 4:
        raise STLInspectionError(f"ASCII STL line {line_number}: vertex requires exactly three values")
    try:
        point = tuple(float(value) for value in parts[1:])
    except ValueError as exc:
        raise STLInspectionError(f"ASCII STL line {line_number}: invalid vertex") from exc
    if any(not math.isfinite(value) for value in point):
        raise STLInspectionError(f"ASCII STL line {line_number}: non-finite vertex")
    return point  # type: ignore[return-value]


def _inspect_ascii(path: Path) -> dict[str, Any]:
    minimum, maximum = _empty_bounds()
    vertices: list[tuple[float, float, float]] = []
    facet_count = 0
    solid_name = ""

    with path.open("r", encoding="utf-8", errors="strict", newline=None) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            keyword = parts[0].lower()
            if keyword == "solid" and not solid_name:
                solid_name = stripped[5:].strip()
            elif keyword == "facet":
                facet_count += 1
            elif keyword == "vertex":
                point = _parse_ascii_vertex(parts, line_number)
                vertices.append(point)
                _extend_bounds(minimum, maximum, point)

    if facet_count == 0:
        raise STLInspectionError("ASCII STL has no facets")
    if len(vertices) != facet_count * 3:
        raise STLInspectionError(
            "ASCII STL facet/vertex mismatch: "
            f"facets={facet_count}, vertices={len(vertices)}, expected_vertices={facet_count * 3}"
        )
    degenerate = sum(
        _triangle_is_degenerate(vertices[index : index + 3])
        for index in range(0, len(vertices), 3)
    )
    return {
        "triangle_count": facet_count,
        "bbox": _bounds_report(minimum, maximum),
        "degenerate_triangles": degenerate,
        "ascii": {"solid_name": solid_name, "vertex_count": len(vertices)},
    }


def audit_topology_vtk(path: Path) -> dict[str, Any]:
    """Audit boundary and non-manifold edges with VTK, when importable."""

    try:
        import vtk  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "backend": "vtk",
            "available": False,
            "audited": False,
            "watertight": None,
            "reason": f"VTK unavailable: {exc}",
        }

    try:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(path))
        reader.MergingOn()
        reader.Update()
        mesh = reader.GetOutput()
        triangle_count = int(mesh.GetNumberOfPolys())
        point_count = int(mesh.GetNumberOfPoints())

        def edge_count(*, boundary: bool = False, non_manifold: bool = False) -> int:
            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(mesh)
            edges.BoundaryEdgesOff()
            edges.FeatureEdgesOff()
            edges.NonManifoldEdgesOff()
            edges.ManifoldEdgesOff()
            if boundary:
                edges.BoundaryEdgesOn()
            if non_manifold:
                edges.NonManifoldEdgesOn()
            edges.Update()
            return int(edges.GetOutput().GetNumberOfCells())

        boundary_edges = edge_count(boundary=True)
        non_manifold_edges = edge_count(non_manifold=True)

        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(mesh)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.Update()
        connected_regions = int(connectivity.GetNumberOfExtractedRegions())

        watertight = triangle_count > 0 and boundary_edges == 0 and non_manifold_edges == 0
        return {
            "backend": "vtk",
            "backend_version": str(vtk.vtkVersion.GetVTKVersion()),
            "available": True,
            "audited": True,
            "watertight": watertight,
            "merged_point_count": point_count,
            "reader_triangle_count": triangle_count,
            "connected_regions": connected_regions,
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
        }
    except Exception as exc:  # pragma: no cover - defensive around optional C++ bindings
        return {
            "backend": "vtk",
            "backend_version": str(vtk.vtkVersion.GetVTKVersion()),
            "available": True,
            "audited": False,
            "watertight": None,
            "reason": f"VTK topology audit failed: {exc}",
        }


def inspect_stl(path: str | os.PathLike[str], *, topology: bool = True) -> dict[str, Any]:
    """Return a JSON-serializable report for an STL file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise STLInspectionError(f"STL does not exist or is not a regular file: {source}")

    stl_format, declared_count, _ = _detect_format(source)
    if stl_format == "binary":
        assert declared_count is not None
        geometry = _inspect_binary(source, declared_count)
    else:
        geometry = _inspect_ascii(source)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "format": stl_format,
        **geometry,
    }
    if topology:
        report["topology"] = audit_topology_vtk(source)
    else:
        report["topology"] = {
            "backend": None,
            "available": False,
            "audited": False,
            "watertight": None,
            "reason": "topology audit disabled",
        }
    return report


def is_confirmed_watertight(report: dict[str, Any]) -> bool:
    """Return true only for a completed audit that confirms watertightness."""

    topology = report.get("topology", {})
    return topology.get("audited") is True and topology.get("watertight") is True


def write_json_report(report: dict[str, Any], destination: str | os.PathLike[str]) -> None:
    """Write stable, strict JSON to stdout or atomically to a file."""

    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if str(destination) == "-":
        sys.stdout.write(payload)
        return

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    os.replace(temporary, target)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="STL file to inspect")
    parser.add_argument(
        "--json",
        dest="json_path",
        default="-",
        metavar="PATH",
        help="JSON report path, or '-' for stdout (default: '-')",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 unless topology is audited and confirmed watertight",
    )
    parser.add_argument(
        "--no-topology",
        action="store_true",
        help="skip optional VTK topology audit (incompatible with a passing --strict check)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = inspect_stl(args.input, topology=not args.no_topology)
    except (OSError, UnicodeError, STLInspectionError) as exc:
        error_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "path": str(Path(args.input).expanduser().resolve()),
            "error": str(exc),
        }
        write_json_report(error_report, args.json_path)
        return EXIT_ERROR

    write_json_report(report, args.json_path)
    if args.strict and not is_confirmed_watertight(report):
        return EXIT_STRICT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

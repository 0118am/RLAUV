#!/usr/bin/env python3
"""Atomically scale a large binary STL without constructing a VTK edge graph.

This utility exists for final high-resolution surfaces whose topology has
already passed ``surfaceCheck``.  It validates the exact binary STL layout,
streams triangle records in NumPy chunks, preserves winding and normals, and
writes transform parameters and bounds to a compact report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np


TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ],
    align=False,
)
RECORD_BYTES = 50


def _read_header(path: Path) -> tuple[bytes, int]:
    with path.open("rb") as stream:
        header = stream.read(80)
        raw_count = stream.read(4)
    if len(header) != 80 or len(raw_count) != 4:
        raise ValueError("input is too short to be a binary STL")
    triangle_count = struct.unpack("<I", raw_count)[0]
    expected_size = 84 + RECORD_BYTES * triangle_count
    if path.stat().st_size != expected_size:
        raise ValueError(
            "input does not have an exact binary STL layout: "
            f"expected {expected_size} bytes, got {path.stat().st_size}"
        )
    if TRIANGLE_DTYPE.itemsize != RECORD_BYTES:
        raise RuntimeError("internal binary STL dtype is not 50 bytes")
    return header, triangle_count


def _bbox_payload(minimum: np.ndarray, maximum: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "extent": [float(value) for value in maximum - minimum],
    }


def _validated_paths(
    source: Path,
    output: Path,
    report_path: Path,
    *,
    scale: float,
    chunk_triangles: int,
    force: bool,
) -> tuple[Path, Path, Path]:
    source, output, report_path = source.resolve(), output.resolve(), report_path.resolve()
    if source == output or report_path in (source, output):
        raise ValueError("input, output and report paths must be distinct")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if chunk_triangles < 1:
        raise ValueError("chunk size must be positive")
    existing = [path for path in (output, report_path) if path.exists()]
    if existing and not force:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, existing)))
    return source, output, report_path


def _temporary_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(name)


def _scale_triangle_records(
    source: Path,
    temporary: Path,
    triangle_count: int,
    scale: float,
    chunk_triangles: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_map = np.memmap(
        source,
        dtype=TRIANGLE_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    header_text = f"AUV CFD wetted surface; uniform scale={scale:g}".encode("ascii")
    with temporary.open("wb") as stream:
        stream.write(header_text[:80].ljust(80, b"\0"))
        stream.write(struct.pack("<I", triangle_count))
        stream.truncate(84 + RECORD_BYTES * triangle_count)
    output_map = np.memmap(
        temporary,
        dtype=TRIANGLE_DTYPE,
        mode="r+",
        offset=84,
        shape=(triangle_count,),
    )
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    try:
        for start in range(0, triangle_count, chunk_triangles):
            stop = min(start + chunk_triangles, triangle_count)
            block = source_map[start:stop]
            vertices = np.asarray(block["vertices"])
            normals = np.asarray(block["normal"])
            if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
                raise ValueError(f"non-finite STL values in triangles {start}:{stop}")
            minimum = np.minimum(minimum, vertices.min(axis=(0, 1)))
            maximum = np.maximum(maximum, vertices.max(axis=(0, 1)))
            output_map[start:stop]["normal"] = normals
            output_map[start:stop]["vertices"] = vertices * scale
            output_map[start:stop]["attribute"] = block["attribute"]
        output_map.flush()
    finally:
        del output_map
        del source_map
    return minimum, maximum


def _scale_report(
    source: Path,
    output: Path,
    temporary: Path,
    triangle_count: int,
    scale: float,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> dict:
    return {
        "schema_version": 1,
        "operation": "streaming_uniform_scale_about_origin",
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "units": "mm",
            "triangle_count": triangle_count,
            "bbox": _bbox_payload(minimum, maximum),
        },
        "transform": {
            "uniform_scale": scale,
            "origin": [0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "axis_map": ["x", "y", "z"],
        },
        "output": {
            "path": str(output),
            "size_bytes": temporary.stat().st_size,
            "units": "m",
            "triangle_count": triangle_count,
            "bbox": _bbox_payload(minimum * scale, maximum * scale),
            "binary_stl_layout_valid": True,
        },
    }


def scale_binary_stl(
    source: Path,
    output: Path,
    report_path: Path,
    *,
    scale: float,
    chunk_triangles: int,
    force: bool,
) -> dict:
    source, output, report_path = _validated_paths(
        source,
        output,
        report_path,
        scale=scale,
        chunk_triangles=chunk_triangles,
        force=force,
    )
    _header, triangle_count = _read_header(source)
    temporary = _temporary_path(output, ".stl")
    temporary_report = _temporary_path(report_path, ".json")
    try:
        minimum, maximum = _scale_triangle_records(
            source,
            temporary,
            triangle_count,
            scale,
            chunk_triangles,
        )
        report = _scale_report(
            source,
            output,
            temporary,
            triangle_count,
            scale,
            minimum,
            maximum,
        )
        temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        os.replace(temporary_report, report_path)
        return report
    finally:
        temporary.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chunk-triangles", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = scale_binary_stl(
            args.input,
            args.output,
            args.report,
            scale=args.scale,
            chunk_triangles=args.chunk_triangles,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"scale_binary_stl: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["output"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

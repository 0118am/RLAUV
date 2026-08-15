"""Parse OpenFOAM surface and mesh diagnostics into reusable audit data."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CORE_TOPOLOGY_CONFIRMATIONS = (
    "Boundary definition OK.",
    "Cell to face addressing OK.",
    "Point usage OK.",
    "Upper triangular ordering OK.",
    "Face vertices OK.",
    "Topological cell zip-up check OK.",
)
_CONFIGURED_FACE_ERROR_PREFIXES = (
    "non-orthogonality",
    "faces with face pyramid volume",
    "faces with face-decomposition tet quality",
    "faces with concavity",
    "faces with skewness",
    "faces with interpolation weights",
    "faces with volume ratio",
    "faces with face twist",
    "faces on cells with determinant",
)


def surface_check_failures(output: str) -> list[str]:
    required = {
        "illegal triangles": r"Surface has no illegal triangles\.",
        "closed two-face edges": r"Surface is closed\. All edges connected to two faces\.",
        "self intersections": r"Surface is not self-intersecting",
    }
    failures = [name for name, pattern in required.items() if re.search(pattern, output) is None]
    parts = re.search(r"Number of unconnected parts\s*:\s*(\d+)", output)
    if parts is None or int(parts.group(1)) != 1:
        failures.append("exactly one connected part")
    zones = re.search(r"Number of zones \(connected area with consistent normal\)\s*:\s*(\d+)", output)
    if zones is None or int(zones.group(1)) != 1:
        failures.append("one consistently oriented normal zone")
    return failures


def _ended_cleanly(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return bool(lines and lines[-1] == "End")


def _fatal_diagnostics(output: str) -> list[str]:
    return [
        line.strip()
        for line in output.splitlines()
        if re.search(r"\b(?:FOAM\s+)?FATAL(?:\s+IO)?\s+ERROR\b", line, re.IGNORECASE)
    ]


def _final_configured_face_errors(output: str) -> tuple[dict[str, int], list[str]]:
    starts = list(re.finditer(r"(?m)^\s*Checking faces in error\s*:\s*$", output))
    if not starts:
        return {}, ["missing final configured face-error block"]
    counts: dict[str, int] = {}
    began = False
    for line in output[starts[-1].end() :].splitlines():
        match = re.match(r"^\s*(.+?)\s*:\s*(\d+)\s*$", line)
        if match:
            began = True
            counts[match.group(1).strip()] = int(match.group(2))
        elif began:
            break
    failures: list[str] = []
    lowered = tuple(label.lower() for label in counts)
    missing = [
        prefix
        for prefix in _CONFIGURED_FACE_ERROR_PREFIXES
        if not any(label.startswith(prefix) for label in lowered)
    ]
    if missing:
        failures.append("incomplete configured face-error block: " + ", ".join(missing))
    nonzero = {label: count for label, count in counts.items() if count != 0}
    if nonzero:
        failures.append(
            "configured mesh-quality limits exceeded: "
            + ", ".join(f"{label}={count}" for label, count in nonzero.items())
        )
    return counts, failures


def _extended_check_mesh_warnings(output: str) -> list[str]:
    return [
        stripped
        for line in output.splitlines()
        if (
            (stripped := line.strip()).startswith("<<")
            or stripped.startswith("***")
            or stripped.startswith("*There are")
            or re.fullmatch(r"Failed\s+\d+\s+mesh checks\.", stripped)
        )
    ]


def snappy_mesh_audit(output: str) -> dict[str, object]:
    counts, failures = _final_configured_face_errors(output)
    fatal = _fatal_diagnostics(output)
    if fatal:
        failures.append("fatal diagnostic: " + " | ".join(fatal))
    ended = _ended_cleanly(output)
    if not ended:
        failures.append("missing terminal End marker")
    return {
        "passed": not failures,
        "hard_failures": failures,
        "warnings": [],
        "ended_cleanly": ended,
        "fatal_diagnostics": fatal,
        "final_configured_face_errors": counts,
    }


def check_mesh_audit(output: str) -> dict[str, object]:
    counts, failures = _final_configured_face_errors(output)
    fatal = _fatal_diagnostics(output)
    if fatal:
        failures.append("fatal diagnostic: " + " | ".join(fatal))
    ended = _ended_cleanly(output)
    if not ended:
        failures.append("missing terminal End marker")
    missing_core = [text for text in _CORE_TOPOLOGY_CONFIRMATIONS if text not in output]
    if missing_core:
        failures.append("missing core topology confirmation(s): " + ", ".join(missing_core))
    regions_match = re.search(r"Number of regions\s*:\s*(\d+)\s*\(OK\)", output)
    regions = int(regions_match.group(1)) if regions_match else None
    if regions != 1:
        failures.append("mesh must contain exactly one connected region")
    minimum_match = re.search(rf"Min volume\s*=\s*({_NUMBER})", output)
    total_match = re.search(rf"Total volume\s*=\s*({_NUMBER})", output)
    minimum_volume = float(minimum_match.group(1)) if minimum_match else None
    total_volume = float(total_match.group(1)) if total_match else None
    if minimum_volume is None or not math.isfinite(minimum_volume) or minimum_volume <= 0.0:
        failures.append("minimum cell volume is missing, non-finite, zero, or negative")
    if total_volume is None or not math.isfinite(total_volume) or total_volume <= 0.0:
        failures.append("total fluid volume is missing, non-finite, zero, or negative")
    warnings = _extended_check_mesh_warnings(output)
    failed_match = re.search(r"Failed\s+(\d+)\s+mesh checks", output)
    return {
        "passed": not failures,
        "hard_failures": failures,
        "warnings": warnings,
        "ended_cleanly": ended,
        "fatal_diagnostics": fatal,
        "core_topology_confirmed": not missing_core,
        "connected_regions": regions,
        "minimum_cell_volume_m3": minimum_volume,
        "total_fluid_volume_m3": total_volume,
        "final_configured_face_errors": counts,
        "extended_checks_reported_failed": int(failed_match.group(1)) if failed_match else 0,
    }


def write_mesh_quality_audit(
    path: Path,
    snappy: dict[str, object] | None,
    check_mesh: dict[str, object] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "passed": bool(
            snappy is not None
            and check_mesh is not None
            and snappy.get("passed") is True
            and check_mesh.get("passed") is True
        ),
        "snappy_hex_mesh": snappy,
        "check_mesh": check_mesh,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mesh_volume_validation(
    block_mesh_output: str,
    check_mesh_output: str,
    expected_displaced_volume_m3: float,
    relative_tolerance: float,
) -> dict[str, float | bool]:
    if not math.isfinite(expected_displaced_volume_m3) or expected_displaced_volume_m3 <= 0.0:
        raise ValueError("expected displaced volume must be positive")
    if not math.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ValueError("mesh volume relative tolerance must lie in (0, 1)")
    bounds = re.search(
        rf"boundingBox\s*:\s*\(\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*\)"
        rf"\s*\(\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*\)",
        block_mesh_output,
    )
    if bounds is None:
        raise RuntimeError("blockMesh log has no parseable boundingBox")
    values = [float(value) for value in bounds.groups()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("blockMesh boundingBox contains a non-finite coordinate")
    extents = [values[index + 3] - values[index] for index in range(3)]
    if not all(extent > 0.0 for extent in extents):
        raise RuntimeError("blockMesh boundingBox must have three positive extents")
    domain_volume = extents[0] * extents[1] * extents[2]
    total = re.search(rf"Total volume\s*=\s*({_NUMBER})", check_mesh_output)
    if total is None:
        raise RuntimeError("checkMesh log has no parseable total volume")
    fluid_volume = float(total.group(1))
    excluded_volume = domain_volume - fluid_volume
    relative_error = abs(excluded_volume - expected_displaced_volume_m3) / expected_displaced_volume_m3
    return {
        "domain_volume_m3": float(domain_volume),
        "fluid_volume_m3": float(fluid_volume),
        "excluded_volume_m3": float(excluded_volume),
        "expected_displaced_volume_m3": float(expected_displaced_volume_m3),
        "relative_error": float(relative_error),
        "relative_tolerance": float(relative_tolerance),
        "passed": bool(
            math.isfinite(fluid_volume)
            and math.isfinite(excluded_volume)
            and fluid_volume > 0.0
            and excluded_volume > 0.0
            and fluid_volume < domain_volume
            and relative_error <= relative_tolerance
        ),
    }

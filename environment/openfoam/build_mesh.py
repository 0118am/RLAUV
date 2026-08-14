#!/usr/bin/env python3
"""Prepare geometry, build a checked snappyHexMesh, and render all motion cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases"
DEFAULT_GEOMETRY = HERE / "geometry" / "processed" / "auv_visual_m.stl"
DEFAULT_PROVENANCE = HERE / "results" / "geometry_provenance.json"
DEFAULT_MESH_VOLUME_RELATIVE_TOLERANCE = 0.055
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Watertight CAD STL to audit and scale")
    parser.add_argument(
        "--expected-sha256",
        help="optional digest check; the actual input digest is always recorded automatically",
    )
    parser.add_argument("--scale", type=float, default=0.001, help="Uniform post-transform scale")
    parser.add_argument(
        "--axis-map",
        default="x,y,z",
        metavar="SIGNED_AXES",
        help="Input axes supplying body x,y,z (default: x,y,z; example: z,-x,y)",
    )
    parser.add_argument(
        "--translate-after-map",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Translation in mapped input units, applied before --scale",
    )
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument(
        "--repair-report",
        type=Path,
        help="STEP repair report used to derive locked-rotor local refinement axes",
    )
    parser.add_argument("--backend", choices=("auto", "vtk", "openfoam"), default="auto")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Debug only: continue after failed VTK/OpenFOAM surface gates",
    )
    parser.add_argument("--force", action="store_true", help="Replace this workflow's generated outputs")
    parser.add_argument(
        "--prepared-input",
        action="store_true",
        help=(
            "Use INPUT directly as metre-scaled body-FLU geometry. The digest and "
            "OpenFOAM surface gate still run, but prepare_geometry.py is skipped."
        ),
    )
    parser.add_argument("--mesh-only", action="store_true", help="Stop after the checked shared mesh")
    parser.add_argument(
        "--expected-displaced-volume-m3",
        type=float,
        required=True,
        help="assembled-vehicle displaced volume used to validate snappy cell removal",
    )
    parser.add_argument(
        "--mesh-volume-relative-tolerance",
        type=float,
        default=DEFAULT_MESH_VOLUME_RELATIVE_TOLERANCE,
        help=(
            "maximum relative error between snappy-excluded and expected displaced volume "
            f"(default: {DEFAULT_MESH_VOLUME_RELATIVE_TOLERANCE:g})"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or running them")
    return parser


def _surface_check_failures(output: str) -> list[str]:
    """Return production-gate failures parsed from OpenCFD ``surfaceCheck``."""

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
    """Require OpenFOAM's terminal marker to be the final non-empty line."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return bool(lines and lines[-1] == "End")


def _fatal_diagnostics(output: str) -> list[str]:
    """Return fatal OpenFOAM diagnostics even when a wrapper reports exit code zero."""

    return [
        line.strip()
        for line in output.splitlines()
        if re.search(r"\b(?:FOAM\s+)?FATAL(?:\s+IO)?\s+ERROR\b", line, re.IGNORECASE)
    ]


def _final_configured_face_errors(output: str) -> tuple[dict[str, int], list[str]]:
    """Parse the last meshQualityDict threshold block, failing closed if incomplete.

    ``snappyHexMesh`` and ``checkMesh -meshQuality`` both print multiple diagnostic
    blocks.  Only the last ``Checking faces in error`` block describes the mesh that
    will be used.  Every printed configured criterion is retained, including future
    OpenFOAM criteria, so a non-zero value cannot be silently ignored.
    """

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
    """Retain diagnostics from optional all-geometry/all-topology checks."""

    warnings: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("<<")
            or stripped.startswith("***")
            or stripped.startswith("*There are")
            or re.fullmatch(r"Failed\s+\d+\s+mesh checks\.", stripped)
        ):
            warnings.append(stripped)
    return warnings


def _snappy_mesh_audit(output: str) -> dict[str, object]:
    """Audit the final snappy mesh against the configured production thresholds."""

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


def _check_mesh_audit(output: str) -> dict[str, object]:
    """Classify strict ``checkMesh`` evidence without conflating thresholds.

    The solver's configured limits are hard gates.  More conservative diagnostics
    enabled by ``-allGeometry -allTopology`` remain visible as warnings when the
    core topology, connected-region, volume, and configured criteria all pass.
    OpenFOAM may summarize those optional diagnostics as ``Failed N mesh checks``;
    that summary alone is therefore not a production-threshold failure.
    """

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
    if (
        minimum_volume is None
        or not math.isfinite(minimum_volume)
        or minimum_volume <= 0.0
    ):
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
        "extended_checks_reported_failed": (
            int(failed_match.group(1)) if failed_match else 0
        ),
    }


def _write_mesh_quality_audit(
    path: Path,
    snappy: dict[str, object] | None,
    check_mesh: dict[str, object] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": bool(
                    snappy is not None
                    and check_mesh is not None
                    and snappy.get("passed") is True
                    and check_mesh.get("passed") is True
                ),
                "snappy_hex_mesh": snappy,
                "check_mesh": check_mesh,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _mesh_volume_validation(
    block_mesh_output: str,
    check_mesh_output: str,
    expected_displaced_volume_m3: float,
    relative_tolerance: float,
) -> dict[str, float | bool]:
    """Compare the volume removed by snappy with measured displacement."""

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
    domain_volume = (
        extents[0]
        * extents[1]
        * extents[2]
    )
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


def _run(command: Sequence[str], *, log: Path | None = None, dry_run: bool = False) -> str:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return ""
    if log is None:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        output = completed.stdout + completed.stderr
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {printable}\n      log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        output = log.read_text(encoding="utf-8", errors="replace")
    if completed.returncode:
        detail = output[-3000:].strip()
        location = f"; see {log}" if log is not None else ""
        raise RuntimeError(f"command failed ({completed.returncode}): {printable}{location}\n{detail}")
    return output


def _require_commands() -> None:
    required = ("surfaceCheck", "blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "checkMesh")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "OpenCFD v2512 environment is not loaded; missing "
            + ", ".join(missing)
            + ". Run: source environment/openfoam/env.sh"
        )
    api = str(os.environ.get("FOAM_API", ""))
    if api != "2512":
        raise RuntimeError(f"FOAM_API={api or 'unset'}; this workflow requires OpenCFD API 2512")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_prepared_input(path: Path, expected_sha256: str | None) -> str:
    if not path.is_file():
        raise RuntimeError(f"prepared input does not exist: {path}")
    if path.suffix.lower() != ".stl":
        raise RuntimeError("prepared input must use the .stl suffix")
    actual = _sha256(path)
    if expected_sha256 is not None and actual != expected_sha256.lower():
        raise RuntimeError(
            f"prepared input SHA-256 mismatch: expected={expected_sha256.lower()}, actual={actual}"
        )
    return actual


def _prepare_command(args: argparse.Namespace, geometry: Path, provenance: Path) -> list[str]:
    source_digest = args.expected_sha256 or _sha256(args.input.resolve())
    command = [
        sys.executable,
        str(HERE / "tools" / "prepare_geometry.py"),
        str(args.input.resolve()),
        str(geometry),
        "--expected-sha256",
        source_digest,
        "--scale",
        f"{args.scale:.17g}",
        "--axis-map",
        args.axis_map,
        "--translate-after-map",
        *(str(value) for value in args.translate_after_map),
        "--backend",
        args.backend,
        "--provenance",
        str(provenance),
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    if args.force:
        command.append("--force")
    return command


def _generator_command(args: argparse.Namespace, *extra: str) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "generate_cases.py"),
        "--config",
        str(args.config.resolve()),
        "--output",
        str(args.cases_dir.resolve()),
        *extra,
    ]
    if args.force:
        command.append("--force")
    if args.repair_report is not None:
        command.extend(("--repair-report", str(args.repair_report.resolve())))
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    geometry = args.input.resolve() if args.prepared_input else args.geometry.resolve()
    provenance = args.provenance.resolve()
    cases_dir = args.cases_dir.resolve()
    mesh_case = cases_dir / "mesh_case"
    mesh_log_dir = mesh_case / "logs"
    block_mesh_output = ""
    mesh_volume_validation: dict[str, float | bool] | None = None
    snappy_mesh_audit: dict[str, object] | None = None
    check_mesh_audit: dict[str, object] | None = None
    mesh_quality_report = mesh_log_dir / "mesh_quality_audit.json"
    geometry_sha256: str | None = None

    try:
        if args.prepared_input:
            ignored_nondefaults = []
            if args.scale != 0.001:
                ignored_nondefaults.append("--scale")
            if args.axis_map != "x,y,z":
                ignored_nondefaults.append("--axis-map")
            if tuple(args.translate_after_map) != (0.0, 0.0, 0.0):
                ignored_nondefaults.append("--translate-after-map")
            if args.geometry != DEFAULT_GEOMETRY:
                ignored_nondefaults.append("--geometry")
            if args.backend != "auto":
                ignored_nondefaults.append("--backend")
            if ignored_nondefaults:
                raise ValueError(
                    "--prepared-input cannot be combined with preparation option(s): "
                    + ", ".join(ignored_nondefaults)
                )
        if args.expected_displaced_volume_m3 is not None and (
            not math.isfinite(args.expected_displaced_volume_m3)
            or args.expected_displaced_volume_m3 <= 0.0
        ):
            raise ValueError("--expected-displaced-volume-m3 must be positive")
        if (
            not math.isfinite(args.mesh_volume_relative_tolerance)
            or not 0.0 < args.mesh_volume_relative_tolerance < 1.0
        ):
            raise ValueError("--mesh-volume-relative-tolerance must lie in (0, 1)")
        if not args.dry_run:
            _require_commands()
        if args.prepared_input:
            geometry_sha256 = _verify_prepared_input(geometry, args.expected_sha256)
            if args.dry_run:
                print(f"recorded-prepared-sha256 {geometry} {geometry_sha256}")
        else:
            _run(_prepare_command(args, geometry, provenance), dry_run=args.dry_run)
            if not args.dry_run:
                geometry_sha256 = _sha256(geometry)

        surface_output = _run(
            ["surfaceCheck", "-checkSelfIntersection", str(geometry)],
            log=provenance.with_name("surfaceCheck.log"),
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            failures = _surface_check_failures(surface_output)
            if failures and not args.allow_dirty:
                raise RuntimeError(
                    "OpenFOAM surface gate failed: "
                    + ", ".join(failures)
                    + f"; see {provenance.with_name('surfaceCheck.log')}"
                )
            if failures:
                print("warning: --allow-dirty bypassed: " + ", ".join(failures), file=sys.stderr)

        _run(
            _generator_command(
                args,
                "--mesh-case-only",
                "--geometry",
                str(geometry),
                "--geometry-mode",
                "symlink",
            ),
            dry_run=args.dry_run,
        )
        for utility, utility_args in (
            ("blockMesh", ()),
            ("surfaceFeatureExtract", ()),
            ("snappyHexMesh", ("-overwrite",)),
            ("checkMesh", ("-allGeometry", "-allTopology", "-meshQuality")),
        ):
            utility_output = _run(
                [utility, *utility_args, "-case", str(mesh_case)],
                log=mesh_log_dir / f"{utility}.log",
                dry_run=args.dry_run,
            )
            if utility == "blockMesh":
                block_mesh_output = utility_output
            if utility == "snappyHexMesh" and not args.dry_run:
                snappy_mesh_audit = _snappy_mesh_audit(utility_output)
                _write_mesh_quality_audit(
                    mesh_quality_report, snappy_mesh_audit, check_mesh_audit
                )
                failures = snappy_mesh_audit["hard_failures"]
                if failures:
                    raise RuntimeError(
                        "snappy final-mesh gate failed: "
                        + ", ".join(str(item) for item in failures)
                        + f"; see {mesh_quality_report}"
                    )
            if utility == "checkMesh" and not args.dry_run:
                check_mesh_audit = _check_mesh_audit(utility_output)
                _write_mesh_quality_audit(
                    mesh_quality_report, snappy_mesh_audit, check_mesh_audit
                )
                if args.expected_displaced_volume_m3 is not None:
                    mesh_volume_validation = _mesh_volume_validation(
                        block_mesh_output,
                        utility_output,
                        args.expected_displaced_volume_m3,
                        args.mesh_volume_relative_tolerance,
                    )
                    volume_report = mesh_log_dir / "mesh_volume_validation.json"
                    volume_report.parent.mkdir(parents=True, exist_ok=True)
                    volume_report.write_text(
                        json.dumps(mesh_volume_validation, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if not mesh_volume_validation["passed"]:
                        raise RuntimeError(
                            "snappy excluded-volume gate failed: "
                            f"actual={mesh_volume_validation['excluded_volume_m3']:.12g} m^3, "
                            f"expected={mesh_volume_validation['expected_displaced_volume_m3']:.12g} m^3, "
                            f"relative_error={mesh_volume_validation['relative_error']:.6g} exceeds "
                            f"tolerance={mesh_volume_validation['relative_tolerance']:.6g}; "
                            f"see {volume_report}"
                        )
                failures = check_mesh_audit["hard_failures"]
                if failures:
                    raise RuntimeError(
                        "mesh quality gate failed: "
                        + ", ".join(str(item) for item in failures)
                        + f"; see {mesh_quality_report}"
                    )
                warnings = check_mesh_audit["warnings"]
                if warnings:
                    print(
                        "warning: checkMesh extended diagnostics were retained "
                        f"({len(warnings)} record(s)); see {mesh_quality_report}",
                        file=sys.stderr,
                    )

        if not args.mesh_only:
            _run(
                _generator_command(
                    args,
                    "--geometry",
                    str(geometry),
                    "--geometry-mode",
                    "symlink",
                    "--base-poly-mesh",
                    str(mesh_case / "constant" / "polyMesh"),
                    "--poly-mesh-mode",
                    "symlink",
                ),
                dry_run=args.dry_run,
            )

        summary = {
            "geometry": str(geometry),
            "geometry_sha256": geometry_sha256,
            "provenance": str(provenance),
            "mesh_case": str(mesh_case),
            "motion_cases_rendered": not args.mesh_only,
            "dirty_override": bool(args.allow_dirty),
            "prepared_input": bool(args.prepared_input),
            "dry_run": bool(args.dry_run),
            "mesh_quality_audit": (
                None if args.dry_run else str(mesh_quality_report)
            ),
            "mesh_volume_validation": mesh_volume_validation,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

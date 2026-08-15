#!/usr/bin/env python3
"""Prepare geometry, build a checked snappyHexMesh, and render all motion cases."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from environment.openfoam.mesh_audit import (
    check_mesh_audit as _check_mesh_audit,
    mesh_volume_validation as _mesh_volume_validation,
    snappy_mesh_audit as _snappy_mesh_audit,
    surface_check_failures as _surface_check_failures,
    write_mesh_quality_audit as _write_mesh_quality_audit,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases"
DEFAULT_GEOMETRY = HERE / "geometry" / "processed" / "auv_visual_m.stl"
DEFAULT_TRANSFORM_REPORT = HERE / "results" / "geometry_transform.json"
DEFAULT_MESH_VOLUME_RELATIVE_TOLERANCE = 0.055
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Watertight CAD STL to audit and scale")
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
    parser.add_argument(
        "--transform-report", type=Path, default=DEFAULT_TRANSFORM_REPORT
    )
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
            "Use INPUT directly as metre-scaled body-FLU geometry. OpenFOAM surface "
            "validation still runs, but prepare_geometry.py is skipped."
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


def _verify_prepared_input(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"prepared input does not exist: {path}")
    if path.suffix.lower() != ".stl":
        raise RuntimeError("prepared input must use the .stl suffix")


def _prepare_command(args: argparse.Namespace, geometry: Path, report: Path) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "tools" / "prepare_geometry.py"),
        str(args.input.resolve()),
        str(geometry),
        "--scale",
        f"{args.scale:.17g}",
        "--axis-map",
        args.axis_map,
        "--translate-after-map",
        *(str(value) for value in args.translate_after_map),
        "--backend",
        args.backend,
        "--report",
        str(report),
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


def _validate_build_args(args: argparse.Namespace) -> None:
    if args.prepared_input:
        ignored = []
        if args.scale != 0.001:
            ignored.append("--scale")
        if args.axis_map != "x,y,z":
            ignored.append("--axis-map")
        if tuple(args.translate_after_map) != (0.0, 0.0, 0.0):
            ignored.append("--translate-after-map")
        if args.geometry != DEFAULT_GEOMETRY:
            ignored.append("--geometry")
        if args.backend != "auto":
            ignored.append("--backend")
        if ignored:
            raise ValueError(
                "--prepared-input cannot be combined with preparation option(s): " + ", ".join(ignored)
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


def _prepare_and_check_surface(
    args: argparse.Namespace,
    geometry: Path,
    transform_report: Path,
    cases_dir: Path,
) -> None:
    if args.prepared_input:
        _verify_prepared_input(geometry)
    else:
        _run(_prepare_command(args, geometry, transform_report), dry_run=args.dry_run)
    output = _run(
        ["surfaceCheck", "-checkSelfIntersection", str(geometry)],
        log=cases_dir / "surfaceCheck.log",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    failures = _surface_check_failures(output)
    if failures and not args.allow_dirty:
        raise RuntimeError(
            "OpenFOAM surface validation failed: "
            + ", ".join(failures)
            + f"; see {cases_dir / 'surfaceCheck.log'}"
        )
    if failures:
        print("warning: --allow-dirty bypassed: " + ", ".join(failures), file=sys.stderr)


def _audit_snappy_output(
    output: str,
    quality_report: Path,
    check_mesh_audit: dict[str, object] | None,
) -> dict[str, object]:
    audit = _snappy_mesh_audit(output)
    _write_mesh_quality_audit(quality_report, audit, check_mesh_audit)
    if audit["hard_failures"]:
        raise RuntimeError(
            "snappy final mesh failed: "
            + ", ".join(str(item) for item in audit["hard_failures"])
            + f"; see {quality_report}"
        )
    return audit


def _audit_check_mesh_output(
    args: argparse.Namespace,
    block_mesh_output: str,
    output: str,
    quality_report: Path,
    snappy_audit: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, float | bool] | None]:
    audit = _check_mesh_audit(output)
    _write_mesh_quality_audit(quality_report, snappy_audit, audit)
    volume_validation = None
    if args.expected_displaced_volume_m3 is not None:
        volume_validation = _mesh_volume_validation(
            block_mesh_output,
            output,
            args.expected_displaced_volume_m3,
            args.mesh_volume_relative_tolerance,
        )
        volume_report = quality_report.parent / "mesh_volume_validation.json"
        volume_report.write_text(
            json.dumps(volume_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not volume_validation["passed"]:
            raise RuntimeError(
                "snappy excluded volume failed: "
                f"actual={volume_validation['excluded_volume_m3']:.12g} m^3, "
                f"expected={volume_validation['expected_displaced_volume_m3']:.12g} m^3, "
                f"relative_error={volume_validation['relative_error']:.6g} exceeds "
                f"tolerance={volume_validation['relative_tolerance']:.6g}; see {volume_report}"
            )
    if audit["hard_failures"]:
        raise RuntimeError(
            "mesh quality failed: "
            + ", ".join(str(item) for item in audit["hard_failures"])
            + f"; see {quality_report}"
        )
    if audit["warnings"]:
        print(
            "warning: checkMesh extended diagnostics were retained "
            f"({len(audit['warnings'])} record(s)); see {quality_report}",
            file=sys.stderr,
        )
    return audit, volume_validation


def _build_shared_mesh(
    args: argparse.Namespace,
    geometry: Path,
    mesh_case: Path,
) -> dict[str, float | bool] | None:
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
    log_dir = mesh_case / "logs"
    quality_report = log_dir / "mesh_quality_audit.json"
    block_mesh_output = ""
    snappy_audit = None
    check_mesh_audit = None
    volume_validation = None
    utilities = (
        ("blockMesh", ()),
        ("surfaceFeatureExtract", ()),
        ("snappyHexMesh", ("-overwrite",)),
        ("checkMesh", ("-allGeometry", "-allTopology", "-meshQuality")),
    )
    for utility, utility_args in utilities:
        output = _run(
            [utility, *utility_args, "-case", str(mesh_case)],
            log=log_dir / f"{utility}.log",
            dry_run=args.dry_run,
        )
        if utility == "blockMesh":
            block_mesh_output = output
        elif utility == "snappyHexMesh" and not args.dry_run:
            snappy_audit = _audit_snappy_output(output, quality_report, check_mesh_audit)
        elif utility == "checkMesh" and not args.dry_run:
            check_mesh_audit, volume_validation = _audit_check_mesh_output(
                args,
                block_mesh_output,
                output,
                quality_report,
                snappy_audit,
            )
    return volume_validation


def _render_motion_cases(args: argparse.Namespace, geometry: Path, mesh_case: Path) -> None:
    if args.mesh_only:
        return
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    geometry = args.input.resolve() if args.prepared_input else args.geometry.resolve()
    transform_report = args.transform_report.resolve()
    cases_dir = args.cases_dir.resolve()
    mesh_case = cases_dir / "mesh_case"
    try:
        _validate_build_args(args)
        if not args.dry_run:
            _require_commands()
        _prepare_and_check_surface(args, geometry, transform_report, cases_dir)
        volume_validation = _build_shared_mesh(args, geometry, mesh_case)
        _render_motion_cases(args, geometry, mesh_case)
        print(
            json.dumps(
                {
                    "geometry": str(geometry),
                    "transform_report": None if args.prepared_input else str(transform_report),
                    "mesh_case": str(mesh_case),
                    "motion_cases_rendered": not args.mesh_only,
                    "dirty_override": bool(args.allow_dirty),
                    "prepared_input": bool(args.prepared_input),
                    "dry_run": bool(args.dry_run),
                    "mesh_quality_audit": (
                        None if args.dry_run else str(mesh_case / "logs" / "mesh_quality_audit.json")
                    ),
                    "mesh_volume_validation": volume_validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

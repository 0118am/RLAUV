#!/usr/bin/env python3
"""Build one no-layer mesh, then attach it to the CFD cases."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.openfoam.case_generation.config import load_config
DEFAULT_REPAIR_REPORT = (
    HERE / "geometry/validated_locked_rotor_v1/selection_report.json"
)

DEFAULT_CASES = HERE / "cases"
MESH_COMPLETION_FILENAME = ".mesh_completed.json"
_CORE_POLY_MESH_FILES = ("boundary", "faces", "neighbour", "owner", "points")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Audited metre-scale body-FLU wetted OBJ"
    )
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--repair-report", type=Path, default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _run(
    command: Sequence[str],
    *,
    log: Path | None = None,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> str:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return ""
    if log is None:
        completed = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False
        )
        output = completed.stdout + completed.stderr
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {printable}\n      log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        output = log.read_text(encoding="utf-8", errors="replace")
    if completed.returncode:
        detail = output[-3000:].strip()
        location = f"; see {log}" if log is not None else ""
        raise RuntimeError(
            f"command failed ({completed.returncode}): {printable}{location}\n{detail}"
        )
    return output


def _require_commands() -> None:
    required = (
        "surfaceTransformPoints",
        "surfaceCheck",
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
    )
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "OpenCFD v2512 environment is not loaded; missing "
            + ", ".join(missing)
            + ". Run: source environment/openfoam/env.sh"
        )
    if os.environ.get("FOAM_API", "") != "2512":
        raise RuntimeError("FOAM_API must be 2512; source environment/openfoam/env.sh")


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
    command.extend(("--repair-report", str(args.repair_report.resolve())))
    return command


def _prepare_surface(
    args: argparse.Namespace,
    geometry: Path,
    cases_dir: Path,
    source_to_output_scale: float,
) -> None:
    if not geometry.is_file() or geometry.suffix.lower() != ".obj":
        raise RuntimeError(f"Prepared OBJ does not exist: {geometry}")
    if not math.isfinite(source_to_output_scale) or source_to_output_scale <= 0.0:
        raise RuntimeError("geometry source/output scale must be positive")
    cases_dir.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    audit_dir = (
        cases_dir / ".surface-audit"
        if args.dry_run
        else Path(tempfile.mkdtemp(prefix=".surface-audit-", dir=cases_dir))
    )
    normalized = audit_dir / "surface_source_scale.stl"
    try:
        _run(
            [
                "surfaceTransformPoints",
                "-write-scale",
                f"{1.0 / source_to_output_scale:.17g}",
                str(geometry),
                str(normalized),
            ],
            log=cases_dir / "surfaceTransformPoints.log",
            dry_run=args.dry_run,
            cwd=audit_dir,
        )
        _run(
            ["surfaceCheck", "-checkSelfIntersection", str(normalized)],
            log=cases_dir / "surfaceCheck.log",
            dry_run=args.dry_run,
            cwd=audit_dir,
        )
    finally:
        if not args.dry_run:
            shutil.rmtree(audit_dir)


def _build_shared_mesh(
    args: argparse.Namespace,
    geometry: Path,
    mesh_case: Path,
) -> dict[str, Any] | None:
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
    logs = mesh_case / "logs"

    _run(
        ["blockMesh", "-case", str(mesh_case)],
        log=logs / "blockMesh.log",
        dry_run=args.dry_run,
    )
    _run(
        ["snappyHexMesh", "-overwrite", "-case", str(mesh_case)],
        log=logs / "snappyHexMesh.log",
        dry_run=args.dry_run,
    )
    _run(
        [
            "checkMesh",
            "-allTopology",
            "-meshQuality",
            "-case",
            str(mesh_case),
        ],
        log=logs / "checkMesh.log",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return None
    return {
        "block_mesh_log": str(logs / "blockMesh.log"),
        "snappy_hex_mesh_log": str(logs / "snappyHexMesh.log"),
        "check_mesh_log": str(logs / "checkMesh.log"),
    }


def _render_motion_cases(
    args: argparse.Namespace, geometry: Path, mesh_case: Path
) -> None:
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
            str(mesh_case / "constant/polyMesh"),
            "--poly-mesh-mode",
            "symlink",
        ),
        dry_run=args.dry_run,
    )


def _completion_payload(
    cases_dir: Path,
    *,
    require_motion_cases: bool,
) -> dict[str, Any]:
    manifest = _load_object(cases_dir / "manifest.json")
    poly_mesh = cases_dir / "mesh_case/constant/polyMesh"
    for name in _CORE_POLY_MESH_FILES:
        if not (poly_mesh / name).is_file():
            raise RuntimeError(f"shared polyMesh is missing {name}")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("campaign manifest cases must be an array")
    if require_motion_cases:
        target = poly_mesh.resolve()
        for record in cases:
            if not isinstance(record, dict):
                raise RuntimeError("campaign manifest case record is not an object")
            name = str(record["case_name"])
            case = cases_dir / name
            if not (case / "case.json").is_file():
                raise RuntimeError(f"{name}: case.json is missing")
            linked = case / "constant/polyMesh"
            if not linked.is_dir() or linked.resolve() != target:
                raise RuntimeError(f"{name}: shared polyMesh link is missing")

    return {
        "status": "completed",
        "motion_cases_rendered": require_motion_cases,
        "case_count": len(cases),
    }


def validate_mesh_completion(
    cases_dir: Path,
    *,
    require_motion_cases: bool = True,
) -> tuple[bool, str]:
    marker_path = cases_dir / MESH_COMPLETION_FILENAME
    if not marker_path.is_file():
        return False, f"missing {MESH_COMPLETION_FILENAME}"
    try:
        actual = _load_object(marker_path)
        _completion_payload(
            cases_dir,
            require_motion_cases=require_motion_cases,
        )
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if actual.get("status") != "completed":
        return False, "mesh completion marker is not completed"
    return True, "shared mesh and case links are usable"


def _write_completion(
    cases_dir: Path,
    *,
    require_motion_cases: bool,
) -> Path:
    path = cases_dir / MESH_COMPLETION_FILENAME
    payload = _completion_payload(
        cases_dir,
        require_motion_cases=require_motion_cases,
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    geometry = args.input.resolve()
    cases_dir = args.cases_dir.resolve()
    mesh_case = cases_dir / "mesh_case"
    try:
        config = load_config(args.config.resolve())
        if args.verify_existing:
            valid, reason = validate_mesh_completion(
                cases_dir,
                require_motion_cases=not args.mesh_only,
            )
            print(json.dumps({"valid": valid, "reason": reason}, indent=2))
            return 0 if valid else 1
        if args.force and cases_dir.exists() and not args.dry_run:
            if cases_dir.is_symlink() or not cases_dir.is_dir():
                cases_dir.unlink()
            else:
                shutil.rmtree(cases_dir)
        if not args.dry_run:
            _require_commands()
        geometry_audit = config["geometry_audit"]
        _prepare_surface(
            args,
            geometry,
            cases_dir,
            float(geometry_audit["source_to_output_coordinate_scale"]),
        )
        mesh_diagnostics = _build_shared_mesh(args, geometry, mesh_case)
        _render_motion_cases(args, geometry, mesh_case)
        marker = None
        if not args.dry_run:
            marker = _write_completion(
                cases_dir,
                require_motion_cases=not args.mesh_only,
            )
        print(
            json.dumps(
                {
                    "geometry": str(geometry),
                    "mesh_case": str(mesh_case),
                    "motion_cases_rendered": not args.mesh_only,
                    "dry_run": args.dry_run,
                    "mesh_diagnostics": mesh_diagnostics,
                    "mesh_completion": None if marker is None else str(marker),
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

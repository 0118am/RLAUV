#!/usr/bin/env python3
"""Render the split OpenFOAM-v2512 damping and added-mass campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.openfoam.case_generation.config import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    CaseSpec,
    load_config,
    campaign_specs,
    stationary_spec,
)
from environment.openfoam.case_generation.renderers import (
    load_locked_rotor_report,
)
from environment.openfoam.case_generation.writer import render_case

DEFAULT_REPAIR_REPORT = (
    HERE / "geometry/validated_locked_rotor_v1/selection_report.json"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--list", action="store_true", help="List case names without writing files.")
    result.add_argument("--dry-run", action="store_true", help="Print planned writes without changing files.")
    result.add_argument("--mesh-case-only", action="store_true", help="Render only one stationary mesh_case.")
    result.add_argument("--force", action="store_true", help="Replace generated case directories that already exist.")
    result.add_argument(
        "--geometry",
        type=Path,
        help="Override the audited metre-scaled OBJ source path.",
    )
    result.add_argument("--geometry-mode", choices=("symlink", "copy", "none"), default="symlink")
    result.add_argument("--base-poly-mesh", type=Path, help="Existing constant/polyMesh shared by motion cases.")
    result.add_argument("--poly-mesh-mode", choices=("symlink", "copy", "none"), default="none")
    result.add_argument(
        "--repair-report",
        type=Path,
        default=DEFAULT_REPAIR_REPORT,
        help="STEP repair report containing measured locked-rotor axes and connector endpoints.",
    )
    return result


def _selected_specs(args: argparse.Namespace, config: dict) -> list[CaseSpec]:
    if args.mesh_case_only:
        return [stationary_spec("mesh_case", "shared_mesh")]
    return campaign_specs(config)


def _locked_rotors(args: argparse.Namespace, config: dict):
    if args.repair_report is not None:
        return load_locked_rotor_report(args.repair_report.resolve())
    if args.mesh_case_only and config["locked_rotor_mesh"].get("enabled"):
        raise ValueError("--repair-report is required to derive locked-rotor mesh refinement axes")
    return None


def _print_dry_run(
    args: argparse.Namespace,
    output: Path,
    geometry: Path,
    specs: list[CaseSpec],
) -> None:
    print(f"output={output}")
    print(f"geometry={geometry} mode={args.geometry_mode}")
    if args.base_poly_mesh:
        print(f"base_poly_mesh={args.base_poly_mesh.resolve()} mode={args.poly_mesh_mode}")
    for spec in specs:
        print(f"would generate {output / spec.name}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config.resolve())
    specs = _selected_specs(args, config)
    if args.list:
        for spec in specs:
            print(spec.name)
        return 0

    locked_rotors = _locked_rotors(args, config)
    output = args.output.resolve()
    default_geometry = HERE / config["geometry_path"]
    geometry = args.geometry.resolve() if args.geometry else default_geometry.resolve()
    if args.dry_run:
        _print_dry_run(args, output, geometry, specs)
        return 0

    output.mkdir(parents=True, exist_ok=True)
    case_records = []
    for spec in specs:
        case_records.append(
            render_case(spec, config, output, geometry, args, locked_rotors)
        )
        print(f"generated {output / spec.name}")
    manifest = {
        "schema_version": 5,
        "case_count": len(specs),
        "geometry_path": str(geometry),
        "repair_report_path": str(args.repair_report.resolve()),
        "cases": case_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

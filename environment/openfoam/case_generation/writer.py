"""Materialize rendered OpenFOAM cases on disk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from environment.openfoam.case_generation.config import CaseSpec, DEFAULT_TEMPLATE
from environment.openfoam.case_generation.renderers import (
    metadata,
    render_block_mesh_dict,
    render_control_dict,
    render_dynamic_mesh_dict,
    render_fv_solution,
    render_point_displacement,
    render_pressure_field,
    render_snappy_hex_mesh_dict,
    render_transport_properties,
    render_velocity_field,
    render_turbulence_field,
)
def attach(source: Path, destination: Path, mode: str) -> None:
    if mode == "none":
        return
    if not source.exists():
        raise FileNotFoundError(f"Required source does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()), target_is_directory=source.is_dir())


def render_case(
    spec: CaseSpec,
    cfg: dict[str, Any],
    output: Path,
    geometry: Path,
    args: argparse.Namespace,
    locked_rotors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case_dir = output / spec.name
    if case_dir.exists():
        if not args.force:
            raise FileExistsError(f"Case already exists (use --force): {case_dir}")
        shutil.rmtree(case_dir)
    shutil.copytree(DEFAULT_TEMPLATE, case_dir)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    # Keep every generated case dictionary complete and parseable even though
    # production motion cases normally consume the checked shared polyMesh.
    (case_dir / "system" / "blockMeshDict").write_text(
        render_block_mesh_dict(cfg), encoding="utf-8"
    )
    (case_dir / "system" / "snappyHexMeshDict").write_text(
        render_snappy_hex_mesh_dict(cfg, locked_rotors), encoding="utf-8"
    )
    (case_dir / "0" / "pointDisplacement").write_text(render_point_displacement(spec, cfg), encoding="utf-8")
    (case_dir / "0" / "U").write_text(
        render_velocity_field(spec, cfg), encoding="utf-8"
    )
    (case_dir / "0" / "p").write_text(
        render_pressure_field(spec, cfg), encoding="utf-8"
    )
    for field_name in ("k", "omega", "nut"):
        (case_dir / "0" / field_name).write_text(
            render_turbulence_field(field_name, cfg), encoding="utf-8"
        )
    (case_dir / "system" / "controlDict").write_text(render_control_dict(spec, cfg), encoding="utf-8")
    (case_dir / "system" / "fvSolution").write_text(
        render_fv_solution(spec, cfg), encoding="utf-8"
    )
    (case_dir / "constant" / "dynamicMeshDict").write_text(
        render_dynamic_mesh_dict(spec, cfg), encoding="utf-8"
    )
    (case_dir / "constant" / "transportProperties").write_text(render_transport_properties(cfg), encoding="utf-8")
    attach(geometry, case_dir / "constant" / "triSurface" / cfg["geometry_filename"], args.geometry_mode)
    if args.poly_mesh_mode != "none":
        if args.base_poly_mesh is None:
            raise ValueError("--base-poly-mesh is required when --poly-mesh-mode is not 'none'.")
        attach(args.base_poly_mesh, case_dir / "constant" / "polyMesh", args.poly_mesh_mode)
    # OpenFOAM utilities are allowed to mutate ``0``. Keep one source tree and
    # let the runner recreate the working
    # tree before and after every attempt.  This is the standard ``0.orig``
    # case layout and also recovers cleanly after SIGKILL/power loss.
    shutil.copytree(case_dir / "0", case_dir / "0.orig")
    case_metadata = metadata(spec, cfg)
    (case_dir / "case.json").write_text(
        json.dumps(case_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return case_metadata

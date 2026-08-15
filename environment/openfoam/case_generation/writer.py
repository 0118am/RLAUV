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
    render_fv_solution,
    render_point_displacement,
    render_snappy_hex_mesh_dict,
    render_transport_properties,
    render_velocity_field,
    render_wall_function_field,
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
) -> None:
    case_dir = output / spec.name
    if case_dir.exists():
        if not args.force:
            raise FileExistsError(f"Case already exists (use --force): {case_dir}")
        shutil.rmtree(case_dir)
    shutil.copytree(DEFAULT_TEMPLATE, case_dir)
    if spec.purpose == "shared_mesh":
        (case_dir / "system" / "blockMeshDict").write_text(
            render_block_mesh_dict(cfg), encoding="utf-8"
        )
        (case_dir / "system" / "snappyHexMeshDict").write_text(
            render_snappy_hex_mesh_dict(cfg, locked_rotors), encoding="utf-8"
        )
    (case_dir / "0" / "pointDisplacement").write_text(render_point_displacement(spec, cfg), encoding="utf-8")
    (case_dir / "0" / "U").write_text(render_velocity_field(cfg), encoding="utf-8")
    for field_name in ("nut", "omega"):
        (case_dir / "0" / field_name).write_text(
            render_wall_function_field(field_name, cfg), encoding="utf-8"
        )
    (case_dir / "system" / "controlDict").write_text(render_control_dict(spec, cfg), encoding="utf-8")
    (case_dir / "system" / "fvSolution").write_text(
        render_fv_solution(cfg), encoding="utf-8"
    )
    (case_dir / "constant" / "transportProperties").write_text(render_transport_properties(cfg), encoding="utf-8")
    (case_dir / "motion.json").write_text(json.dumps(metadata(spec, cfg), indent=2) + "\n", encoding="utf-8")
    attach(geometry, case_dir / "constant" / "triSurface" / cfg["geometry_filename"], args.geometry_mode)
    if args.poly_mesh_mode != "none":
        if args.base_poly_mesh is None:
            raise ValueError("--base-poly-mesh is required when --poly-mesh-mode is not 'none'.")
        attach(args.base_poly_mesh, case_dir / "constant" / "polyMesh", args.poly_mesh_mode)

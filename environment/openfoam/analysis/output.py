"""Configuration loading and durable files for fitted matrices."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

from .motion import DOF_NAMES, WRENCH_NAMES
from .types import HydroFitResult

def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wrench/dof", *DOF_NAMES))
        for name, row in zip(WRENCH_NAMES, matrix):
            writer.writerow((name, *(f"{float(value):.17g}" for value in row)))


def write_fit_outputs(result: HydroFitResult, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Immutable CFD result already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    paths = {
        "report": staging / "hydrodynamic_fit.json",
        "config_updates": staging / "config_updates.json",
        "added_mass": staging / "added_mass.csv",
        "linear_damping": staging / "linear_damping.csv",
        "quadratic_damping": staging / "quadratic_damping.csv",
    }
    try:
        with paths["report"].open("w", encoding="utf-8") as stream:
            json.dump(result.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        with paths["config_updates"].open("w", encoding="utf-8") as stream:
            json.dump(result.config_updates(), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        _write_matrix_csv(paths["added_mass"], result.added_mass)
        _write_matrix_csv(paths["linear_damping"], result.linear_damping)
        _write_matrix_csv(paths["quadratic_damping"], result.quadratic_damping)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {name: str(destination / path.name) for name, path in paths.items()}

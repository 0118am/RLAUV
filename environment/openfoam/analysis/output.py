"""Configuration loading and durable files for fitted matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .motion import DOF_NAMES, WRENCH_NAMES
from .types import HydroFitResult

def load_analysis_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Analysis config must contain a JSON object")
    return value


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wrench/dof", *DOF_NAMES))
        for name, row in zip(WRENCH_NAMES, matrix):
            writer.writerow((name, *(f"{float(value):.17g}" for value in row)))


def write_fit_outputs(result: HydroFitResult, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": destination / "hydrodynamic_fit.json",
        "config_updates": destination / "config_updates.json",
        "added_mass": destination / "added_mass.csv",
        "added_mass_raw": destination / "added_mass_raw.csv",
        "linear_damping": destination / "linear_damping.csv",
        "quadratic_damping": destination / "quadratic_damping.csv",
    }
    with paths["report"].open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with paths["config_updates"].open("w", encoding="utf-8") as stream:
        json.dump(result.config_updates(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _write_matrix_csv(paths["added_mass"], result.added_mass)
    _write_matrix_csv(paths["added_mass_raw"], result.added_mass_raw)
    _write_matrix_csv(paths["linear_damping"], result.linear_damping)
    _write_matrix_csv(paths["quadratic_damping"], result.quadratic_damping)
    return {name: str(path) for name, path in paths.items()}


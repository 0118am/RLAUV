#!/usr/bin/env python3
"""Copy three fitted 6x6 CFD matrices into the PhysX profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from environment.profile import EnvironmentProfile

DEFAULT_PROFILE = (
    REPOSITORY_ROOT
    / "environment/hydrodynamics/coefficients/auv_open_water_openfoam_full_hydrodynamics_v2.json"
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _matrix(report: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(report["matrices"][name], dtype=float)
    if value.shape != (6, 6) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 6x6 matrix")
    return value


def verify_report(report_path: Path) -> dict[str, Any]:
    report = _load_object(report_path)
    matrices = {
        name: _matrix(report, name)
        for name in ("added_mass", "linear_damping", "quadratic_damping")
    }
    report["_verified_matrices"] = {
        name: value.tolist() for name, value in matrices.items()
    }
    return report


def publish(
    report_path: Path,
    profile_path: Path,
) -> None:
    report = verify_report(report_path)
    profile = _load_object(profile_path)
    matrices = report.pop("_verified_matrices")
    hydrodynamics = profile["hydrodynamics"]
    hydrodynamics["added_mass"] = matrices["added_mass"]
    hydrodynamics["linear_damping"] = matrices["linear_damping"]
    hydrodynamics["quadratic_damping"] = matrices["quadratic_damping"]
    profile["name"] = "auv-openfoam-open-water-full-hydrodynamics-v2"
    profile["description"] = (
        "Full-response locked-rotor open-water CFD matrices for RL below 0.4 m/s."
    )
    EnvironmentProfile.model_validate(profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=profile_path.parent,
        prefix=f".{profile_path.name}.",
        delete=False,
    ) as stream:
        json.dump(profile, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, profile_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="hydrodynamic_fit.json")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    publish(
        args.report.resolve(),
        args.profile.resolve(),
    )
    print(args.profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

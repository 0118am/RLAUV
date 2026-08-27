"""Minimal completion handling for generated OpenFOAM cases."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_FOAM_API = "2512"


def _read_case_metadata(case: Path) -> dict[str, Any]:
    path = case / "case.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{case.name}: cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 5:
        raise RuntimeError(f"{case.name}: {path} must use schema_version 5")
    try:
        end_time = float(value["end_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{case.name}: invalid end_time_s") from exc
    if not math.isfinite(end_time) or end_time < 0.0:
        raise RuntimeError(f"{case.name}: invalid end_time_s")
    return value


def _validated_completion(case: Path, solver: str) -> tuple[bool, str]:
    """Treat a matching completion marker as sufficient for ``--resume``."""

    marker_path = case / ".completed"
    if not marker_path.is_file():
        return False, "missing .completed marker"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        _read_case_metadata(case)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if not isinstance(marker, dict):
        return False, ".completed is not an object"
    required = {
        "status": "completed",
        "case": case.name,
        "solver": solver,
        "foam_api": _FOAM_API,
    }
    for name, expected in required.items():
        if marker.get(name) != expected:
            return False, f".completed {name} differs"
    return True, "completed"

"""Execute one planned OpenFOAM case and publish its completion marker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import time

from environment.openfoam.case_execution.planning import _command_plan
from environment.openfoam.case_execution.validation import (
    _FOAM_API,
    _MARKER_SCHEMA_VERSION,
    _read_motion,
    _validate_case_outputs,
)

def _run_one(
    case: Path,
    solver: str,
    ranks: int,
    reconstruct: bool,
    dry_run: bool,
    bind_to_core: bool = False,
    cpu_set: str | None = None,
) -> tuple[str, float]:
    started = time.monotonic()
    plan = _command_plan(case, solver, ranks, reconstruct, bind_to_core, cpu_set)
    if dry_run:
        print(json.dumps({"case": case.name, "commands": [command for command, _ in plan]}, ensure_ascii=False))
        return case.name, 0.0

    marker_path = case / ".completed"
    marker_path.unlink(missing_ok=True)
    print(f"[start] {case.name}", flush=True)
    for command, log_path in plan:
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"{case.name}: {' '.join(command)} failed; see {log_path}")

    metadata = _read_motion(case)
    validation = _validate_case_outputs(case, solver, metadata)
    marker = {
        "schema_version": _MARKER_SCHEMA_VERSION,
        "status": "completed",
        "case": case.name,
        "solver": solver,
        "foam_api": _FOAM_API,
        "mpi_ranks": ranks,
        "bind_to_core": bool(bind_to_core and ranks > 1),
        "cpu_set": cpu_set if ranks > 1 else None,
        "motion": metadata,
        "validation": validation,
        "elapsed_s": time.monotonic() - started,
    }
    temporary_marker = marker_path.with_name(f"{marker_path.name}.tmp")
    temporary_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_marker, marker_path)
    print(f"[done]  {case.name} ({marker['elapsed_s']:.1f} s)", flush=True)
    return case.name, float(marker["elapsed_s"])


def _run_one_with_cpu_slot(
    cpu_slots: queue.Queue[str],
    case: Path,
    solver: str,
    ranks: int,
    reconstruct: bool,
    dry_run: bool,
    bind_to_core: bool,
) -> tuple[str, float]:
    """Lease one exclusive CPU set for the duration of a case run."""

    cpu_set = cpu_slots.get()
    try:
        return _run_one(
            case,
            solver,
            ranks,
            reconstruct,
            dry_run,
            bind_to_core,
            cpu_set,
        )
    finally:
        cpu_slots.put(cpu_set)


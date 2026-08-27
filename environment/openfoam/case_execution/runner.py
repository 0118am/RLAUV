"""Execute one planned OpenFOAM case and publish its completion marker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import time

from environment.openfoam.case_execution.planning import _command_plan
from environment.openfoam.case_execution.validation import (
    _FOAM_API,
    _read_case_metadata,
)


_PROCESSOR_DIRECTORY_RE = re.compile(r"^processor\d+$")


def _is_generated_time_directory(path: Path) -> bool:
    """Return whether ``path`` is a solver time other than the working zero."""

    if not path.is_dir() or path.name == "0":
        return False
    try:
        value = float(path.name)
    except ValueError:
        return False
    return value > 0.0


def _reset_working_initial_fields(case: Path) -> None:
    """Atomically recreate mutable ``0`` from ``0.orig``."""

    source = case / "0.orig"
    if not source.is_dir():
        raise RuntimeError(f"{case.name}: missing immutable initial fields {source}")
    staging = case / ".0.reset"
    if staging.exists() or staging.is_symlink():
        shutil.rmtree(staging) if staging.is_dir() and not staging.is_symlink() else staging.unlink()
    shutil.copytree(source, staging, symlinks=True)
    working = case / "0"
    if working.exists() or working.is_symlink():
        shutil.rmtree(working) if working.is_dir() and not working.is_symlink() else working.unlink()
    os.replace(staging, working)


def _clean_generated_outputs(case: Path) -> None:
    """Remove only artifacts owned by a previous execution attempt."""

    for child in case.iterdir():
        if (
            child.name in {"postProcessing", ".execution"}
            or _PROCESSOR_DIRECTORY_RE.fullmatch(child.name) is not None
            or _is_generated_time_directory(child)
        ):
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        elif child.is_file() and (
            child.name.startswith("log.")
            or child.name in {".completed", ".completed.tmp"}
        ):
            child.unlink()


def _remove_parallel_partitions(case: Path) -> None:
    """Discard reproducible processor fields after a successful run."""

    for child in case.iterdir():
        if _PROCESSOR_DIRECTORY_RE.fullmatch(child.name) is not None:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def _write_runtime_decomposition(case: Path, ranks: int) -> dict[str, object] | None:
    if ranks <= 1:
        return None
    directory = case / ".execution"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "decomposeParDict"
    path.write_text(
        """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains %d;
method          scotch;
distributed     no;
roots           ();
""" % ranks,
        encoding="utf-8",
    )
    return {
        "path": str(path.relative_to(case)),
        "number_of_subdomains": ranks,
        "method": "scotch",
    }

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
    if dry_run:
        plan = _command_plan(case, solver, ranks, reconstruct, bind_to_core, cpu_set)
        print(json.dumps({"case": case.name, "commands": [command for command, _ in plan]}, ensure_ascii=False))
        return case.name, 0.0

    _read_case_metadata(case)
    _clean_generated_outputs(case)
    _reset_working_initial_fields(case)
    runtime_decomposition = _write_runtime_decomposition(case, ranks)
    plan = _command_plan(case, solver, ranks, reconstruct, bind_to_core, cpu_set)
    marker_path = case / ".completed"
    print(f"[start] {case.name}", flush=True)
    try:
        for command, log_path in plan:
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(f"{case.name}: {' '.join(command)} failed; see {log_path}")
        _remove_parallel_partitions(case)
    finally:
        # A normal failure and a later retry both leave the rendered case in
        # the same pristine state.  SIGKILL is recovered at the next attempt,
        # which resets ``0`` before invoking any OpenFOAM utility.
        _reset_working_initial_fields(case)
    marker = {
        "status": "completed",
        "case": case.name,
        "solver": solver,
        "foam_api": _FOAM_API,
        "mpi_ranks": ranks,
        "bind_to_core": bool(bind_to_core and ranks > 1),
        "cpu_set": cpu_set if ranks > 1 else None,
        "runtime_decomposition": runtime_decomposition,
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

"""Discover runnable cases and build serial/MPI command plans."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import re
import sys

from environment.openfoam.case_execution.validation import _validated_completion

_CPU_LIST_ITEM_RE = re.compile(r"^(\d+)(?:-(\d+))?$")

def _parse_cpu_set(value: str) -> tuple[tuple[int, int], ...]:
    """Parse the comma/range form accepted by ``taskset --cpu-list``."""

    if not value:
        raise ValueError("must not be empty")
    intervals: list[tuple[int, int]] = []
    for item in value.split(","):
        match = _CPU_LIST_ITEM_RE.fullmatch(item)
        if match is None:
            raise ValueError("must use comma-separated CPU numbers or inclusive ranges")
        start = int(match.group(1))
        end = start if match.group(2) is None else int(match.group(2))
        if end < start:
            raise ValueError(f"range {item!r} ends before it starts")
        intervals.append((start, end))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] <= previous[1]:
            raise ValueError("contains duplicate or overlapping CPUs")
    return tuple(intervals)


def _validate_cpu_sets(values: list[str], jobs: int) -> None:
    if len(values) != jobs:
        raise ValueError(f"received {len(values)} --cpu-set values, but --jobs is {jobs}")

    parsed: list[tuple[tuple[int, int], ...]] = []
    for value in values:
        try:
            parsed.append(_parse_cpu_set(value))
        except ValueError as exc:
            raise ValueError(f"invalid --cpu-set {value!r}: {exc}") from exc

    for left_index, left in enumerate(parsed):
        for right_index in range(left_index + 1, len(parsed)):
            right = parsed[right_index]
            if any(
                left_start <= right_end and right_start <= left_end
                for left_start, left_end in left
                for right_start, right_end in right
            ):
                raise ValueError(
                    "--cpu-set values must be mutually disjoint: "
                    f"{values[left_index]!r} overlaps {values[right_index]!r}"
                )


def _discover(cases_dir: Path, patterns: list[str], resume: bool, solver: str = "pimpleFoam") -> list[Path]:
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory does not exist: {cases_dir}")
    cases = []
    skipped = 0
    for metadata in sorted(cases_dir.glob("*/case.json")):
        case = metadata.parent
        try:
            case_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{case.name}: cannot read {metadata}: {exc}") from exc
        if not isinstance(case_metadata, dict):
            raise RuntimeError(f"{case.name}: {metadata} must contain a JSON object")
        if case_metadata.get("purpose") == "shared_mesh":
            continue
        if patterns and not any(fnmatch.fnmatch(case.name, pattern) for pattern in patterns):
            continue
        if resume and (case / ".completed").is_file():
            valid, reason = _validated_completion(case, solver)
            if valid:
                skipped += 1
                print(f"[resume] skip {case.name}: {reason}", flush=True)
                continue
            print(f"[resume] rerun {case.name}: {reason}", file=sys.stderr, flush=True)
        if not (case / "constant" / "polyMesh" / "boundary").is_file():
            raise FileNotFoundError(f"{case}: missing constant/polyMesh; build/distribute the mesh first")
        cases.append(case)
    if not cases:
        if skipped:
            return []
        raise RuntimeError("No runnable cases matched.")
    return cases


def _command_plan(
    case: Path,
    solver: str,
    ranks: int,
    reconstruct: bool,
    bind_to_core: bool = False,
    cpu_set: str | None = None,
) -> list[tuple[list[str], Path]]:
    if cpu_set is not None:
        if ranks <= 1:
            raise ValueError("cpu_set requires MPI execution")
        if not bind_to_core:
            raise ValueError("cpu_set requires bind_to_core")
        _parse_cpu_set(cpu_set)

    metadata = json.loads((case / "case.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 5:
        raise ValueError(f"{case}: only case schema_version 5 is runnable")
    steady = metadata.get("case_family") == "steady_damping"
    plan: list[tuple[list[str], Path]] = []
    if steady:
        plan.append(
            (
                ["potentialFoam", "-writePhi", "-case", str(case)],
                case / "log.potentialFoam",
            )
        )
    if ranks > 1:
        runtime_decomposition = case / ".execution" / "decomposeParDict"
        plan.append(
            (
                [
                    "decomposePar",
                    "-force",
                    "-decomposeParDict",
                    str(runtime_decomposition),
                    "-case",
                    str(case),
                ],
                case / "log.decomposePar",
            )
        )
        mpi_command = []
        if cpu_set is not None:
            mpi_command.extend(("taskset", "-c", cpu_set))
        mpi_command.extend(("mpirun", "-np", str(ranks)))
        if bind_to_core:
            mpi_command.extend(("--map-by", "core", "--bind-to", "core"))
        mpi_command.extend((solver, "-parallel", "-case", str(case)))
        plan.append((mpi_command, case / f"log.{solver}"))
        if reconstruct:
            plan.append((["reconstructPar", "-case", str(case)], case / "log.reconstructPar"))
    else:
        plan.append(([solver, "-case", str(case)], case / f"log.{solver}"))
    return plan

#!/usr/bin/env python3
"""Run generated forced-oscillation cases with bounded local concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import time
from typing import Any


_FOAM_API = "2512"
_MARKER_SCHEMA_VERSION = 2
_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|(?:nan|inf(?:inity)?))"
_NUMBER_RE = re.compile(_NUMBER, re.IGNORECASE)
_TIME_LINE_RE = re.compile(rf"^\s*Time\s*=\s*({_NUMBER})\s*$", re.IGNORECASE)
_FORCE_SPLIT_RE = re.compile(r"^force(?:_(\d+))?\.dat$")
_MOMENT_SPLIT_RE = re.compile(r"^moment(?:_(\d+))?\.dat$")
_NONFINITE_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?(?:nan|inf(?:inity)?)(?![A-Za-z0-9_])", re.IGNORECASE)
_FATAL_LOG_RE = re.compile(
    r"FOAM\s+FATAL|\bfatal\s+(?:io\s+)?error\b|\bMPI_ABORT\b|"
    r"floating\s+point\s+exception(?!\s+trapping\s+enabled)|segmentation\s+fault|\bcore\s+dumped\b|"
    r"\bSIG(?:FPE|SEGV|ABRT)\b",
    re.IGNORECASE,
)
_NEGATIVE_VOLUME_PHRASE_RE = re.compile(
    r"\bnegative\s+(?:(?:or\s+zero|cell)\s+)*volumes?\b|"
    r"\bnegative\s*volume(?:s|cells)?\b|"
    r"\bcells?\s+with\s+negative\s+volume\b|"
    r"\bvolumes?\b.{0,24}\b(?:is|are|was|were|=|:)\s*negative\b",
    re.IGNORECASE,
)
_NEGATIVE_VOLUME_VALUE_RE = re.compile(
    r"\b(?:(?:minimum|min|cell)\s+)?volume\b\s*(?:=|:)\s*-\s*(?:\d|\.)",
    re.IGNORECASE,
)
_BENIGN_NEGATIVE_VOLUME_RE = re.compile(
    r"\b(?:no|zero|0)\s+(?:cells?\s+with\s+)?negative"
    r"(?:\s+(?:(?:or\s+zero|cell)\s+)*|\s*)volume(?:s|cells)?\b|"
    r"\bnon[-\s]?negative\b",
    re.IGNORECASE,
)
_CPU_LIST_ITEM_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


def _has_negative_volume(line: str) -> bool:
    if _NEGATIVE_VOLUME_VALUE_RE.search(line):
        return True
    without_benign_reports = _BENIGN_NEGATIVE_VOLUME_RE.sub("", line)
    return _NEGATIVE_VOLUME_PHRASE_RE.search(without_benign_reports) is not None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=Path(__file__).with_name("cases"))
    parser.add_argument("--only", action="append", default=[], help="Case-name glob; repeatable.")
    parser.add_argument("--np", type=int, default=4, help="MPI ranks per case; use 1 for serial.")
    parser.add_argument("--jobs", type=int, default=1, help="Cases to execute concurrently.")
    parser.add_argument("--solver", default="pimpleFoam")
    parser.add_argument(
        "--bind-to-core",
        action="store_true",
        help="Map each MPI rank to one physical core and bind it there.",
    )
    parser.add_argument(
        "--cpu-set",
        action="append",
        dest="cpu_sets",
        default=[],
        metavar="LIST",
        help=(
            "Linux CPU list reserved for one concurrent job (for example 0-7,16-23); "
            "repeat once per --jobs slot. Requires MPI and --bind-to-core."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases whose .completed marker, solver log, and force outputs still validate.",
    )
    parser.add_argument("--reconstruct", action="store_true", help="Run reconstructPar after parallel solve.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


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


def _read_motion(case: Path) -> dict[str, Any]:
    path = case / "motion.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{case.name}: cannot read {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{case.name}: {path} must contain a JSON object")
    value = metadata.get("end_time_s")
    if isinstance(value, bool):
        raise RuntimeError(f"{case.name}: motion.json end_time_s must be a finite non-negative number")
    try:
        end_time_s = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{case.name}: motion.json end_time_s must be a finite non-negative number"
        ) from exc
    if not math.isfinite(end_time_s) or end_time_s < 0.0:
        raise RuntimeError(f"{case.name}: motion.json end_time_s must be a finite non-negative number")
    return metadata


def _time_tolerance(expected: float) -> float:
    """Cover decimal rendering noise without accepting a missing time step."""

    return 1.0e-9 * max(1.0, abs(float(expected)))


def _numeric_path_key(path: Path) -> tuple[tuple[int, float | str], ...]:
    key: list[tuple[int, float | str]] = []
    for part in path.parts:
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part))
    return tuple(key)


def _require_time_coverage(case: Path, label: str, actual: float, expected: float) -> None:
    if actual + _time_tolerance(expected) < expected:
        raise RuntimeError(
            f"{case.name}: {label} ends at {actual:.12g} s, before motion end_time_s={expected:.12g} s"
        )


def _scan_solver_log(case: Path, solver: str) -> float:
    path = case / f"log.{solver}"
    if not path.is_file():
        raise RuntimeError(f"{case.name}: missing solver log {path}")

    last_nonempty = ""
    last_time_s: float | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            last_nonempty = stripped
            if _NONFINITE_RE.search(stripped):
                raise RuntimeError(f"{case.name}: non-finite value in {path}:{line_number}")
            if _FATAL_LOG_RE.search(stripped):
                raise RuntimeError(f"{case.name}: fatal solver output in {path}:{line_number}: {stripped}")
            if _has_negative_volume(stripped):
                raise RuntimeError(
                    f"{case.name}: negative-volume solver output in {path}:{line_number}: {stripped}"
                )
            match = _TIME_LINE_RE.fullmatch(stripped)
            if match is not None:
                value = float(match.group(1))
                if not math.isfinite(value):
                    raise RuntimeError(f"{case.name}: non-finite solver time in {path}:{line_number}")
                last_time_s = value

    if last_nonempty not in {"End", "Finalising parallel run"}:
        raise RuntimeError(f"{case.name}: solver log does not end normally with 'End': {path}")
    if last_time_s is None:
        raise RuntimeError(f"{case.name}: solver log contains no 'Time = ...' entry: {path}")
    return last_time_s


def _scan_v2512_vector_file(case: Path, path: Path) -> float:
    latest: float | None = None
    v2512_header = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                lowered = stripped.lower()
                if all(field in lowered for field in ("time", "total_x", "total_y", "total_z")):
                    v2512_header = True
                continue
            data = stripped.split("#", 1)[0].split("//", 1)[0].rstrip()
            if _NONFINITE_RE.search(data):
                raise RuntimeError(f"{case.name}: non-finite force output in {path}:{line_number}")
            time_match = _NUMBER_RE.match(data)
            if time_match is None or time_match.end() == len(data) or not data[time_match.end()].isspace():
                raise RuntimeError(f"{case.name}: malformed v2512 force output in {path}:{line_number}")
            time_s = float(time_match.group(0))
            columns = data[time_match.end() :]
            number_tokens = _NUMBER_RE.findall(columns)
            residue = _NUMBER_RE.sub("", columns)
            values = [float(value) for value in number_tokens]
            if (
                not math.isfinite(time_s)
                or len(values) not in (9, 12)
                or residue.strip()
                or not all(math.isfinite(value) for value in values)
            ):
                raise RuntimeError(f"{case.name}: malformed v2512 force output in {path}:{line_number}")
            latest = time_s if latest is None else max(latest, time_s)
    if latest is None:
        raise RuntimeError(f"{case.name}: no data rows in v2512 force output {path}")
    if not v2512_header:
        raise RuntimeError(f"{case.name}: missing v2512 total-vector header in force output {path}")
    return latest


def _validate_case_outputs(
    case: Path,
    solver: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the durable evidence required before a case is complete."""

    motion = metadata if metadata is not None else _read_motion(case)
    try:
        end_time_s = float(motion["end_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{case.name}: invalid motion.json end_time_s") from exc
    if not math.isfinite(end_time_s) or end_time_s < 0.0:
        raise RuntimeError(f"{case.name}: invalid motion.json end_time_s")

    solver_end_time_s = _scan_solver_log(case, solver)
    _require_time_coverage(case, "solver log", solver_end_time_s, end_time_s)

    root = case / "postProcessing" / "forces"
    force_candidates = sorted(root.glob("**/force*.dat")) if root.is_dir() else []
    moment_candidates = sorted(root.glob("**/moment*.dat")) if root.is_dir() else []

    def keyed_segments(
        paths: list[Path], pattern: re.Pattern[str]
    ) -> dict[tuple[Path, int], Path]:
        segments: dict[tuple[Path, int], Path] = {}
        for path in paths:
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            suffix_priority = -1 if match.group(1) is None else int(match.group(1))
            segments[(path.parent, suffix_priority)] = path
        return segments

    force_segments = keyed_segments(force_candidates, _FORCE_SPLIT_RE)
    moment_segments = keyed_segments(moment_candidates, _MOMENT_SPLIT_RE)
    force_files = sorted(force_segments.values())
    moment_files = sorted(moment_segments.values())
    if not force_files or not moment_files:
        raise RuntimeError(
            f"{case.name}: missing OpenCFD v2512 postProcessing/forces/**/"
            "{force*.dat,moment*.dat} output"
        )

    unpaired = sorted(set(force_segments) ^ set(moment_segments))
    if unpaired:
        relative = ", ".join(
            str((force_segments.get(key) or moment_segments[key]).relative_to(case))
            for key in unpaired
        )
        raise RuntimeError(f"{case.name}: unpaired force/moment restart segment: {relative}")

    pair_end_times: list[tuple[Path, float, float]] = []
    ordered_segments = sorted(
        force_segments,
        key=lambda key: (_numeric_path_key(key[0].relative_to(root)), key[1]),
    )
    for key in ordered_segments:
        force_end = _scan_v2512_vector_file(case, force_segments[key])
        moment_end = _scan_v2512_vector_file(case, moment_segments[key])
        pair_end_times.append((force_segments[key], force_end, moment_end))

    # The newest restart segment is authoritative.  An older complete file
    # must not hide a later truncated `_N` segment.
    force_end_time_s = pair_end_times[-1][1]
    moment_end_time_s = pair_end_times[-1][2]
    for force_path, force_end, moment_end in pair_end_times:
        alignment_tolerance = 1.0e-9 * max(1.0, abs(force_end), abs(moment_end))
        if abs(force_end - moment_end) > alignment_tolerance:
            raise RuntimeError(
                f"{case.name}: terminal force.dat/moment.dat times do not align in "
                f"{force_path.parent.relative_to(case)} ({force_path.name}): "
                f"{force_end:.12g} vs {moment_end:.12g} s"
            )
    _require_time_coverage(case, "force.dat", force_end_time_s, end_time_s)
    _require_time_coverage(case, "moment.dat", moment_end_time_s, end_time_s)

    return {
        "end_time_s": end_time_s,
        "solver_end_time_s": solver_end_time_s,
        "force_end_time_s": force_end_time_s,
        "moment_end_time_s": moment_end_time_s,
        "force_files": [str(path.relative_to(case)) for path in force_files],
        "moment_files": [str(path.relative_to(case)) for path in moment_files],
    }


def _validated_completion(case: Path, solver: str) -> tuple[bool, str]:
    marker_path = case / ".completed"
    if not marker_path.is_file():
        return False, "missing .completed marker"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable .completed marker: {exc}"
    if not isinstance(marker, dict):
        return False, ".completed marker is not a JSON object"

    metadata = _read_motion(case)
    required = {
        "schema_version": _MARKER_SCHEMA_VERSION,
        "status": "completed",
        "case": case.name,
        "solver": solver,
        "foam_api": _FOAM_API,
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            return False, f".completed {key}={marker.get(key)!r}, expected {expected!r}"
    if marker.get("motion") != metadata:
        return False, ".completed motion metadata no longer matches motion.json"
    if not isinstance(marker.get("validation"), dict):
        return False, ".completed marker has no validation evidence"
    try:
        elapsed_s = float(marker["elapsed_s"])
    except (KeyError, TypeError, ValueError):
        return False, ".completed elapsed_s is invalid"
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        return False, ".completed elapsed_s is invalid"

    try:
        current_validation = _validate_case_outputs(case, solver, metadata)
    except (OSError, RuntimeError, ValueError) as exc:
        return False, str(exc)
    recorded_validation = marker["validation"]
    for key in ("end_time_s", "solver_end_time_s", "force_end_time_s", "moment_end_time_s"):
        try:
            recorded = float(recorded_validation[key])
            current = float(current_validation[key])
        except (KeyError, TypeError, ValueError):
            return False, f".completed validation.{key} is invalid"
        tolerance = 1.0e-12 * max(1.0, abs(recorded), abs(current))
        if not math.isfinite(recorded) or abs(recorded - current) > tolerance:
            return False, f".completed validation.{key} no longer matches current output"
    for key in ("force_files", "moment_files"):
        if recorded_validation.get(key) != current_validation[key]:
            return False, f".completed validation.{key} no longer matches current output"
    return True, "completion marker and outputs validated"


def _discover(cases_dir: Path, patterns: list[str], resume: bool, solver: str = "pimpleFoam") -> list[Path]:
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory does not exist: {cases_dir}")
    cases = []
    skipped = 0
    for metadata in sorted(cases_dir.glob("*/motion.json")):
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

    plan: list[tuple[list[str], Path]] = []
    if ranks > 1:
        plan.append(
            (
                [
                    "foamDictionary",
                    str(case / "system" / "decomposeParDict"),
                    "-entry",
                    "numberOfSubdomains",
                    "-set",
                    str(ranks),
                ],
                case / "log.foamDictionary",
            )
        )
        plan.append((["decomposePar", "-force", "-case", str(case)], case / "log.decomposePar"))
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


def main() -> int:
    args = _parser().parse_args()
    if args.np < 1 or args.jobs < 1:
        raise SystemExit("--np and --jobs must be positive")
    if args.bind_to_core and args.np == 1:
        raise SystemExit("--bind-to-core requires --np greater than 1")
    if args.cpu_sets:
        if args.np == 1:
            raise SystemExit("--cpu-set requires --np greater than 1")
        if not args.bind_to_core:
            raise SystemExit("--cpu-set requires --bind-to-core")
        try:
            _validate_cpu_sets(args.cpu_sets, args.jobs)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    required = [args.solver]
    if args.np > 1:
        required.extend(("foamDictionary", "decomposePar", "mpirun"))
        if args.cpu_sets:
            required.append("taskset")
        if args.reconstruct:
            required.append("reconstructPar")
    if not args.dry_run:
        api = os.environ.get("FOAM_API", "")
        if api != _FOAM_API:
            raise SystemExit(
                f"FOAM_API={api or 'unset'}; this workflow requires OpenCFD API {_FOAM_API}; "
                "source environment/openfoam/env.sh first"
            )
        missing = [command for command in required if shutil.which(command) is None]
        if missing:
            raise SystemExit(f"Missing commands: {', '.join(missing)}; source environment/openfoam/env.sh first")

    cases = _discover(args.cases_dir.resolve(), args.only, args.resume, args.solver)
    if not cases:
        print("[resume] all matched cases already have validated completion evidence")
        return 0
    available = os.cpu_count() or 1
    requested = args.np * args.jobs
    if requested > available:
        print(f"warning: requested {requested} ranks across jobs, but only {available} CPUs are visible", file=sys.stderr)
    if args.bind_to_core and args.jobs > 1 and not args.cpu_sets:
        print(
            "warning: each concurrent mpirun maps independently; use disjoint external CPU sets "
            "to prevent jobs from binding to the same cores",
            file=sys.stderr,
        )

    failures: list[str] = []
    cpu_slots: queue.Queue[str] | None = None
    if args.cpu_sets:
        cpu_slots = queue.Queue()
        for cpu_set in args.cpu_sets:
            cpu_slots.put(cpu_set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        if cpu_slots is None:
            futures = {
                pool.submit(
                    _run_one,
                    case,
                    args.solver,
                    args.np,
                    args.reconstruct,
                    args.dry_run,
                    args.bind_to_core,
                ): case
                for case in cases
            }
        else:
            futures = {
                pool.submit(
                    _run_one_with_cpu_slot,
                    cpu_slots,
                    case,
                    args.solver,
                    args.np,
                    args.reconstruct,
                    args.dry_run,
                    args.bind_to_core,
                ): case
                for case in cases
            }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # keep independent cases running
                failures.append(f"{futures[future].name}: {exc}")
                print(f"[fail]  {failures[-1]}", file=sys.stderr, flush=True)

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

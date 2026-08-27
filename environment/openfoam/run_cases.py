#!/usr/bin/env python3
"""Run generated forced-oscillation cases with bounded local concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.openfoam.case_execution.planning import (
    _discover,
    _validate_cpu_sets,
)
from environment.openfoam.case_execution.runner import _run_one, _run_one_with_cpu_slot
from environment.openfoam.case_execution.validation import _FOAM_API
from environment.openfoam.build_mesh import validate_mesh_completion
from environment.openfoam.case_generation.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=Path(__file__).with_name("cases"))
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Schema-5 preliminary campaign configuration.",
    )
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
        help="Skip cases that have a matching .completed marker.",
    )
    parser.add_argument("--reconstruct", action="store_true", help="Run reconstructPar after parallel solve.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


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
    cases_root = args.cases_dir.resolve()
    config_path = args.config.resolve()
    load_config(config_path)
    mesh_valid, mesh_reason = validate_mesh_completion(
        cases_root,
    )
    if not mesh_valid:
        raise RuntimeError(f"shared mesh is unusable: {mesh_reason}")
    cases = _discover(cases_root, args.only, args.resume, args.solver)
    if not cases:
        print("[resume] all matched cases are completed")
        return 0
    required = [args.solver]
    if any(
        json.loads((case / "case.json").read_text(encoding="utf-8")).get("case_family")
        == "steady_damping"
        for case in cases
    ):
        required.append("potentialFoam")
    if args.np > 1:
        required.extend(("decomposePar", "mpirun"))
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

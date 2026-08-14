"""Aggregate held-out pool replay validation reports into one evidence gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.envs.auv.validation.replay import aggregate_replay_validation_reports  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate independent held-out pool replay reports.")
    parser.add_argument("reports", type=Path, nargs="+", help="JSON reports from validate_pool_replay.py.")
    parser.add_argument("--output", type=Path, required=True, help="Campaign summary JSON path.")
    parser.add_argument("--min-held-out-cases", type=int, default=3)
    parser.add_argument(
        "--allow-missing-action-gate",
        action="store_true",
        help="Permit reports that do not prove measured and simulated actions match.",
    )
    return parser


def summarize_report_files(
    report_paths: Sequence[Path],
    min_held_out_cases: int = 3,
    require_action_gate: bool = True,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for path in report_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Replay report does not exist: {path}")
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        if not isinstance(report, dict):
            raise ValueError(f"Replay report must contain a JSON object: {path}")
        reports.append(report)
    summary = aggregate_replay_validation_reports(
        reports,
        min_held_out_cases=min_held_out_cases,
        require_action_gate=require_action_gate,
    )
    summary["source_reports"] = [str(path) for path in report_paths]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = summarize_report_files(
        args.reports,
        args.min_held_out_cases,
        require_action_gate=not args.allow_missing_action_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

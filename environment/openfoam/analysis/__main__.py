"""Command-line entry point for ``python -m openfoam.analysis``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .fit import analyze_cases, write_fit_outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit full 6x6 AUV added-mass and damping matrices from OpenFOAM cases.",
    )
    parser.add_argument("case_dirs", nargs="*", type=Path, help="Generated case directories containing motion.json")
    parser.add_argument(
        "--cases-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively discover motion.json files below this directory (repeatable)",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON analysis/experiment configuration")
    parser.add_argument("--output-dir", type=Path, default=Path("environment/openfoam/results"))
    parser.add_argument("--bootstrap-samples", type=int, help="Override bootstrap cycle resample count")
    parser.add_argument("--passivity-samples", type=int, help="Override random passivity sample count")
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument("--project-added-mass-psd", action="store_true", dest="project_psd")
    projection.add_argument("--raw-added-mass", action="store_false", dest="project_psd")
    parser.set_defaults(project_psd=None)
    return parser


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_dirs = list(args.case_dirs)
    for root in args.cases_root:
        case_dirs.extend(sorted(path.parent for path in root.rglob("motion.json")))
    # Preserve order but avoid fitting a case twice when roots overlap.
    unique_dirs = list(dict.fromkeys(path.resolve() for path in case_dirs))
    if not unique_dirs:
        raise SystemExit("No cases supplied; pass case directories or --cases-root")

    config = _load_config(args.config)
    section = dict(config.get("analysis", config))
    if args.bootstrap_samples is not None:
        section["bootstrap_samples"] = args.bootstrap_samples
    if args.passivity_samples is not None:
        section["passivity_samples"] = args.passivity_samples
    if args.project_psd is not None:
        section["project_added_mass_psd"] = args.project_psd
    result = analyze_cases(unique_dirs, config={"analysis": section})
    paths = write_fit_outputs(result, args.output_dir)
    print(json.dumps({"case_count": len(result.case_summaries), "outputs": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

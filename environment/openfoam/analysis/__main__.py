"""Command-line entry point for ``python -m openfoam.analysis``."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from .fit import analyze_cases
from environment.openfoam.case_generation.config import load_config

from .output import write_fit_outputs


def _discover_analysis_cases(roots: list[Path]) -> list[Path]:
    """Discover schema-5 identification cases."""

    discovered: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("case.json")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(metadata, dict)
                and metadata.get("schema_version") == 5
                and metadata.get("case_family") != "shared_mesh"
            ):
                discovered.append(path.parent)
    return discovered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit full-response 6x6 AUV matrices from 24 OpenFOAM cases.",
    )
    parser.add_argument("case_dirs", nargs="*", type=Path, help="Generated case directories containing case.json")
    parser.add_argument(
        "--cases-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively discover case.json files below this directory (repeatable)",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON analysis/experiment configuration")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("environment/openfoam/results"),
        help="Create a timestamped result directory below this root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_dirs = list(args.case_dirs)
    case_dirs.extend(_discover_analysis_cases(args.cases_root))
    # Preserve order but avoid fitting a case twice when roots overlap.
    unique_dirs = list(dict.fromkeys(path.resolve() for path in case_dirs))
    if not unique_dirs:
        raise SystemExit("No cases supplied; pass case directories or --cases-root")

    config_path = args.config or Path(__file__).resolve().parents[1] / "config.json"
    config = load_config(config_path.resolve())
    result = analyze_cases(unique_dirs, config=config)
    destination = args.output_root / datetime.now().strftime("fit_%Y%m%d_%H%M%S_%f")
    paths = write_fit_outputs(result, destination)
    print(
        json.dumps(
            {
                "case_count": len(result.case_summaries),
                "outputs": paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

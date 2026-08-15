#!/usr/bin/env python3
"""Identify frequency-resolved diagonal 6-DOF FLU models from PMM data.

The implementation is split into configuration/preflight, kinematics, trial
reconstruction, robust fitting, and reporting modules. This file preserves the
original public import surface and command-line entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environment.pmm.pmm_common import *
from environment.pmm.pmm_kinematics import (
    FourierFit, fit_fourier, motion_origin_to_com_kinematics, r2_score,
    rigid_body_wrench_at_com, rotate_twist_H_to_B, rotate_wrench_H_to_B,
    rotation_x, skew, translate_wrench_origin_to_com, validate_rotation,
)
from environment.pmm.pmm_config import (
    AuditRow, PreflightResult, SixDofConfig, TrialPlan, _count_rows_and_width,
    _finite, _positive, _require_mapping, expected_trials, load_config, preflight,
)
from environment.pmm.pmm_trials import (
    SixDofTrial, _block_average, _body_to_H_rotation, _fourier_sse, build_trial,
    estimate_trial_frequency, project_trial_harmonics, read_raw_pair, residualize_trial,
)
from environment.pmm.pmm_fitting import (
    _frequency_key, _stack, _timing_fit_rows, build_frequency_resolved_matrices,
    build_six_dof_diagonal, coefficient_metadata, coefficient_rows, direct_term,
    direct_unit, fit_dof, mount_sign_invariant_coefficients, robust_fit,
)
from environment.pmm.pmm_reporting import (
    _atomic_write_text, _make_plot, _metadata, _report, _write_csv, diagnostic_rows,
)


def fit_by_frequency(
    trials_by_dof: Mapping[str, Sequence[SixDofTrial]],
    config: SixDofConfig,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Compatibility wrapper retaining module-level ``fit_dof`` substitution."""

    results: dict[tuple[str, float], dict[str, Any]] = {}
    minimum = int(config.quality["minimum_repeats_per_frequency"])
    for dof in DOFS:
        grouped: dict[float, list[SixDofTrial]] = {}
        for trial in trials_by_dof[dof]:
            grouped.setdefault(_frequency_key(trial.plan.nominal_frequency_hz), []).append(trial)
        for nominal_frequency_hz, trials in sorted(grouped.items()):
            if len(trials) < minimum:
                raise ValueError(
                    f"{dof} nominal frequency {nominal_frequency_hz:g} Hz has "
                    f"{len(trials)} trials; requires {minimum}"
                )
            result = fit_dof(trials, config)
            estimated = np.asarray([trial.frequency_hz for trial in trials], dtype=float)
            result.update(
                {
                    "dof": dof,
                    "trials": trials,
                    "nominal_frequency_hz": nominal_frequency_hz,
                    "mean_estimated_frequency_hz": float(np.mean(estimated)),
                    "min_estimated_frequency_hz": float(np.min(estimated)),
                    "max_estimated_frequency_hz": float(np.max(estimated)),
                    "estimated_frequency_std_hz": float(np.std(estimated)),
                    "included_repeats": len(trials),
                }
            )
            results[(dof, nominal_frequency_hz)] = result
    return results


def run_analysis(
    root: Path,
    config: SixDofConfig,
    output: Path,
    *,
    write_timing: bool = True,
    write_plot: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    checked = preflight(root, config)
    trials_by_dof: dict[str, list[SixDofTrial]] = {dof: [] for dof in DOFS}
    built: dict[tuple[str, int, int], SixDofTrial] = {}
    for plan in checked.included:
        trial = build_trial(plan, config)
        trials_by_dof[plan.dof].append(trial)
        built[(plan.dof, plan.repeat, plan.file_id)] = trial
    results = fit_by_frequency(trials_by_dof, config)
    rows = coefficient_rows(results, config)
    diagnostics = diagnostic_rows(checked, built, config)
    metadata = _metadata(config, checked, trials_by_dof, results)
    matrices = build_frequency_resolved_matrices(rows, metadata, config)
    # All expensive/error-prone core work above completes before the first file
    # is opened. Individual files are then atomically replaced.
    output.mkdir(parents=True, exist_ok=True)
    coefficient_path = output / "hydrodynamic_coefficients.csv"
    diagnostic_path = output / "trial_diagnostics.csv"
    metadata_path = output / "identification_metadata.json"
    matrices_path = output / "fossen_6x6_by_frequency.json"
    report_path = output / "REPORT.md"
    _write_csv(coefficient_path, COEFFICIENT_FIELDS, rows)
    _write_csv(diagnostic_path, DIAGNOSTIC_FIELDS, diagnostics)
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(matrices_path, json.dumps(matrices, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(report_path, _report(rows, diagnostics, metadata))
    artifacts: dict[str, Path] = {
        "coefficients": coefficient_path,
        "diagnostics": diagnostic_path,
        "metadata": metadata_path,
        "fossen_by_frequency": matrices_path,
        "report": report_path,
    }
    if write_timing:
        timing_path = output / "timing_sensitivity.csv"
        _write_csv(timing_path, TIMING_FIELDS, _timing_fit_rows(checked.included, config))
        artifacts["timing_sensitivity"] = timing_path
    if write_plot:
        plot_path = output / "fit_diagnostics.png"
        panels = [
            (f"{dof} {frequency:g} Hz", result)
            for (dof, frequency), result in sorted(
                results.items(), key=lambda item: (DOFS.index(item[0][0]), item[0][1])
            )
        ]
        try:
            _make_plot(plot_path, panels)
        except (ImportError, AttributeError) as error:
            print(
                f"Warning: optional fit plot was skipped because Matplotlib could not load: {error}",
                file=sys.stderr,
            )
        else:
            artifacts["plot"] = plot_path
    return artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=script_root)
    parser.add_argument("--config", type=Path, default=script_root / "six_dof_config.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_root / "hydro_results" / "six_dof_diagonal" / "model_scale",
    )
    parser.add_argument("--skip-timing-sensitivity", action="store_true")
    parser.add_argument(
        "--write-plot",
        action="store_true",
        help="write the optional fit plot (requires a Matplotlib build compatible with NumPy)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    artifacts = run_analysis(
        args.root,
        config,
        args.output,
        write_timing=not args.skip_timing_sensitivity,
        write_plot=args.write_plot,
        overwrite=args.overwrite,
    )
    print(
        "Completed model-scale, per-frequency 6x6 diagonal identification; "
        f"wrote {len(artifacts)} artifacts."
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

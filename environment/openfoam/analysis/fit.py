"""Orchestrate hydrodynamic matrix fitting from loaded OpenFOAM cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cycles import OddProjectedCase, odd_project
from .diagnostics import (
    _cycle_convergence_diagnostic,
    _full_model_diagnostics,
    _passivity_diagnostics,
    _raw_intercept_diagnostic,
)
from .matrix_fit import bootstrap, fit_odd_groups
from .motion import CaseData, load_case_data
from .output import load_analysis_config, write_fit_outputs
from .regression import project_symmetric_psd
from .types import FitOptions, HydroFitResult


def _apply_matrix_structure(
    added_mass: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    options: FitOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if options.diagonal_only:
        mask = np.eye(6, dtype=bool)
        structure = {
            "name": "diagonal",
            "assumption": "user-selected model reduction",
            "allowed_mask": mask.tolist(),
        }
    elif options.port_starboard_symmetry:
        # Body-FLU reflection parity for [u,v,w,p,q,r] and [X,Y,Z,K,M,N].
        parity = np.asarray((1, -1, 1, -1, 1, -1), dtype=int)
        mask = parity[:, None] == parity[None, :]
        structure = {
            "name": "port_starboard_reflection_symmetric",
            "frame": "body FLU",
            "generalized_parity": parity.tolist(),
            "even_block": ["u", "w", "q"],
            "odd_block": ["v", "p", "r"],
            "allowed_mask": mask.tolist(),
        }
    else:
        mask = np.ones((6, 6), dtype=bool)
        structure = {"name": "full", "allowed_mask": mask.tolist()}
    return (
        np.where(mask, added_mass, 0.0),
        np.where(mask, linear, 0.0),
        np.where(mask, quadratic, 0.0),
        mask,
        structure,
    )


def _project_added_mass(
    added_mass: np.ndarray,
    mask: np.ndarray,
    options: FitOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    if options.project_added_mass_psd:
        projected, diagnostics = project_symmetric_psd(
            added_mass,
            options.min_added_mass_eigenvalue,
        )
    else:
        projected = added_mass.copy()
        diagnostics = {
            "enabled": False,
            "raw_asymmetry_frobenius": float(np.linalg.norm(added_mass - added_mass.T)),
        }
    # Projection acts on the whole matrix and can repopulate forbidden entries.
    return np.where(mask, projected, 0.0), diagnostics


def _fit_diagnostics(
    cases: Sequence[CaseData],
    odd_cases: Sequence[OddProjectedCase],
    added_mass: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    options: FitOptions,
    matrix_structure: Mapping[str, Any],
    fit_by_dof: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "matrix_structure": dict(matrix_structure),
        "fit_by_excited_dof": dict(fit_by_dof),
        "cycle_convergence_by_case": [
            _cycle_convergence_diagnostic(item) for item in odd_cases
        ],
        "raw_intercept_fits": [_raw_intercept_diagnostic(case) for case in cases],
        "full_model_case_fits": _full_model_diagnostics(
            cases,
            added_mass,
            linear,
            quadratic,
        ),
        "added_mass_projection": dict(projection),
        "passivity": _passivity_diagnostics(cases, linear, quadratic, options),
    }


def _case_summaries(cases: Sequence[CaseData]) -> list[dict[str, Any]]:
    return [
        {
            "case_name": case.motion.case_name,
            "case_dir": case.case_dir,
            "dof": case.motion.dof,
            "amplitude_si": case.motion.amplitude_si,
            "omega_rad_s": case.motion.omega_rad_s,
            "settle_cycles": case.motion.settle_cycles,
            "sample_cycles": case.motion.sample_cycles,
            "force_files": list(case.force_series.source_files),
        }
        for case in cases
    ]


def fit_case_data(cases: Sequence[CaseData], options: FitOptions | None = None) -> HydroFitResult:
    """Fit all 36 entries of each requested matrix from loaded cases."""

    if not cases:
        raise ValueError("At least one case is required")
    fit_options = options or FitOptions()
    odd_cases = [
        odd_project(
            case,
            fit_options.phase_samples_per_cycle,
            include_rotation_attitude_term=(
                fit_options.include_rotation_attitude_term
                and (case.motion.dof != "p" or fit_options.include_roll_attitude_term)
            ),
        )
        for case in cases
    ]
    groups: dict[int, list[OddProjectedCase]] = {}
    for item in odd_cases:
        groups.setdefault(item.dof_index, []).append(item)
    added_raw, linear, quadratic, fit_diagnostics = fit_odd_groups(
        groups,
        fit_options.minimum_samples_per_dof,
    )
    added_raw, linear, quadratic, mask, matrix_structure = _apply_matrix_structure(
        added_raw,
        linear,
        quadratic,
        fit_options,
    )
    added_mass, projection = _project_added_mass(added_raw, mask, fit_options)
    diagnostics = _fit_diagnostics(
        cases,
        odd_cases,
        added_mass,
        linear,
        quadratic,
        fit_options,
        matrix_structure,
        fit_diagnostics,
        projection,
    )
    return HydroFitResult(
        added_raw,
        added_mass,
        linear,
        quadratic,
        diagnostics,
        bootstrap(groups, fit_options),
        _case_summaries(cases),
        fit_options,
    )



def analyze_cases(
    case_dirs: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
) -> HydroFitResult:
    """Load generated cases, fit matrices and optionally write result files."""

    config_data = load_analysis_config(config)
    analysis_config = config_data.get("analysis", config_data)
    options = FitOptions.from_mapping(analysis_config)
    overrides = {
        key: analysis_config[key]
        for key in ("settle_cycles", "sample_cycles")
        if key in analysis_config
    }
    cases: list[CaseData] = []
    for path in case_dirs:
        root = Path(path)
        with (root / "motion.json").open("r", encoding="utf-8") as stream:
            motion_metadata = json.load(stream)
        kind = str(motion_metadata.get("motion_kind", motion_metadata.get("kind", ""))).lower()
        dof = motion_metadata.get("dof")
        if (
            kind in {"baseline", "rest", "static"}
            or dof is None
            or motion_metadata.get("include_in_fit") is False
        ):
            continue
        cases.append(load_case_data(root, config_overrides=overrides))
    if not cases:
        raise ValueError("No oscillatory single-DOF cases were supplied (baseline cases are skipped)")
    result = fit_case_data(cases, options)
    if output_dir is not None:
        write_fit_outputs(result, output_dir)
    return result

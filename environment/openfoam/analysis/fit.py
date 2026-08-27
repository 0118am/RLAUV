"""Load the 24 completed cases and fit three full-response 6x6 matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from environment.openfoam.case_generation.config import campaign_specs, load_config

from .cycles import odd_project
from .identification import (
    fit_low_amplitude_added_mass,
    fit_rotational_coefficients,
    fit_translational_damping,
    summarize_steady_cases,
)
from .motion import CaseData, SteadyCaseData, load_case_data, load_steady_case_data
from .output import write_fit_outputs
from .types import FitOptions, HydroFitResult

ANALYSIS_ROOT = Path(__file__).resolve().parent

_GENERALIZED_REFLECTION_PARITY = np.asarray((1, -1, 1, -1, 1, -1))
_PORT_STARBOARD_MASK = (
    _GENERALIZED_REFLECTION_PARITY[:, None]
    == _GENERALIZED_REFLECTION_PARITY[None, :]
)


def _case_summaries(
    steady_cases: Sequence[SteadyCaseData],
    oscillatory_cases: Sequence[CaseData],
) -> list[dict[str, Any]]:
    records = [
        {
            "case_name": case.case_name,
            "case_dir": case.case_dir,
            "case_family": case.case_family,
            "dof": case.dof,
            "body_velocity_b_m_s": case.body_velocity_b_m_s.tolist(),
            "settle_end_s": case.settle_end_s,
            "end_time_s": case.end_time_s,
            "force_files": list(case.force_series.source_files),
        }
        for case in steady_cases
    ]
    records.extend(
        {
            "case_name": case.motion.case_name,
            "case_dir": case.case_dir,
            "case_family": case.motion.case_family,
            "dof": case.motion.dof,
            "amplitude_si": case.motion.amplitude_si,
            "omega_rad_s": case.motion.omega_rad_s,
            "ramp_duration_s": case.motion.ramp_duration_s,
            "settle_cycles": case.motion.settle_cycles,
            "sample_cycles": case.motion.sample_cycles,
            "force_files": list(case.force_series.source_files),
        }
        for case in oscillatory_cases
    )
    return sorted(records, key=lambda item: item["case_name"])


def fit_case_data(
    steady_cases: Sequence[SteadyCaseData],
    oscillatory_cases: Sequence[CaseData],
    config: Mapping[str, Any],
    *,
    options: FitOptions | None = None,
) -> HydroFitResult:
    fit_options = options or FitOptions.from_mapping(config["analysis"])
    if len(steady_cases) != 12:
        raise ValueError(f"Full-response fit requires 12 steady cases, got {len(steady_cases)}")
    added_cases = [
        case for case in oscillatory_cases if case.motion.case_family == "added_mass"
    ]
    rotation_cases = [
        case
        for case in oscillatory_cases
        if case.motion.case_family == "oscillatory_damping"
    ]
    if len(added_cases) != 6 or len(rotation_cases) != 6 or len(oscillatory_cases) != 12:
        raise ValueError(
            "Full-response fit requires six added-mass and six rotational-damping cases"
        )
    added_odd = [
        odd_project(case, fit_options.phase_samples_per_cycle)
        for case in added_cases
    ]
    rotation_odd = [
        odd_project(case, fit_options.phase_samples_per_cycle)
        for case in rotation_cases
    ]
    reference_length_m = float(config["reference_length_m"])
    low_amplitude_added_mass, added_diagnostics = fit_low_amplitude_added_mass(
        added_odd, reference_length_m
    )
    steady_summaries = summarize_steady_cases(steady_cases)
    linear_translation, quadratic_translation, steady_diagnostics = (
        fit_translational_damping(steady_summaries, reference_length_m)
    )
    rotational_added_mass, linear_rotation, quadratic_rotation, rotation_diagnostics = (
        fit_rotational_coefficients(rotation_odd, reference_length_m)
    )
    added_mass_raw = low_amplitude_added_mass.copy()
    for excitation in range(3, 6):
        added_mass_raw[:, excitation] = rotational_added_mass[:, excitation]
    added_mass_raw = np.where(_PORT_STARBOARD_MASK, added_mass_raw, 0.0)
    # Potential-flow added mass obeys reciprocity. Independent column fits do
    # not satisfy it exactly, so publish their least-squares symmetric average.
    added_mass = 0.5 * (added_mass_raw + added_mass_raw.T)
    linear = np.where(
        _PORT_STARBOARD_MASK,
        linear_translation + linear_rotation,
        0.0,
    )
    quadratic = np.where(
        _PORT_STARBOARD_MASK,
        quadratic_translation + quadratic_rotation,
        0.0,
    )
    added_mass_eigenvalues = np.linalg.eigvalsh(added_mass)
    linear_symmetric_eigenvalues = np.linalg.eigvalsh(0.5 * (linear + linear.T))
    diagnostics = {
        "matrix_structure": {
            "name": "port_starboard_reflection_symmetric_full_response",
            "reason": (
                "each single-axis case contributes all six measured wrench responses; "
                "reflection-forbidden couplings are zero and allowed off-diagonal "
                "coefficients retain the CFD fit"
            ),
            "generalized_reflection_parity": _GENERALIZED_REFLECTION_PARITY.tolist(),
            "allowed_mask": _PORT_STARBOARD_MASK.tolist(),
            "cross_axis_load_normalization": (
                "moments are divided by reference_length_m before RMS ratios"
            ),
            "reference_length_m": reference_length_m,
        },
        "low_amplitude_added_mass": added_diagnostics,
        "steady_translational_damping": steady_diagnostics,
        "joint_oscillatory_rotational_coefficients": rotation_diagnostics,
        "added_mass_reciprocity": {
            "method": "symmetric_average_of_independent_full_response_columns",
            "raw_asymmetry_frobenius": float(
                np.linalg.norm(added_mass_raw - added_mass_raw.T)
            ),
            "published_eigenvalues": added_mass_eigenvalues.tolist(),
        },
        "passivity": {
            "linear_symmetric_eigenvalues": linear_symmetric_eigenvalues.tolist(),
            "linear_minimum_symmetric_eigenvalue": float(
                linear_symmetric_eigenvalues[0]
            ),
            "minimum_linear_diagonal": float(np.min(np.diag(linear))),
            "minimum_quadratic_diagonal": float(np.min(np.diag(quadratic))),
        },
    }
    return HydroFitResult(
        added_mass,
        linear,
        quadratic,
        diagnostics,
        _case_summaries(steady_cases, oscillatory_cases),
        fit_options,
    )


def _load_case_metadata(path: Path) -> dict[str, Any]:
    value = json.loads((path / "case.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 5:
        raise ValueError(f"{path / 'case.json'} must use schema_version 5")
    return value


def analyze_cases(
    case_dirs: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
) -> HydroFitResult:
    if config is None:
        config_data = load_config(ANALYSIS_ROOT.parent / "config.json")
    elif isinstance(config, Mapping):
        config_data = dict(config)
    else:
        config_data = load_config(Path(config).resolve())
    paths = sorted({Path(path).resolve() for path in case_dirs})
    expected_names = {spec.name for spec in campaign_specs(config_data)}
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        raise ValueError(
            "Campaign case set mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    steady: list[SteadyCaseData] = []
    oscillatory: list[CaseData] = []
    for path in paths:
        metadata = _load_case_metadata(path)
        if metadata["case_family"] == "steady_damping":
            case = load_steady_case_data(path)
            steady.append(case)
        elif metadata["case_family"] in {"added_mass", "oscillatory_damping"}:
            case = load_case_data(path)
            oscillatory.append(case)
        else:
            raise ValueError(f"{path.name}: unsupported case family")
    result = fit_case_data(
        steady,
        oscillatory,
        config_data,
    )
    if output_dir is not None:
        write_fit_outputs(result, output_dir)
    return result

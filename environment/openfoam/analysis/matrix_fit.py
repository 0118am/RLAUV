"""Per-DOF matrix regression and cycle bootstrap."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .cycles import OddProjectedCase
from .motion import DOF_NAMES
from .regression import scaled_lstsq
from .types import FitOptions

def fit_odd_groups(
    groups: Mapping[int, Sequence[OddProjectedCase]],
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    added_mass = np.zeros((6, 6), dtype=float)
    linear = np.zeros((6, 6), dtype=float)
    quadratic = np.zeros((6, 6), dtype=float)
    diagnostics: dict[str, Any] = {}
    for dof_index, dof_name in enumerate(DOF_NAMES):
        datasets = list(groups.get(dof_index, ()))
        if not datasets:
            raise ValueError(f"No single-DOF cases supplied for {dof_name}")
        design = np.concatenate([item.design for item in datasets], axis=0)
        target = np.concatenate([item.target for item in datasets], axis=0)
        attitude = None
        if all(item.attitude_feature is not None for item in datasets):
            attitude = np.concatenate(
                [np.asarray(item.attitude_feature, dtype=float) for item in datasets]
            )
            design_fit = np.column_stack((design, attitude))
        else:
            design_fit = design
        if design.shape[0] < minimum_samples:
            raise ValueError(
                f"DOF {dof_name} has {design.shape[0]} odd samples; at least {minimum_samples} required"
            )
        coefficients_fit, dof_diagnostics = scaled_lstsq(design_fit, target)
        required_rank = 4 if attitude is not None else 3
        if dof_diagnostics["rank"] < required_rank:
            raise ValueError(
                f"DOF {dof_name} regression rank is below {required_rank}; "
                "rotation attitude and added-mass terms require multiple frequencies"
            )
        coefficients = coefficients_fit[:3]
        added_mass[:, dof_index] = coefficients[0]
        linear[:, dof_index] = coefficients[1]
        quadratic[:, dof_index] = coefficients[2]
        residual = target - design_fit @ coefficients_fit
        even = np.concatenate([item.even_target for item in datasets], axis=0)
        dof_diagnostics.update(
            {
                "case_names": [item.case_name for item in datasets],
                "sample_count": int(design.shape[0]),
                "weighting": "equal uniform phase rows per complete cycle",
                "complete_cycles_by_case": {
                    item.case_name: int(np.unique(item.cycle_id).size)
                    for item in datasets
                },
                "odd_samples_by_case": {
                    item.case_name: int(item.design.shape[0]) for item in datasets
                },
                "odd_residual_rms_by_wrench": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                "even_component_rms_by_wrench": np.sqrt(np.mean(even**2, axis=0)).tolist(),
                "rotation_attitude_coefficient_by_wrench": (
                    None if attitude is None else coefficients_fit[3].tolist()
                ),
            }
        )
        diagnostics[dof_name] = dof_diagnostics
    return added_mass, linear, quadratic, diagnostics


def bootstrap(
    groups: Mapping[int, Sequence[OddProjectedCase]],
    options: FitOptions,
) -> dict[str, Any]:
    if options.bootstrap_samples <= 0:
        return {}
    rng = np.random.default_rng(options.bootstrap_seed)
    samples = np.empty((options.bootstrap_samples, 3, 6, 6), dtype=float)
    for sample_index in range(options.bootstrap_samples):
        resampled: dict[int, list[OddProjectedCase]] = {}
        for dof_index, datasets in groups.items():
            resampled[dof_index] = []
            for item in datasets:
                cycles = np.unique(item.cycle_id)
                selected_cycles = rng.choice(cycles, size=cycles.size, replace=True)
                row_indices = np.concatenate([np.flatnonzero(item.cycle_id == cycle) for cycle in selected_cycles])
                resampled[dof_index].append(
                    OddProjectedCase(
                        item.case_name,
                        item.dof_index,
                        item.design[row_indices],
                        item.target[row_indices],
                        item.even_target[row_indices],
                        item.cycle_id[row_indices],
                        (
                            None
                            if item.attitude_feature is None
                            else item.attitude_feature[row_indices]
                        ),
                    )
                )
        matrices = fit_odd_groups(resampled, 3)[:3]
        samples[sample_index] = np.stack(matrices)
    lower, median, upper = np.percentile(samples, (2.5, 50.0, 97.5), axis=0)
    result: dict[str, Any] = {
        "method": "stratified bootstrap by complete cycle within case",
        "samples": options.bootstrap_samples,
        "seed": options.bootstrap_seed,
    }
    for index, name in enumerate(("added_mass_raw", "linear_damping", "quadratic_damping")):
        result[name] = {
            "lower_2p5": lower[index].tolist(),
            "median": median[index].tolist(),
            "upper_97p5": upper[index].tolist(),
        }
    return result


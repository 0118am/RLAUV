#!/usr/bin/env python3
"""Compare one identical forced-oscillation case across convergence variants.

The coefficient extraction intentionally reuses the production analysis path:
OpenFOAM forces are transformed to the moving body frame, complete cycles are
odd-projected on a uniform phase grid, and the one excited matrix column is
fit to ``[-nudot, -nu, -abs(nu)*nu]``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from environment.openfoam.analysis.cycles import odd_project
from environment.openfoam.analysis.motion import CaseData, WRENCH_NAMES, load_case_data
from environment.openfoam.analysis.regression import scaled_lstsq
from environment.openfoam.convergence.report import render_markdown as _markdown, write_report
VARIANTS = ("coarse", "nominal", "fine", "dt", "domain")
GRID_VARIANTS = ("coarse", "nominal", "fine")
HERE = Path(__file__).resolve().parent
DEFAULT_GRID_CONFIGS = {
    "coarse": HERE / "configs" / "mesh_coarse.json",
    "nominal": HERE / "configs" / "mesh_nominal.json",
    "fine": HERE / "configs" / "mesh_fine.json",
}
DEFAULT_VARIANT_CONFIGS = {
    **DEFAULT_GRID_CONFIGS,
    "dt": HERE / "configs" / "dt800.json",
    "domain": HERE / "configs" / "domain_expanded.json",
}
PHASE_SAMPLES_PER_CYCLE = 256
def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _phase_delta_rad(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def _assert_close(name: str, first: float, second: float) -> None:
    if not math.isclose(first, second, rel_tol=1.0e-11, abs_tol=1.0e-12):
        raise ValueError(f"motion mismatch for {name}: {first:.17g} != {second:.17g}")


def _validate_identical_motion(
    cases: Mapping[str, CaseData], metadata: Mapping[str, Mapping[str, Any]]
) -> None:
    reference = cases["nominal"].motion
    reference_meta = metadata["nominal"]
    for variant in VARIANTS:
        motion = cases[variant].motion
        if motion.case_name != reference.case_name:
            raise ValueError(
                "case_name mismatch: "
                f"nominal={reference.case_name!r}, {variant}={motion.case_name!r}"
            )
        for name in ("dof", "dof_index", "motion_kind"):
            if getattr(motion, name) != getattr(reference, name):
                raise ValueError(
                    f"motion mismatch for {name}: nominal={getattr(reference, name)!r}, "
                    f"{variant}={getattr(motion, name)!r}"
                )
        if not np.allclose(motion.axis, reference.axis, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"motion mismatch for axis: nominal={reference.axis.tolist()}, "
                f"{variant}={motion.axis.tolist()}"
            )
        for name in (
            "amplitude_si",
            "omega_rad_s",
            "settle_cycles",
            "sample_cycles",
        ):
            first = getattr(reference, name)
            second = getattr(motion, name)
            if first is None or second is None:
                if first != second:
                    raise ValueError(
                        f"motion mismatch for {name}: nominal={first!r}, {variant}={second!r}"
                    )
            else:
                _assert_close(name, float(first), float(second))
        if abs(_phase_delta_rad(motion.phase_rad, reference.phase_rad)) > 1.0e-12:
            raise ValueError(
                f"motion mismatch for phase_rad: nominal={reference.phase_rad:.17g}, "
                f"{variant}={motion.phase_rad:.17g}"
            )
        for name in ("cofr_global_m", "com_initial_global_m"):
            if not np.allclose(
                getattr(motion, name), getattr(reference, name), rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"motion mismatch for {name} in {variant}")

        # These are not needed by MotionSpec, but changing them invalidates a
        # numerical-convergence comparison just as surely as changing motion.
        for name in ("rho_kg_m3", "nu_m2_s", "force_patch"):
            reference_value = reference_meta.get(name)
            value = metadata[variant].get(name)
            if reference_value is None and value is None:
                continue
            if isinstance(reference_value, (int, float)) and isinstance(value, (int, float)):
                _assert_close(name, float(reference_value), float(value))
            elif value != reference_value:
                raise ValueError(
                    f"metadata mismatch for {name}: nominal={reference_value!r}, "
                    f"{variant}={value!r}"
                )


def _require_complete_cycles(case: CaseData, expected_delta_t_s: float) -> int:
    motion = case.motion
    if motion.sample_cycles is None:
        raise ValueError(f"{motion.case_name}: sample_cycles is required for convergence")
    cycle_count = int(round(motion.sample_cycles))
    if cycle_count <= 0 or not math.isclose(
        motion.sample_cycles, cycle_count, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError(
            f"{motion.case_name}: sample_cycles must be a positive integer, "
            f"got {motion.sample_cycles!r}"
        )
    time = np.asarray(case.time_s, dtype=float)
    if time.size < 4 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{motion.case_name}: force time history must be finite and increasing")
    start = motion.settle_cycles * motion.period_s
    stop = start + cycle_count * motion.period_s
    tolerance = 1.0e-9 * max(1.0, motion.period_s, abs(stop))
    if time[0] > start + tolerance or time[-1] < stop - tolerance:
        raise ValueError(
            f"{motion.case_name}: incomplete analysis window; need [{start:.12g}, {stop:.12g}] s, "
            f"have [{time[0]:.12g}, {time[-1]:.12g}] s"
        )
    analysis_time = time[(time >= start - tolerance) & (time <= stop + tolerance)]
    if analysis_time.size < 4:
        raise ValueError(f"{motion.case_name}: too few samples in requested analysis window")
    # Include the requested window boundaries in this global check. Checking
    # one cycle at a time misses a restart hole straddling a cycle boundary.
    clipped = np.clip(analysis_time, start, stop)
    augmented = np.unique(np.concatenate(([start], clipped, [stop])))
    gaps = np.diff(augmented)
    if not math.isfinite(expected_delta_t_s) or expected_delta_t_s <= 0.0:
        raise ValueError(f"{motion.case_name}: delta_t_s must be finite and positive")
    configured_gap_limit = expected_delta_t_s * (1.0 + 1.0e-6) + tolerance
    maximum_gap = float(np.max(gaps))
    if maximum_gap > configured_gap_limit:
        raise ValueError(
            f"{motion.case_name}: raw time gap {maximum_gap:.9g}s exceeds configured "
            f"delta_t_s {expected_delta_t_s:.9g}s (limit {configured_gap_limit:.9g}s)"
        )
    positive_raw_gaps = np.diff(analysis_time)
    positive_raw_gaps = positive_raw_gaps[positive_raw_gaps > tolerance]
    if positive_raw_gaps.size == 0:
        raise ValueError(f"{motion.case_name}: insufficient time resolution")
    nominal_delta = float(np.median(positive_raw_gaps))
    phase_delta = motion.period_s / PHASE_SAMPLES_PER_CYCLE
    maximum_allowed_gap = max(2.0 * phase_delta, 4.0 * nominal_delta)
    if maximum_gap > maximum_allowed_gap * (1.0 + 1.0e-10):
        raise ValueError(
            f"{motion.case_name}: large raw time gap would be interpolated: "
            f"max {maximum_gap:.9g}s > limit {maximum_allowed_gap:.9g}s "
            f"(median {nominal_delta:.9g}s, phase {phase_delta:.9g}s)"
        )
    return cycle_count


def _uniform_main_load(case: CaseData, cycle_count: int) -> dict[str, float]:
    """Fundamental of measured main load, equally weighted by phase and cycle."""

    motion = case.motion
    samples = PHASE_SAMPLES_PER_CYCLE
    start = motion.settle_cycles * motion.period_s
    offsets = np.arange(samples, dtype=float) * motion.period_s / samples
    sample_time = np.concatenate(
        [start + cycle * motion.period_s + offsets for cycle in range(cycle_count)]
    )
    row = motion.dof_index
    measured = np.interp(sample_time, case.time_s, case.wrench_body[:, row])
    local_axis = motion.dof_index if motion.dof_index < 3 else motion.dof_index - 3
    direction_phase = 0.0 if motion.axis[local_axis] > 0.0 else math.pi
    phase = motion.omega_rad_s * sample_time + motion.phase_rad + direction_phase
    mean = float(np.mean(measured))
    sine = float(2.0 * np.mean(measured * np.sin(phase)))
    cosine = float(2.0 * np.mean(measured * np.cos(phase)))
    amplitude = math.hypot(sine, cosine)
    phase_deg = math.degrees(math.atan2(cosine, sine))
    prediction = mean + sine * np.sin(phase) + cosine * np.cos(phase)
    return {
        "mean": mean,
        "sine_coefficient": sine,
        "cosine_coefficient": cosine,
        "amplitude": amplitude,
        "phase_deg_relative_to_displacement_sine": phase_deg,
        "fundamental_reconstruction_residual_rms": float(
            np.sqrt(np.mean((measured - prediction) ** 2))
        ),
    }


def _analyze_variant(case: CaseData, expected_delta_t_s: float) -> dict[str, Any]:
    cycle_count = _require_complete_cycles(case, expected_delta_t_s)
    if not np.all(np.isfinite(case.wrench_body)):
        raise ValueError(f"{case.motion.case_name}: force/moment history contains non-finite values")
    # This also rejects a missing cycle or a restart gap that interpolation
    # would otherwise conceal.
    odd = odd_project(case, PHASE_SAMPLES_PER_CYCLE)
    coefficients, fit_diagnostics = scaled_lstsq(odd.design, odd.target)
    if fit_diagnostics["rank"] != 3:
        raise ValueError(f"{case.motion.case_name}: single-column regression rank is below 3")
    row = case.motion.dof_index
    main_coefficients = coefficients[:, row]
    prediction = odd.design @ coefficients
    residual = odd.target - prediction
    main_residual_rms = float(np.sqrt(np.mean(residual[:, row] ** 2)))
    peak_velocity = case.motion.amplitude_si * case.motion.omega_rad_s
    main_load = _uniform_main_load(case, cycle_count)
    relative_residual = (
        None
        if main_load["amplitude"] <= np.finfo(float).tiny
        else 100.0 * main_residual_rms / main_load["amplitude"]
    )
    return {
        "case_dir": str(Path(case.case_dir).resolve()),
        "force_files": list(case.force_series.source_files),
        "complete_cycles": cycle_count,
        "raw_sample_count": int(case.time_s.size),
        "odd_uniform_phase_sample_count": int(odd.design.shape[0]),
        "peak_speed_si": peak_velocity,
        "coefficients": {
            "added_mass": float(main_coefficients[0]),
            "linear_damping": float(main_coefficients[1]),
            "quadratic_damping": float(main_coefficients[2]),
            "effective_damping_at_peak_speed": float(
                main_coefficients[1] + main_coefficients[2] * peak_velocity
            ),
        },
        "main_load": main_load,
        "fit": {
            "rank": fit_diagnostics["rank"],
            "condition_number_scaled": fit_diagnostics["condition_number_scaled"],
            "odd_model_residual_rms": main_residual_rms,
            "odd_model_residual_percent_of_fundamental_amplitude": relative_residual,
        },
    }


def _mesh_characteristic_size(config_path: Path) -> float:
    config = _load_json(config_path)
    lower, upper, cells = _block_mesh_spec(config, str(config_path))
    cell_volume = float(np.prod(upper - lower) / np.prod(cells))
    return cell_volume ** (1.0 / 3.0)


def _block_mesh_spec(
    config: Mapping[str, Any], context: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = config.get("block_mesh")
    if not isinstance(block, Mapping):
        raise ValueError(f"{context}: missing block_mesh object")
    lower = np.asarray(block.get("domain_min"), dtype=float)
    upper = np.asarray(block.get("domain_max"), dtype=float)
    cells = np.asarray(block.get("base_cells"), dtype=float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or cells.shape != (3,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(cells))
        or np.any(upper <= lower)
        or np.any(cells <= 0.0)
    ):
        raise ValueError(f"{context}: invalid block_mesh geometry/cell counts")
    if not np.all(cells == np.floor(cells)):
        raise ValueError(f"{context}: block_mesh.base_cells must be integers")
    return lower, upper, cells.astype(int)


_FOAM_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
def _relative_difference(candidate: float, reference: float) -> dict[str, float | None]:
    difference = candidate - reference
    scale = abs(reference)
    return {
        "candidate": candidate,
        "reference": reference,
        "signed_difference": difference,
        "absolute_difference": abs(difference),
        "signed_relative_difference_percent": (
            None if scale <= np.finfo(float).tiny else 100.0 * difference / scale
        ),
        "absolute_relative_difference_percent": (
            None if scale <= np.finfo(float).tiny else 100.0 * abs(difference) / scale
        ),
    }


def _phase_difference(candidate: float, reference: float) -> dict[str, float]:
    signed = math.degrees(_phase_delta_rad(math.radians(candidate), math.radians(reference)))
    return {
        "candidate_deg": candidate,
        "reference_deg": reference,
        "signed_difference_deg": signed,
        "absolute_difference_deg": abs(signed),
    }


def _grid_gci(values: Mapping[str, float], sizes: Mapping[str, float]) -> dict[str, Any]:
    coarse, nominal, fine = (values[name] for name in GRID_VARIANTS)
    h_coarse, h_nominal, h_fine = (sizes[name] for name in GRID_VARIANTS)
    if not (h_coarse > h_nominal > h_fine > 0.0):
        raise ValueError(
            "grid characteristic sizes must satisfy coarse > nominal > fine; "
            f"got {h_coarse:.9g}, {h_nominal:.9g}, {h_fine:.9g}"
        )
    r_coarse_nominal = h_coarse / h_nominal
    r_nominal_fine = h_nominal / h_fine
    result: dict[str, Any] = {
        "method": "three-grid Richardson/Roache GCI, safety factor 1.25",
        "refinement_ratio_coarse_to_nominal": r_coarse_nominal,
        "refinement_ratio_nominal_to_fine": r_nominal_fine,
        "monotonic": False,
        "available": False,
    }
    epsilon_coarse_nominal = coarse - nominal
    epsilon_nominal_fine = nominal - fine
    scale = max(1.0, abs(coarse), abs(nominal), abs(fine))
    exact_tolerance = 1.0e-14 * scale
    if abs(epsilon_coarse_nominal) <= exact_tolerance and abs(epsilon_nominal_fine) <= exact_tolerance:
        result.update(
            {
                "monotonic": True,
                "available": True,
                "status": "grid-identical within numerical tolerance",
                "observed_order": None,
                "richardson_extrapolated": fine,
                "fine_grid_gci_absolute": 0.0,
                "fine_grid_gci_percent": 0.0,
            }
        )
        return result
    monotonic = epsilon_coarse_nominal * epsilon_nominal_fine > 0.0
    result["monotonic"] = monotonic
    if not monotonic:
        result["status"] = "GCI unavailable: sequence is not monotonic"
        return result
    if not math.isclose(r_coarse_nominal, r_nominal_fine, rel_tol=1.0e-6, abs_tol=1.0e-12):
        result["status"] = "GCI unavailable: unequal refinement ratios are not supported"
        return result
    error_ratio = abs(epsilon_coarse_nominal / epsilon_nominal_fine)
    observed_order = math.log(error_ratio) / math.log(r_nominal_fine)
    if not math.isfinite(observed_order) or observed_order <= 0.0:
        result["status"] = "GCI unavailable: monotonic sequence has no positive observed order"
        result["observed_order"] = observed_order
        return result
    denominator = r_nominal_fine**observed_order - 1.0
    correction = (fine - nominal) / denominator
    gci_absolute = 1.25 * abs(correction)
    result.update(
        {
            "available": True,
            "status": "available",
            "observed_order": observed_order,
            "richardson_extrapolated": fine + correction,
            "fine_grid_gci_absolute": gci_absolute,
            "fine_grid_gci_percent": (
                None
                if abs(fine) <= np.finfo(float).tiny
                else 100.0 * gci_absolute / abs(fine)
            ),
        }
    )
    return result


def _scalar_grid_comparison(
    values: Mapping[str, float], sizes: Mapping[str, float]
) -> dict[str, Any]:
    return {
        "values": {name: values[name] for name in GRID_VARIANTS},
        "coarse_vs_nominal": _relative_difference(values["coarse"], values["nominal"]),
        "nominal_vs_fine": _relative_difference(values["nominal"], values["fine"]),
        "coarse_vs_fine": _relative_difference(values["coarse"], values["fine"]),
        "gci": _grid_gci(values, sizes),
    }


def _phase_grid_comparison(
    values: Mapping[str, float], sizes: Mapping[str, float]
) -> dict[str, Any]:
    # Unwrap from fine -> nominal -> coarse before applying Richardson/GCI.
    fine = values["fine"]
    nominal = fine + _phase_difference(values["nominal"], fine)["signed_difference_deg"]
    coarse = nominal + _phase_difference(values["coarse"], nominal)["signed_difference_deg"]
    unwrapped = {"coarse": coarse, "nominal": nominal, "fine": fine}
    return {
        "values_deg": {name: values[name] for name in GRID_VARIANTS},
        "unwrapped_values_deg_for_gci": unwrapped,
        "coarse_vs_nominal": _phase_difference(values["coarse"], values["nominal"]),
        "nominal_vs_fine": _phase_difference(values["nominal"], values["fine"]),
        "coarse_vs_fine": _phase_difference(values["coarse"], values["fine"]),
        "gci_degrees": _grid_gci(unwrapped, sizes),
    }


def _metric_values(variants: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    extractors = {
        "added_mass": lambda item: item["coefficients"]["added_mass"],
        "effective_damping_at_peak_speed": lambda item: item["coefficients"][
            "effective_damping_at_peak_speed"
        ],
        "main_load_fundamental_amplitude": lambda item: item["main_load"]["amplitude"],
        "odd_model_residual_rms": lambda item: item["fit"]["odd_model_residual_rms"],
    }
    return {
        metric: {variant: float(extract(item)) for variant, item in variants.items()}
        for metric, extract in extractors.items()
    }


def compare_cases(
    case_dirs: Mapping[str, str | Path],
    *,
    variant_config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Analyze and compare one case across coarse/nominal/fine/dt/domain."""

    missing = sorted(set(VARIANTS) - set(case_dirs))
    unknown = sorted(set(case_dirs) - set(VARIANTS))
    if missing or unknown:
        raise ValueError(f"case variants mismatch; missing={missing}, unknown={unknown}")
    paths = {name: Path(case_dirs[name]).resolve() for name in VARIANTS}
    metadata = {name: _load_json(path / "motion.json") for name, path in paths.items()}
    supplied_configs = variant_config_paths or DEFAULT_VARIANT_CONFIGS
    missing_configs = sorted(set(VARIANTS) - set(supplied_configs))
    unknown_configs = sorted(set(supplied_configs) - set(VARIANTS))
    if missing_configs or unknown_configs:
        raise ValueError(
            f"config variants mismatch; missing={missing_configs}, unknown={unknown_configs}"
        )
    config_paths = {
        name: Path(supplied_configs[name]).resolve() for name in VARIANTS
    }
    if len(set(paths.values())) != len(VARIANTS):
        raise ValueError("coarse/nominal/fine/dt/domain must be five distinct case directories")
    cases = {name: load_case_data(path) for name, path in paths.items()}
    _validate_identical_motion(cases, metadata)
    variants = {
        name: _analyze_variant(cases[name], float(metadata[name]["delta_t_s"]))
        for name in VARIANTS
    }

    sizes = {name: _mesh_characteristic_size(config_paths[name]) for name in GRID_VARIANTS}
    metrics = _metric_values(variants)
    grid_metrics = {
        name: _scalar_grid_comparison(values, sizes) for name, values in metrics.items()
    }
    phase_values = {
        variant: float(item["main_load"]["phase_deg_relative_to_displacement_sine"])
        for variant, item in variants.items()
    }
    grid_metrics["main_load_phase"] = _phase_grid_comparison(phase_values, sizes)

    def two_variant(candidate: str, reference: str) -> dict[str, Any]:
        comparison = {
            metric: _relative_difference(values[candidate], values[reference])
            for metric, values in metrics.items()
        }
        comparison["main_load_phase"] = _phase_difference(
            phase_values[candidate], phase_values[reference]
        )
        return comparison

    time_step_comparison = two_variant("nominal", "dt")
    domain_comparison = two_variant("nominal", "domain")
    motion = cases["nominal"].motion
    return {
        "schema_version": 1,
        "case": {
            "case_name": motion.case_name,
            "dof": motion.dof,
            "dof_index": motion.dof_index,
            "main_wrench": WRENCH_NAMES[motion.dof_index],
            "motion_kind": motion.motion_kind,
            "amplitude_si": motion.amplitude_si,
            "omega_rad_s": motion.omega_rad_s,
            "frequency_hz": motion.omega_rad_s / (2.0 * math.pi),
            "settle_cycles": motion.settle_cycles,
            "sample_cycles": motion.sample_cycles,
        },
        "definitions": {
            "force_sign": "fluid-on-body",
            "single_column_model": "tau = -MA*nudot - DL*nu - DQ*abs(nu)*nu",
            "effective_damping": "D_eff = DL + DQ*v_peak, v_peak = amplitude*omega",
            "load_phase": "measured main-load fundamental relative to the signed generalized displacement",
            "relative_difference": "candidate minus reference, divided by abs(reference)",
            "phase_sampling": f"{PHASE_SAMPLES_PER_CYCLE} uniform samples per complete cycle",
        },
        "variants": variants,
        "comparisons": {
            "grid": {
                "config_paths": {name: str(config_paths[name]) for name in GRID_VARIANTS},
                "characteristic_cell_size_m": sizes,
                "metrics": grid_metrics,
            },
            "time_step": {
                "candidate": "nominal",
                "reference": "dt",
                "metrics": time_step_comparison,
            },
            "domain": {
                "candidate": "nominal",
                "reference": "domain",
                "metrics": domain_comparison,
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one identical OpenFOAM oscillation case across five convergence variants."
    )
    for variant in VARIANTS:
        parser.add_argument(f"--{variant}", required=True, type=Path, help=f"{variant} case directory")
    for variant in VARIANTS:
        parser.add_argument(
            f"--{variant}-config",
            type=Path,
            default=DEFAULT_VARIANT_CONFIGS[variant],
            help=f"configuration that generated the {variant} case",
        )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_dirs = {name: getattr(args, name) for name in VARIANTS}
    configs = {name: getattr(args, f"{name}_config") for name in VARIANTS}
    report = compare_cases(case_dirs, variant_config_paths=configs)
    paths = write_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "case_name": report["case"]["case_name"],
                "dof": report["case"]["dof"],
                "outputs": paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

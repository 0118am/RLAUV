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
import re
import sys
from typing import Any, Mapping

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from environment.openfoam.analysis.fit import _odd_project, _scaled_lstsq
from environment.openfoam.analysis.motion import CaseData, WRENCH_NAMES, load_case_data
from environment.openfoam.run_cases import _validated_completion


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
ACCEPTANCE_LIMITS = {
    "grid": {
        "added_mass_percent": 2.0,
        "effective_damping_percent": 5.0,
        "main_load_amplitude_percent": 3.0,
    },
    "time_step": {
        "added_mass_percent": 2.0,
        "effective_damping_percent": 3.0,
        "main_load_amplitude_percent": 2.0,
        "main_load_phase_deg": 1.0,
    },
    "domain": {
        "added_mass_percent": 1.0,
        "effective_damping_percent": 1.0,
        "main_load_amplitude_percent": 1.0,
        "main_load_phase_deg": 1.0,
    },
}


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
    odd = _odd_project(case, PHASE_SAMPLES_PER_CYCLE)
    coefficients, fit_diagnostics = _scaled_lstsq(odd.design, odd.target)
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


def _foam_entry(text: str, key: str, context: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s+([^;]+);", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{context}: missing OpenFOAM entry {key}")
    return match.group(1).strip()


def _foam_block(text: str, key: str, context: str) -> str:
    match = re.search(rf"\b{re.escape(key)}\s*\{{", text)
    if match is None:
        raise ValueError(f"{context}: missing OpenFOAM block {key}")
    opening = text.find("{", match.start())
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise ValueError(f"{context}: unterminated OpenFOAM block {key}")


def _foam_scalar(text: str, key: str, context: str) -> float:
    value = _foam_entry(text, key, context)
    if re.fullmatch(_FOAM_NUMBER, value) is None:
        raise ValueError(f"{context}: {key} is not a scalar: {value!r}")
    return float(value)


def _foam_vector(text: str, key: str, context: str) -> np.ndarray:
    value = _foam_entry(text, key, context)
    match = re.fullmatch(
        rf"\(\s*({_FOAM_NUMBER})\s+({_FOAM_NUMBER})\s+({_FOAM_NUMBER})\s*\)", value
    )
    if match is None:
        raise ValueError(f"{context}: {key} is not a three-vector: {value!r}")
    return np.asarray(tuple(map(float, match.groups())))


def _validate_solver_inputs(
    variant: str,
    case: Path,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    control_path = case / "system" / "controlDict"
    transport_path = case / "constant" / "transportProperties"
    displacement_path = case / "0" / "pointDisplacement"
    for path in (control_path, transport_path, displacement_path):
        if not path.is_file():
            raise ValueError(f"{variant}: missing solver input {path}")
    control = control_path.read_text(encoding="utf-8", errors="replace")
    context = str(control_path)
    if _foam_entry(control, "application", context) != config["solver"]:
        raise ValueError(f"{variant}: controlDict application does not match config")
    scalar_expectations = {
        "endTime": float(metadata["end_time_s"]),
        "deltaT": float(metadata.get("initial_delta_t_s", metadata["delta_t_s"])),
        "maxDeltaT": float(metadata["delta_t_s"]),
        "maxCo": float(config["max_co"]),
        "purgeWrite": float(config["purge_write"]),
    }
    for key, expected in scalar_expectations.items():
        actual = _foam_scalar(control, key, context)
        if not math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(
                f"{variant}: controlDict {key}={actual:.17g} does not match {expected:.17g}"
            )
    forces = _foam_block(control, "forces", context)
    for key, expected in (
        ("executeControl", "timeStep"),
        ("executeInterval", "1"),
        ("writeControl", "timeStep"),
        ("writeInterval", "1"),
    ):
        if _foam_entry(forces, key, f"{context}:forces") != expected:
            raise ValueError(f"{variant}: forces {key} must be {expected}")
    expected_start = float(
        metadata.get(
            "settle_end_s",
            float(metadata["settle_cycles"]) / float(metadata["frequency_hz"]),
        )
    )
    if not math.isclose(
        _foam_scalar(forces, "timeStart", f"{context}:forces"),
        expected_start,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{variant}: forces timeStart does not match settling window")
    patches = _foam_entry(forces, "patches", f"{context}:forces")
    if patches != f"({config['force_patch']})":
        raise ValueError(f"{variant}: forces patches does not match force_patch")
    if not math.isclose(
        _foam_scalar(forces, "rhoInf", f"{context}:forces"),
        float(config["rho_kg_m3"]),
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{variant}: forces rhoInf does not match config")
    expected_cofr = np.asarray(config["centre_of_rotation_m"], dtype=float)
    if not np.allclose(
        _foam_vector(forces, "CofR", f"{context}:forces"),
        expected_cofr,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"{variant}: forces CofR does not match config")

    transport = transport_path.read_text(encoding="utf-8", errors="replace")
    if not math.isclose(
        _foam_scalar(transport, "nu", str(transport_path)),
        float(config["nu_m2_s"]),
        rel_tol=1.0e-10,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{variant}: transportProperties nu does not match config")

    displacement = displacement_path.read_text(encoding="utf-8", errors="replace")
    patch = _foam_block(displacement, str(config["force_patch"]), str(displacement_path))
    if not math.isclose(
        _foam_scalar(patch, "omega", str(displacement_path)),
        float(metadata["omega_rad_s"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{variant}: pointDisplacement omega does not match motion.json")
    dof_index = int(metadata["dof_index"])
    axis = np.asarray(metadata["axis"], dtype=float)
    amplitude = float(
        metadata["amplitude_m"] if dof_index < 3 else metadata["amplitude_rad"]
    )
    if dof_index < 3:
        if _foam_entry(patch, "type", str(displacement_path)) != "oscillatingDisplacement":
            raise ValueError(f"{variant}: translation patch has the wrong motion type")
        actual_amplitude = _foam_vector(patch, "amplitude", str(displacement_path))
        if not np.allclose(actual_amplitude, amplitude * axis, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError(f"{variant}: translation amplitude does not match motion.json")
    else:
        if _foam_entry(patch, "type", str(displacement_path)) != "angularOscillatingDisplacement":
            raise ValueError(f"{variant}: rotation patch has the wrong motion type")
        if not np.allclose(
            _foam_vector(patch, "axis", str(displacement_path)), axis, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"{variant}: rotation axis does not match motion.json")
        if not math.isclose(
            _foam_scalar(patch, "amplitude", str(displacement_path)),
            amplitude,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{variant}: rotation amplitude does not match motion.json")
    return {
        "control_dict": str(control_path),
        "transport_properties": str(transport_path),
        "point_displacement": str(displacement_path),
    }


def _read_block_mesh_dict(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    vertices_match = re.search(r"\bvertices\s*\((.*?)\)\s*;", text, re.DOTALL)
    if vertices_match is None:
        raise ValueError(f"{path}: could not parse vertices")
    vector_pattern = re.compile(
        rf"\(\s*({_FOAM_NUMBER})\s+({_FOAM_NUMBER})\s+({_FOAM_NUMBER})\s*\)"
    )
    vertices = np.asarray(
        [tuple(map(float, match.groups())) for match in vector_pattern.finditer(vertices_match.group(1))],
        dtype=float,
    )
    if vertices.shape != (8, 3):
        raise ValueError(f"{path}: expected eight block vertices, got shape {vertices.shape}")
    block_match = re.search(
        r"\bhex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text
    )
    if block_match is None:
        raise ValueError(f"{path}: could not parse hex base-cell counts")
    return np.min(vertices, axis=0), np.max(vertices, axis=0), np.asarray(
        tuple(map(int, block_match.groups())), dtype=int
    )


def _metadata_matches_config(
    variant: str,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    for name in (
        "openfoam_version",
        "solver",
        "rho_kg_m3",
        "nu_m2_s",
        "force_patch",
        "settle_cycles",
        "sample_cycles",
        "max_co",
    ):
        expected = config.get(name)
        actual = metadata.get(name)
        if expected is None and actual is None:
            continue
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if not math.isclose(float(actual), float(expected), rel_tol=1.0e-11, abs_tol=1.0e-12):
                raise ValueError(
                    f"{variant}: motion.json {name}={actual!r} does not match config {expected!r}"
                )
        elif actual != expected:
            raise ValueError(
                f"{variant}: motion.json {name}={actual!r} does not match config {expected!r}"
            )
    frequency = metadata.get("frequency_hz")
    steps = config.get("steps_per_cycle")
    delta_t = metadata.get("delta_t_s")
    initial_delta_t = metadata.get("initial_delta_t_s")
    if not isinstance(frequency, (int, float)) or float(frequency) <= 0.0:
        raise ValueError(f"{variant}: motion.json requires a positive frequency_hz")
    if type(steps) is not int or steps <= 0:
        raise ValueError(f"{variant}: config requires a positive integer steps_per_cycle")
    expected_delta_t = 1.0 / (float(frequency) * steps)
    if not isinstance(delta_t, (int, float)) or not math.isclose(
        float(delta_t), expected_delta_t, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{variant}: motion.json delta_t_s={delta_t!r} does not match "
            f"period/steps_per_cycle={expected_delta_t:.17g}"
        )
    expected_initial_delta_t = expected_delta_t * float(config["initial_delta_t_fraction"])
    if not isinstance(initial_delta_t, (int, float)) or not math.isclose(
        float(initial_delta_t), expected_initial_delta_t, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{variant}: motion.json initial_delta_t_s={initial_delta_t!r} does not match "
            f"delta_t_s*initial_delta_t_fraction={expected_initial_delta_t:.17g}"
        )


def _strip_config_fields(
    config: Mapping[str, Any], fields: tuple[tuple[str, ...], ...]
) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    for field in fields:
        parent: Any = result
        for key in field[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                parent = None
                break
            parent = parent[key]
        if isinstance(parent, dict):
            parent.pop(field[-1], None)
    return result


def _validate_config_family(configs: Mapping[str, Mapping[str, Any]]) -> None:
    grid_allowed = (
        ("block_mesh", "base_cells"),
        ("snappy", "max_local_cells"),
        ("snappy", "max_global_cells"),
    )
    nominal_grid = _strip_config_fields(configs["nominal"], grid_allowed)
    for variant in ("coarse", "fine"):
        if _strip_config_fields(configs[variant], grid_allowed) != nominal_grid:
            raise ValueError(
                f"{variant} config differs from nominal outside base cells/snappy cell caps"
            )
    time_allowed = (("steps_per_cycle",), ("max_co",))
    if _strip_config_fields(configs["dt"], time_allowed) != _strip_config_fields(
        configs["nominal"], time_allowed
    ):
        raise ValueError("dt config differs from nominal outside steps_per_cycle/max_co")
    domain_allowed = (("block_mesh",),)
    if _strip_config_fields(configs["domain"], domain_allowed) != _strip_config_fields(
        configs["nominal"], domain_allowed
    ):
        raise ValueError("domain config differs from nominal outside block_mesh")


def _validate_case_provenance(
    paths: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
    config_paths: Mapping[str, Path],
) -> dict[str, Any]:
    if len(set(paths.values())) != len(VARIANTS):
        raise ValueError("coarse/nominal/fine/dt/domain must be five distinct case directories")
    mesh_paths: dict[str, Path] = {}
    geometry_paths: dict[str, Path] = {}
    mesh_quality: dict[str, dict[str, Any]] = {}
    solver_inputs: dict[str, dict[str, str]] = {}
    configs = {variant: _load_json(config_paths[variant]) for variant in VARIANTS}
    _validate_config_family(configs)
    for variant in VARIANTS:
        config = configs[variant]
        _metadata_matches_config(variant, metadata[variant], config)
        solver_inputs[variant] = _validate_solver_inputs(
            variant, paths[variant], metadata[variant], config
        )
        solver = config.get("solver")
        if not isinstance(solver, str) or not solver:
            raise ValueError(f"{config_paths[variant]}: missing solver")
        completed, reason = _validated_completion(paths[variant], solver)
        if not completed:
            raise ValueError(f"{variant}: case completion evidence is invalid: {reason}")
        geometry_filename = config.get("geometry_filename")
        if not isinstance(geometry_filename, str) or not geometry_filename:
            raise ValueError(f"{config_paths[variant]}: missing geometry_filename")
        geometry = paths[variant] / "constant" / "triSurface" / geometry_filename
        try:
            geometry_paths[variant] = geometry.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"{variant}: missing configured geometry {geometry}") from error

        poly_mesh = paths[variant] / "constant" / "polyMesh"
        try:
            resolved_mesh = poly_mesh.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"{variant}: missing constant/polyMesh") from error
        required_mesh_files = ("points", "faces", "owner", "neighbour", "boundary")
        missing_files = [name for name in required_mesh_files if not (resolved_mesh / name).is_file()]
        if missing_files:
            raise ValueError(f"{variant}: incomplete polyMesh; missing {missing_files}")
        mesh_paths[variant] = resolved_mesh
        if resolved_mesh.name != "polyMesh" or resolved_mesh.parent.name != "constant":
            raise ValueError(f"{variant}: unexpected resolved polyMesh path {resolved_mesh}")
        block_dict = resolved_mesh.parent.parent / "system" / "blockMeshDict"
        if not block_dict.is_file():
            raise ValueError(f"{variant}: missing source mesh dictionary {block_dict}")
        actual_lower, actual_upper, actual_cells = _read_block_mesh_dict(block_dict)
        expected_lower, expected_upper, expected_cells = _block_mesh_spec(
            config, str(config_paths[variant])
        )
        if (
            not np.allclose(actual_lower, expected_lower, rtol=0.0, atol=1.0e-12)
            or not np.allclose(actual_upper, expected_upper, rtol=0.0, atol=1.0e-12)
            or not np.array_equal(actual_cells, expected_cells)
        ):
            raise ValueError(
                f"{variant}: source blockMeshDict does not match {config_paths[variant]}"
            )
        log_dir = resolved_mesh.parent.parent / "logs"
        check_mesh_path = log_dir / "checkMesh.log"
        if not check_mesh_path.is_file():
            raise ValueError(f"{variant}: missing mesh-quality evidence {check_mesh_path}")
        check_mesh_text = check_mesh_path.read_text(encoding="utf-8", errors="replace")
        if "Mesh OK." not in check_mesh_text or re.search(
            r"Failed\s+[1-9]\d*\s+mesh checks", check_mesh_text
        ):
            raise ValueError(f"{variant}: checkMesh evidence does not pass strictly")
        cell_match = re.search(r"^\s*cells:\s*(\d+)\s*$", check_mesh_text, re.MULTILINE)
        if cell_match is None or int(cell_match.group(1)) <= 0:
            raise ValueError(f"{variant}: checkMesh evidence has no positive cell count")
        volume_path = log_dir / "mesh_volume_validation.json"
        volume = _load_json(volume_path)
        if volume.get("passed") is not True:
            raise ValueError(f"{variant}: displaced-volume mesh validation did not pass")
        mesh_quality[variant] = {
            "check_mesh_log": str(check_mesh_path),
            "cell_count": int(cell_match.group(1)),
            "mesh_volume_validation": volume,
        }

    independent_meshes = {mesh_paths[name] for name in ("coarse", "nominal", "fine", "domain")}
    if len(independent_meshes) != 4:
        raise ValueError("coarse/nominal/fine/domain must resolve to four distinct meshes")
    if mesh_paths["dt"] != mesh_paths["nominal"]:
        raise ValueError("dt case must reuse the exact nominal checked mesh")
    grid_cell_counts = [mesh_quality[name]["cell_count"] for name in GRID_VARIANTS]
    if not grid_cell_counts[0] < grid_cell_counts[1] < grid_cell_counts[2]:
        raise ValueError(
            "actual checked grid cell counts must increase coarse < nominal < fine; "
            f"got {grid_cell_counts}"
        )
    if len(set(geometry_paths.values())) != 1:
        raise ValueError("all convergence variants must resolve to the same geometry file")
    return {
        "config_paths": {name: str(config_paths[name]) for name in VARIANTS},
        "resolved_poly_mesh_paths": {name: str(mesh_paths[name]) for name in VARIANTS},
        "resolved_geometry_path": str(geometry_paths["nominal"]),
        "mesh_quality": mesh_quality,
        "solver_inputs": solver_inputs,
    }


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


def _acceptance_assessment(
    grid_metrics: Mapping[str, Mapping[str, Any]],
    time_step: Mapping[str, Mapping[str, Any]],
    domain: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def usable(value: Any) -> float | None:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def grid_check(metric: str, limit_name: str, label: str) -> None:
        item = grid_metrics[metric]
        gci = item["gci"]
        limit = ACCEPTANCE_LIMITS["grid"][limit_name]
        gci_value = usable(gci.get("fine_grid_gci_percent"))
        if gci["available"] and gci_value is not None:
            value = gci_value
            provisional = False
            source = "fine_grid_gci_percent"
            reason = f"fine-grid GCI {value:.6g}% {'<=' if value <= limit else '>'} {limit:.6g}%"
        else:
            value = usable(item["nominal_vs_fine"]["absolute_relative_difference_percent"])
            provisional = True
            source = "nominal_vs_fine_absolute_relative_difference_percent"
            if value is None:
                reason = f"{gci['status']}; relative difference is undefined because reference is zero"
            else:
                reason = (
                    f"{gci['status']}; provisional nominal-vs-fine difference "
                    f"{value:.6g}% {'<=' if value <= limit else '>'} {limit:.6g}%"
                )
        passed = value is not None and value <= limit
        checks[f"grid_{label}"] = {
            "pass": passed,
            "provisional": provisional,
            "status": (
                "provisional_pass" if provisional and passed else
                "provisional_fail" if provisional else
                "pass" if passed else "fail"
            ),
            "value": value,
            "limit": limit,
            "units": "percent",
            "source": source,
            "reason": reason,
        }

    def relative_check(
        section: str,
        values: Mapping[str, Mapping[str, Any]],
        metric: str,
        limit_name: str,
        label: str,
    ) -> None:
        value = usable(values[metric]["absolute_relative_difference_percent"])
        limit = ACCEPTANCE_LIMITS[section][limit_name]
        passed = value is not None and value <= limit
        checks[f"{section}_{label}"] = {
            "pass": passed,
            "provisional": False,
            "status": "pass" if passed else "fail",
            "value": value,
            "limit": limit,
            "units": "percent",
            "source": "absolute_relative_difference_percent",
            "reason": (
                "relative difference is undefined because reference is zero"
                if value is None
                else f"absolute relative difference {value:.6g}% {'<=' if passed else '>'} {limit:.6g}%"
            ),
        }

    def phase_check(
        section: str, values: Mapping[str, Mapping[str, Any]], label: str
    ) -> None:
        value = float(values["main_load_phase"]["absolute_difference_deg"])
        limit = ACCEPTANCE_LIMITS[section]["main_load_phase_deg"]
        passed = value <= limit
        checks[f"{section}_{label}"] = {
            "pass": passed,
            "provisional": False,
            "status": "pass" if passed else "fail",
            "value": value,
            "limit": limit,
            "units": "degree",
            "source": "absolute_circular_phase_difference_deg",
            "reason": f"absolute phase difference {value:.6g} deg {'<=' if passed else '>'} {limit:.6g} deg",
        }

    grid_check("added_mass", "added_mass_percent", "added_mass")
    grid_check(
        "effective_damping_at_peak_speed",
        "effective_damping_percent",
        "effective_damping",
    )
    grid_check(
        "main_load_fundamental_amplitude",
        "main_load_amplitude_percent",
        "main_load_amplitude",
    )
    for section, values in (("time_step", time_step), ("domain", domain)):
        relative_check(section, values, "added_mass", "added_mass_percent", "added_mass")
        relative_check(
            section,
            values,
            "effective_damping_at_peak_speed",
            "effective_damping_percent",
            "effective_damping",
        )
        relative_check(
            section,
            values,
            "main_load_fundamental_amplitude",
            "main_load_amplitude_percent",
            "main_load_amplitude",
        )
        phase_check(section, values, "main_load_phase")

    all_numeric_limits_pass = all(item["pass"] for item in checks.values())
    provisional = any(item["provisional"] for item in checks.values())
    overall_pass = all_numeric_limits_pass and not provisional
    return {
        "overall_pass": overall_pass,
        "status": (
            "provisional_pass" if all_numeric_limits_pass and provisional else
            "pass" if overall_pass else "fail"
        ),
        "provisional": provisional,
        "all_numeric_limits_pass": all_numeric_limits_pass,
        "limits": ACCEPTANCE_LIMITS,
        "checks": checks,
        "residual_note": "odd-fit residual is reported as a diagnostic and is not a convergence gate",
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
    provenance = _validate_case_provenance(paths, metadata, config_paths)
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
    acceptance = _acceptance_assessment(
        grid_metrics, time_step_comparison, domain_comparison
    )
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
        "acceptance": acceptance,
        "provenance_validation": provenance,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.7g}"


def _markdown(report: Mapping[str, Any]) -> str:
    case = report["case"]
    variants = report["variants"]
    lines = [
        f"# Convergence report: {case['case_name']}",
        "",
        (
            f"Excitation `{case['dof']}` / main load `{case['main_wrench']}`; "
            f"amplitude `{case['amplitude_si']:.7g}` SI, frequency "
            f"`{case['frequency_hz']:.7g} Hz`. Force sign is fluid-on-body."
        ),
        "",
        "All variants passed config/mesh/geometry identity, durable completion, identical-motion, complete-cycle, and restart-gap checks.",
        "",
        "| Variant | MA | DL | DQ | D_eff at v_peak | Load amplitude | Load phase (deg) | Odd-fit residual RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        item = variants[name]
        coefficient = item["coefficients"]
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    _format_number(coefficient["added_mass"]),
                    _format_number(coefficient["linear_damping"]),
                    _format_number(coefficient["quadratic_damping"]),
                    _format_number(coefficient["effective_damping_at_peak_speed"]),
                    _format_number(item["main_load"]["amplitude"]),
                    _format_number(
                        item["main_load"]["phase_deg_relative_to_displacement_sine"]
                    ),
                    _format_number(item["fit"]["odd_model_residual_rms"]),
                )
            )
            + " |"
        )

    acceptance = report["acceptance"]
    lines.extend(
        [
            "",
            "## Machine-readable acceptance",
            "",
            f"Overall: **{acceptance['status']}** (`overall_pass={str(acceptance['overall_pass']).lower()}`).",
            "",
            "| Check | Status | Value | Limit | Units | Reason |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for name, check in acceptance["checks"].items():
        lines.append(
            f"| {name} | {check['status']} | {_format_number(check['value'])} | "
            f"{_format_number(check['limit'])} | {check['units']} | {check['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Grid convergence",
            "",
            "| Metric | coarse vs nominal (%) | nominal vs fine (%) | GCI fine (%) | observed order | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    grid_metrics = report["comparisons"]["grid"]["metrics"]
    for name in (
        "added_mass",
        "effective_damping_at_peak_speed",
        "main_load_fundamental_amplitude",
        "odd_model_residual_rms",
    ):
        item = grid_metrics[name]
        gci = item["gci"]
        lines.append(
            f"| {name} | "
            f"{_format_number(item['coarse_vs_nominal']['absolute_relative_difference_percent'])} | "
            f"{_format_number(item['nominal_vs_fine']['absolute_relative_difference_percent'])} | "
            f"{_format_number(gci.get('fine_grid_gci_percent'))} | "
            f"{_format_number(gci.get('observed_order'))} | {gci['status']} |"
        )
    phase = grid_metrics["main_load_phase"]
    phase_gci = phase["gci_degrees"]
    lines.append(
        "| main_load_phase (absolute deg) | "
        f"{_format_number(phase['coarse_vs_nominal']['absolute_difference_deg'])} | "
        f"{_format_number(phase['nominal_vs_fine']['absolute_difference_deg'])} | "
        f"{_format_number(phase_gci.get('fine_grid_gci_absolute'))} | "
        f"{_format_number(phase_gci.get('observed_order'))} | {phase_gci['status']} |"
    )

    lines.extend(
        [
            "",
            "## Time-step and domain checks",
            "",
            "Percentages use the refined-time-step or expanded-domain result as reference; phase is an absolute circular difference.",
            "",
            "| Metric | nominal vs refined dt | nominal vs expanded domain |",
            "|---|---:|---:|",
        ]
    )
    timestep = report["comparisons"]["time_step"]["metrics"]
    domain = report["comparisons"]["domain"]["metrics"]
    for name in (
        "added_mass",
        "effective_damping_at_peak_speed",
        "main_load_fundamental_amplitude",
        "odd_model_residual_rms",
    ):
        lines.append(
            f"| {name} (%) | "
            f"{_format_number(timestep[name]['absolute_relative_difference_percent'])} | "
            f"{_format_number(domain[name]['absolute_relative_difference_percent'])} |"
        )
    lines.append(
        "| main_load_phase (deg) | "
        f"{_format_number(timestep['main_load_phase']['absolute_difference_deg'])} | "
        f"{_format_number(domain['main_load_phase']['absolute_difference_deg'])} |"
    )
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "convergence_report.json"
    markdown_path = destination / "convergence_report.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


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
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return exit status 1 unless acceptance.overall_pass is true",
    )
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
                "overall_pass": report["acceptance"]["overall_pass"],
                "acceptance_status": report["acceptance"]["status"],
                "outputs": paths,
            },
            indent=2,
        )
    )
    return 1 if args.require_pass and not report["acceptance"]["overall_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

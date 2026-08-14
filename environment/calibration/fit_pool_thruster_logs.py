"""Fit dynamic thruster and interaction updates from validated pool CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.identification.fitters import (  # noqa: E402
    fit_battery_voltage_sag,
    fit_thruster_first_order_response,
    fit_thruster_reaction_torque_coefficient,
    fit_thruster_voltage_exponent,
    fit_thruster_wake_loss_coefficient,
)
from environment.profiles.pool_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolDynamicsProfile,
    PoolProfileAuditOptions,
    load_pool_dynamics_profile_json,
    pool_profile_calibration_log_schemas,
    validate_pool_calibration_log_directory,
)


THRUSTER_LOG_FILENAMES = (
    "thruster_step_response.csv",
    "battery_voltage_thrust_samples.csv",
    "thruster_wake_interaction.csv",
    "thruster_reaction_torque.csv",
)


@dataclass(frozen=True)
class ThrusterCalibrationPipelineResult:
    cfg_updates: dict[str, Any]
    diagnostics: dict[str, Any]
    source_files: tuple[str, ...]
    domain_randomization_updates: dict[str, Any] | None = None

    def update_payload(self) -> dict[str, Any]:
        return {
            "cfg_updates": self.cfg_updates,
            "domain_randomization_updates": self.domain_randomization_updates or {},
        }

    def report_dict(self) -> dict[str, Any]:
        return {
            "source_files": list(self.source_files),
            "cfg_updates": self.cfg_updates,
            "domain_randomization_updates": self.domain_randomization_updates or {},
            "diagnostics": self.diagnostics,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit thruster response, voltage, wake, and reaction-torque updates from calibration CSV logs.",
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing thruster_*.csv calibration logs.")
    parser.add_argument("--base-profile", type=Path, help="Profile JSON providing wake geometry and spin directions.")
    parser.add_argument("--output", type=Path, required=True, help="Output builder-compatible updates JSON path.")
    parser.add_argument("--report", type=Path, help="Optional detailed fit diagnostics JSON path.")
    parser.add_argument(
        "--physics-dt",
        type=float,
        help="Physics timestep used to convert fitted response delay seconds into command delay steps.",
    )
    parser.add_argument(
        "--delay-candidates",
        type=int,
        default=128,
        help="Number of response-delay candidates evaluated for step-response fitting.",
    )
    parser.add_argument("--nominal-voltage", type=float, default=16.0, help="Nominal voltage for thrust scaling.")
    parser.add_argument("--wake-length", type=float, help="Override the base-profile wake length in m.")
    parser.add_argument("--wake-radius", type=float, help="Override the base-profile wake base radius in m.")
    parser.add_argument("--wake-expansion-rate", type=float, help="Override the base-profile wake radial expansion rate.")
    parser.add_argument("--wake-min-scale", type=float, help="Override the base-profile minimum target-thrust scale.")
    return parser


def fit_thruster_calibration_logs(
    log_dir: Path,
    *,
    base_profile: PoolDynamicsProfile | None = None,
    physics_dt_s: float | None = None,
    delay_candidate_count: int = 128,
    nominal_voltage: float = 16.0,
    wake_length: float | None = None,
    wake_radius: float | None = None,
    wake_expansion_rate: float | None = None,
    wake_min_scale: float | None = None,
) -> ThrusterCalibrationPipelineResult:
    if physics_dt_s is not None and float(physics_dt_s) <= 0.0:
        raise ValueError("physics_dt_s must be positive when provided.")
    if int(delay_candidate_count) != delay_candidate_count or int(delay_candidate_count) < 1:
        raise ValueError("delay_candidate_count must be a positive integer.")
    if float(nominal_voltage) <= 0.0:
        raise ValueError("nominal_voltage must be positive.")

    profile = NOMINAL_POOL_DYNAMICS_PROFILE if base_profile is None else base_profile
    profile.validate()
    resolved_wake_length = profile.thrusters.wake_length if wake_length is None else float(wake_length)
    resolved_wake_radius = profile.thrusters.wake_radius if wake_radius is None else float(wake_radius)
    resolved_wake_expansion = (
        profile.thrusters.wake_expansion_rate
        if wake_expansion_rate is None
        else float(wake_expansion_rate)
    )
    resolved_wake_min_scale = profile.thrusters.wake_min_scale if wake_min_scale is None else float(wake_min_scale)
    if resolved_wake_length <= 0.0 or resolved_wake_radius <= 0.0:
        raise ValueError("Wake length and radius must be positive.")
    if resolved_wake_expansion < 0.0:
        raise ValueError("Wake expansion rate must be non-negative.")
    if not 0.0 <= resolved_wake_min_scale <= 1.0:
        raise ValueError("Wake minimum scale must be in [0, 1].")

    schemas = pool_profile_calibration_log_schemas(
        NOMINAL_POOL_DYNAMICS_PROFILE,
        PoolProfileAuditOptions(domain_randomization_expected=False),
    )
    schema_by_filename = {schema.filename: schema for schema in schemas}
    source_files = tuple(filename for filename in THRUSTER_LOG_FILENAMES if (log_dir / filename).is_file())
    if not source_files:
        raise ValueError(f"No supported thruster calibration logs found in {log_dir}.")

    source_schemas = tuple(schema_by_filename[filename] for filename in source_files)
    validation = validate_pool_calibration_log_directory(log_dir, source_schemas)
    if not validation.is_valid:
        messages = "; ".join(
            f"{issue.filename}:{issue.row_number or '-'}:{issue.column or '-'} {issue.message}"
            for issue in validation.issues
            if issue.severity == "error"
        )
        raise ValueError(f"Thruster calibration log validation failed: {messages}")

    cfg_updates: dict[str, Any] = {}
    domain_randomization_updates: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"validation": validation.to_dict()}

    response_path = log_dir / "thruster_step_response.csv"
    if response_path.is_file():
        rows = _read_csv_rows(response_path)
        time_s = _float_column(rows, "time_s")
        commands = _float_column(rows, "pwm_us")
        measured_thrust = _float_column(rows, "measured_thrust_n")
        step_time = _infer_step_time(time_s, commands)
        fit = fit_thruster_first_order_response(
            time_s,
            measured_thrust,
            command_step_time_s=step_time,
            delay_candidate_count=delay_candidate_count,
        )
        cfg_updates.update(fit.to_cfg_updates(physics_dt_s=physics_dt_s))
        diagnostics["first_order_response"] = {
            "command_step_time_s": step_time,
            "time_constant_s": fit.time_constant_s,
            "response_delay_s": fit.response_delay_s,
            "initial_thrust_n": fit.initial_thrust,
            "steady_state_thrust_n": fit.steady_state_thrust,
            "residual_rms_n": fit.residual_rms,
            "sample_count": fit.sample_count,
            "physics_dt_s": physics_dt_s,
            "command_delay_steps": cfg_updates.get("thruster_command_delay_steps"),
        }

    battery_path = log_dir / "battery_voltage_thrust_samples.csv"
    if battery_path.is_file():
        rows = _read_csv_rows(battery_path)
        sag_fit = fit_battery_voltage_sag(
            _float_column(rows, "time_s"),
            _float_column(rows, "voltage_v"),
        )
        cfg_updates.update(sag_fit.to_cfg_updates())
        diagnostics["battery_voltage_sag"] = {
            "initial_voltage_v": sag_fit.initial_voltage,
            "min_observed_voltage_v": sag_fit.min_observed_voltage,
            "voltage_drop_per_s": sag_fit.voltage_drop_per_s,
            "residual_rms_v": sag_fit.residual_rms,
            "sample_count": sag_fit.sample_count,
            "time_origin_s": sag_fit.time_origin_s,
        }
        scale_rows = _rows_with_value(rows, "thrust_scale")
        if scale_rows:
            exponent_fit = fit_thruster_voltage_exponent(
                _float_column(scale_rows, "voltage_v"),
                _float_column(scale_rows, "thrust_scale"),
                nominal_voltage=nominal_voltage,
            )
            cfg_updates.update(exponent_fit.to_cfg_updates())
            diagnostics["battery_thrust_scaling"] = {
                "nominal_voltage_v": exponent_fit.nominal_voltage,
                "thrust_exponent": exponent_fit.thrust_exponent,
                "residual_rms": exponent_fit.residual_rms,
                "sample_count": exponent_fit.sample_count,
            }

    wake_path = log_dir / "thruster_wake_interaction.csv"
    if wake_path.is_file():
        rows = _read_csv_rows(wake_path)
        fit = fit_thruster_wake_loss_coefficient(
            _float_column(rows, "source_thrust_n"),
            _float_column(rows, "source_reference_thrust_n"),
            _float_column(rows, "axial_distance_m"),
            _float_column(rows, "radial_distance_m"),
            _float_column(rows, "measured_target_thrust_scale"),
            wake_length=resolved_wake_length,
            wake_radius=resolved_wake_radius,
            expansion_rate=resolved_wake_expansion,
            min_scale=resolved_wake_min_scale,
        )
        cfg_updates.update(fit.to_cfg_updates())
        domain_randomization_updates.update(fit.to_domain_randomization_updates())
        diagnostics["wake_interaction"] = {
            "loss_coefficient": fit.loss_coefficient,
            "loss_coefficient_std": fit.loss_coefficient_std,
            "wake_length_m": fit.wake_length,
            "wake_radius_m": fit.wake_radius,
            "expansion_rate": fit.expansion_rate,
            "min_scale": fit.min_scale,
            "residual_rms_scale": fit.residual_rms,
            "sample_count": fit.sample_count,
            "informative_sample_count": fit.informative_sample_count,
        }

    reaction_path = log_dir / "thruster_reaction_torque.csv"
    if reaction_path.is_file():
        rows = _read_csv_rows(reaction_path)
        fit = fit_thruster_reaction_torque_coefficient(
            _float_column(rows, "thrust_n"),
            _float_column(rows, "reaction_torque_nm"),
            _float_column(rows, "spin_direction"),
        )
        cfg_updates.update(fit.to_cfg_updates())
        domain_randomization_updates.update(fit.to_domain_randomization_updates())
        spin_directions, spin_source = _vehicle_spin_directions_from_rows(rows, profile)
        cfg_updates["thruster_spin_directions"] = spin_directions
        diagnostics["reaction_torque"] = {
            "torque_coefficient_m": fit.torque_coefficient_m,
            "torque_coefficient_std_m": fit.torque_coefficient_std_m,
            "residual_rms_nm": fit.residual_rms_nm,
            "sample_count": fit.sample_count,
            "spin_directions": spin_directions,
            "spin_direction_source": spin_source,
        }

    return ThrusterCalibrationPipelineResult(
        cfg_updates,
        diagnostics,
        source_files,
        domain_randomization_updates,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return rows


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _required_float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    values: list[float] = []
    for row_index, row in enumerate(rows, start=1):
        raw = row.get(name, "")
        if not raw.strip():
            raise ValueError(f"Column {name} must be populated for every row when present; row {row_index} is empty.")
        values.append(float(raw))
    return values


def _rows_with_value(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(name, "").strip()]


def _vehicle_spin_directions_from_rows(
    rows: list[dict[str, str]],
    profile: PoolDynamicsProfile,
) -> tuple[list[float], str]:
    grouped: dict[str, set[float]] = {}
    for row in rows:
        grouped.setdefault(row["thruster_index"], set()).add(float(row["spin_direction"]))
    expected = {str(index) for index in range(8)}
    if set(grouped) != expected:
        return [float(value) for value in profile.thrusters.spin_directions], "base_profile"
    if any(values not in ({-1.0}, {1.0}) for values in grouped.values()):
        raise ValueError("Each thruster_index must use one consistent spin_direction of -1 or 1.")
    return [next(iter(grouped[str(index)])) for index in range(8)], "reaction_torque_log"


def _infer_step_time(time_s: list[float], commands: list[float], tolerance: float = 1.0e-6) -> float:
    if len(time_s) != len(commands) or len(time_s) < 2:
        raise ValueError("Step-response time and command arrays must have matching length >= 2.")
    baseline = commands[0]
    for time_value, command in zip(time_s[1:], commands[1:]):
        if abs(command - baseline) > float(tolerance):
            return float(time_value)
    raise ValueError("Could not infer a command step from thruster_step_response.csv.")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        base_profile = (
            load_pool_dynamics_profile_json(args.base_profile)
            if args.base_profile is not None
            else NOMINAL_POOL_DYNAMICS_PROFILE
        )
        result = fit_thruster_calibration_logs(
            args.log_dir,
            base_profile=base_profile,
            physics_dt_s=args.physics_dt,
            delay_candidate_count=args.delay_candidates,
            nominal_voltage=args.nominal_voltage,
            wake_length=args.wake_length,
            wake_radius=args.wake_radius,
            wake_expansion_rate=args.wake_expansion_rate,
            wake_min_scale=args.wake_min_scale,
        )
        _write_json(args.output, result.update_payload())
        if args.report is not None:
            _write_json(args.report, result.report_dict())
    except Exception as exc:
        print(f"Failed to fit thruster calibration logs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit rigid-body and hydrostatic profile updates from validated pool CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.identification.fitters import (  # noqa: E402
    compound_pendulum_moments_from_periods,
    fit_buoyancy_volume_from_forces,
    fit_com_to_cob_offset_from_static_torques,
    fit_inertia_tensor_from_axis_moments,
    fit_mass_from_scale_readings,
)
from environment.profiles.pool_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolProfileAuditOptions,
    pool_profile_calibration_log_schemas,
    validate_pool_calibration_log_directory,
)


STATIC_LOG_FILENAMES = (
    "rigid_body_mass_readings.csv",
    "rigid_body_buoyancy_forces.csv",
    "rigid_body_static_buoyancy_torques.csv",
    "rigid_body_axis_moments.csv",
    "rigid_body_compound_pendulum_periods.csv",
)


@dataclass(frozen=True)
class StaticCalibrationPipelineResult:
    cfg_updates: dict[str, Any]
    diagnostics: dict[str, Any]
    source_files: tuple[str, ...]

    def update_payload(self) -> dict[str, Any]:
        return {
            "cfg_updates": self.cfg_updates,
            "domain_randomization_updates": {},
        }

    def report_dict(self) -> dict[str, Any]:
        return {
            "source_files": list(self.source_files),
            "cfg_updates": self.cfg_updates,
            "diagnostics": self.diagnostics,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit rigid-body/hydrostatic PoolDynamicsProfile updates from calibration CSV logs.",
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing rigid_body_*.csv calibration logs.")
    parser.add_argument("--output", type=Path, required=True, help="Output builder-compatible updates JSON path.")
    parser.add_argument("--report", type=Path, help="Optional detailed fit diagnostics JSON path.")
    parser.add_argument("--gravity-z", type=float, default=-9.81, help="World-frame gravity z component in m/s^2.")
    parser.add_argument(
        "--min-inertia-eigenvalue",
        type=float,
        default=0.0,
        help="Minimum eigenvalue used when projecting the fitted inertia tensor.",
    )
    return parser


def fit_static_calibration_logs(
    log_dir: Path,
    *,
    gravity_z: float = -9.81,
    min_inertia_eigenvalue: float = 0.0,
) -> StaticCalibrationPipelineResult:
    if float(gravity_z) >= 0.0:
        raise ValueError("gravity_z must be negative for the default world-frame convention.")
    if float(min_inertia_eigenvalue) < 0.0:
        raise ValueError("min_inertia_eigenvalue must be non-negative.")

    # The AUV nominal profile already contains the full CAD inertia tensor,
    # but this tool must still accept raw inertia experiment logs.  Request the
    # corresponding schemas explicitly instead of letting the audit infer that
    # no inertia work is needed from the nominal profile.
    schema_profile = replace(
        NOMINAL_POOL_DYNAMICS_PROFILE,
        rigid_body=replace(NOMINAL_POOL_DYNAMICS_PROFILE.rigid_body, inertia_diag=(1.0, 1.0, 1.0)),
    )
    schemas = pool_profile_calibration_log_schemas(
        schema_profile,
        PoolProfileAuditOptions(domain_randomization_expected=False),
    )
    schema_by_filename = {schema.filename: schema for schema in schemas}
    source_files = tuple(filename for filename in STATIC_LOG_FILENAMES if (log_dir / filename).is_file())
    if not source_files:
        raise ValueError(f"No supported static calibration logs found in {log_dir}.")

    source_schemas = tuple(schema_by_filename[filename] for filename in source_files)
    validation = validate_pool_calibration_log_directory(log_dir, source_schemas)
    if not validation.is_valid:
        messages = "; ".join(
            f"{issue.filename}:{issue.row_number or '-'}:{issue.column or '-'} {issue.message}"
            for issue in validation.issues
            if issue.severity == "error"
        )
        raise ValueError(f"Static calibration log validation failed: {messages}")

    cfg_updates: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"validation": validation.to_dict()}
    volume_fit = None

    mass_path = log_dir / "rigid_body_mass_readings.csv"
    if mass_path.is_file():
        rows = _read_csv_rows(mass_path)
        fit = fit_mass_from_scale_readings(_float_column(rows, "mass_kg"))
        cfg_updates.update(fit.to_cfg_updates())
        diagnostics["mass"] = {
            "mass_kg": fit.mass,
            "residual_rms_kg": fit.residual_rms,
            "sample_count": fit.sample_count,
        }

    buoyancy_path = log_dir / "rigid_body_buoyancy_forces.csv"
    if buoyancy_path.is_file():
        rows = _read_csv_rows(buoyancy_path)
        density = _mean_column(rows, "water_density_kg_m3")
        gravity_from_log = _mean_column(rows, "gravity_w_z_mps2")
        forces = _vector_columns(
            rows,
            ("buoyancy_force_w_x_n", "buoyancy_force_w_y_n", "buoyancy_force_w_z_n"),
        )
        volume_fit = fit_buoyancy_volume_from_forces(
            forces,
            water_density=density,
            gravity_w=(0.0, 0.0, gravity_from_log),
        )
        cfg_updates.update(volume_fit.to_cfg_updates())
        diagnostics["buoyancy_volume"] = {
            "volume_m3": volume_fit.volume,
            "water_density_kg_m3": volume_fit.water_density,
            "mean_buoyancy_force_w_n": volume_fit.mean_buoyancy_force_w.detach().cpu().tolist(),
            "residual_rms_n": volume_fit.residual_rms,
            "sample_count": volume_fit.sample_count,
            "gravity_z_mps2": gravity_from_log,
        }

    axes: list[list[float]] = []
    moments: list[float] = []
    axis_moment_path = log_dir / "rigid_body_axis_moments.csv"
    if axis_moment_path.is_file():
        rows = _read_csv_rows(axis_moment_path)
        axes.extend(_vector_columns(rows, ("axis_b_x", "axis_b_y", "axis_b_z")))
        moments.extend(_float_column(rows, "moment_kg_m2"))

    pendulum_path = log_dir / "rigid_body_compound_pendulum_periods.csv"
    if pendulum_path.is_file():
        rows = _read_csv_rows(pendulum_path)
        pendulum_axes = _vector_columns(rows, ("axis_b_x", "axis_b_y", "axis_b_z"))
        pendulum_mass = _mean_column(rows, "mass_kg")
        pendulum_moments = compound_pendulum_moments_from_periods(
            _float_column(rows, "period_s"),
            mass=pendulum_mass,
            pivot_to_com_distance_samples=_float_column(rows, "pivot_to_com_distance_m"),
            gravity_mps2=abs(float(gravity_z)),
        )
        axes.extend(pendulum_axes)
        moments.extend(pendulum_moments.detach().cpu().tolist())

    if axes:
        fit = fit_inertia_tensor_from_axis_moments(
            axes,
            moments,
            min_eigenvalue=min_inertia_eigenvalue,
            project_to_psd=True,
        )
        cfg_updates.update(fit.to_cfg_updates())
        diagnostics["inertia"] = {
            "inertia_tensor_kg_m2": fit.inertia_tensor.detach().cpu().tolist(),
            "residual_rms_kg_m2": fit.residual_rms,
            "sample_count": fit.sample_count,
            "design_rank": fit.design_rank,
            "min_eigenvalue_before_projection": fit.min_eigenvalue_before_projection,
            "min_eigenvalue_after_projection": fit.min_eigenvalue_after_projection,
        }

    torque_path = log_dir / "rigid_body_static_buoyancy_torques.csv"
    if torque_path.is_file():
        rows = _read_csv_rows(torque_path)
        volume = volume_fit.volume if volume_fit is not None else _mean_column(rows, "volume_m3")
        density = volume_fit.water_density if volume_fit is not None else _mean_column(rows, "water_density_kg_m3")
        quats = _vector_columns(rows, ("quat_w", "quat_x", "quat_y", "quat_z"))
        torques = _vector_columns(
            rows,
            ("buoyancy_torque_b_x_nm", "buoyancy_torque_b_y_nm", "buoyancy_torque_b_z_nm"),
        )
        fit = fit_com_to_cob_offset_from_static_torques(
            quats,
            torques,
            volume=volume,
            water_density=density,
            gravity_w=(0.0, 0.0, float(gravity_z)),
        )
        cfg_updates.update(fit.to_cfg_updates())
        diagnostics["center_of_buoyancy"] = {
            "com_to_cob_offset_m": fit.com_to_cob_offset.detach().cpu().tolist(),
            "residual_rms_nm": fit.residual_rms,
            "sample_count": fit.sample_count,
            "design_rank": fit.design_rank,
            "volume_m3": volume,
            "water_density_kg_m3": density,
        }

    return StaticCalibrationPipelineResult(cfg_updates, diagnostics, source_files)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return rows


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _mean_column(rows: list[dict[str, str]], name: str) -> float:
    values = torch.tensor(_float_column(rows, name), dtype=torch.float64)
    return float(torch.mean(values).item())


def _vector_columns(rows: list[dict[str, str]], names: tuple[str, ...]) -> list[list[float]]:
    return [[float(row[name]) for name in names] for row in rows]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = fit_static_calibration_logs(
            args.log_dir,
            gravity_z=args.gravity_z,
            min_inertia_eigenvalue=args.min_inertia_eigenvalue,
        )
        _write_json(args.output, result.update_payload())
        if args.report is not None:
            _write_json(args.report, result.report_dict())
    except Exception as exc:
        print(f"Failed to fit static calibration logs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

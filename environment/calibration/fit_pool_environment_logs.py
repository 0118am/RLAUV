"""Fit water-current, pool-boundary, and free-surface updates from CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.identification.fitters import (  # noqa: E402
    fit_free_surface_effect_scales,
    fit_pool_boundary_effect_scales,
    fit_rectangular_pool_sloshing_modes,
    fit_water_current_field_grid,
    fit_water_current_process,
)
from environment.profiles.pool_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolProfileAuditOptions,
    pool_profile_calibration_log_schemas,
    validate_pool_calibration_log_directory,
)


ENVIRONMENT_LOG_FILENAMES = (
    "water_current_timeseries.csv",
    "water_current_field_samples.csv",
    "pool_boundary_effect_samples.csv",
    "free_surface_effect_samples.csv",
    "free_surface_wave_gauge.csv",
)


@dataclass(frozen=True)
class EnvironmentCalibrationPipelineResult:
    cfg_updates: dict[str, Any]
    domain_randomization_updates: dict[str, Any]
    diagnostics: dict[str, Any]
    source_files: tuple[str, ...]

    def update_payload(self) -> dict[str, Any]:
        return {
            "cfg_updates": self.cfg_updates,
            "domain_randomization_updates": self.domain_randomization_updates,
        }

    def report_dict(self) -> dict[str, Any]:
        return {
            "source_files": list(self.source_files),
            "cfg_updates": self.cfg_updates,
            "domain_randomization_updates": self.domain_randomization_updates,
            "diagnostics": self.diagnostics,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit current, boundary, and free-surface profile updates from calibration CSV logs.",
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing environment calibration CSV logs.")
    parser.add_argument("--output", type=Path, required=True, help="Output builder-compatible updates JSON path.")
    parser.add_argument("--report", type=Path, help="Optional detailed fit diagnostics JSON path.")
    parser.add_argument("--current-stages", type=int, default=1, help="Number of current curriculum stages.")
    parser.add_argument(
        "--current-grid-shape",
        type=int,
        nargs=3,
        default=(5, 5, 3),
        metavar=("NX", "NY", "NZ"),
        help="Spatial current grid shape.",
    )
    parser.add_argument(
        "--current-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Optional explicit spatial current grid bounds; inferred from samples when omitted.",
    )
    parser.add_argument("--current-k-neighbors", type=int, default=8, help="IDW neighbors for current grid fitting.")
    parser.add_argument("--current-interpolation-power", type=float, default=2.0, help="IDW interpolation power.")
    parser.add_argument(
        "--pool-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Required when pool_boundary_effect_samples.csv is present.",
    )
    parser.add_argument("--boundary-effect-distance", type=float, default=0.75, help="Boundary model distance in m.")
    parser.add_argument("--surface-z", type=float, help="Required when free_surface_effect_samples.csv is present.")
    parser.add_argument("--surface-effect-distance", type=float, default=0.5, help="Free-surface model distance in m.")
    parser.add_argument(
        "--sloshing-mode",
        type=int,
        nargs=2,
        action="append",
        metavar=("M", "N"),
        help="Rectangular-pool standing-wave mode; repeat for multiple modes.",
    )
    parser.add_argument("--sloshing-water-depth", type=float, help="Still-water depth required for wave gauges.")
    parser.add_argument("--sloshing-depth-axis-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    return parser


def fit_environment_calibration_logs(
    log_dir: Path,
    *,
    current_stage_count: int = 1,
    current_grid_shape: Sequence[int] = (5, 5, 3),
    current_bounds: Sequence[float] | None = None,
    current_k_neighbors: int = 8,
    current_interpolation_power: float = 2.0,
    pool_bounds: Sequence[float] | None = None,
    boundary_effect_distance: float = 0.75,
    surface_z: float | None = None,
    surface_effect_distance: float = 0.5,
    sloshing_modes: Sequence[Sequence[int]] | None = None,
    sloshing_water_depth: float | None = None,
    sloshing_depth_axis_sign: float = 1.0,
) -> EnvironmentCalibrationPipelineResult:
    if int(current_stage_count) != current_stage_count or int(current_stage_count) < 1:
        raise ValueError("current_stage_count must be a positive integer.")
    if int(current_k_neighbors) != current_k_neighbors or int(current_k_neighbors) < 1:
        raise ValueError("current_k_neighbors must be a positive integer.")
    if float(current_interpolation_power) <= 0.0:
        raise ValueError("current_interpolation_power must be positive.")
    if float(boundary_effect_distance) <= 0.0:
        raise ValueError("boundary_effect_distance must be positive.")
    if float(surface_effect_distance) <= 0.0:
        raise ValueError("surface_effect_distance must be positive.")
    if sloshing_water_depth is not None and float(sloshing_water_depth) <= 0.0:
        raise ValueError("sloshing_water_depth must be positive.")
    if float(sloshing_depth_axis_sign) not in (-1.0, 1.0):
        raise ValueError("sloshing_depth_axis_sign must be -1 or 1.")

    schemas = pool_profile_calibration_log_schemas(
        NOMINAL_POOL_DYNAMICS_PROFILE,
        PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            sloshing_expected=True,
            spatial_current_expected=True,
            domain_randomization_expected=False,
        ),
    )
    schema_by_filename = {schema.filename: schema for schema in schemas}
    source_files = tuple(filename for filename in ENVIRONMENT_LOG_FILENAMES if (log_dir / filename).is_file())
    if not source_files:
        raise ValueError(f"No supported environment calibration logs found in {log_dir}.")

    source_schemas = tuple(schema_by_filename[filename] for filename in source_files)
    validation = validate_pool_calibration_log_directory(log_dir, source_schemas)
    if not validation.is_valid:
        messages = "; ".join(
            f"{issue.filename}:{issue.row_number or '-'}:{issue.column or '-'} {issue.message}"
            for issue in validation.issues
            if issue.severity == "error"
        )
        raise ValueError(f"Environment calibration log validation failed: {messages}")

    cfg_updates: dict[str, Any] = {}
    domain_updates: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"validation": validation.to_dict()}

    current_path = log_dir / "water_current_timeseries.csv"
    if current_path.is_file():
        rows = _read_csv_rows(current_path)
        fit = fit_water_current_process(
            _float_column(rows, "time_s"),
            _vector_columns(rows, ("current_w_x_mps", "current_w_y_mps", "current_w_z_mps")),
        )
        cfg_updates.update(fit.to_cfg_updates())
        domain_updates.update(fit.to_domain_randomization_updates(stage_count=current_stage_count))
        diagnostics["water_current_process"] = {
            "mean_current_w_mps": fit.mean_current_w.detach().cpu().tolist(),
            "residual_std_w_mps": fit.residual_std_w.detach().cpu().tolist(),
            "tau_s": fit.tau_s,
            "estimated_alpha": fit.estimated_alpha,
            "variation_std_mps": fit.variation_std,
            "horizontal_max_mps": fit.horizontal_max,
            "vertical_max_mps": fit.vertical_max,
            "sample_count": fit.sample_count,
        }

    field_path = log_dir / "water_current_field_samples.csv"
    if field_path.is_file():
        rows = _read_csv_rows(field_path)
        measured_field_current = torch.tensor(
            _vector_columns(rows, ("current_w_x_mps", "current_w_y_mps", "current_w_z_mps")),
            dtype=torch.float32,
        )
        # Runtime composes the spatial field additively with water_current_w.
        # Store only the spatial residual when a global mean was fitted in the
        # same batch, otherwise the mean would be counted twice.
        base_current = torch.tensor(cfg_updates.get("water_current_w", [0.0, 0.0, 0.0]), dtype=torch.float32)
        fit = fit_water_current_field_grid(
            _vector_columns(rows, ("pos_x_m", "pos_y_m", "pos_z_m")),
            measured_field_current - base_current.reshape(1, 3),
            grid_shape=current_grid_shape,
            bounds=current_bounds,
            k_neighbors=current_k_neighbors,
            interpolation_power=current_interpolation_power,
        )
        cfg_updates.update(fit.to_cfg_updates())
        diagnostics["water_current_field"] = {
            "value_semantics": "spatial residual added to water_current_w",
            "subtracted_global_current_w_mps": base_current.tolist(),
            "bounds": fit.bounds.detach().cpu().tolist(),
            "grid_shape": list(fit.grid_shape),
            "sample_count": fit.sample_count,
            "k_neighbors": fit.k_neighbors,
            "interpolation_power": fit.interpolation_power,
        }

    boundary_path = log_dir / "pool_boundary_effect_samples.csv"
    if boundary_path.is_file():
        if pool_bounds is None:
            raise ValueError("pool_bounds must be provided when pool_boundary_effect_samples.csv is present.")
        rows = _read_csv_rows(boundary_path)
        cfg_updates.update(
            {
                "pool_boundary_effects_enabled": True,
                "pool_bounds": [float(value) for value in pool_bounds],
                "pool_boundary_effect_distance": float(boundary_effect_distance),
            }
        )
        diagnostics["pool_boundary"] = {}
        measured = False
        boundary_columns = {
            "damping_scale": ("pool_boundary_damping_scale", "damping_scale_at_boundary", 0),
            "added_mass_scale": ("pool_boundary_added_mass_scale", "added_mass_scale_at_boundary", 1),
            "thrust_scale": ("pool_boundary_thrust_scale", "thrust_scale_at_boundary", 2),
        }
        for column_name, (cfg_key, fit_attribute, residual_index) in boundary_columns.items():
            selected = _rows_with_value(rows, column_name)
            if not selected:
                continue
            measured = True
            kwargs = {f"{column_name}_samples": _float_column(selected, column_name)}
            fit = fit_pool_boundary_effect_scales(
                _vector_columns(selected, ("pos_x_m", "pos_y_m", "pos_z_m")),
                bounds=pool_bounds,
                effect_distance=boundary_effect_distance,
                **kwargs,
            )
            cfg_updates[cfg_key] = float(getattr(fit, fit_attribute))
            diagnostics["pool_boundary"][column_name] = {
                "scale_at_boundary": float(getattr(fit, fit_attribute)),
                "residual_rms": float(fit.residual_rms[residual_index]),
                "sample_count": int(fit.sample_count[residual_index]),
            }
        if not measured:
            raise ValueError("pool_boundary_effect_samples.csv contains no measured scale columns.")

    surface_path = log_dir / "free_surface_effect_samples.csv"
    if surface_path.is_file():
        if surface_z is None:
            raise ValueError("surface_z must be provided when free_surface_effect_samples.csv is present.")
        rows = _read_csv_rows(surface_path)
        cfg_updates.update(
            {
                "free_surface_effects_enabled": True,
                "free_surface_z": float(surface_z),
                "free_surface_effect_distance": float(surface_effect_distance),
            }
        )
        diagnostics["free_surface"] = {}
        measured = False
        surface_columns = {
            "heave_damping_scale": ("free_surface_heave_damping_scale", "heave_damping_scale", 0),
            "roll_pitch_damping_scale": ("free_surface_roll_pitch_damping_scale", "roll_pitch_damping_scale", 1),
            "added_mass_scale": ("free_surface_added_mass_scale", "added_mass_scale", 2),
            "buoyancy_scale": ("free_surface_buoyancy_scale", "buoyancy_scale", 3),
            "thrust_scale": ("free_surface_thrust_scale", "thrust_scale", 4),
        }
        for column_name, (cfg_key, fit_attribute, residual_index) in surface_columns.items():
            selected = _rows_with_value(rows, column_name)
            if not selected:
                continue
            measured = True
            kwargs = {f"{column_name}_samples": _float_column(selected, column_name)}
            fit = fit_free_surface_effect_scales(
                [[0.0, 0.0, float(row["pos_z_m"])] for row in selected],
                surface_z=surface_z,
                effect_distance=surface_effect_distance,
                **kwargs,
            )
            cfg_updates[cfg_key] = float(getattr(fit, fit_attribute))
            diagnostics["free_surface"][column_name] = {
                "scale_at_surface": float(getattr(fit, fit_attribute)),
                "residual_rms": float(fit.residual_rms[residual_index]),
                "sample_count": int(fit.sample_count[residual_index]),
            }
        if not measured:
            raise ValueError("free_surface_effect_samples.csv contains no measured scale columns.")

    wave_path = log_dir / "free_surface_wave_gauge.csv"
    if wave_path.is_file():
        if surface_z is None or pool_bounds is None or sloshing_water_depth is None or not sloshing_modes:
            raise ValueError(
                "free_surface_wave_gauge.csv requires surface_z, pool_bounds, sloshing_water_depth, and sloshing_modes."
            )
        rows = _read_csv_rows(wave_path)
        fit = fit_rectangular_pool_sloshing_modes(
            _float_column(rows, "time_s"),
            _vector_columns(rows, ("gauge_x_m", "gauge_y_m")),
            _float_column(rows, "surface_z_m"),
            base_surface_z=surface_z,
            pool_bounds=pool_bounds,
            water_depth=sloshing_water_depth,
            mode_numbers=sloshing_modes,
            depth_axis_sign=sloshing_depth_axis_sign,
        )
        cfg_updates.update(fit.to_cfg_updates())
        diagnostics["free_surface_sloshing"] = {
            "mode_numbers": fit.mode_numbers.detach().cpu().tolist(),
            "amplitudes_m": fit.amplitudes_m.detach().cpu().tolist(),
            "phases_rad": fit.phases_rad.detach().cpu().tolist(),
            "angular_frequencies_rad_s": fit.angular_frequencies_rad_s.detach().cpu().tolist(),
            "residual_rms_m": fit.residual_rms,
            "sample_count": fit.sample_count,
            "design_rank": fit.design_rank,
        }

    return EnvironmentCalibrationPipelineResult(cfg_updates, domain_updates, diagnostics, source_files)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return rows


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _vector_columns(rows: list[dict[str, str]], names: tuple[str, ...]) -> list[list[float]]:
    return [[float(row[name]) for name in names] for row in rows]


def _rows_with_value(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(name, "").strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = fit_environment_calibration_logs(
            args.log_dir,
            current_stage_count=args.current_stages,
            current_grid_shape=args.current_grid_shape,
            current_bounds=args.current_bounds,
            current_k_neighbors=args.current_k_neighbors,
            current_interpolation_power=args.current_interpolation_power,
            pool_bounds=args.pool_bounds,
            boundary_effect_distance=args.boundary_effect_distance,
            surface_z=args.surface_z,
            surface_effect_distance=args.surface_effect_distance,
            sloshing_modes=args.sloshing_mode,
            sloshing_water_depth=args.sloshing_water_depth,
            sloshing_depth_axis_sign=args.sloshing_depth_axis_sign,
        )
        _write_json(args.output, result.update_payload())
        if args.report is not None:
            _write_json(args.report, result.report_dict())
    except Exception as exc:
        print(f"Failed to fit environment calibration logs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

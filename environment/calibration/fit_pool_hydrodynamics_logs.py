"""Fit 6-DOF added mass and damping updates from calibrated motion/wrench CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.identification.fitters import (  # noqa: E402
    calculate_damping_dissipated_power,
    damping_is_dissipative_for_samples,
    finite_difference,
    fit_diagonal_added_mass_linear_quadratic_damping,
    fit_full_matrix_added_mass_linear_quadratic_damping,
    fit_high_order_hydrodynamic_residual_damping,
    fit_speed_dependent_damping_scales,
    project_added_mass_to_physical,
    project_linear_damping_to_dissipative,
)
from environment.profiles.pool_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolDynamicsProfile,
    PoolProfileAuditOptions,
    load_pool_dynamics_profile_json,
    pool_profile_calibration_log_schemas,
    validate_pool_calibration_log_directory,
)
from robot.dynamics.rigid_body import inertia_matrix_tensor  # noqa: E402


MOTION_LOG_FILENAME = "hydrodynamics_motion_wrench_log.csv"
NU_COLUMNS = (
    "nu_r_u_mps",
    "nu_r_v_mps",
    "nu_r_w_mps",
    "nu_r_p_radps",
    "nu_r_q_radps",
    "nu_r_r_radps",
)
WRENCH_COLUMNS = (
    "wrench_x_n",
    "wrench_y_n",
    "wrench_z_n",
    "wrench_k_nm",
    "wrench_m_nm",
    "wrench_n_nm",
)
ACCEL_COLUMNS = (
    "nu_r_dot_u_mps2",
    "nu_r_dot_v_mps2",
    "nu_r_dot_w_mps2",
    "nu_r_dot_p_radps2",
    "nu_r_dot_q_radps2",
    "nu_r_dot_r_radps2",
)


@dataclass(frozen=True)
class HydrodynamicsCalibrationPipelineResult:
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
        description="Fit 6-DOF added mass and damping updates from motion/wrench calibration logs.",
    )
    parser.add_argument("log_dir", type=Path, help=f"Directory containing {MOTION_LOG_FILENAME}.")
    parser.add_argument("--base-profile", type=Path, help="Profile JSON providing rigid-body mass and inertia.")
    parser.add_argument(
        "--fit-mode",
        choices=("full", "diagonal", "high-order-residual"),
        default="full",
        help=(
            "'high-order-residual' expects wrench_* columns to contain CFD/experiment residual wrench "
            "(measured fluid wrench minus nominal Fossen wrench), not the total applied wrench."
        ),
    )
    parser.add_argument("--regularization", type=float, default=0.0, help="Ridge regularization for full-matrix fit.")
    parser.add_argument("--min-added-mass-eigenvalue", type=float, default=0.0)
    parser.add_argument("--min-linear-damping-eigenvalue", type=float, default=0.0)
    parser.add_argument("--allow-rank-deficient", action="store_true")
    parser.add_argument("--allow-nonpassive", action="store_true")
    parser.add_argument(
        "--damping-speed-points",
        type=float,
        nargs="+",
        help="Fit per-DOF damping scales at these speed-bin centers; requires diagonal fit mode.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output builder-compatible updates JSON path.")
    parser.add_argument("--report", type=Path, help="Optional detailed fit diagnostics JSON path.")
    return parser


def fit_hydrodynamics_calibration_logs(
    log_dir: Path,
    *,
    base_profile: PoolDynamicsProfile | None = None,
    fit_mode: str = "full",
    regularization: float = 0.0,
    min_added_mass_eigenvalue: float = 0.0,
    min_linear_damping_eigenvalue: float = 0.0,
    allow_rank_deficient: bool = False,
    allow_nonpassive: bool = False,
    damping_speed_points: list[float] | None = None,
) -> HydrodynamicsCalibrationPipelineResult:
    if fit_mode not in {"full", "diagonal", "high-order-residual"}:
        raise ValueError("fit_mode must be 'full', 'diagonal', or 'high-order-residual'.")
    if float(regularization) < 0.0:
        raise ValueError("regularization must be non-negative.")
    if float(min_added_mass_eigenvalue) < 0.0 or float(min_linear_damping_eigenvalue) < 0.0:
        raise ValueError("Projection eigenvalue floors must be non-negative.")
    if damping_speed_points is not None and fit_mode != "diagonal":
        raise ValueError(
            "Speed-dependent damping calibration currently requires fit_mode='diagonal' so its "
            "per-DOF scale curves match the identified coefficient parameterization."
        )

    profile = NOMINAL_POOL_DYNAMICS_PROFILE if base_profile is None else base_profile
    profile.validate()
    schemas = pool_profile_calibration_log_schemas(
        NOMINAL_POOL_DYNAMICS_PROFILE,
        PoolProfileAuditOptions(domain_randomization_expected=False),
    )
    schema = next(schema for schema in schemas if schema.filename == MOTION_LOG_FILENAME)
    validation = validate_pool_calibration_log_directory(log_dir, (schema,))
    if not validation.is_valid:
        messages = "; ".join(
            f"{issue.filename}:{issue.row_number or '-'}:{issue.column or '-'} {issue.message}"
            for issue in validation.issues
            if issue.severity == "error"
        )
        raise ValueError(f"Hydrodynamics calibration log validation failed: {messages}")

    rows = _read_csv_rows(log_dir / MOTION_LOG_FILENAME)
    time_s = torch.tensor(_float_column(rows, "time_s"), dtype=torch.float32)
    nu_r = torch.tensor(_vector_columns(rows, NU_COLUMNS), dtype=torch.float32)
    applied_wrench = torch.tensor(_vector_columns(rows, WRENCH_COLUMNS), dtype=torch.float32)
    relative_acceleration = _optional_acceleration(rows)
    acceleration_for_rank = relative_acceleration if relative_acceleration is not None else finite_difference(time_s, nu_r)
    rigid_mass_matrix = _rigid_body_mass_matrix(profile)

    cfg_updates: dict[str, Any]
    domain_randomization_updates: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {
        "validation": validation.to_dict(),
        "fit_mode": fit_mode,
        "velocity_reference": "root/COM body-frame velocity, matching AUV root_lin_vel_b/root_ang_vel_b",
        "acceleration_source": "csv" if relative_acceleration is not None else "finite_difference",
        "rigid_body_mass_matrix": rigid_mass_matrix.detach().cpu().tolist(),
    }

    if fit_mode == "high-order-residual":
        speed = torch.linalg.vector_norm(nu_r, dim=1, keepdim=True)
        design = torch.cat((nu_r, speed * nu_r, speed.square() * nu_r), dim=1)
        design_rank = int(torch.linalg.matrix_rank(design).item())
        diagnostics["design_rank"] = design_rank
        diagnostics["required_design_rank"] = 18
        diagnostics["wrench_interpretation"] = (
            "CFD/experiment fluid-wrench residual after subtracting the nominal Fossen prediction"
        )
        if design_rank < 18 and not allow_rank_deficient:
            raise ValueError(
                f"High-order residual design rank is {design_rank}, below 18; collect richer multi-axis CFD "
                "or tank excitation, or pass allow_rank_deficient=True for diagnosis only."
            )
        fit = fit_high_order_hydrodynamic_residual_damping(
            nu_r,
            applied_wrench,
            regularization=regularization,
        )
        cfg_updates = fit.to_cfg_updates(enabled=True)
        diagnostics["fit"] = {
            "sample_count": fit.sample_count,
            "regularization": fit.regularization,
            "residual_rms_by_dof": fit.residual_rms.detach().cpu().tolist(),
            "raw_linear_damping": fit.raw_linear_damping.detach().cpu().tolist(),
            "raw_quadratic_damping": fit.raw_quadratic_damping.detach().cpu().tolist(),
            "raw_cubic_damping": fit.raw_cubic_damping.detach().cpu().tolist(),
            "factor_parameterization": "D_k = L_k @ L_k.T",
        }
    elif fit_mode == "full":
        design = torch.cat((acceleration_for_rank, nu_r, torch.abs(nu_r) * nu_r), dim=1)
        design_rank = int(torch.linalg.matrix_rank(design).item())
        diagnostics["design_rank"] = design_rank
        diagnostics["required_design_rank"] = 18
        if design_rank < 18 and not allow_rank_deficient:
            raise ValueError(
                f"Full-matrix hydrodynamics design rank is {design_rank}, below 18; "
                "collect richer multi-axis excitation or pass allow_rank_deficient=True."
            )

        fit = fit_full_matrix_added_mass_linear_quadratic_damping(
            time_s,
            nu_r,
            applied_wrench,
            rigid_body_inertia=rigid_mass_matrix,
            relative_acceleration=relative_acceleration,
            regularization=regularization,
            symmetrize_added_mass=True,
        )
        added_projection = project_added_mass_to_physical(
            fit.added_mass,
            min_eigenvalue=min_added_mass_eigenvalue,
        )
        linear_projection = project_linear_damping_to_dissipative(
            fit.linear_damping,
            min_eigenvalue=min_linear_damping_eigenvalue,
            preserve_skew=True,
        )
        added_mass = added_projection.projected_matrix
        linear_damping = linear_projection.projected_matrix
        quadratic_damping = fit.quadratic_damping
        dissipated_power = calculate_damping_dissipated_power(
            nu_r,
            linear_damping=linear_damping,
            quadratic_damping=quadratic_damping,
        )
        sampled_passive = damping_is_dissipative_for_samples(
            nu_r,
            linear_damping=linear_damping,
            quadratic_damping=quadratic_damping,
            tolerance=1.0e-5,
        )
        if not sampled_passive and not allow_nonpassive:
            raise ValueError(
                "Fitted full-matrix damping injects energy for measured velocity samples after linear projection; "
                "collect richer data, constrain the fit, or pass allow_nonpassive=True for diagnosis only."
            )

        cfg_updates = {
            "added_mass_diag": added_mass.detach().cpu().tolist(),
            "linear_damping": linear_damping.detach().cpu().tolist(),
            "quadratic_damping": quadratic_damping.detach().cpu().tolist(),
        }
        diagnostics["fit"] = {
            "sample_count": fit.sample_count,
            "regularization": fit.regularization,
            "residual_rms_by_dof": fit.residual_rms.detach().cpu().tolist(),
            "raw_added_mass": fit.added_mass.detach().cpu().tolist(),
            "raw_linear_damping": fit.linear_damping.detach().cpu().tolist(),
            "quadratic_damping": quadratic_damping.detach().cpu().tolist(),
        }
        diagnostics["added_mass_projection"] = {
            "original_min_eigenvalue": added_projection.original_min_eigenvalue,
            "projected_min_eigenvalue": added_projection.projected_min_eigenvalue,
            "correction_frobenius_norm": added_projection.correction_frobenius_norm,
        }
        diagnostics["linear_damping_projection"] = {
            "original_min_eigenvalue": linear_projection.original_min_eigenvalue,
            "projected_min_eigenvalue": linear_projection.projected_min_eigenvalue,
            "correction_frobenius_norm": linear_projection.correction_frobenius_norm,
            "preserved_skew": linear_projection.preserved_skew,
        }
        diagnostics["sampled_passivity"] = {
            "is_passive": sampled_passive,
            "minimum_dissipated_power": float(torch.min(dissipated_power).item()),
            "maximum_dissipated_power": float(torch.max(dissipated_power).item()),
        }
    else:
        rigid_diagonal = torch.diagonal(rigid_mass_matrix)
        fit = fit_diagonal_added_mass_linear_quadratic_damping(
            time_s,
            nu_r,
            applied_wrench,
            rigid_body_inertia=rigid_diagonal,
            relative_acceleration=relative_acceleration,
            nonnegative=True,
        )
        cfg_updates = fit.to_cfg_updates()
        sampled_passive = damping_is_dissipative_for_samples(
            nu_r,
            linear_damping=fit.linear_damping,
            quadratic_damping=fit.quadratic_damping,
        )
        diagnostics["fit"] = {
            "sample_count_by_dof": fit.sample_count.detach().cpu().tolist(),
            "residual_rms_by_dof": fit.residual_rms.detach().cpu().tolist(),
            "effective_inertia": fit.effective_inertia.detach().cpu().tolist(),
        }
        diagnostics["sampled_passivity"] = {"is_passive": sampled_passive}
        if damping_speed_points is not None:
            speed_fit = fit_speed_dependent_damping_scales(
                time_s,
                nu_r,
                applied_wrench,
                effective_mass=fit.effective_inertia,
                speed_points=damping_speed_points,
                nominal_linear_damping=fit.linear_damping,
                nominal_quadratic_damping=fit.quadratic_damping,
                relative_acceleration=relative_acceleration,
                require_identifiable=True,
            )
            cfg_updates.update(speed_fit.to_cfg_updates(damping_speed_points))
            domain_randomization_updates.update(speed_fit.to_domain_randomization_updates())
            diagnostics["speed_dependent_damping"] = {
                "speed_points": [float(value) for value in damping_speed_points],
                "linear_scales": speed_fit.linear_scales.detach().cpu().tolist(),
                "quadratic_scales": speed_fit.quadratic_scales.detach().cpu().tolist(),
                "linear_scale_std": speed_fit.linear_scale_std.detach().cpu().tolist(),
                "quadratic_scale_std": speed_fit.quadratic_scale_std.detach().cpu().tolist(),
                "residual_rms_by_bin_dof": speed_fit.residual_rms.detach().cpu().tolist(),
                "sample_count_by_bin_dof": speed_fit.sample_count.detach().cpu().tolist(),
                "design_rank_by_bin_dof": speed_fit.design_rank.detach().cpu().tolist(),
                "required_rank_by_bin_dof": speed_fit.required_rank.detach().cpu().tolist(),
                "nominal_source": "same-log diagonal added-mass/damping fit",
            }

    return HydrodynamicsCalibrationPipelineResult(
        cfg_updates,
        diagnostics,
        (MOTION_LOG_FILENAME,),
        domain_randomization_updates,
    )


def _rigid_body_mass_matrix(profile: PoolDynamicsProfile) -> torch.Tensor:
    matrix = torch.zeros((6, 6), dtype=torch.float32)
    matrix[0:3, 0:3] = torch.eye(3, dtype=torch.float32) * float(profile.rigid_body.mass)
    matrix[3:6, 3:6] = inertia_matrix_tensor(
        profile.rigid_body.inertia_diag,
        torch.device("cpu"),
        torch.float32,
    )
    return matrix


def _optional_acceleration(rows: list[dict[str, str]]) -> torch.Tensor | None:
    header = set(rows[0])
    present = [name in header for name in ACCEL_COLUMNS]
    if any(present) and not all(present):
        raise ValueError("Acceleration columns must be all present or all absent.")
    if not any(present):
        return None
    populated = [[bool(row.get(name, "").strip()) for name in ACCEL_COLUMNS] for row in rows]
    if all(not any(row_flags) for row_flags in populated):
        return None
    if not all(all(row_flags) for row_flags in populated):
        raise ValueError("Acceleration columns must be fully populated for every row or left entirely empty.")
    return torch.tensor(_vector_columns(rows, ACCEL_COLUMNS), dtype=torch.float32)


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
        result = fit_hydrodynamics_calibration_logs(
            args.log_dir,
            base_profile=base_profile,
            fit_mode=args.fit_mode,
            regularization=args.regularization,
            min_added_mass_eigenvalue=args.min_added_mass_eigenvalue,
            min_linear_damping_eigenvalue=args.min_linear_damping_eigenvalue,
            allow_rank_deficient=args.allow_rank_deficient,
            allow_nonpassive=args.allow_nonpassive,
            damping_speed_points=args.damping_speed_points,
        )
        _write_json(args.output, result.update_payload())
        if args.report is not None:
            _write_json(args.report, result.report_dict())
    except Exception as exc:
        print(f"Failed to fit hydrodynamics calibration logs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

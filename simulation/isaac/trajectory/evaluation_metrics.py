"""Scalar summaries derived from trajectory evaluation logs."""

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from simulation.isaac.trajectory.evaluation_cases import sanitize_evaluation_label


def _finite_statistic(values: Any, reducer: Callable[[np.ndarray], Any]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(reducer(finite))


def _reference_metrics(log: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    path_lengths: list[float] = []
    speed_p95_by_env: list[float] = []
    for _, curve_rows in log.groupby("env_id", sort=True):
        ordered = curve_rows.sort_values("time")
        positions = ordered[["desired_x", "desired_y", "desired_z"]].to_numpy(dtype=np.float64)
        path_lengths.append(float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
        speed_p95_by_env.append(float(np.quantile(ordered["target_speed_mps"].to_numpy(), 0.95)))

    speed_max = float(log["target_speed_mps"].max())
    acceleration_max = float(log["target_acceleration_mps2"].max())
    jerk_max = float(log["target_jerk_mps3"].max())
    orientation_rate_max = float(log["target_orientation_rate_radps"].max())
    within_envelope = (
        speed_max <= float(cfg.trajectory_max_speed_mps) * 1.01
        and acceleration_max <= float(cfg.trajectory_max_acceleration_mps2) * 1.01
        and jerk_max <= float(cfg.trajectory_max_jerk_mps3) * 1.01
        and orientation_rate_max <= float(cfg.trajectory_max_orientation_rate_radps) * 1.01
    )
    reference_valid = (
        bool(path_lengths)
        and min(path_lengths) >= 0.10
        and min(speed_p95_by_env) >= 0.05
        and within_envelope
    )
    requested_periods = [
        float(value) for value in log.groupby("env_id", sort=True)["requested_period_s"].first().tolist()
    ]
    effective_periods = [
        float(value) for value in log.groupby("env_id", sort=True)["effective_period_s"].first().tolist()
    ]
    return {
        "reference_generator_version": str(cfg.trajectory_generator_version),
        "reference_valid": int(reference_valid),
        "reference_within_kinematic_envelope": int(within_envelope),
        "reference_path_length_mean_m": float(np.mean(path_lengths)),
        "reference_path_length_m_by_env_json": json.dumps(path_lengths),
        "min_reference_path_length_m": float(min(path_lengths)),
        "target_speed_p95_mps_by_env_json": json.dumps(speed_p95_by_env),
        "min_curve_target_speed_p95_mps": float(min(speed_p95_by_env)),
        "target_speed_mean_mps": float(log["target_speed_mps"].mean()),
        "target_speed_p95_mps": float(log["target_speed_mps"].quantile(0.95)),
        "target_speed_max_mps": speed_max,
        "target_acceleration_mean_mps2": float(log["target_acceleration_mps2"].mean()),
        "target_acceleration_p95_mps2": float(log["target_acceleration_mps2"].quantile(0.95)),
        "target_acceleration_max_mps2": acceleration_max,
        "target_jerk_mean_mps3": float(log["target_jerk_mps3"].mean()),
        "target_jerk_p95_mps3": float(log["target_jerk_mps3"].quantile(0.95)),
        "target_jerk_max_mps3": jerk_max,
        "target_curvature_mean_m_inv": float(log["target_curvature_m_inv"].mean()),
        "target_curvature_p95_m_inv": float(log["target_curvature_m_inv"].quantile(0.95)),
        "target_curvature_max_m_inv": float(log["target_curvature_m_inv"].max()),
        "target_orientation_rate_mean_radps": float(log["target_orientation_rate_radps"].mean()),
        "target_orientation_rate_p95_radps": float(log["target_orientation_rate_radps"].quantile(0.95)),
        "target_orientation_rate_max_radps": orientation_rate_max,
        "requested_period_mean_s": float(log["requested_period_s"].mean()),
        "requested_period_s_by_env_json": json.dumps(requested_periods),
        "effective_period_mean_s": float(log["effective_period_s"].mean()),
        "effective_period_max_s": float(log["effective_period_s"].max()),
        "effective_period_s_by_env_json": json.dumps(effective_periods),
        "retimed_curve_fraction": float(log.groupby("env_id")["trajectory_retimed"].first().mean()),
    }


def _tracking_metrics(log: pd.DataFrame) -> dict[str, Any]:
    position_errors = log["position_error"].to_numpy()
    velocity_errors = log["velocity_error"].to_numpy()
    heading_mean = _finite_statistic(log["command_heading_error_rad"], np.mean)
    sideslip_mean = _finite_statistic(log["motion_sideslip_error_rad"], np.mean)
    return {
        "position_rmse": float(np.sqrt(np.mean(position_errors**2))),
        "position_error_p95": float(np.quantile(position_errors, 0.95)),
        "position_mae": float(np.mean(position_errors)),
        "max_position_error": float(np.max(position_errors)),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_errors**2))),
        "mean_command_heading_error_deg": None if heading_mean is None else float(np.degrees(heading_mean)),
        "mean_motion_sideslip_error_deg": None if sideslip_mean is None else float(np.degrees(sideslip_mean)),
        "mean_reward_per_step": float(log["reward"].mean()),
    }


def _actuator_metrics(log: pd.DataFrame) -> dict[str, Any]:
    force_columns = [
        "realized_wrench_force_x_n",
        "realized_wrench_force_y_n",
        "realized_wrench_force_z_n",
    ]
    return {
        "mean_action_rms": float(log["action_rms"].mean()),
        "mean_action_rate_rms": float(log["action_rate_rms"].mean()),
        "raw_policy_action_clip_fraction": float(log["raw_policy_action_clip_fraction"].mean()),
        "mean_requested_to_applied_action_rms": _finite_statistic(
            log["requested_to_applied_action_rms"], np.mean
        ),
        "mean_applied_action_rate_rms": _finite_statistic(log["applied_action_rate_rms"], np.mean),
        "mean_realized_thruster_force_abs_n": _finite_statistic(
            log["realized_thruster_force_abs_mean_n"], np.mean
        ),
        "max_realized_thruster_force_abs_n": _finite_statistic(
            log["realized_thruster_force_abs_max_n"], np.max
        ),
        "mean_realized_wrench_force_norm_n": _finite_statistic(
            np.linalg.norm(log[force_columns].to_numpy(), axis=1), np.mean
        ),
    }


def _domain_metrics(domain_samples: pd.DataFrame) -> dict[str, float]:
    return {
        "thruster_time_constant_mean_s": float(domain_samples["sampled_thruster_time_constant_s"].mean()),
        "thruster_command_delay_mean_s": float(domain_samples["sampled_thruster_delay_s"].mean()),
        "thruster_command_delay_max_s": float(domain_samples["sampled_thruster_delay_s"].max()),
        "thruster_command_rate_limit_mean_per_s": float(
            domain_samples["sampled_thruster_max_command_rate_per_s"].mean()
        ),
        "thruster_command_resolution_mean": float(
            domain_samples["sampled_thruster_command_resolution_mean"].mean()
        ),
        "thruster_command_dropout_probability_mean": float(
            domain_samples["sampled_thruster_command_dropout_probability_mean"].mean()
        ),
    }


def _disturbance_metrics(log: pd.DataFrame, args: Any) -> dict[str, float | str | int]:
    current_norm = np.linalg.norm(
        log[["water_current_x", "water_current_y", "water_current_z"]].to_numpy(),
        axis=1,
    )
    return {
        "mean_water_current_norm": float(np.mean(current_norm)),
        "max_water_current_norm": float(np.max(current_norm)),
        "eval_damping_scale": float(args.eval_damping_scale),
        "eval_thruster_scale": float(args.eval_thruster_scale),
        "eval_thruster_tau_scale": float(args.eval_thruster_tau_scale),
        "evaluation_label": sanitize_evaluation_label(args.evaluation_label),
        "eval_disturbance_stage": int(args.eval_disturbance_stage),
    }


def _failure_metrics(
    log: pd.DataFrame,
    termination_events: int,
    any_failure: torch.Tensor,
    first_failure_time_s: torch.Tensor,
) -> dict[str, float | int | None]:
    return {
        "failure_events": int(termination_events),
        "any_failure_rate": float(any_failure.to(dtype=torch.float32).mean().cpu().item()),
        "terminations_per_episode": float(termination_events / max(1, int(log["env_id"].nunique()))),
        "mean_time_to_first_failure_s": (
            float(torch.nanmean(first_failure_time_s).cpu().item()) if bool(torch.any(any_failure)) else None
        ),
        "survival_rate": float((~any_failure).to(dtype=torch.float32).mean().cpu().item()),
    }


def build_evaluation_summary(
    log: pd.DataFrame,
    domain_samples: pd.DataFrame,
    env: Any,
    args: Any,
    disturbance_label: str,
    *,
    termination_events: int,
    any_failure: torch.Tensor,
    first_failure_time_s: torch.Tensor,
) -> dict[str, Any]:
    cfg = env.unwrapped.cfg
    summary: dict[str, Any] = {
        "controller": args.controller,
        "trajectory": args.trajectory,
        "reward_profile": args.reward_profile,
        "seed": int(cfg.seed),
        "environment_profile_name": cfg.environment_profile_name,
        "domain_randomization_spec_name": getattr(cfg, "domain_randomization_spec_name", None) or "",
        "disturbance": disturbance_label or "nominal",
        "num_curves": int(log["env_id"].nunique()),
    }
    summary.update(_domain_metrics(domain_samples))
    summary.update(_reference_metrics(log, cfg))
    summary.update(_tracking_metrics(log))
    summary.update(_actuator_metrics(log))
    summary.update(_disturbance_metrics(log, args))
    summary.update(_failure_metrics(log, termination_events, any_failure, first_failure_time_s))
    return summary


def write_evaluation_summary(summary: dict[str, Any], path: str) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)

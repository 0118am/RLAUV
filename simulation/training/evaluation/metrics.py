"""Scalar summaries derived from policy-evaluation logs."""

import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from robot.dynamics.parameters import AUV
from simulation.training.evaluation.config import sanitize_evaluation_label
from simulation.training.evaluation.campaign import evaluation_paths
from simulation.training.recipe import ExperimentSpec
from simulation.training.campaign import checkpoint_iter


def _finite_statistic(values: Any, reducer: Callable[[np.ndarray], Any]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(reducer(finite))


def _reference_metrics(log: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    evaluated_distances: list[float] = []
    cycle_lengths: list[float] = []
    speed_p95_by_env: list[float] = []
    for _, curve_rows in log.groupby("env_id", sort=True):
        ordered = curve_rows.sort_values("time")
        positions = ordered[["desired_x", "desired_y", "desired_z"]].to_numpy(dtype=np.float64)
        evaluated_distances.append(float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
        speed_p95_by_env.append(float(np.quantile(ordered["target_speed_mps"].to_numpy(), 0.95)))
        effective_period_s = float(ordered["effective_period_s"].iloc[0])
        cycle_end_time_s = effective_period_s + 0.5 * float(
            cfg.trajectory_startup_duration_s
        )
        if float(ordered["time"].iloc[-1]) >= cycle_end_time_s:
            cycle_positions = ordered.loc[
                ordered["time"] <= cycle_end_time_s,
                ["desired_x", "desired_y", "desired_z"],
            ].to_numpy(dtype=np.float64)
            cycle_lengths.append(
                float(np.sum(np.linalg.norm(np.diff(cycle_positions, axis=0), axis=1)))
            )

    speed_max = float(log["target_speed_mps"].max())
    acceleration_max = float(log["target_acceleration_mps2"].max())
    jerk_max = float(log["target_jerk_mps3"].max())
    yaw_rate_max = float(log["target_yaw_rate_radps"].max())
    within_envelope = (
        speed_max <= float(cfg.trajectory_max_speed_mps) * 1.01
        and acceleration_max <= float(cfg.trajectory_max_acceleration_mps2) * 1.01
        and jerk_max <= float(cfg.trajectory_max_jerk_mps3) * 1.01
        and yaw_rate_max <= float(cfg.trajectory_max_yaw_rate_radps) * 1.01
    )
    reference_valid = (
        bool(evaluated_distances)
        and min(evaluated_distances) >= 0.10
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
        "reference_evaluated_distance_mean_m": float(np.mean(evaluated_distances)),
        "reference_evaluated_distance_m_by_env_json": json.dumps(evaluated_distances),
        "min_reference_evaluated_distance_m": float(min(evaluated_distances)),
        "reference_complete_cycle_fraction": float(
            len(cycle_lengths) / len(evaluated_distances)
        ),
        "reference_cycle_length_mean_m": (
            None if not cycle_lengths else float(np.mean(cycle_lengths))
        ),
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
        "target_yaw_rate_mean_radps": float(log["target_yaw_rate_radps"].mean()),
        "target_yaw_rate_p95_radps": float(log["target_yaw_rate_radps"].quantile(0.95)),
        "target_yaw_rate_max_radps": yaw_rate_max,
        "requested_period_mean_s": float(log["requested_period_s"].mean()),
        "requested_period_s_by_env_json": json.dumps(requested_periods),
        "effective_period_mean_s": float(log["effective_period_s"].mean()),
        "effective_period_max_s": float(log["effective_period_s"].max()),
        "effective_period_s_by_env_json": json.dumps(effective_periods),
        "retimed_curve_fraction": float(log.groupby("env_id")["trajectory_retimed"].first().mean()),
    }


def _window_tracking_metrics(log: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    position_error_w = (
        log[["desired_x", "desired_y", "desired_z"]].to_numpy(dtype=np.float64)
        - log[["true_x", "true_y", "true_z"]].to_numpy(dtype=np.float64)
    )
    velocity_error_w = (
        log[["desired_vx", "desired_vy", "desired_vz"]].to_numpy(dtype=np.float64)
        - log[["true_vx", "true_vy", "true_vz"]].to_numpy(dtype=np.float64)
    )
    position_error = np.linalg.norm(position_error_w, axis=1)
    velocity_error = np.linalg.norm(velocity_error_w, axis=1)
    bias = np.mean(position_error_w, axis=0)
    target_velocity = log[["desired_vx", "desired_vy", "desired_vz"]].to_numpy(
        dtype=np.float64
    )
    target_speed = np.linalg.norm(target_velocity, axis=1)
    moving = target_speed > 1.0e-6
    tangent_error = np.sum(
        position_error_w[moving] * target_velocity[moving] / target_speed[moving, None],
        axis=1,
    )
    cross_track_error = np.sqrt(
        np.maximum(
            position_error[moving] ** 2 - tangent_error**2,
            0.0,
        )
    )
    heading_mean = _finite_statistic(log["nose_to_target_heading_angle_rad"], np.mean)
    heading_p95 = _finite_statistic(
        log["nose_to_target_heading_angle_rad"], lambda x: np.quantile(x, 0.95)
    )
    motion_angle_mean = _finite_statistic(log["nose_to_motion_heading_angle_rad"], np.mean)
    motion_angle_p95 = _finite_statistic(
        log["nose_to_motion_heading_angle_rad"], lambda x: np.quantile(x, 0.95)
    )
    attitude_error = log["attitude_error"].to_numpy(dtype=np.float64)
    curvature = log["target_curvature_m_inv"].to_numpy(dtype=np.float64)
    curvature_correlation = (
        None
        if np.std(curvature) == 0.0 or np.std(position_error) == 0.0
        else float(np.corrcoef(curvature, position_error)[0, 1])
    )
    result: dict[str, Any] = {
        "position_rmse": float(np.sqrt(np.mean(position_error**2))),
        "position_error_p95": float(np.quantile(position_error, 0.95)),
        "position_mae": float(np.mean(position_error)),
        "max_position_error": float(np.max(position_error)),
        "position_bias_x_m": float(bias[0]),
        "position_bias_y_m": float(bias[1]),
        "position_bias_z_m": float(bias[2]),
        "position_bias_norm_m": float(np.linalg.norm(bias)),
        "position_x_rmse_m": float(np.sqrt(np.mean(position_error_w[:, 0] ** 2))),
        "position_y_rmse_m": float(np.sqrt(np.mean(position_error_w[:, 1] ** 2))),
        "position_z_rmse_m": float(np.sqrt(np.mean(position_error_w[:, 2] ** 2))),
        "tangential_position_error_rmse_m": float(np.sqrt(np.mean(tangent_error**2))),
        "cross_track_position_error_rmse_m": float(np.sqrt(np.mean(cross_track_error**2))),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_error**2))),
        "velocity_x_rmse_mps": float(np.sqrt(np.mean(velocity_error_w[:, 0] ** 2))),
        "velocity_y_rmse_mps": float(np.sqrt(np.mean(velocity_error_w[:, 1] ** 2))),
        "velocity_z_rmse_mps": float(np.sqrt(np.mean(velocity_error_w[:, 2] ** 2))),
        "attitude_error_rmse_deg": float(np.degrees(np.sqrt(np.mean(attitude_error**2)))),
        "attitude_error_p95_deg": float(np.degrees(np.quantile(attitude_error, 0.95))),
        "mean_nose_to_target_heading_angle_deg": None if heading_mean is None else float(np.degrees(heading_mean)),
        "nose_to_target_heading_angle_p95_deg": None if heading_p95 is None else float(np.degrees(heading_p95)),
        "mean_nose_to_motion_heading_angle_deg": None if motion_angle_mean is None else float(np.degrees(motion_angle_mean)),
        "nose_to_motion_heading_angle_p95_deg": None if motion_angle_p95 is None else float(np.degrees(motion_angle_p95)),
        "position_error_curvature_correlation": curvature_correlation,
        "mean_reward_per_step": float(log["reward"].mean()),
    }
    return {f"{prefix}{name}": value for name, value in result.items()}


def _tracking_metrics(
    log: pd.DataFrame,
    cfg: Any,
    domain_samples: pd.DataFrame,
) -> dict[str, Any]:
    # The target finishes its smooth startup first; three actuator time
    # constants plus the configured sensor delay then cover 95% response.
    steady_state_start_s = (
        float(cfg.trajectory_startup_duration_s)
        + 3.0 * float(domain_samples["sampled_thruster_time_constant_s"].max())
        + float(domain_samples["pose_sensor_delay_s"].max())
    )
    steady = log.loc[log["time"] >= steady_state_start_s]
    result = _window_tracking_metrics(log)
    result.update(_window_tracking_metrics(steady, prefix="steady_"))
    result["steady_state_start_s"] = steady_state_start_s
    return result


def _actuator_metrics(log: pd.DataFrame) -> dict[str, Any]:
    thruster_force_columns = [
        "thruster_wrench_b_force_x_n",
        "thruster_wrench_b_force_y_n",
        "thruster_wrench_b_force_z_n",
    ]
    physx_force_columns = [
        "physx_applied_wrench_b_force_x_n",
        "physx_applied_wrench_b_force_y_n",
        "physx_applied_wrench_b_force_z_n",
    ]
    action_columns = sorted(
        (name for name in log.columns if re.fullmatch(r"action_\d+", name)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    actions = log[action_columns].to_numpy(dtype=np.float64)
    deadband = AUV.thruster_pwm_deadband_us / AUV.thruster_pwm_half_range_us
    return {
        "mean_action_rms": float(log["action_rms"].mean()),
        "mean_action_rate_rms_per_s": float(log["action_rate_rms_per_s"].mean()),
        "mean_action_acceleration_rms_per_s2": float(
            log["action_acceleration_rms_per_s2"].mean()
        ),
        "mean_action_saturation_fraction": float(log["action_saturation_fraction"].mean()),
        "mean_realized_thruster_force_abs_n": _finite_statistic(
            log["realized_thruster_force_abs_mean_n"], np.mean
        ),
        "max_realized_thruster_force_abs_n": _finite_statistic(
            log["realized_thruster_force_abs_max_n"], np.max
        ),
        "mean_thruster_wrench_force_norm_n": _finite_statistic(
            np.linalg.norm(log[thruster_force_columns].to_numpy(), axis=1), np.mean
        ),
        "mean_physx_applied_wrench_force_norm_n": _finite_statistic(
            np.linalg.norm(log[physx_force_columns].to_numpy(), axis=1), np.mean
        ),
        "action_deadband_fraction": float(np.mean(np.abs(actions) <= deadband)),
    }


def _boundary_metrics(log: pd.DataFrame, cfg: Any) -> dict[str, float]:
    positions = log[
        ["position_local_x_m", "position_local_y_m", "position_local_z_m"]
    ].to_numpy(dtype=np.float64)
    quaternions = log[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(
        dtype=np.float64
    )
    w, x, y, z = quaternions.T
    rotation = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
            2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    body_half_extents = 0.5 * np.asarray(cfg.body_bounds_size_m, dtype=np.float64)
    world_half_extents = np.abs(rotation) @ body_half_extents
    bounds = np.asarray(cfg.pool_bounds, dtype=np.float64)
    clearance = np.stack(
        (
            positions[:, 0] - world_half_extents[:, 0] - bounds[0],
            bounds[1] - positions[:, 0] - world_half_extents[:, 0],
            positions[:, 1] - world_half_extents[:, 1] - bounds[2],
            bounds[3] - positions[:, 1] - world_half_extents[:, 1],
            positions[:, 2] - world_half_extents[:, 2] - bounds[4],
            bounds[5] - positions[:, 2] - world_half_extents[:, 2],
        ),
        axis=1,
    )
    minimum = np.min(clearance, axis=1)
    return {
        "minimum_vehicle_boundary_clearance_m": float(np.min(minimum)),
        "vehicle_boundary_violation_fraction": float(np.mean(minimum < 0.0)),
    }


def _domain_metrics(domain_samples: pd.DataFrame) -> dict[str, float]:
    fluid_added_mass_scales = domain_samples[
        [
            "sampled_fluid_added_mass_scale_surge",
            "sampled_fluid_added_mass_scale_sway",
            "sampled_fluid_added_mass_scale_heave",
            "sampled_fluid_added_mass_scale_roll",
            "sampled_fluid_added_mass_scale_pitch",
            "sampled_fluid_added_mass_scale_yaw",
        ]
    ].to_numpy()
    return {
        "fluid_added_mass_scale_mean": float(np.mean(fluid_added_mass_scales)),
        "fluid_added_mass_scale_min": float(np.min(fluid_added_mass_scales)),
        "fluid_added_mass_scale_max": float(np.max(fluid_added_mass_scales)),
        "thruster_time_constant_mean_s": float(domain_samples["sampled_thruster_time_constant_s"].mean()),
        "pose_sensor_delay_s": float(domain_samples["pose_sensor_delay_s"].mean()),
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
        "evaluation_log_schema_version": int(log["log_schema_version"].iloc[0]),
        "trajectory": args.trajectory,
        "reward_profile": args.reward_profile,
        "seed": int(cfg.seed),
        "environment_profile_name": cfg.environment_profile_name,
        "domain_randomization_spec_name": cfg.domain_randomization_spec_name or "",
        "disturbance": disturbance_label or "nominal",
        "num_curves": int(log["env_id"].nunique()),
    }
    summary.update(_domain_metrics(domain_samples))
    summary.update(_reference_metrics(log, cfg))
    summary.update(_tracking_metrics(log, cfg, domain_samples))
    summary.update(_actuator_metrics(log))
    summary.update(_boundary_metrics(log, cfg))
    summary.update(_disturbance_metrics(log, args))
    summary.update(_failure_metrics(log, termination_events, any_failure, first_failure_time_s))
    return summary


def write_evaluation_summary(summary: dict[str, Any], path: str) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)


def parse_eval_dir(path: str | Path) -> tuple[int | None, str | None]:
    path = Path(path)
    name = path.parent.name if path.name == "summary_metrics.csv" else path.name
    match = re.match(r"model_(\d+)(?:_(.*))?_trajectory_eval$", name)
    if not match:
        return None, None
    return int(match.group(1)), "lissajous" if match.group(2) is None else match.group(2)


def collect_summary_df(
    spec: ExperimentSpec,
    run_name: str,
    *,
    case_label: str | None = None,
) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(spec.results_root(run_name).glob("model_*_trajectory_eval/summary_metrics.csv")):
        iteration, inferred_trajectory = parse_eval_dir(csv_path)
        if iteration is None:
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        if case_label is not None:
            expected_case = case_label or "nominal"
            raw_label = row.get("evaluation_label")
            evaluation_label = "" if pd.isna(raw_label) else str(raw_label).strip()
            raw_disturbance = row.get("disturbance", "nominal")
            disturbance = "nominal" if pd.isna(raw_disturbance) else str(raw_disturbance).strip()
            actual_case = evaluation_label or disturbance or "nominal"
            if actual_case != expected_case:
                continue
        row.update(
            {
                "run": run_name,
                "checkpoint": iteration,
                "checkpoint_name": f"model_{iteration}.pt",
                "trajectory": row.get("trajectory", inferred_trajectory),
                "summary_path": str(csv_path),
                "logs_path": str(csv_path.parent / "logs.csv"),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["run", "checkpoint", "checkpoint_name", "trajectory"])
    summary = pd.DataFrame(rows)
    for column in ("position_rmse", "position_mae", "max_position_error", "velocity_rmse", "num_curves"):
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    return summary.sort_values(["trajectory", "checkpoint"]).reset_index(drop=True)


def save_summary_table(
    spec: ExperimentSpec,
    run_name: str,
    summary_df: pd.DataFrame,
    *,
    case_label: str = "",
) -> Path:
    suffix = f"_{case_label}" if case_label else ""
    output = spec.results_root(run_name) / f"checkpoint_rmse_summary{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output, index=False)
    print(f"Saved: {output}")
    return output


def plot_rmse_summary(
    spec: ExperimentSpec,
    run_name: str,
    summary_df: pd.DataFrame,
    *,
    case_label: str = "",
    save: bool = True,
):
    import matplotlib.pyplot as plt

    if summary_df.empty:
        raise ValueError("summary_df is empty. Run eval first.")
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for trajectory, group in summary_df.groupby("trajectory"):
        group = group.sort_values("checkpoint")
        ax.plot(group["checkpoint"], group["position_rmse"], marker="o", linewidth=2, label=trajectory)
    ax.set_title(f"Trajectory Tracking Position RMSE: {run_name}")
    ax.set_xlabel("checkpoint iteration")
    ax.set_ylabel("position RMSE [m]")
    ax.legend(ncol=2)
    suffix = f"_{case_label}" if case_label else ""
    output = spec.results_root(run_name) / f"checkpoint_rmse_curve{suffix}.png"
    if save:
        fig.savefig(output, dpi=180)
        print(f"Saved: {output}")
    return fig, ax


def plot_rmse_heatmap(
    spec: ExperimentSpec,
    run_name: str,
    summary_df: pd.DataFrame,
    *,
    case_label: str = "",
    save: bool = True,
):
    import matplotlib.pyplot as plt

    if summary_df.empty:
        raise ValueError("summary_df is empty.")
    pivot = summary_df.pivot_table(
        index="checkpoint_name", columns="trajectory", values="position_rmse", aggfunc="mean"
    )
    pivot = pivot.reindex(sorted(pivot.index, key=checkpoint_iter))
    fig, ax = plt.subplots(
        figsize=(max(8, 1.15 * len(pivot.columns)), max(4, 0.36 * len(pivot))), constrained_layout=True
    )
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Position RMSE heatmap [m]")
    fig.colorbar(image, ax=ax, label="position RMSE [m]")
    suffix = f"_{case_label}" if case_label else ""
    output = spec.results_root(run_name) / f"checkpoint_rmse_heatmap{suffix}.png"
    if save:
        fig.savefig(output, dpi=180)
        print(f"Saved: {output}")
    return fig, ax, pivot


def best_by_trajectory(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    indices = summary_df.groupby("trajectory")["position_rmse"].idxmin()
    columns = [
        "trajectory",
        "checkpoint_name",
        "position_rmse",
        "position_mae",
        "max_position_error",
        "velocity_rmse",
    ]
    return summary_df.loc[indices, columns].sort_values("trajectory").reset_index(drop=True)


def resolve_detail_trajectory(summary_df: pd.DataFrame, preferred: str) -> str:
    if summary_df.empty:
        raise ValueError("summary_df is empty.")
    available = sorted(summary_df["trajectory"].unique().tolist())
    if preferred in available:
        return preferred
    print(f"Trajectory {preferred!r} is unavailable; using {available[0]!r}.")
    return available[0]


def resolve_detail_checkpoint(summary_df: pd.DataFrame, trajectory: str, selection: str) -> str:
    subset = summary_df[summary_df["trajectory"] == trajectory].sort_values("checkpoint")
    if subset.empty:
        raise FileNotFoundError(f"No evaluated summaries for trajectory={trajectory!r}")
    if selection == "latest":
        return str(subset.iloc[-1]["checkpoint_name"])
    if selection == "best":
        return str(subset.loc[subset["position_rmse"].idxmin()]["checkpoint_name"])
    return selection


def load_eval_log(
    spec: ExperimentSpec,
    run_name: str,
    checkpoint: str,
    trajectory: str,
    case_label: str = "",
) -> tuple[pd.DataFrame, Path]:
    path = evaluation_paths(
        spec.results_root(run_name), checkpoint, trajectory, case_label
    ).logs_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing logs.csv: {path}")
    frame = pd.read_csv(path)
    if "env_id" not in frame.columns:
        frame["env_id"] = 0
    return frame, path


def plot_eval_detail(
    spec: ExperimentSpec,
    run_name: str,
    log_df: pd.DataFrame,
    checkpoint: str,
    trajectory: str,
    *,
    case_label: str = "",
    env_id: int = 0,
    save: bool = True,
):
    import matplotlib.pyplot as plt

    frame = log_df[log_df["env_id"] == env_id].copy() if "env_id" in log_df.columns else log_df.copy()
    if frame.empty:
        raise ValueError(f"No rows for env_id={env_id}")
    pos_rmse = float(np.sqrt(np.mean(frame["position_error"] ** 2)))
    vel_rmse = float(np.sqrt(np.mean(frame["velocity_error"] ** 2)))

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)
    ax_xy = fig.add_subplot(grid[0:2, 0])
    ax_xy.plot(frame["desired_x"], frame["desired_y"], label="desired", linewidth=2.2)
    ax_xy.plot(frame["true_x"], frame["true_y"], label="actual", linewidth=1.7)
    ax_xy.set(title=f"XY path: {checkpoint} / {trajectory} / env {env_id}", xlabel="x [m]", ylabel="y [m]")
    ax_xy.axis("equal")
    ax_xy.legend()

    ax_position = fig.add_subplot(grid[0, 1])
    for axis in ("x", "y", "z"):
        ax_position.plot(frame["time"], frame[f"desired_{axis}"], "--", alpha=0.75, label=f"{axis} desired")
        ax_position.plot(frame["time"], frame[f"true_{axis}"], label=f"{axis} actual")
    ax_position.set(title="Position tracking", xlabel="time [s]", ylabel="position [m]")
    ax_position.legend(ncol=2, fontsize=8)

    ax_pos_error = fig.add_subplot(grid[1, 1])
    ax_pos_error.plot(frame["time"], frame["position_error"], label=f"position RMSE {pos_rmse:.3f} m")
    ax_pos_error.set(title="Position error", xlabel="time [s]", ylabel="error [m]")
    ax_pos_error.legend()

    ax_velocity = fig.add_subplot(grid[2, 0])
    for axis in ("x", "y", "z"):
        ax_velocity.plot(frame["time"], frame[f"desired_v{axis}"], "--", alpha=0.75, label=f"v{axis} desired")
        ax_velocity.plot(frame["time"], frame[f"true_v{axis}"], label=f"v{axis} actual")
    ax_velocity.set(title="Velocity tracking", xlabel="time [s]", ylabel="velocity [m/s]")
    ax_velocity.legend(ncol=2, fontsize=8)

    ax_control = fig.add_subplot(grid[2, 1])
    ax_control.plot(frame["time"], frame["velocity_error"], label=f"velocity RMSE {vel_rmse:.3f} m/s")
    ax_control.plot(frame["time"], frame["action_norm"], alpha=0.75, label="bounded action norm")
    ax_control.plot(
        frame["time"],
        frame["latent_policy_mean_norm"],
        "--",
        alpha=0.55,
        label="latent policy mean norm",
    )
    if "reward" in frame.columns:
        ax_control.plot(frame["time"], frame["reward"], alpha=0.65, label="reward")
    ax_control.set(title="Velocity error, action and reward", xlabel="time [s]")
    ax_control.legend()

    output = evaluation_paths(
        spec.results_root(run_name), checkpoint, trajectory, case_label
    ).directory / f"tracking_eval_env{env_id}.png"
    if save:
        fig.savefig(output, dpi=180)
        print(f"Saved: {output}")
    return fig


def plot_checkpoint_gallery(
    spec: ExperimentSpec,
    run_name: str,
    summary_df: pd.DataFrame,
    checkpoint: str,
    *,
    case_label: str = "",
    save: bool = True,
):
    import matplotlib.pyplot as plt

    trajectories = sorted(
        summary_df.loc[summary_df["checkpoint_name"] == checkpoint, "trajectory"].unique().tolist()
    )
    if not trajectories:
        raise FileNotFoundError(f"No evaluated trajectories for {checkpoint}")
    fig, axes = plt.subplots(len(trajectories), 2, figsize=(13, 3.2 * len(trajectories)), constrained_layout=True)
    if len(trajectories) == 1:
        axes = np.array([axes])
    for row, trajectory in enumerate(trajectories):
        frame, _ = load_eval_log(spec, run_name, checkpoint, trajectory, case_label)
        frame = frame[frame["env_id"] == frame["env_id"].min()].copy()
        ax_xy, ax_error = axes[row]
        ax_xy.plot(frame["desired_x"], frame["desired_y"], label="desired", linewidth=2)
        ax_xy.plot(frame["true_x"], frame["true_y"], label="actual", linewidth=1.5)
        ax_xy.set(title=f"{trajectory}: XY", xlabel="x [m]", ylabel="y [m]")
        ax_xy.axis("equal")
        ax_xy.legend()
        rmse = float(np.sqrt(np.mean(frame["position_error"] ** 2)))
        ax_error.plot(frame["time"], frame["position_error"], label=f"position RMSE {rmse:.3f} m")
        ax_error.plot(frame["time"], frame["velocity_error"], alpha=0.75, label="velocity error")
        ax_error.set(title=f"{trajectory}: errors", xlabel="time [s]")
        ax_error.legend()
    suffix = f"_{case_label}" if case_label else ""
    output = spec.results_root(run_name) / f"{Path(checkpoint).stem}{suffix}_trajectory_gallery.png"
    if save:
        fig.savefig(output, dpi=180)
        print(f"Saved: {output}")
    return fig


def quick_numeric_report(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    return (
        summary_df.groupby("trajectory")
        .agg(
            evaluated_checkpoints=("checkpoint_name", "nunique"),
            best_position_rmse=("position_rmse", "min"),
            median_position_rmse=("position_rmse", "median"),
            best_velocity_rmse=("velocity_rmse", "min"),
        )
        .reset_index()
    )

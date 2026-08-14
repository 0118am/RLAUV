"""Metrics and alignment for measured-versus-simulated AUV replay logs."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal, Sequence

import torch


FrameAlignmentMode = Literal["none", "initial_pose"]


@dataclass(frozen=True)
class ReplayTrajectory:
    time_s: torch.Tensor
    position_w: torch.Tensor
    quaternion_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    angular_velocity_b: torch.Tensor
    actions: torch.Tensor | None = None

    def validate(self, name: str = "trajectory") -> None:
        time = torch.as_tensor(self.time_s)
        if time.ndim != 1 or time.numel() < 2:
            raise ValueError(f"{name}.time_s must be one-dimensional with at least two samples.")
        if not torch.all(torch.isfinite(time)) or torch.any(time[1:] <= time[:-1]):
            raise ValueError(f"{name}.time_s must be finite and strictly increasing.")
        _validate_rows(self.position_w, time.numel(), 3, f"{name}.position_w")
        quaternion = _validate_rows(self.quaternion_wxyz, time.numel(), 4, f"{name}.quaternion_wxyz")
        if torch.any(torch.linalg.norm(quaternion, dim=-1) <= 1.0e-8):
            raise ValueError(f"{name}.quaternion_wxyz contains a zero quaternion.")
        _validate_rows(self.linear_velocity_w, time.numel(), 3, f"{name}.linear_velocity_w")
        _validate_rows(self.angular_velocity_b, time.numel(), 3, f"{name}.angular_velocity_b")
        if self.actions is not None:
            actions = torch.as_tensor(self.actions)
            if actions.ndim != 2 or actions.shape[0] != time.numel() or actions.shape[1] < 1:
                raise ValueError(f"{name}.actions must have shape (N, A).")
            if not torch.all(torch.isfinite(actions)):
                raise ValueError(f"{name}.actions must contain only finite values.")


@dataclass(frozen=True)
class AlignedReplay:
    measured_time_s: torch.Tensor
    measured_position_w: torch.Tensor
    simulated_position_w: torch.Tensor
    measured_quaternion_wxyz: torch.Tensor
    simulated_quaternion_wxyz: torch.Tensor
    measured_linear_velocity_w: torch.Tensor
    simulated_linear_velocity_w: torch.Tensor
    measured_angular_velocity_b: torch.Tensor
    simulated_angular_velocity_b: torch.Tensor
    measured_actions: torch.Tensor | None
    simulated_actions: torch.Tensor | None
    simulation_time_offset_s: float
    frame_rotation_wxyz: torch.Tensor
    frame_translation_w: torch.Tensor


@dataclass(frozen=True)
class ReplayMetricThresholds:
    max_position_rmse_m: float | None = None
    max_attitude_rmse_deg: float | None = None
    max_linear_velocity_rmse_mps: float | None = None
    max_angular_velocity_rmse_radps: float | None = None
    max_action_rmse: float | None = None
    min_overlap_duration_s: float | None = None

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class ReplayValidationResult:
    metrics: dict[str, Any]
    gates: tuple[dict[str, Any], ...]
    passed: bool
    aligned: AlignedReplay

    def report_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": "passed" if self.passed else ("failed" if self.gates else "metrics_only"),
            "metrics": self.metrics,
            "gates": list(self.gates),
            "alignment": {
                "simulation_time_offset_s": self.aligned.simulation_time_offset_s,
                "frame_rotation_wxyz": self.aligned.frame_rotation_wxyz.detach().cpu().tolist(),
                "frame_translation_w_m": self.aligned.frame_translation_w.detach().cpu().tolist(),
            },
        }


def aggregate_replay_validation_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    min_held_out_cases: int = 1,
    require_action_gate: bool = True,
) -> dict[str, Any]:
    """Aggregate independent held-out reports without counting fit-set evidence."""

    if int(min_held_out_cases) != min_held_out_cases or int(min_held_out_cases) < 1:
        raise ValueError("min_held_out_cases must be a positive integer.")
    if len(reports) == 0:
        raise ValueError("At least one replay validation report is required.")
    seen_ids: set[str] = set()
    held_out: list[Mapping[str, Any]] = []
    excluded: list[str] = []
    for index, report in enumerate(reports):
        experiment_id = str(report.get("experiment_id", f"report-{index}"))
        if experiment_id in seen_ids:
            raise ValueError(f"Duplicate replay experiment_id: {experiment_id}.")
        seen_ids.add(experiment_id)
        if report.get("split") == "held_out":
            held_out.append(report)
        else:
            excluded.append(experiment_id)

    metric_paths = {
        "position_rmse_m": ("position", "rmse"),
        "attitude_rmse_deg": ("attitude", "rmse_deg"),
        "linear_velocity_rmse_mps": ("linear_velocity", "rmse"),
        "angular_velocity_rmse_radps": ("angular_velocity", "rmse"),
    }
    aggregate_metrics: dict[str, Any] = {}
    if held_out:
        sample_counts = [_report_sample_count(report) for report in held_out]
        total_samples = sum(sample_counts)
        for output_name, path in metric_paths.items():
            values = [_nested_report_float(report, ("metrics", *path)) for report in held_out]
            weighted_rmse = (
                sum(count * value**2 for count, value in zip(sample_counts, values)) / total_samples
            ) ** 0.5
            worst_index = max(range(len(values)), key=values.__getitem__)
            aggregate_metrics[output_name] = {
                "sample_weighted_rmse": weighted_rmse,
                "worst_case": values[worst_index],
                "worst_experiment_id": str(held_out[worst_index].get("experiment_id")),
            }
        action_reports = [report for report in held_out if "actions" in report.get("metrics", {})]
        if action_reports:
            action_counts = [_report_sample_count(report) for report in action_reports]
            action_values = [_nested_report_float(report, ("metrics", "actions", "rmse")) for report in action_reports]
            aggregate_metrics["action_rmse"] = {
                "sample_weighted_rmse": (
                    sum(count * value**2 for count, value in zip(action_counts, action_values))
                    / sum(action_counts)
                )
                ** 0.5,
                "case_count": len(action_reports),
            }
    case_results = []
    for report in held_out:
        gates = report.get("gates", ())
        has_gates = isinstance(gates, (list, tuple)) and len(gates) > 0
        has_action_gate = has_gates and any(
            isinstance(gate, Mapping) and gate.get("metric") == "actions.rmse"
            for gate in gates
        )
        gates_passed = has_gates and all(
            isinstance(gate, Mapping) and bool(gate.get("passed", False))
            for gate in gates
        )
        case_results.append(
            {
                "experiment_id": str(report.get("experiment_id")),
                "report_passed": bool(report.get("passed", False)),
                "has_explicit_gates": has_gates,
                "has_action_gate": has_action_gate,
                "all_gates_passed": gates_passed,
                "passed": bool(report.get("passed", False))
                and has_gates
                and gates_passed
                and (has_action_gate or not require_action_gate),
                "sample_count": _report_sample_count(report),
            }
        )
    enough_cases = len(held_out) >= int(min_held_out_cases)
    all_cases_passed = bool(held_out) and all(case["passed"] for case in case_results)
    return {
        "passed": enough_cases and all_cases_passed,
        "held_out_case_count": len(held_out),
        "minimum_held_out_cases": int(min_held_out_cases),
        "enough_held_out_cases": enough_cases,
        "all_held_out_cases_passed": all_cases_passed,
        "require_action_gate": bool(require_action_gate),
        "excluded_non_held_out_experiments": excluded,
        "case_results": case_results,
        "aggregate_metrics": aggregate_metrics,
    }


def validate_pool_replay(
    measured: ReplayTrajectory,
    simulated: ReplayTrajectory,
    *,
    max_time_offset_s: float = 0.5,
    time_offset_resolution_s: float | None = None,
    alignment_window_s: float | None = 5.0,
    frame_alignment: FrameAlignmentMode = "initial_pose",
    thresholds: ReplayMetricThresholds | None = None,
    min_overlap_samples: int = 10,
) -> ReplayValidationResult:
    """Align two state streams and calculate 6-DOF replay error metrics.

    The selected offset follows ``sim_query_time = measured_time + offset``.
    Positive values therefore mean the matching simulated state has a later
    timestamp. Frame alignment is a single rigid transform, never a scale or
    trajectory-shape fit.
    """

    measured.validate("measured")
    simulated.validate("simulated")
    if float(max_time_offset_s) < 0.0:
        raise ValueError("max_time_offset_s must be non-negative.")
    if int(min_overlap_samples) != min_overlap_samples or int(min_overlap_samples) < 2:
        raise ValueError("min_overlap_samples must be an integer >= 2.")
    if alignment_window_s is not None and float(alignment_window_s) <= 0.0:
        raise ValueError("alignment_window_s must be positive when provided.")
    if frame_alignment not in ("none", "initial_pose"):
        raise ValueError("frame_alignment must be 'none' or 'initial_pose'.")

    measured = _trajectory_to_common_dtype(measured)
    simulated = _trajectory_to_common_dtype(simulated, measured.time_s)
    if time_offset_resolution_s is None:
        measured_dt = torch.median(measured.time_s[1:] - measured.time_s[:-1])
        simulated_dt = torch.median(simulated.time_s[1:] - simulated.time_s[:-1])
        resolution = float(torch.minimum(measured_dt, simulated_dt).item())
    else:
        resolution = float(time_offset_resolution_s)
    if resolution <= 0.0:
        raise ValueError("time_offset_resolution_s must be positive.")

    offsets = _offset_candidates(float(max_time_offset_s), resolution, measured.time_s)
    best_key: tuple[float, int, float] | None = None
    best_aligned: AlignedReplay | None = None
    for offset in offsets:
        candidate_query_time = measured.time_s + float(offset.item())
        if not torch.any(
            (candidate_query_time >= simulated.time_s[0]) & (candidate_query_time <= simulated.time_s[-1])
        ):
            continue
        aligned = align_replay_trajectories(
            measured,
            simulated,
            simulation_time_offset_s=float(offset.item()),
            frame_alignment=frame_alignment,
        )
        if aligned.measured_time_s.numel() < int(min_overlap_samples):
            continue
        coverage_penalty = (measured.time_s.numel() / aligned.measured_time_s.numel()) ** 0.5
        score = _alignment_score(aligned, alignment_window_s) * coverage_penalty
        # Deterministic tie-breaking prefers more evidence, then the smallest
        # clock correction. Candidate order must not bias a static trajectory
        # toward the negative search boundary.
        candidate_key = (
            score,
            -int(aligned.measured_time_s.numel()),
            abs(float(offset.item())),
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_aligned = aligned
    if best_aligned is None:
        raise ValueError("No time-offset candidate retained the required overlap samples.")

    metrics = calculate_replay_metrics(best_aligned)
    threshold_values = thresholds or ReplayMetricThresholds()
    threshold_values.validate()
    gates = _evaluate_replay_gates(metrics, threshold_values)
    return ReplayValidationResult(
        metrics=metrics,
        gates=tuple(gates),
        # Metrics without acceptance thresholds are diagnostics, not
        # validation evidence, and must not pass through ``all([])``.
        passed=bool(gates) and all(bool(gate["passed"]) for gate in gates),
        aligned=best_aligned,
    )


def align_replay_trajectories(
    measured: ReplayTrajectory,
    simulated: ReplayTrajectory,
    *,
    simulation_time_offset_s: float,
    frame_alignment: FrameAlignmentMode = "initial_pose",
) -> AlignedReplay:
    query_time = measured.time_s + float(simulation_time_offset_s)
    overlap = (query_time >= simulated.time_s[0]) & (query_time <= simulated.time_s[-1])
    if not torch.any(overlap):
        raise ValueError("Measured and simulated trajectories do not overlap at this offset.")
    measured_indices = torch.nonzero(overlap, as_tuple=False).reshape(-1)
    measured_time = measured.time_s[measured_indices]
    query_time = query_time[measured_indices]
    sim_position = _interpolate_rows(simulated.time_s, simulated.position_w, query_time)
    sim_quaternion = _interpolate_quaternions(simulated.time_s, simulated.quaternion_wxyz, query_time)
    sim_linear_velocity = _interpolate_rows(simulated.time_s, simulated.linear_velocity_w, query_time)
    sim_angular_velocity = _interpolate_rows(simulated.time_s, simulated.angular_velocity_b, query_time)
    sim_actions = None
    measured_actions = None
    if simulated.actions is not None and measured.actions is not None:
        if simulated.actions.shape[1] != measured.actions.shape[1]:
            raise ValueError("Measured and simulated action dimensions do not match.")
        sim_actions = _sample_zero_order_hold(simulated.time_s, simulated.actions, query_time)
        measured_actions = measured.actions[measured_indices]

    measured_position = measured.position_w[measured_indices]
    measured_quaternion = _normalize_quaternion(measured.quaternion_wxyz[measured_indices])
    measured_linear_velocity = measured.linear_velocity_w[measured_indices]
    measured_angular_velocity = measured.angular_velocity_b[measured_indices]
    frame_rotation = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=measured_time.dtype,
        device=measured_time.device,
    )
    frame_translation = torch.zeros(3, dtype=measured_time.dtype, device=measured_time.device)
    if frame_alignment == "initial_pose":
        frame_rotation = _quat_multiply(
            measured_quaternion[0:1],
            _quat_conjugate(sim_quaternion[0:1]),
        )[0]
        rotated_initial = _quat_apply(frame_rotation.reshape(1, 4), sim_position[0:1])[0]
        frame_translation = measured_position[0] - rotated_initial
        repeated_rotation = frame_rotation.reshape(1, 4).repeat(sim_position.shape[0], 1)
        sim_position = _quat_apply(repeated_rotation, sim_position) + frame_translation
        sim_quaternion = _normalize_quaternion(
            _quat_multiply(repeated_rotation, sim_quaternion)
        )
        sim_linear_velocity = _quat_apply(repeated_rotation, sim_linear_velocity)

    return AlignedReplay(
        measured_time_s=measured_time,
        measured_position_w=measured_position,
        simulated_position_w=sim_position,
        measured_quaternion_wxyz=measured_quaternion,
        simulated_quaternion_wxyz=sim_quaternion,
        measured_linear_velocity_w=measured_linear_velocity,
        simulated_linear_velocity_w=sim_linear_velocity,
        measured_angular_velocity_b=measured_angular_velocity,
        simulated_angular_velocity_b=sim_angular_velocity,
        measured_actions=measured_actions,
        simulated_actions=sim_actions,
        simulation_time_offset_s=float(simulation_time_offset_s),
        frame_rotation_wxyz=frame_rotation,
        frame_translation_w=frame_translation,
    )


def calculate_replay_metrics(aligned: AlignedReplay) -> dict[str, Any]:
    position_error = aligned.simulated_position_w - aligned.measured_position_w
    linear_velocity_error = aligned.simulated_linear_velocity_w - aligned.measured_linear_velocity_w
    angular_velocity_error = aligned.simulated_angular_velocity_b - aligned.measured_angular_velocity_b
    attitude_error = _quaternion_geodesic_error(
        aligned.measured_quaternion_wxyz,
        aligned.simulated_quaternion_wxyz,
    )
    duration = float((aligned.measured_time_s[-1] - aligned.measured_time_s[0]).item())
    metrics: dict[str, Any] = {
        "sample_count": int(aligned.measured_time_s.numel()),
        "overlap_duration_s": duration,
        "simulation_time_offset_s": aligned.simulation_time_offset_s,
        "position": _vector_error_metrics(position_error, aligned.measured_position_w),
        "attitude": {
            "rmse_rad": _root_mean_square(attitude_error),
            "rmse_deg": _root_mean_square(attitude_error) * 180.0 / torch.pi,
            "mean_rad": float(torch.mean(attitude_error).item()),
            "mean_deg": float(torch.mean(attitude_error).item()) * 180.0 / torch.pi,
            "max_rad": float(torch.max(attitude_error).item()),
            "max_deg": float(torch.max(attitude_error).item()) * 180.0 / torch.pi,
        },
        "linear_velocity": _vector_error_metrics(linear_velocity_error, aligned.measured_linear_velocity_w),
        "angular_velocity": _vector_error_metrics(angular_velocity_error, aligned.measured_angular_velocity_b),
    }
    if aligned.measured_actions is not None and aligned.simulated_actions is not None:
        action_error = aligned.simulated_actions - aligned.measured_actions
        metrics["actions"] = _vector_error_metrics(action_error, aligned.measured_actions)
    return metrics


def _alignment_score(aligned: AlignedReplay, alignment_window_s: float | None) -> float:
    if alignment_window_s is None:
        count = aligned.measured_time_s.numel()
    else:
        elapsed = aligned.measured_time_s - aligned.measured_time_s[0]
        count = int(torch.sum(elapsed <= float(alignment_window_s)).item())
        count = max(2, count)
    position_error = aligned.simulated_position_w[:count] - aligned.measured_position_w[:count]
    velocity_error = aligned.simulated_linear_velocity_w[:count] - aligned.measured_linear_velocity_w[:count]
    attitude_error = _quaternion_geodesic_error(
        aligned.measured_quaternion_wxyz[:count],
        aligned.simulated_quaternion_wxyz[:count],
    )
    position_scale = max(float(torch.std(aligned.measured_position_w[:count], correction=0).item()), 0.1)
    velocity_scale = max(float(torch.std(aligned.measured_linear_velocity_w[:count], correction=0).item()), 0.05)
    score = (
        _root_mean_square(torch.linalg.norm(position_error, dim=-1)) / position_scale
        + 0.25 * _root_mean_square(torch.linalg.norm(velocity_error, dim=-1)) / velocity_scale
        + 0.1 * _root_mean_square(attitude_error)
    )
    if aligned.measured_actions is not None and aligned.simulated_actions is not None:
        action_error = aligned.simulated_actions[:count] - aligned.measured_actions[:count]
        score += 0.1 * _root_mean_square(action_error)
    return float(score)


def _vector_error_metrics(error: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    norm = torch.linalg.norm(error, dim=-1)
    axis_rmse = torch.sqrt(torch.mean(error.square(), dim=0))
    reference_std = torch.std(reference, dim=0, correction=0)
    nrmse = axis_rmse / torch.clamp(reference_std, min=1.0e-8)
    return {
        "rmse": _root_mean_square(norm),
        "mae": float(torch.mean(norm).item()),
        "max": float(torch.max(norm).item()),
        "bias": torch.mean(error, dim=0).detach().cpu().tolist(),
        "axis_rmse": axis_rmse.detach().cpu().tolist(),
        "axis_nrmse_by_reference_std": nrmse.detach().cpu().tolist(),
        "final_error": error[-1].detach().cpu().tolist(),
    }


def _evaluate_replay_gates(
    metrics: dict[str, Any],
    thresholds: ReplayMetricThresholds,
) -> list[dict[str, Any]]:
    gate_specs = (
        ("position.rmse", metrics["position"]["rmse"], thresholds.max_position_rmse_m, "m"),
        ("attitude.rmse", metrics["attitude"]["rmse_deg"], thresholds.max_attitude_rmse_deg, "deg"),
        (
            "linear_velocity.rmse",
            metrics["linear_velocity"]["rmse"],
            thresholds.max_linear_velocity_rmse_mps,
            "m/s",
        ),
        (
            "angular_velocity.rmse",
            metrics["angular_velocity"]["rmse"],
            thresholds.max_angular_velocity_rmse_radps,
            "rad/s",
        ),
        ("overlap.duration", metrics["overlap_duration_s"], thresholds.min_overlap_duration_s, "s"),
    )
    gates: list[dict[str, Any]] = []
    for name, actual, threshold, unit in gate_specs:
        if threshold is None:
            continue
        minimum_gate = name == "overlap.duration"
        passed = float(actual) >= float(threshold) if minimum_gate else float(actual) <= float(threshold)
        gates.append(
            {
                "metric": name,
                "actual": float(actual),
                "threshold": float(threshold),
                "comparison": ">=" if minimum_gate else "<=",
                "unit": unit,
                "passed": bool(passed),
            }
        )
    if thresholds.max_action_rmse is not None:
        action_metrics = metrics.get("actions")
        if action_metrics is None:
            gates.append(
                {
                    "metric": "actions.rmse",
                    "actual": None,
                    "threshold": float(thresholds.max_action_rmse),
                    "comparison": "<=",
                    "unit": "normalized command",
                    "passed": False,
                    "reason": "Both logs must contain actions for the same-input gate.",
                }
            )
        else:
            actual = float(action_metrics["rmse"])
            gates.append(
                {
                    "metric": "actions.rmse",
                    "actual": actual,
                    "threshold": float(thresholds.max_action_rmse),
                    "comparison": "<=",
                    "unit": "normalized command",
                    "passed": actual <= float(thresholds.max_action_rmse),
                }
            )
    return gates


def _trajectory_to_common_dtype(
    trajectory: ReplayTrajectory,
    reference: torch.Tensor | None = None,
) -> ReplayTrajectory:
    if reference is None:
        time = torch.as_tensor(trajectory.time_s)
        dtype = time.dtype if torch.is_floating_point(time) else torch.float64
        device = time.device
    else:
        dtype = reference.dtype
        device = reference.device

    def convert(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=dtype, device=device)

    return ReplayTrajectory(
        time_s=convert(trajectory.time_s),
        position_w=convert(trajectory.position_w),
        quaternion_wxyz=_normalize_quaternion(convert(trajectory.quaternion_wxyz)),
        linear_velocity_w=convert(trajectory.linear_velocity_w),
        angular_velocity_b=convert(trajectory.angular_velocity_b),
        actions=convert(trajectory.actions),
    )


def _offset_candidates(max_offset: float, resolution: float, reference: torch.Tensor) -> torch.Tensor:
    if max_offset <= 0.0:
        return torch.zeros(1, dtype=reference.dtype, device=reference.device)
    count = int(torch.floor(torch.tensor(max_offset / resolution)).item())
    offsets = torch.arange(-count, count + 1, dtype=reference.dtype, device=reference.device) * resolution
    endpoints = torch.tensor([-max_offset, 0.0, max_offset], dtype=reference.dtype, device=reference.device)
    return torch.unique(torch.cat((offsets, endpoints)), sorted=True)


def _interpolate_rows(time: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    upper = torch.searchsorted(time, query.contiguous(), right=True)
    upper = torch.clamp(upper, min=1, max=time.numel() - 1)
    lower = upper - 1
    fraction = ((query - time[lower]) / torch.clamp(time[upper] - time[lower], min=1.0e-12)).unsqueeze(-1)
    return values[lower] + fraction * (values[upper] - values[lower])


def _interpolate_quaternions(time: torch.Tensor, quaternions: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    quaternion = _normalize_quaternion(quaternions)
    upper = torch.searchsorted(time, query.contiguous(), right=True)
    upper = torch.clamp(upper, min=1, max=time.numel() - 1)
    lower = upper - 1
    fraction = ((query - time[lower]) / torch.clamp(time[upper] - time[lower], min=1.0e-12)).unsqueeze(-1)
    left = quaternion[lower]
    right = quaternion[upper]
    right = torch.where(torch.sum(left * right, dim=-1, keepdim=True) < 0.0, -right, right)
    return _normalize_quaternion(left + fraction * (right - left))


def _sample_zero_order_hold(time: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    median_dt = torch.median(time[1:] - time[:-1])
    boundary_tolerance = torch.clamp(median_dt * 1.0e-6, min=torch.finfo(time.dtype).eps * 16.0)
    indices = torch.searchsorted(time, (query + boundary_tolerance).contiguous(), right=True) - 1
    indices = torch.clamp(indices, min=0, max=time.numel() - 1)
    return values[indices]


def _quaternion_geodesic_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(_normalize_quaternion(left) * _normalize_quaternion(right), dim=-1))
    return 2.0 * torch.acos(torch.clamp(dot, min=0.0, max=1.0))


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.clamp(torch.linalg.norm(quaternion, dim=-1, keepdim=True), min=1.0e-12)


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[:, 0:1], -quaternion[:, 1:]), dim=-1)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_w, left_xyz = left[:, 0:1], left[:, 1:]
    right_w, right_xyz = right[:, 0:1], right[:, 1:]
    return torch.cat(
        (
            left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + torch.cross(left_xyz, right_xyz, dim=-1),
        ),
        dim=-1,
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[:, 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[:, 0:1] * uv + uuv)


def _root_mean_square(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.square())).item())


def _report_sample_count(report: Mapping[str, Any]) -> int:
    try:
        count = int(report["metrics"]["sample_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Replay report is missing a valid metrics.sample_count.") from exc
    if count < 1:
        raise ValueError("Replay report metrics.sample_count must be positive.")
    return count


def _nested_report_float(report: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = report
    try:
        for key in path:
            value = value[key]
        result = float(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Replay report is missing numeric field {'.'.join(path)}.") from exc
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"Replay report field {'.'.join(path)} must be finite.")
    return result


def _validate_rows(value: torch.Tensor, rows: int, width: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 2 or tensor.shape != (rows, width):
        raise ValueError(f"{name} must have shape ({rows}, {width}).")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor

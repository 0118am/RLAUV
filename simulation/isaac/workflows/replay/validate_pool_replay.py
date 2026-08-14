"""Compare measured and simulated AUV replay logs with explicit accuracy gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.envs.auv.validation.replay import (  # noqa: E402
    ReplayMetricThresholds,
    ReplayTrajectory,
    ReplayValidationResult,
    validate_pool_replay,
)


REPLAY_STATE_COLUMNS = {
    "position_w": ("position_w_x_m", "position_w_y_m", "position_w_z_m"),
    "quaternion_wxyz": ("quat_w", "quat_x", "quat_y", "quat_z"),
    "linear_velocity_w": ("linear_velocity_w_x_mps", "linear_velocity_w_y_mps", "linear_velocity_w_z_mps"),
    "angular_velocity_b": (
        "angular_velocity_b_x_radps",
        "angular_velocity_b_y_radps",
        "angular_velocity_b_z_radps",
    ),
}
REQUIRED_REPLAY_COLUMNS = ("time_s", *[column for columns in REPLAY_STATE_COLUMNS.values() for column in columns])
ACTION_COLUMN_PATTERN = re.compile(r"^action_(\d+)$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align measured/simulated replay logs and gate 6-DOF trajectory accuracy.",
    )
    parser.add_argument("measured_log", type=Path, help="Trusted pool measurement CSV.")
    parser.add_argument("simulated_log", type=Path, help="Isaac replay CSV generated from the same commands.")
    parser.add_argument("--output", type=Path, required=True, help="JSON validation report path.")
    parser.add_argument("--aligned-output", type=Path, help="Optional aligned sample-by-sample CSV output.")
    parser.add_argument("--experiment-id", default="pool-replay", help="Experiment label stored in the report.")
    parser.add_argument("--measured-env-id", type=int, help="Select env_id when the measured CSV contains many streams.")
    parser.add_argument("--simulated-env-id", type=int, help="Select env_id when the simulated CSV contains many streams.")
    parser.add_argument(
        "--split",
        choices=("fit", "validation", "held_out"),
        default="held_out",
        help="Whether the experiment was used for fitting or held out for validation.",
    )
    parser.add_argument("--max-time-offset", type=float, default=0.5, help="Maximum absolute alignment offset in s.")
    parser.add_argument("--time-offset-resolution", type=float, help="Offset search resolution in s.")
    parser.add_argument("--alignment-window", type=float, default=5.0, help="Initial duration used to select offset.")
    parser.add_argument(
        "--frame-alignment",
        choices=("none", "initial_pose"),
        default="initial_pose",
        help="Single rigid world-frame registration applied to simulated states.",
    )
    parser.add_argument("--min-overlap-samples", type=int, default=10)
    parser.add_argument("--max-position-rmse", type=float, help="Position norm RMSE gate in m.")
    parser.add_argument("--max-attitude-rmse-deg", type=float, help="SO(3) attitude RMSE gate in deg.")
    parser.add_argument("--max-linear-velocity-rmse", type=float, help="Linear velocity norm RMSE gate in m/s.")
    parser.add_argument("--max-angular-velocity-rmse", type=float, help="Angular velocity norm RMSE gate in rad/s.")
    parser.add_argument("--max-action-rmse", type=float, help="Same-input normalized-command RMSE gate.")
    parser.add_argument("--min-overlap-duration", type=float, help="Minimum aligned duration gate in s.")
    return parser


def load_replay_csv(path: Path, env_id: int | None = None) -> ReplayTrajectory:
    if not path.is_file():
        raise FileNotFoundError(f"Replay log does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(set(REQUIRED_REPLAY_COLUMNS) - set(fieldnames))
        if missing:
            raise ValueError(f"Replay log {path} is missing columns: {', '.join(missing)}.")
        action_columns = _action_columns(fieldnames)
        rows = [row for row in reader if any((value or "").strip() for value in row.values())]
    if "env_id" in fieldnames:
        available_env_ids = sorted({_parse_env_id(row.get("env_id"), path) for row in rows})
        if env_id is None:
            if len(available_env_ids) > 1:
                raise ValueError(
                    f"Replay log {path} contains env_id values {available_env_ids}; select one explicitly."
                )
            env_id = available_env_ids[0]
        if env_id not in available_env_ids:
            raise ValueError(f"Replay log {path} does not contain env_id={env_id}.")
        rows = [row for row in rows if _parse_env_id(row.get("env_id"), path) == env_id]
    if len(rows) < 2:
        raise ValueError(f"Replay log {path} must contain at least two data rows.")

    def column(name: str) -> list[float]:
        values: list[float] = []
        for row_number, row in enumerate(rows, start=2):
            raw = (row.get(name) or "").strip()
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"Replay log {path}:{row_number} column {name} is not numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Replay log {path}:{row_number} column {name} is not finite.")
            values.append(value)
        return values

    def matrix(names: Sequence[str]) -> torch.Tensor:
        return torch.tensor([column(name) for name in names], dtype=torch.float64).transpose(0, 1)

    trajectory = ReplayTrajectory(
        time_s=torch.tensor(column("time_s"), dtype=torch.float64),
        position_w=matrix(REPLAY_STATE_COLUMNS["position_w"]),
        quaternion_wxyz=matrix(REPLAY_STATE_COLUMNS["quaternion_wxyz"]),
        linear_velocity_w=matrix(REPLAY_STATE_COLUMNS["linear_velocity_w"]),
        angular_velocity_b=matrix(REPLAY_STATE_COLUMNS["angular_velocity_b"]),
        actions=matrix(action_columns) if action_columns else None,
    )
    trajectory.validate(path.name)
    return trajectory


def run_replay_validation(
    measured_log: Path,
    simulated_log: Path,
    *,
    experiment_id: str = "pool-replay",
    split: str = "held_out",
    max_time_offset_s: float = 0.5,
    time_offset_resolution_s: float | None = None,
    alignment_window_s: float | None = 5.0,
    frame_alignment: str = "initial_pose",
    min_overlap_samples: int = 10,
    thresholds: ReplayMetricThresholds | None = None,
    measured_env_id: int | None = None,
    simulated_env_id: int | None = None,
) -> tuple[ReplayValidationResult, dict[str, Any]]:
    if split not in {"fit", "validation", "held_out"}:
        raise ValueError("split must be fit, validation, or held_out.")
    measured = load_replay_csv(measured_log, measured_env_id)
    simulated = load_replay_csv(simulated_log, simulated_env_id)
    result = validate_pool_replay(
        measured,
        simulated,
        max_time_offset_s=max_time_offset_s,
        time_offset_resolution_s=time_offset_resolution_s,
        alignment_window_s=alignment_window_s,
        frame_alignment=frame_alignment,
        thresholds=thresholds,
        min_overlap_samples=min_overlap_samples,
    )
    report = result.report_dict()
    report["experiment_id"] = str(experiment_id)
    report["split"] = split
    report["measured_log"] = str(measured_log)
    report["simulated_log"] = str(simulated_log)
    report["measured_env_id"] = measured_env_id
    report["simulated_env_id"] = simulated_env_id
    report["evidence_scope"] = (
        "held-out replay validation" if split == "held_out" else f"{split} replay; not independent hold-out evidence"
    )
    return result, report


def write_aligned_replay_csv(result: ReplayValidationResult, path: Path) -> None:
    aligned = result.aligned
    action_count = 0 if aligned.measured_actions is None else int(aligned.measured_actions.shape[1])
    header = [
        "time_s",
        *[f"measured_position_{axis}_m" for axis in "xyz"],
        *[f"simulated_position_{axis}_m" for axis in "xyz"],
        "position_error_norm_m",
        *[f"measured_quat_{axis}" for axis in "wxyz"],
        *[f"simulated_quat_{axis}" for axis in "wxyz"],
        "attitude_error_rad",
        *[f"measured_linear_velocity_{axis}_mps" for axis in "xyz"],
        *[f"simulated_linear_velocity_{axis}_mps" for axis in "xyz"],
        *[f"measured_angular_velocity_{axis}_radps" for axis in "xyz"],
        *[f"simulated_angular_velocity_{axis}_radps" for axis in "xyz"],
        *[f"measured_action_{index}" for index in range(action_count)],
        *[f"simulated_action_{index}" for index in range(action_count)],
    ]
    quaternion_dot = torch.abs(
        torch.sum(aligned.measured_quaternion_wxyz * aligned.simulated_quaternion_wxyz, dim=-1)
    )
    attitude_error = 2.0 * torch.acos(torch.clamp(quaternion_dot, min=0.0, max=1.0))
    position_error = torch.linalg.norm(aligned.simulated_position_w - aligned.measured_position_w, dim=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index in range(aligned.measured_time_s.numel()):
            row: list[float] = [float(aligned.measured_time_s[index].item())]
            for tensor in (aligned.measured_position_w, aligned.simulated_position_w):
                row.extend(float(value) for value in tensor[index].detach().cpu().tolist())
            row.append(float(position_error[index].item()))
            for tensor in (aligned.measured_quaternion_wxyz, aligned.simulated_quaternion_wxyz):
                row.extend(float(value) for value in tensor[index].detach().cpu().tolist())
            row.append(float(attitude_error[index].item()))
            for tensor in (
                aligned.measured_linear_velocity_w,
                aligned.simulated_linear_velocity_w,
                aligned.measured_angular_velocity_b,
                aligned.simulated_angular_velocity_b,
            ):
                row.extend(float(value) for value in tensor[index].detach().cpu().tolist())
            if action_count:
                row.extend(float(value) for value in aligned.measured_actions[index].detach().cpu().tolist())
                row.extend(float(value) for value in aligned.simulated_actions[index].detach().cpu().tolist())
            writer.writerow(row)


def _action_columns(fieldnames: Sequence[str]) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    for name in fieldnames:
        match = ACTION_COLUMN_PATTERN.match(name)
        if match:
            indexed.append((int(match.group(1)), name))
    indexed.sort()
    if indexed and [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError("Replay action columns must be contiguous from action_0.")
    return tuple(name for _, name in indexed)


def _parse_env_id(raw: str | None, path: Path) -> int:
    try:
        value = int((raw or "").strip())
    except ValueError as exc:
        raise ValueError(f"Replay log {path} contains an invalid env_id value {raw!r}.") from exc
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    thresholds = ReplayMetricThresholds(
        max_position_rmse_m=args.max_position_rmse,
        max_attitude_rmse_deg=args.max_attitude_rmse_deg,
        max_linear_velocity_rmse_mps=args.max_linear_velocity_rmse,
        max_angular_velocity_rmse_radps=args.max_angular_velocity_rmse,
        max_action_rmse=args.max_action_rmse,
        min_overlap_duration_s=args.min_overlap_duration,
    )
    result, report = run_replay_validation(
        args.measured_log,
        args.simulated_log,
        experiment_id=args.experiment_id,
        split=args.split,
        max_time_offset_s=args.max_time_offset,
        time_offset_resolution_s=args.time_offset_resolution,
        alignment_window_s=args.alignment_window,
        frame_alignment=args.frame_alignment,
        min_overlap_samples=args.min_overlap_samples,
        thresholds=thresholds,
        measured_env_id=args.measured_env_id,
        simulated_env_id=args.simulated_env_id,
    )
    _write_json(args.output, report)
    if args.aligned_output is not None:
        write_aligned_replay_csv(result, args.aligned_output)
    print(json.dumps({"passed": result.passed, "metrics": result.metrics, "gates": list(result.gates)}, indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Build and launch evaluation campaigns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from simulation.domain_randomization import load_domain_randomization_spec_json
from robot.control.trajectory import EVALUATION_TRAJECTORY_NAMES
from simulation.training.evaluation.config import (
    DEFAULT_CURRENT_TAU_S,
    DEFAULT_DYNAMICS_SCALE,
    build_evaluation_case_label,
    resolve_random_smooth_ranges,
)
from simulation.training.recipe import EvalRequest, ExperimentSpec, run_input_paths
from simulation.training.campaign import run_command
from simulation.training.campaign import (
    checkpoints_in_run_dir,
    select_checkpoints,
)


def eval_request_case_label(
    request: EvalRequest,
    *,
    domain_randomization_spec: str | Path | None = None,
) -> str:
    spec = None
    if request.sample_domain_randomization:
        selected_spec = request.domain_randomization_spec or domain_randomization_spec
        if selected_spec is not None:
            spec = load_domain_randomization_spec_json(selected_spec)
    return build_evaluation_case_label(
        evaluation_label=request.evaluation_label,
        disturbance_name=request.disturbance_name,
        sample_domain_randomization=request.sample_domain_randomization,
        domain_randomization_name=spec.name if spec is not None else None,
        seed=request.seed,
        current_w=request.eval_current,
        smooth_current=request.eval_smooth_current,
        current_variation_std=request.eval_current_variation_std,
        damping_scale=request.eval_damping_scale,
        thruster_scale=request.eval_thruster_scale,
        thruster_tau_scale=request.eval_thruster_tau_scale,
    )


@dataclass(frozen=True)
class EvaluationPaths:
    directory: Path
    logs_csv: Path
    summary_csv: Path
    domain_samples_csv: Path


def evaluation_paths(
    results_root: str | Path,
    checkpoint: str,
    trajectory: str,
    case_label: str = "",
) -> EvaluationPaths:
    parts = [Path(checkpoint).stem]
    if trajectory != "lissajous":
        parts.append(trajectory)
    if case_label:
        parts.append(case_label)
    directory = Path(results_root) / ("_".join(parts) + "_trajectory_eval")
    return EvaluationPaths(
        directory=directory,
        logs_csv=directory / "logs.csv",
        summary_csv=directory / "summary_metrics.csv",
        domain_samples_csv=directory / "domain_samples.csv",
    )


def validate_trajectories(trajectories: str | Sequence[str]) -> list[str]:
    normalized = [trajectories] if isinstance(trajectories, str) else list(trajectories)
    unknown = [name for name in normalized if name not in EVALUATION_TRAJECTORY_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown trajectories: {unknown}. Choices: {list(EVALUATION_TRAJECTORY_NAMES)}"
        )
    return normalized


def _eval_base_command(
    spec: ExperimentSpec,
    request: EvalRequest,
    checkpoint: str,
    trajectory: str,
    run_dir: Path,
) -> list[str]:
    return [
        "./isaaclab.sh",
        "-p",
        spec.eval_script,
        "--task",
        spec.task_name,
        "--checkpoint",
        str(run_dir / checkpoint),
        "--trajectory",
        trajectory,
        "--duration",
        str(request.duration_s),
        "--seed",
        str(request.seed),
    ]


def _append_eval_context(command: list[str], request: EvalRequest) -> None:
    optional_paths = (
        ("--environment_profile", request.environment_profile),
        ("--domain_randomization_spec", request.domain_randomization_spec),
    )
    for flag, value in optional_paths:
        if value is not None:
            command.extend((flag, str(value)))
    if request.sample_domain_randomization:
        command.append("--eval_domain_randomization")
    if request.eval_disturbance_stage >= 0:
        command.extend(("--eval_disturbance_stage", str(request.eval_disturbance_stage)))
    if request.evaluation_label:
        command.extend(("--evaluation_label", request.evaluation_label))
    if request.num_envs is not None:
        command.extend(("--num_envs", str(request.num_envs)))
    if request.headless:
        command.append("--headless")
    if request.align_initial_target:
        command.append("--align_initial_target")


def _append_trajectory_options(
    command: list[str],
    request: EvalRequest,
    trajectory: str,
    random_smooth_ranges: dict[str, tuple[float, float]],
) -> None:
    if trajectory == "random_smooth":
        command.extend(("--random_curve_count", str(request.random_curve_count)))
    optional_pairs = (
        ("--trajectory_amp_x", request.trajectory_amp_x),
        ("--trajectory_amp_y", request.trajectory_amp_y),
        ("--trajectory_amp_z", request.trajectory_amp_z),
        ("--trajectory_period", request.trajectory_period),
        ("--trajectory_radius_min", request.trajectory_radius_min),
        ("--trajectory_radius_max", request.trajectory_radius_max),
    )
    range_backed_flags = {
        "--trajectory_amp_x",
        "--trajectory_amp_y",
        "--trajectory_amp_z",
        "--trajectory_period",
    }
    for flag, value in optional_pairs:
        if trajectory == "random_smooth" and flag in range_backed_flags:
            continue
        if value is not None:
            command.extend((flag, str(value)))
    if trajectory == "random_smooth":
        for name, value_range in random_smooth_ranges.items():
            command.extend((f"--{name}", str(value_range[0]), str(value_range[1])))


def _append_disturbance_options(command: list[str], request: EvalRequest) -> None:
    if request.eval_current is not None:
        command.extend(("--eval_current", *(str(value) for value in request.eval_current)))
    if request.eval_smooth_current:
        command.append("--eval_smooth_current")
    disturbance_pairs = (
        ("--eval_current_variation_std", request.eval_current_variation_std, 0.0),
        ("--eval_current_tau", request.eval_current_tau, DEFAULT_CURRENT_TAU_S),
        ("--eval_damping_scale", request.eval_damping_scale, DEFAULT_DYNAMICS_SCALE),
        ("--eval_thruster_scale", request.eval_thruster_scale, DEFAULT_DYNAMICS_SCALE),
        ("--eval_thruster_tau_scale", request.eval_thruster_tau_scale, DEFAULT_DYNAMICS_SCALE),
    )
    for flag, value, default in disturbance_pairs:
        if value != default:
            command.extend((flag, str(value)))
    if request.disturbance_name:
        command.extend(("--disturbance_name", request.disturbance_name))


def _assemble_eval_command(
    spec: ExperimentSpec,
    request: EvalRequest,
    checkpoint: str,
    trajectory: str,
    run_dir: Path,
) -> list[str]:
    random_smooth_ranges = (
        resolve_random_smooth_ranges(
            trajectory_amp_x=request.trajectory_amp_x,
            trajectory_amp_y=request.trajectory_amp_y,
            trajectory_amp_z=request.trajectory_amp_z,
            trajectory_period=request.trajectory_period,
            trajectory_amp_x_range=request.trajectory_amp_x_range,
            trajectory_amp_y_range=request.trajectory_amp_y_range,
            trajectory_amp_z_range=request.trajectory_amp_z_range,
            trajectory_period_range=request.trajectory_period_range,
        )
        if trajectory == "random_smooth"
        else {}
    )
    command = _eval_base_command(spec, request, checkpoint, trajectory, run_dir)
    _append_eval_context(command, request)
    _append_trajectory_options(command, request, trajectory, random_smooth_ranges)
    _append_disturbance_options(command, request)
    return command


def run_eval_matrix(
    spec: ExperimentSpec,
    request: EvalRequest,
    *,
    load_run: str = "",
    execute: bool = False,
) -> tuple[str, list[list[str]]]:
    if not load_run:
        raise ValueError("Evaluation requires an explicit run directory.")
    run_dir = spec.logs_root / load_run
    run = run_dir.name
    inputs = run_input_paths(run_dir)
    checkpoints = select_checkpoints(
        checkpoints_in_run_dir(run_dir),
        request.checkpoint,
    )
    trajectories = validate_trajectories(request.trajectories)
    case_label = eval_request_case_label(
        request,
        domain_randomization_spec=inputs.domain_randomization,
    )
    console_log = spec.results_root(run) / "evaluation_console.log"
    if execute:
        console_log.parent.mkdir(parents=True, exist_ok=True)
        console_log.write_text("", encoding="utf-8")
    commands: list[list[str]] = []
    for checkpoint in checkpoints:
        for trajectory in trajectories:
            output_summary = evaluation_paths(
                spec.results_root(run), checkpoint, trajectory, case_label
            ).summary_csv
            if request.skip_existing and output_summary.exists():
                print(f"[SKIP] {checkpoint} / {trajectory}: {output_summary}")
                continue
            command = _assemble_eval_command(
                spec,
                request,
                checkpoint,
                trajectory,
                run_dir,
            )
            commands.append(command)
            run_command(
                command,
                cwd=spec.isaaclab_root,
                execute=execute,
                label=f"EVAL {checkpoint} / {trajectory}",
                # Isaac/Kit startup logs are verbose. Keep them on disk rather
                # than embedding them in a VS Code notebook WebView.
                log_path=console_log if execute else None,
            )
    return run, commands

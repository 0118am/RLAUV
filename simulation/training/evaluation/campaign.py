"""Build and launch evaluation campaigns."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from environment.profiles.domain_randomization import load_domain_randomization_spec_json
from robot.control.trajectory import EVALUATION_TRAJECTORY_NAMES
from simulation.training.evaluation.config import (
    DEFAULT_CURRENT_TAU_S,
    DEFAULT_DYNAMICS_SCALE,
    build_evaluation_case_label,
    resolve_random_smooth_ranges,
    validate_evaluation_parameters,
)
from simulation.training.recipe import EvalRequest, ExperimentSpec
from simulation.training.campaign import run_command
from simulation.training.campaign import (
    checkpoint_iter, resolve_checkpoints, resolve_run,
)
from simulation.training.manifest import (
    RunManifest,
    load_run_manifest,
    validate_manifest_selection,
)


def eval_request_case_label(
    request: EvalRequest,
    *,
    domain_randomization_spec: str | Path | None = None,
) -> str:
    validate_evaluation_parameters(
        duration_s=request.duration_s,
        current_w=request.eval_current,
        current_variation_std=request.eval_current_variation_std,
        current_tau=request.eval_current_tau,
        damping_scale=request.eval_damping_scale,
        thruster_scale=request.eval_thruster_scale,
        thruster_tau_scale=request.eval_thruster_tau_scale,
        num_envs=request.num_envs,
        random_curve_count=request.random_curve_count,
    )
    spec = None
    if request.sample_domain_randomization:
        selected_spec = request.domain_randomization_spec or domain_randomization_spec
        if selected_spec is not None:
            spec = load_domain_randomization_spec_json(selected_spec)
    return build_evaluation_case_label(
        evaluation_label=request.evaluation_label,
        disturbance_name=request.disturbance_name,
        sample_domain_randomization=request.sample_domain_randomization,
        domain_randomization_name=spec.name if spec is not None else "run_manifest",
        seed=request.seed,
        current_w=request.eval_current,
        smooth_current=request.eval_smooth_current,
        current_variation_std=request.eval_current_variation_std,
        damping_scale=request.eval_damping_scale,
        thruster_scale=request.eval_thruster_scale,
        thruster_tau_scale=request.eval_thruster_tau_scale,
    )


def eval_dir_name(checkpoint: str, trajectory: str, case_label: str = "") -> str:
    parts = [Path(checkpoint).stem]
    if trajectory != "lissajous":
        parts.append(trajectory)
    if case_label:
        parts.append(case_label)
    return "_".join(parts) + "_trajectory_eval"


def eval_dir(
    spec: ExperimentSpec,
    run_name: str,
    checkpoint: str,
    trajectory: str,
    case_label: str = "",
) -> Path:
    return spec.results_root(run_name) / eval_dir_name(checkpoint, trajectory, case_label)


def summary_path(
    spec: ExperimentSpec,
    run_name: str,
    checkpoint: str,
    trajectory: str,
    case_label: str = "",
) -> Path:
    return eval_dir(spec, run_name, checkpoint, trajectory, case_label) / "summary_metrics.csv"


def logs_path(
    spec: ExperimentSpec,
    run_name: str,
    checkpoint: str,
    trajectory: str,
    case_label: str = "",
) -> Path:
    return eval_dir(spec, run_name, checkpoint, trajectory, case_label) / "logs.csv"


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
    run_name: str,
    checkpoint: str,
    trajectory: str,
    manifest: RunManifest,
) -> list[str]:
    return [
        "./isaaclab.sh",
        "-p",
        spec.eval_script,
        "--task",
        spec.task_name,
        "--experiment_name",
        spec.rsl_experiment_name,
        "--load_run",
        run_name,
        "--checkpoint",
        checkpoint,
        "--run_manifest",
        str(manifest.source_path),
        "--reward_profile",
        manifest.reward_profile,
        "--trajectory",
        trajectory,
        "--duration",
        str(request.duration_s),
        "--seed",
        str(request.seed),
        "--mlp_architecture",
        spec.architecture.name,
    ]


def _append_eval_context(
    command: list[str], request: EvalRequest, checkpoint: str, manifest: RunManifest
) -> None:
    optional_paths = (
        (
            "--environment_profile",
            request.environment_profile or manifest.input_path("environment"),
        ),
        (
            "--domain_randomization_spec",
            request.domain_randomization_spec or manifest.input_path("domain_randomization"),
        ),
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
    if request.keep_boundaries:
        command.append("--keep_boundaries")
    if request.num_envs is not None:
        command.extend(("--num_envs", str(request.num_envs)))
    if checkpoint_iter(checkpoint) == 0:
        if not request.include_initial_checkpoint:
            raise ValueError("model_0.pt is excluded from tracking evaluation by default.")
        command.append("--allow_initial_checkpoint")
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
        if len(request.eval_current) != 3:
            raise ValueError("eval_current must contain exactly three world-frame components.")
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


def build_eval_command(
    spec: ExperimentSpec,
    request: EvalRequest,
    run_name: str,
    checkpoint: str,
    trajectory: str,
) -> list[str]:
    # Validate every command-producing path, including callers that bypass
    # ``run_eval_matrix`` and invoke this helper directly.
    manifest = load_run_manifest(spec.logs_root / run_name)
    eval_request_case_label(
        request,
        domain_randomization_spec=manifest.input_path("domain_randomization"),
    )
    validate_manifest_selection(
        manifest,
        mlp_architecture=spec.mlp_architecture,
        reward_profile=request.reward_profile,
    )
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
    command = _eval_base_command(spec, request, run_name, checkpoint, trajectory, manifest)
    _append_eval_context(command, request, checkpoint, manifest)
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
        raise ValueError("Evaluation requires an explicit run selected from its run manifest.")
    manifest = load_run_manifest(spec.logs_root / load_run)
    validate_manifest_selection(
        manifest,
        mlp_architecture=spec.mlp_architecture,
        reward_profile=request.reward_profile,
    )
    run = resolve_run(spec, load_run, manifest.reward_profile)
    checkpoints = resolve_checkpoints(
        spec,
        request.checkpoint,
        run,
        reward_profile=manifest.reward_profile,
        include_initial=request.include_initial_checkpoint,
    )
    trajectories = validate_trajectories(request.trajectories)
    case_label = eval_request_case_label(
        request,
        domain_randomization_spec=manifest.input_path("domain_randomization"),
    )
    console_log = spec.results_root(run) / "evaluation_console.log"
    if execute:
        console_log.parent.mkdir(parents=True, exist_ok=True)
        console_log.write_text("", encoding="utf-8")
    commands: list[list[str]] = []
    for checkpoint in checkpoints:
        checkpoint_path = spec.logs_root / run / checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        for trajectory in trajectories:
            output_summary = summary_path(spec, run, checkpoint, trajectory, case_label)
            if request.skip_existing and output_summary.exists():
                print(f"[SKIP] {checkpoint} / {trajectory}: {output_summary}")
                continue
            command = build_eval_command(spec, request, run, checkpoint, trajectory)
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

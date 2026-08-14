"""Reusable train/eval/notebook utilities for trajectory experiments.

This module intentionally has no Isaac Sim imports. Notebook kernels can use
it to inspect runs, build commands, collect CSV metrics, and plot results while
Isaac Sim remains isolated in the launched train/eval subprocess.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.isaac.agents.ppo.architectures import MlpArchitecture, get_mlp_architecture
from environment.profiles.domain_randomization import load_domain_randomization_spec_json
from simulation.isaac.workflows.common.evaluation_cases import (
    build_evaluation_case_label,
    validate_evaluation_parameters,
)


TRAJECTORY_NAMES = (
    "lissajous",
    "helix",
    "spiral",
    "chirp",
    "racetrack",
    "random_smooth",
    "lateral_sine",
    "vertical_sine",
    "spatial_helix",
)
REWARD_POLICY_ALIASES = {"baseline": "policy_0", "heading_v1": "policy_1"}


@dataclass(frozen=True)
class ExperimentSpec:
    """Filesystem contract shared by train and eval."""

    isaaclab_root: Path
    # A named feed-forward profile is the only architecture selector.
    mlp_architecture: str = "mlp_history_5"
    task_name: str = "Isaac-AUV-Traj-Direct-v1"
    train_script: str = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/isaac/workflows/train/trajectory.py"
    )
    eval_script: str = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/isaac/workflows/evaluate/trajectory.py"
    )

    @property
    def architecture(self) -> MlpArchitecture:
        return get_mlp_architecture(self.mlp_architecture)

    @property
    def policy_architecture(self) -> str:
        return self.architecture.name

    @property
    def experiment_name(self) -> str:
        return self.architecture.experiment_name

    @property
    def logs_root(self) -> Path:
        return self.isaaclab_root / "logs" / "rsl_rl" / self.experiment_name

    def results_root(self, run_name: str) -> Path:
        # Keep runtime artifacts outside IsaacLab's ``source`` extension scan.
        # A directory directly under ``source`` is treated as an extension and
        # causes a missing extension.toml warning on every launch.
        return self.isaaclab_root / "results" / "rsl_rl" / self.experiment_name / run_name


@dataclass(frozen=True)
class TrajectoryCurriculumRequest:
    """Trajectory curriculum selected explicitly by the training notebook."""

    enabled: bool
    amplitude_x_range: tuple[float, float]
    amplitude_y_range: tuple[float, float]
    amplitude_z_range: tuple[float, float]
    period_range: tuple[float, float]
    stage_steps: tuple[int, ...]
    stage_0_types: tuple[int, ...]
    stage_1_types: tuple[int, ...]
    stage_2_types: tuple[int, ...]
    stage_3_types: tuple[int, ...]
    amplitude_scales: tuple[float, ...]
    vertical_amplitude_scales: tuple[float, ...]
    period_min_by_stage: tuple[float, ...]
    period_max_by_stage: tuple[float, ...]
    speed_levels_mps: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class TrainRequest:
    """Parameters controlled by the training notebook."""

    reward_profile: str
    seed: int = 42
    num_envs: int = 4096
    run_name: str = "trajectory"
    headless: bool = True
    extra_args: tuple[str, ...] = ("--logger", "tensorboard")
    max_iterations: int | None = None
    rollout_steps_per_env: int | None = None
    pool_dynamics_profile: str | Path | None = None
    domain_randomization_spec: str | Path | None = None
    # ``None`` preserves the recipe's feature selection. An empty tuple is a
    # valid explicit request for deterministic reset/step physics while still
    # retaining the recipe identity in the run manifest.
    domain_randomization_features: tuple[str, ...] | None = None
    trajectory_curriculum: TrajectoryCurriculumRequest | None = None
    # Competence-gated campaigns force one stage for each resumed segment.
    # ``None`` preserves the ordinary step-based curriculum.
    curriculum_gate_stage: int | None = None
    # Policy steps completed by earlier fresh-process segments.  This keeps
    # the DR schedule monotonic when competence-gated training resumes.
    disturbance_curriculum_global_step_offset: int | None = None
    resume_load_run: str = ""
    resume_checkpoint: str = ""


@dataclass(frozen=True)
class EvalRequest:
    """Parameters controlled by the evaluation notebook."""

    reward_profile: str
    seed: int = 42
    checkpoint: str | Sequence[str] = "latest"
    trajectories: tuple[str, ...] = TRAJECTORY_NAMES
    duration_s: float = 32.0
    headless: bool = True
    skip_existing: bool = True
    include_initial_checkpoint: bool = False
    align_initial_target: bool = True
    random_curve_count: int = 8
    trajectory_amp_x: float | None = None
    trajectory_amp_y: float | None = None
    trajectory_amp_z: float | None = None
    trajectory_period: float | None = None
    trajectory_amp_x_range: tuple[float, float] | None = None
    trajectory_amp_y_range: tuple[float, float] | None = None
    trajectory_amp_z_range: tuple[float, float] | None = None
    trajectory_period_range: tuple[float, float] | None = None
    trajectory_radius_min: float | None = None
    trajectory_radius_max: float | None = None
    pool_dynamics_profile: str | Path | None = None
    domain_randomization_spec: str | Path | None = None
    sample_domain_randomization: bool = False
    num_envs: int | None = None
    eval_disturbance_stage: int = -1
    evaluation_label: str = ""
    keep_boundaries: bool = False
    # Deterministic, current-only diagnostics.  These deliberately bypass the
    # sampled DR recipe so a fixed current can be compared on identical curves.
    eval_current: tuple[float, float, float] | None = None
    eval_smooth_current: bool = False
    eval_current_variation_std: float = 0.0
    eval_current_tau: float = 12.0
    eval_damping_scale: float = 1.0
    eval_thruster_scale: float = 1.0
    eval_thruster_tau_scale: float = 1.0
    disturbance_name: str | None = None


def configure_plots() -> None:
    plt.rcParams.update({"figure.figsize": (10, 5), "axes.grid": True, "grid.alpha": 0.25})


def shell_join(command: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _display_path(path: str, cwd: Path) -> str:
    """Render an absolute path relative to the subprocess working directory."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return path
    try:
        return os.path.relpath(candidate, start=cwd)
    except ValueError:
        # Different Windows drives cannot be relativized. Keep the executable
        # command valid while limiting this fallback to that platform edge case.
        return path


def display_command(command: Sequence[object], *, cwd: Path) -> str:
    """Format a command for notebooks without leaking long absolute paths.

    The returned value is presentation-only. ``run_command`` always receives
    the original absolute values, which keeps profile/checkpoint resolution
    independent of the notebook's working directory.
    """

    display_parts: list[str] = []
    for value in command:
        part = str(value)
        name, separator, assigned_value = part.partition("=")
        if separator and assigned_value.startswith(os.path.sep):
            display_parts.append(f"{name}={_display_path(assigned_value, cwd)}")
        else:
            display_parts.append(_display_path(part, cwd))
    return shell_join(display_parts)


def run_command(
    command: Sequence[object],
    *,
    cwd: Path,
    execute: bool = False,
    label: str | None = None,
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> int | None:
    """Preview or run a subprocess, optionally keeping verbose output out of a notebook."""

    normalized = [str(part) for part in command]
    displayed_command = display_command(normalized, cwd=cwd)
    if label:
        print(f"[{label}]")
    print(displayed_command)
    if not execute:
        return None

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("TERM", "xterm")
    if extra_env:
        env.update(extra_env)

    def terminate_process_group(process: subprocess.Popen) -> None:
        """Stop a command and all of its children after an interrupted supervisor."""

        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        except ProcessLookupError:
            pass

    started = time.time()
    process: subprocess.Popen | None = None
    try:
        if log_path is None:
            process = subprocess.Popen(
                normalized,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
            return_code = process.wait()
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{label or 'COMMAND'}]\n{shell_join(normalized)}\n")
                process = subprocess.Popen(
                    normalized,
                    cwd=str(cwd),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            terminate_process_group(process)
        print("\n[interrupted] terminated command process group")
        raise
    print(f"\n[exit={return_code}] elapsed={(time.time() - started) / 60.0:.1f} min")
    if return_code != 0:
        if log_path is not None and log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8_000:]
            print(f"\n[last output from {_display_path(str(log_path), cwd)}]\n{tail}")
        raise RuntimeError(f"Command failed with exit code {return_code}: {displayed_command}")
    return return_code


def checkpoint_iter(name: str | Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", str(name))
    return int(match.group(1)) if match else -1


def list_runs(spec: ExperimentSpec) -> list[Path]:
    if not spec.logs_root.exists():
        return []
    return sorted(
        (path for path in spec.logs_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def has_trained_checkpoint(run_dir: Path) -> bool:
    return any(checkpoint_iter(path.name) > 0 for path in run_dir.glob("model_*.pt"))


def reward_profile_for_run(run_dir: Path) -> str:
    """Read and canonicalize the persisted reward policy for a run."""

    config_path = run_dir / "params" / "env.yaml"
    if not config_path.exists():
        return "unknown"
    match = re.search(
        r"^\s*tracking_reward_profile:\s*['\"]?([^\s#'\"]+)",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    stored_name = match.group(1) if match else "policy_0"
    return REWARD_POLICY_ALIASES.get(stored_name, stored_name)


def is_completed_run(run_dir: Path, reward_profile: str | None = None) -> bool:
    """Return whether a run has a trained checkpoint for the requested reward profile."""

    profile_matches = reward_profile is None or reward_profile_for_run(run_dir) == reward_profile
    return has_trained_checkpoint(run_dir) and profile_matches


def latest_run_dir(spec: ExperimentSpec, reward_profile: str | None = None) -> Path:
    for run_dir in list_runs(spec):
        if is_completed_run(run_dir, reward_profile):
            return run_dir
    profile_note = f" and reward profile {reward_profile!r}" if reward_profile else ""
    raise FileNotFoundError(
        f"No completed trajectory {spec.policy_architecture.upper()} run{profile_note} exists under "
        f"{spec.logs_root}."
    )


def resolve_run(spec: ExperimentSpec, load_run: str = "", reward_profile: str | None = None) -> str:
    if not load_run:
        return latest_run_dir(spec, reward_profile).name
    run_dir = spec.logs_root / load_run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not is_completed_run(run_dir, reward_profile):
        actual_profile = reward_profile_for_run(run_dir)
        raise ValueError(
            f"Run {load_run!r} has no trained checkpoint or uses a different reward profile. "
            f"Stored reward profile: {actual_profile!r}."
        )
    return load_run


def checkpoints_for_run(
    spec: ExperimentSpec,
    run_name: str,
    *,
    reward_profile: str | None = None,
    include_initial: bool = False,
) -> list[str]:
    run = resolve_run(spec, run_name, reward_profile)
    run_dir = spec.logs_root / run
    checkpoints = sorted((path.name for path in run_dir.glob("model_*.pt")), key=checkpoint_iter)
    if not include_initial:
        checkpoints = [name for name in checkpoints if checkpoint_iter(name) > 0]
    if not checkpoints:
        raise FileNotFoundError(f"No eligible checkpoint found in {run_dir}")
    return checkpoints


def resolve_checkpoints(
    spec: ExperimentSpec,
    selection: str | Sequence[str],
    run_name: str,
    *,
    reward_profile: str | None = None,
    include_initial: bool = False,
) -> list[str]:
    available = checkpoints_for_run(
        spec,
        run_name,
        reward_profile=reward_profile,
        include_initial=include_initial,
    )
    if isinstance(selection, str):
        if selection == "all":
            return available
        if selection == "latest":
            return [available[-1]]
        if selection not in available:
            raise ValueError(f"Checkpoint {selection!r} is unavailable. Choices: {available}")
        return [selection]
    if isinstance(selection, Iterable):
        resolved: list[str] = []
        for item in selection:
            resolved.extend(
                resolve_checkpoints(
                    spec,
                    item,
                    run_name,
                    reward_profile=reward_profile,
                    include_initial=include_initial,
                )
            )
        return list(dict.fromkeys(resolved))
    raise TypeError(f"Unsupported checkpoint selection: {selection!r}")


def runs_dataframe(spec: ExperimentSpec, reward_profile: str | None = None) -> pd.DataFrame:
    rows = []
    for run_dir in list_runs(spec):
        checkpoints = sorted((path.name for path in run_dir.glob("model_*.pt")), key=checkpoint_iter)
        completed = is_completed_run(run_dir, reward_profile)
        if completed:
            status = "ready"
        elif has_trained_checkpoint(run_dir):
            status = "reward profile mismatch"
        else:
            status = "model_0-only"
        rows.append(
            {
                "run": run_dir.name,
                "modified": pd.to_datetime(run_dir.stat().st_mtime, unit="s"),
                "reward_profile": reward_profile_for_run(run_dir),
                "num_checkpoints": len(checkpoints),
                "latest_checkpoint": checkpoints[-1] if checkpoints else "",
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def show_active_paths(spec: ExperimentSpec, load_run: str = "", reward_profile: str | None = None) -> None:
    try:
        run = resolve_run(spec, load_run, reward_profile)
    except (FileNotFoundError, ValueError) as error:
        print(f"No completed run selected yet: {error}")
        return
    print("IsaacLab root: .")
    print(f"Reward profile: {reward_profile_for_run(spec.logs_root / run)}")
    print(f"Active run: {run}")
    print(f"Run logs: {_display_path(str(spec.logs_root / run), spec.isaaclab_root)}")
    print(f"Run results: {_display_path(str(spec.results_root(run)), spec.isaaclab_root)}")


def _trajectory_curriculum_overrides(curriculum: TrajectoryCurriculumRequest) -> list[str]:
    values = {
        "trajectory_curriculum": curriculum.enabled,
        "trajectory_amp_x_range": curriculum.amplitude_x_range,
        "trajectory_amp_y_range": curriculum.amplitude_y_range,
        "trajectory_amp_z_range": curriculum.amplitude_z_range,
        "trajectory_period_range": curriculum.period_range,
        "trajectory_speed_levels_mps": curriculum.speed_levels_mps,
        "trajectory_curriculum_stage_steps": curriculum.stage_steps,
        "trajectory_curriculum_stage_0_types": curriculum.stage_0_types,
        "trajectory_curriculum_stage_1_types": curriculum.stage_1_types,
        "trajectory_curriculum_stage_2_types": curriculum.stage_2_types,
        "trajectory_curriculum_stage_3_types": curriculum.stage_3_types,
        "trajectory_curriculum_amp_scales": curriculum.amplitude_scales,
        "trajectory_curriculum_z_amp_scales": curriculum.vertical_amplitude_scales,
        "trajectory_curriculum_period_min": curriculum.period_min_by_stage,
        "trajectory_curriculum_period_max": curriculum.period_max_by_stage,
    }
    return [
        f"env.{name}={json.dumps(value, separators=(',', ':'))}"
        for name, value in values.items()
    ]


def _mlp_architecture_overrides(spec: ExperimentSpec) -> list[str]:
    """Forward one named MLP recipe to both IsaacLab and RSL-RL.

    The environment owns the causal history buffer, while the runner owns the
    layer widths.  Passing them together makes a checkpoint self-describing in
    its saved Hydra configuration and prevents train/eval input-shape drift.
    """

    architecture = spec.architecture
    values = {
        "env.mlp_architecture": architecture.name,
        "agent.experiment_name": architecture.experiment_name,
        "agent.policy.actor_hidden_dims": list(architecture.actor_hidden_dims),
        "agent.policy.critic_hidden_dims": list(architecture.critic_hidden_dims),
    }
    return [
        f"{name}={json.dumps(value, separators=(',', ':'))}"
        for name, value in values.items()
    ]


def build_train_command(spec: ExperimentSpec, request: TrainRequest) -> list[str]:
    command = [
        "./isaaclab.sh",
        "-p",
        spec.train_script,
        "--task",
        spec.task_name,
        "--num_envs",
        str(request.num_envs),
        "--seed",
        str(request.seed),
    ]
    command.extend(_mlp_architecture_overrides(spec))
    if request.max_iterations is not None:
        command.extend(["--max_iterations", str(request.max_iterations)])
    if request.run_name:
        command.extend(["--run_name", request.run_name])
    if request.headless:
        command.append("--headless")
    command.extend(request.extra_args)
    if request.rollout_steps_per_env is not None:
        command.append(f"agent.num_steps_per_env={request.rollout_steps_per_env}")
    command.append(f"env.tracking_reward_profile={request.reward_profile}")
    if request.pool_dynamics_profile is not None:
        command.append(f"env.pool_dynamics_profile={request.pool_dynamics_profile}")
    if request.domain_randomization_spec is not None:
        command.append(f"env.domain_randomization_spec={request.domain_randomization_spec}")
    if request.domain_randomization_features is not None:
        command.append("env.domain_randomization_feature_override_enabled=true")
        command.append(
            "env.domain_randomization.enabled_features="
            + json.dumps(list(request.domain_randomization_features), separators=(",", ":"))
        )
    if request.trajectory_curriculum is not None:
        command.extend(_trajectory_curriculum_overrides(request.trajectory_curriculum))
    if request.curriculum_gate_stage is not None:
        command.append(f"env.curriculum_gate_stage={request.curriculum_gate_stage}")
    if request.disturbance_curriculum_global_step_offset is not None:
        if request.disturbance_curriculum_global_step_offset < 0:
            raise ValueError("disturbance_curriculum_global_step_offset must be non-negative.")
        command.append(
            "env.disturbance_curriculum_global_step_offset="
            f"{request.disturbance_curriculum_global_step_offset}"
        )
    if request.resume_load_run:
        if not request.resume_checkpoint:
            raise ValueError("resume_load_run requires resume_checkpoint.")
        # These are argparse flags of IsaacLab's RSL-RL launcher, rather than
        # Hydra fields.  In particular, the launcher's ``--resume`` default
        # otherwise overwrites ``agent.resume=true`` after Hydra resolves it.
        command.extend(
            (
                "--resume",
                "--load_run",
                request.resume_load_run,
                "--checkpoint",
                request.resume_checkpoint,
            )
        )
    return command


def train_policy(spec: ExperimentSpec, request: TrainRequest, *, execute: bool = False) -> tuple[int | None, str | None]:
    result = run_command(build_train_command(spec, request), cwd=spec.isaaclab_root, execute=execute, label="TRAIN")
    selected = latest_run_dir(spec, request.reward_profile).name if execute else None
    if selected:
        print(f"Selected completed training run: {selected}")
    return result, selected


def launch_training_detached(
    spec: ExperimentSpec,
    request: TrainRequest,
    *,
    execute: bool = False,
) -> tuple[int | None, Path | None]:
    """Launch training in its own session so closing VSCode cannot stop it.

    Standard output is redirected to a durable launcher log.  The returned PID
    is the process-group leader, so it can be stopped later with
    ``kill -- -<pid>`` after checking that it is the intended training job.
    """

    command = build_train_command(spec, request)
    if not execute:
        run_command(command, cwd=spec.isaaclab_root, execute=False, label="DETACHED TRAIN")
        return None, None

    launcher_dir = spec.logs_root / "_launcher"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launch_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{request.run_name}"
    log_path = launcher_dir / f"{launch_name}.log"
    pid_path = launcher_dir / f"{launch_name}.pid"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("TERM", "xterm")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[DETACHED TRAIN]\n{shell_join(command)}\n")
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(spec.isaaclab_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Detached training PID: {process.pid}")
    print(f"Launcher log: {_display_path(str(log_path), spec.isaaclab_root)}")
    print(f"Stop only this run: kill -- -{process.pid}")
    return process.pid, log_path


def build_gpu_benchmark_commands(
    spec: ExperimentSpec,
    request: TrainRequest,
    env_candidates: Sequence[int],
) -> list[list[str]]:
    return [
        build_train_command(
            spec,
            TrainRequest(
                reward_profile=request.reward_profile,
                seed=request.seed,
                num_envs=num_envs,
                run_name=f"gpu_bench_{request.reward_profile}_{num_envs}",
                headless=request.headless,
                extra_args=request.extra_args,
                max_iterations=1,
                rollout_steps_per_env=request.rollout_steps_per_env,
                pool_dynamics_profile=request.pool_dynamics_profile,
                domain_randomization_spec=request.domain_randomization_spec,
                trajectory_curriculum=request.trajectory_curriculum,
            ),
        )
        for num_envs in env_candidates
    ]


def benchmark_gpu_throughput(
    spec: ExperimentSpec,
    request: TrainRequest,
    env_candidates: Sequence[int],
    *,
    execute: bool = False,
) -> None:
    for command in build_gpu_benchmark_commands(spec, request, env_candidates):
        run_command(command, cwd=spec.isaaclab_root, execute=execute, label="GPU BENCHMARK")


def eval_request_case_label(request: EvalRequest) -> str:
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
        if request.domain_randomization_spec is None:
            raise ValueError("sample_domain_randomization=True requires domain_randomization_spec.")
        spec = load_domain_randomization_spec_json(request.domain_randomization_spec)
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
    unknown = [name for name in normalized if name not in TRAJECTORY_NAMES]
    if unknown:
        raise ValueError(f"Unknown trajectories: {unknown}. Choices: {list(TRAJECTORY_NAMES)}")
    return normalized


def _validated_positive_range(
    request: EvalRequest,
    *,
    scalar_name: str,
    range_name: str,
) -> tuple[float, float] | None:
    """Resolve a scalar/range request without allowing ambiguous overrides."""

    scalar = getattr(request, scalar_name)
    value_range = getattr(request, range_name)
    if scalar is not None and value_range is not None:
        raise ValueError(f"Specify only one of {scalar_name} or {range_name}.")
    if scalar is not None:
        value = float(scalar)
        if value <= 0.0:
            raise ValueError(f"{scalar_name} must be positive.")
        return value, value
    if value_range is None:
        return None
    if len(value_range) != 2:
        raise ValueError(f"{range_name} must contain exactly two values.")
    lower, upper = (float(value_range[0]), float(value_range[1]))
    if lower <= 0.0 or upper < lower:
        raise ValueError(f"{range_name} must satisfy 0 < lower <= upper.")
    return lower, upper


def resolve_random_smooth_ranges(request: EvalRequest) -> dict[str, tuple[float, float]]:
    """Return the mandatory non-static random-smooth evaluation envelope."""

    names = (
        ("trajectory_amp_x", "trajectory_amp_x_range"),
        ("trajectory_amp_y", "trajectory_amp_y_range"),
        ("trajectory_amp_z", "trajectory_amp_z_range"),
        ("trajectory_period", "trajectory_period_range"),
    )
    resolved = {
        range_name: _validated_positive_range(request, scalar_name=scalar_name, range_name=range_name)
        for scalar_name, range_name in names
    }
    missing = [name for name, value in resolved.items() if value is None]
    if missing:
        raise ValueError(
            "random_smooth evaluation requires explicit positive amplitude and period ranges; missing "
            + ", ".join(missing)
            + "."
        )
    return {name: value for name, value in resolved.items() if value is not None}


def build_eval_command(
    spec: ExperimentSpec,
    request: EvalRequest,
    run_name: str,
    checkpoint: str,
    trajectory: str,
) -> list[str]:
    # Validate every command-producing path, including callers that bypass
    # ``run_eval_matrix`` and invoke this helper directly.
    eval_request_case_label(request)
    if request.sample_domain_randomization and request.domain_randomization_spec is None:
        raise ValueError("sample_domain_randomization=True requires domain_randomization_spec.")
    random_smooth_ranges = resolve_random_smooth_ranges(request) if trajectory == "random_smooth" else {}
    command = [
        "./isaaclab.sh",
        "-p",
        spec.eval_script,
        "--task",
        spec.task_name,
        "--experiment_name",
        spec.experiment_name,
        "--load_run",
        run_name,
        "--checkpoint",
        checkpoint,
        "--reward_profile",
        request.reward_profile,
        "--trajectory",
        trajectory,
        "--duration",
        str(request.duration_s),
        "--seed",
        str(request.seed),
    ]
    # This standalone evaluator uses argparse rather than Hydra.  It resolves
    # the same profile internally instead of accepting raw Hydra overrides.
    command.extend(["--mlp_architecture", spec.architecture.name])
    if request.pool_dynamics_profile is not None:
        command.extend(["--pool_dynamics_profile", str(request.pool_dynamics_profile)])
    if request.domain_randomization_spec is not None:
        command.extend(["--domain_randomization_spec", str(request.domain_randomization_spec)])
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
    if trajectory == "random_smooth":
        command.extend(["--random_curve_count", str(request.random_curve_count)])
    optional_pairs = (
        ("--trajectory_amp_x", request.trajectory_amp_x),
        ("--trajectory_amp_y", request.trajectory_amp_y),
        ("--trajectory_amp_z", request.trajectory_amp_z),
        ("--trajectory_period", request.trajectory_period),
        ("--trajectory_radius_min", request.trajectory_radius_min),
        ("--trajectory_radius_max", request.trajectory_radius_max),
    )
    for flag, value in optional_pairs:
        if trajectory == "random_smooth" and flag in {
            "--trajectory_amp_x",
            "--trajectory_amp_y",
            "--trajectory_amp_z",
            "--trajectory_period",
        }:
            continue
        if value is not None:
            command.extend([flag, str(value)])
    if trajectory == "random_smooth":
        for name, value_range in random_smooth_ranges.items():
            command.extend([f"--{name}", str(value_range[0]), str(value_range[1])])
    if request.eval_current is not None:
        if len(request.eval_current) != 3:
            raise ValueError("eval_current must contain exactly three world-frame components.")
        command.extend(["--eval_current", *(str(value) for value in request.eval_current)])
    if request.eval_smooth_current:
        command.append("--eval_smooth_current")
    disturbance_pairs = (
        ("--eval_current_variation_std", request.eval_current_variation_std, 0.0),
        ("--eval_current_tau", request.eval_current_tau, 12.0),
        ("--eval_damping_scale", request.eval_damping_scale, 1.0),
        ("--eval_thruster_scale", request.eval_thruster_scale, 1.0),
        ("--eval_thruster_tau_scale", request.eval_thruster_tau_scale, 1.0),
    )
    for flag, value, default in disturbance_pairs:
        if value != default:
            command.extend([flag, str(value)])
    if request.disturbance_name:
        command.extend(["--disturbance_name", request.disturbance_name])
    return command


def run_eval_matrix(
    spec: ExperimentSpec,
    request: EvalRequest,
    *,
    load_run: str = "",
    execute: bool = False,
) -> tuple[str, list[list[str]]]:
    run = resolve_run(spec, load_run, request.reward_profile)
    checkpoints = resolve_checkpoints(
        spec,
        request.checkpoint,
        run,
        reward_profile=request.reward_profile,
        include_initial=request.include_initial_checkpoint,
    )
    trajectories = validate_trajectories(request.trajectories)
    case_label = eval_request_case_label(request)
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


@dataclass(frozen=True)
class CompetenceGateCriteria:
    """Stage-specific held-out limits used to promote a trajectory curriculum.

    The limits intentionally use task metrics rather than reward.  Each tuple
    is indexed by the four trajectory stages.  They are conservative starting
    values and remain notebook-overridable as pool identification improves.
    """

    nominal_position_error_p95_m: tuple[float, ...] = (0.65, 0.75, 0.90, 1.05)
    nominal_velocity_rmse_mps: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65)
    robust_position_error_p95_m: tuple[float, ...] = (0.80, 0.95, 1.10, 1.25)
    robust_velocity_rmse_mps: tuple[float, ...] = (0.45, 0.55, 0.65, 0.75)
    max_any_failure_rate: float = 0.02
    min_reference_path_length_m: float = 0.10
    min_curve_target_speed_p95_mps: float = 0.05
    max_target_speed_mps: float = 0.60
    max_target_acceleration_mps2: float = 0.45
    max_target_orientation_rate_radps: float = 0.80
    max_target_jerk_mps3: float = 0.36
    consecutive_passes_required: int = 2


@dataclass(frozen=True)
class CompetenceGateDecision:
    """Serializable result of one nominal-plus-robust checkpoint evaluation."""

    checkpoint: str
    evaluated_stage: int
    nominal_passed: bool
    robust_passed: bool
    passed: bool
    consecutive_passes: int
    next_stage: int
    promoted: bool
    metrics: dict[str, float]


def _stage_value(values: Sequence[float], stage: int, name: str) -> float:
    if not values:
        raise ValueError(f"{name} must contain at least one stage value.")
    return float(values[min(max(stage, 0), len(values) - 1)])


def curriculum_eval_requests(
    train_request: TrainRequest,
    *,
    stage: int,
) -> tuple[EvalRequest, EvalRequest]:
    """Build two fixed, held-out sets for one candidate curriculum stage.

    ``nominal`` uses deterministic pool physics. ``robust`` uses an independent
    seed and the recipe's final DR level, so passing requires robustness beyond
    the training distribution. Random-smooth curves are held out from the
    training type lists.
    """

    curriculum = train_request.trajectory_curriculum
    if curriculum is None or not curriculum.enabled:
        raise ValueError("Competence evaluation requires an enabled TrajectoryCurriculumRequest.")
    if not 0 <= stage < len(curriculum.amplitude_scales):
        raise ValueError(f"Invalid curriculum stage {stage}.")

    amp_scale = curriculum.amplitude_scales[stage]
    z_scale = curriculum.vertical_amplitude_scales[stage]
    amp_x = curriculum.amplitude_x_range[1] * amp_scale
    amp_y = curriculum.amplitude_y_range[1] * amp_scale
    amp_z = curriculum.amplitude_z_range[1] * z_scale
    period = curriculum.period_min_by_stage[stage]
    common = dict(
        reward_profile=train_request.reward_profile,
        checkpoint="latest",
        trajectories=("random_smooth",),
        # The horizon matches the training episode. Re-timing may lengthen
        # individual periods, so summaries record actual path length instead
        # of assuming an integer number of laps.
        duration_s=40.0,
        headless=train_request.headless,
        skip_existing=False,
        align_initial_target=True,
        random_curve_count=32,
        num_envs=32,
        trajectory_amp_x=amp_x,
        trajectory_amp_y=amp_y,
        trajectory_amp_z=amp_z,
        trajectory_period=period,
        pool_dynamics_profile=train_request.pool_dynamics_profile,
        domain_randomization_spec=train_request.domain_randomization_spec,
        keep_boundaries=True,
    )
    nominal = EvalRequest(seed=10_173, evaluation_label="curve_v2_curriculum_nominal", **common)
    # Keep this fixed at the recipe's final level. The robust set is a genuine
    # held-out acceptance test, not another copy of the current curriculum.
    robust = EvalRequest(
        seed=20_971,
        evaluation_label="curve_v2_curriculum_robust",
        sample_domain_randomization=True,
        eval_disturbance_stage=4,
        **common,
    )
    return nominal, robust


def curriculum_current_sweep_requests(
    train_request: TrainRequest,
    *,
    stage: int,
    magnitudes_mps: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20),
    direction_w: Sequence[float] = (1.0, 0.0, 0.0),
) -> tuple[tuple[EvalRequest, ...], EvalRequest]:
    """Build matched fixed-current and full-DR diagnostics for one stage.

    Every current-only request shares the nominal held-out seed and curves;
    only the specified world-frame current changes.  The final returned request
    is the ordinary stage-4 robust gate, retained as the full-DR comparison.
    """

    nominal, robust = curriculum_eval_requests(train_request, stage=stage)
    direction = np.asarray(direction_w, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("direction_w must contain exactly three components.")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("direction_w must have non-zero magnitude.")
    unit_direction = direction / norm

    requests: list[EvalRequest] = []
    for magnitude in magnitudes_mps:
        if magnitude < 0.0:
            raise ValueError("Current sweep magnitudes must be non-negative.")
        current = tuple(float(magnitude * component) for component in unit_direction)
        label = f"curve_v2_curriculum_current_only_{float(magnitude):.3f}mps"
        requests.append(
            replace(
                nominal,
                evaluation_label=label,
                eval_current=current,
                disturbance_name=label,
            )
        )
    return tuple(requests), robust


def run_curriculum_evaluations(
    spec: ExperimentSpec,
    train_request: TrainRequest,
    *,
    run_name: str,
    checkpoint: str,
    stage: int,
    execute: bool = False,
) -> tuple[Path, Path]:
    """Run the two independent held-out sets after a training process exits."""

    nominal_request, robust_request = curriculum_eval_requests(train_request, stage=stage)
    output_paths: list[Path] = []
    for request in (nominal_request, robust_request):
        selected = replace(request, checkpoint=checkpoint)
        run_eval_matrix(spec, selected, load_run=run_name, execute=execute)
        case_label = eval_request_case_label(selected)
        output_paths.append(summary_path(spec, run_name, checkpoint, "random_smooth", case_label))
    return output_paths[0], output_paths[1]


def _read_gate_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing curriculum evaluation summary: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}.")
    row = frame.iloc[0]
    required = (
        "position_error_p95",
        "velocity_rmse",
        "any_failure_rate",
        "reference_valid",
        "reference_within_kinematic_envelope",
        "min_reference_path_length_m",
        "min_curve_target_speed_p95_mps",
        "target_speed_max_mps",
        "target_acceleration_max_mps2",
        "target_orientation_rate_max_radps",
        "target_jerk_max_mps3",
    )
    missing = [name for name in required if name not in row or pd.isna(row[name])]
    if missing:
        raise ValueError(f"Curriculum summary {path} is missing {missing}.")
    return {name: float(row[name]) for name in required}


def assess_competence_gate(
    *,
    checkpoint: str,
    stage: int,
    nominal_summary: Path,
    robust_summary: Path,
    previous_consecutive_passes: int = 0,
    criteria: CompetenceGateCriteria = CompetenceGateCriteria(),
) -> CompetenceGateDecision:
    """Assess one checkpoint without depending on reward scale or train logs."""

    nominal = _read_gate_metrics(nominal_summary)
    robust = _read_gate_metrics(robust_summary)
    def reference_passed(metrics: dict[str, float]) -> bool:
        return (
            metrics["reference_valid"] >= 1.0
            and metrics["reference_within_kinematic_envelope"] >= 1.0
            and metrics["min_reference_path_length_m"] >= criteria.min_reference_path_length_m
            and metrics["min_curve_target_speed_p95_mps"] >= criteria.min_curve_target_speed_p95_mps
            and metrics["target_speed_max_mps"] <= criteria.max_target_speed_mps * 1.01
            and metrics["target_acceleration_max_mps2"] <= criteria.max_target_acceleration_mps2 * 1.01
            and metrics["target_orientation_rate_max_radps"]
            <= criteria.max_target_orientation_rate_radps * 1.01
            and metrics["target_jerk_max_mps3"] <= criteria.max_target_jerk_mps3 * 1.01
        )

    nominal_passed = (
        nominal["position_error_p95"] <= _stage_value(criteria.nominal_position_error_p95_m, stage, "nominal p95")
        and nominal["velocity_rmse"] <= _stage_value(criteria.nominal_velocity_rmse_mps, stage, "nominal velocity")
        and nominal["any_failure_rate"] <= criteria.max_any_failure_rate
        and reference_passed(nominal)
    )
    robust_passed = (
        robust["position_error_p95"] <= _stage_value(criteria.robust_position_error_p95_m, stage, "robust p95")
        and robust["velocity_rmse"] <= _stage_value(criteria.robust_velocity_rmse_mps, stage, "robust velocity")
        and robust["any_failure_rate"] <= criteria.max_any_failure_rate
        and reference_passed(robust)
    )
    passed = nominal_passed and robust_passed
    consecutive = previous_consecutive_passes + 1 if passed else 0
    final_stage = max(
        len(criteria.nominal_position_error_p95_m),
        len(criteria.robust_position_error_p95_m),
    ) - 1
    promoted = passed and consecutive >= criteria.consecutive_passes_required and stage < final_stage
    next_stage = stage + 1 if promoted else stage
    return CompetenceGateDecision(
        checkpoint=checkpoint,
        evaluated_stage=stage,
        nominal_passed=nominal_passed,
        robust_passed=robust_passed,
        passed=passed,
        consecutive_passes=0 if promoted else consecutive,
        next_stage=next_stage,
        promoted=promoted,
        metrics={
            **{f"nominal_{name}": value for name, value in nominal.items()},
            **{f"robust_{name}": value for name, value in robust.items()},
        },
    )


def write_competence_gate_decision(path: Path, decision: CompetenceGateDecision) -> Path:
    """Persist the latest gate result for a restart-safe segmented campaign."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": decision.checkpoint,
        "evaluated_stage": decision.evaluated_stage,
        "nominal_passed": decision.nominal_passed,
        "robust_passed": decision.robust_passed,
        "passed": decision.passed,
        "consecutive_passes": decision.consecutive_passes,
        "next_stage": decision.next_stage,
        "promoted": decision.promoted,
        "metrics": decision.metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def curriculum_segment_request(
    request: TrainRequest,
    *,
    stage: int,
    segment_iterations: int,
    completed_iterations: int = 0,
    resume_load_run: str = "",
    resume_checkpoint: str = "",
) -> TrainRequest:
    """Return one checkpoint-gated segment with persistent DR progress."""

    if segment_iterations <= 0:
        raise ValueError("segment_iterations must be positive.")
    if completed_iterations < 0:
        raise ValueError("completed_iterations must be non-negative.")
    if completed_iterations and request.rollout_steps_per_env is None:
        raise ValueError("Resumed curriculum segments require rollout_steps_per_env for DR progress.")
    global_step_offset = completed_iterations * int(request.rollout_steps_per_env or 0)
    return replace(
        request,
        max_iterations=segment_iterations,
        curriculum_gate_stage=stage,
        disturbance_curriculum_global_step_offset=global_step_offset,
        resume_load_run=resume_load_run,
        resume_checkpoint=resume_checkpoint,
    )


def competence_gate_state_path(spec: ExperimentSpec, campaign_name: str) -> Path:
    """Return the durable, run-independent state file for a gated campaign."""

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", campaign_name).strip("_") or "trajectory"
    return spec.logs_root / "_curriculum" / f"{safe_name}_gate.json"


def run_competence_gate_cycle(
    spec: ExperimentSpec,
    request: TrainRequest,
    *,
    stage: int,
    segment_iterations: int,
    completed_iterations: int = 0,
    previous_consecutive_passes: int = 0,
    resume_load_run: str = "",
    resume_checkpoint: str = "",
    criteria: CompetenceGateCriteria = CompetenceGateCriteria(),
    execute: bool = False,
) -> tuple[str | None, str | None, CompetenceGateDecision | None]:
    """Run one isolated train/evaluate/promote cycle.

    This deliberately launches evaluation only *after* the training subprocess
    exits.  Therefore the two held-out sets own a separate Isaac Sim process
    and cannot alter the training rollout, CUDA RNG stream, or physics state.
    Call this once per segment; feed its selected run/checkpoint and decision
    into the next call's ``resume_*``, ``stage``, and pass-count arguments.
    """

    if resume_load_run:
        previous_path = spec.logs_root / resume_load_run / resume_checkpoint
        if not previous_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {previous_path}")

    segment = curriculum_segment_request(
        request,
        stage=stage,
        segment_iterations=segment_iterations,
        completed_iterations=completed_iterations,
        resume_load_run=resume_load_run,
        resume_checkpoint=resume_checkpoint,
    )
    _, run_name = train_policy(spec, segment, execute=execute)
    if not execute or run_name is None:
        return run_name, None, None
    checkpoints = checkpoints_for_run(spec, run_name, reward_profile=request.reward_profile)
    checkpoint = checkpoints[-1]
    nominal_path, robust_path = run_curriculum_evaluations(
        spec,
        request,
        run_name=run_name,
        checkpoint=checkpoint,
        stage=stage,
        execute=True,
    )
    decision = assess_competence_gate(
        checkpoint=checkpoint,
        stage=stage,
        nominal_summary=nominal_path,
        robust_summary=robust_path,
        previous_consecutive_passes=previous_consecutive_passes,
        criteria=criteria,
    )
    state_path = competence_gate_state_path(spec, request.run_name)
    write_competence_gate_decision(state_path, decision)
    print(
        "[CURRICULUM GATE] "
        f"stage={stage} checkpoint={checkpoint} passed={decision.passed} "
        f"streak={decision.consecutive_passes} promoted={decision.promoted} next_stage={decision.next_stage}"
    )
    print(f"[CURRICULUM GATE] state: {_display_path(str(state_path), spec.isaaclab_root)}")
    return run_name, checkpoint, decision


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
    path = logs_path(spec, run_name, checkpoint, trajectory, case_label)
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
    ax_control.plot(frame["time"], frame["action_norm"], alpha=0.75, label="clipped command norm")
    if "raw_policy_action_norm" in frame.columns:
        ax_control.plot(
            frame["time"],
            frame["raw_policy_action_norm"],
            "--",
            alpha=0.55,
            label="raw policy action norm",
        )
    if "reward" in frame.columns:
        ax_control.plot(frame["time"], frame["reward"], alpha=0.65, label="reward")
    ax_control.set(title="Velocity error, action and reward", xlabel="time [s]")
    ax_control.legend()

    output = eval_dir(spec, run_name, checkpoint, trajectory, case_label) / f"tracking_eval_env{env_id}.png"
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

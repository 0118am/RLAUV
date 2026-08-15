"""Detached process lifecycle for one AUV training campaign."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from robot.control.trajectory import TrajectoryKinematicLimits
from .manifest import load_run_manifest
from .recipe import (
    DEFAULT_TRAINING_RECIPE,
    PROJECT_ROOT,
    ExperimentSpec,
    TrainingRecipe,
    TrainRequest,
    TrajectoryCurriculumRequest,
    load_training_recipe,
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

@dataclass(frozen=True)
class TrainingCampaign:
    """Configuration for one directly managed training worker."""

    experiment: ExperimentSpec
    train: TrainRequest
    recipe: TrainingRecipe

    @property
    def run_name(self) -> str:
        return self.train.run_name

    @property
    def launcher_dir(self) -> Path:
        return self.experiment.logs_root / "_launcher"


def build_training_campaign(
    *,
    isaaclab_root: str | Path,
    recipe_path: str | Path = DEFAULT_TRAINING_RECIPE,
    rlpolicy_root: str | Path = PROJECT_ROOT / "simulation/rlpolicy",
    seed: int = 42,
    num_envs: int = 1024,
    run_name: str = "t60_policy_6",
    headless: bool = True,
) -> TrainingCampaign:
    """Build one campaign from a single versioned behavior recipe."""

    selected_recipe_path = Path(recipe_path).expanduser().resolve()
    recipe = load_training_recipe(selected_recipe_path)
    experiment = ExperimentSpec(
        isaaclab_root=Path(isaaclab_root).expanduser().resolve(),
        rlpolicy_root=Path(rlpolicy_root).expanduser().resolve(),
        mlp_architecture=recipe.mlp_architecture,
    )
    train = TrainRequest(
        reward_profile=recipe.reward_profile,
        training_recipe=selected_recipe_path,
        seed=seed,
        num_envs=num_envs,
        run_name=run_name,
        headless=headless,
        extra_args=("--logger", "tensorboard"),
        max_iterations=recipe.max_iterations,
        rollout_steps_per_env=recipe.rollout_steps_per_env,
        trajectory_curriculum=recipe.trajectory_curriculum,
    )
    return TrainingCampaign(experiment=experiment, train=train, recipe=recipe)


def _is_campaign_command(command: str, campaign: TrainingCampaign) -> bool:
    is_train = (
        "simulation/training/train.py" in command
        and f"--run_name {campaign.run_name}" in command
    )
    is_eval = (
        "simulation/training/evaluation/cli.py" in command
        and f"--load_run {campaign.run_name}" in command
    )
    return is_train or is_eval


@dataclass(frozen=True)
class CampaignProcess:
    pid: int
    process_group: int | None
    command: str
    pid_path: Path | None = None


def campaign_processes(campaign: TrainingCampaign) -> tuple[list[CampaignProcess], dict[Path, int]]:
    """Find live worker processes and all launcher PID records for a campaign."""

    pid_records: dict[Path, int] = {}
    for pid_path in campaign.launcher_dir.glob(f"*_{campaign.run_name}.pid"):
        try:
            pid_records[pid_path] = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue

    found: dict[int, CampaignProcess] = {}
    for pid_path, pid in pid_records.items():
        command = process_command(pid)
        if process_state(pid) != "running" or not _is_campaign_command(command, campaign):
            continue
        try:
            process_group = os.getpgid(pid)
        except ProcessLookupError:
            process_group = None
        found[pid] = CampaignProcess(pid, process_group, command, pid_path)

    listing = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in listing.stdout.splitlines():
        try:
            pid_text, command = line.strip().split(maxsplit=1)
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in found or not _is_campaign_command(command, campaign) or process_state(pid) != "running":
            continue
        try:
            process_group = os.getpgid(pid)
        except ProcessLookupError:
            process_group = None
        found[pid] = CampaignProcess(pid, process_group, command)
    return sorted(found.values(), key=lambda item: item.pid), pid_records


def stop_campaign(
    campaign: TrainingCampaign,
    *,
    clean_stale: bool = False,
    timeout_s: float = 8.0,
) -> tuple[list[int], list[Path]]:
    """Stop only process groups verified as belonging to ``campaign``."""

    processes, pid_records = campaign_processes(campaign)
    own_process_group = os.getpgrp()
    process_groups: dict[int, int] = {}
    for process in processes:
        if process.process_group is None:
            continue
        if process.process_group == own_process_group:
            raise RuntimeError(
                f"Refusing to signal the manager's process group {own_process_group} (PID {process.pid})."
            )
        process_groups.setdefault(process.process_group, process.pid)

    for process_group, leader_pid in process_groups.items():
        terminate_process_group(
            process_group,
            leader_pid,
            timeout_s=timeout_s,
        )

    removed: list[Path] = []
    if clean_stale:
        for pid_path, pid in pid_records.items():
            if process_state(pid) != "running":
                pid_path.unlink(missing_ok=True)
                removed.append(pid_path)
    return sorted(process_groups), sorted(removed)


def _validate_launch_paths(campaign: TrainingCampaign) -> None:
    launcher = campaign.experiment.isaaclab_root / "isaaclab.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"IsaacLab launcher not found: {launcher}")
    if not campaign.train.training_recipe.is_file():
        raise FileNotFoundError(f"Training recipe not found: {campaign.train.training_recipe}")


def launch_campaign(
    campaign: TrainingCampaign,
    *,
    replace_running: bool = True,
) -> tuple[int, Path]:
    """Launch one training worker in a detached session."""

    _validate_launch_paths(campaign)
    running, _ = campaign_processes(campaign)
    if running and not replace_running:
        raise RuntimeError(f"Campaign {campaign.run_name!r} already has running processes.")
    if running:
        stop_campaign(campaign, clean_stale=True)

    command = build_train_command(campaign.experiment, campaign.train)
    launch_suffix = campaign.run_name
    label = "DETACHED TRAIN"

    campaign.launcher_dir.mkdir(parents=True, exist_ok=True)
    launch_id = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{launch_suffix}"
    log_path = campaign.launcher_dir / f"{launch_id}.log"
    pid_path = campaign.launcher_dir / f"{launch_id}.pid"
    launch_env = experiment_environment()
    launch_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{label}]\n{' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=campaign.experiment.isaaclab_root,
            env=launch_env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return process.pid, log_path


def launch_or_attach_campaign(campaign: TrainingCampaign) -> tuple[int, Path, bool]:
    """Attach to a tracked live campaign, or launch it when none is running."""

    running, _ = campaign_processes(campaign)
    tracked = [process for process in running if process.pid_path is not None]
    if tracked:
        selected = max(tracked, key=lambda process: process.pid_path.name)
        return selected.pid, selected.pid_path.with_suffix(".log"), False
    if running:
        raise RuntimeError(
            f"Campaign {campaign.run_name!r} is running without a launcher log; "
            "refusing to replace it."
        )
    pid, log_path = launch_campaign(campaign, replace_running=False)
    return pid, log_path, True


def _emit_log_delta(log_path: Path, offset: int) -> int:
    """Print newly appended launcher-log bytes and return the next offset."""

    try:
        with log_path.open("rb") as log_file:
            log_file.seek(offset)
            payload = log_file.read()
            next_offset = log_file.tell()
    except FileNotFoundError:
        return offset
    if payload:
        text = payload.decode("utf-8", errors="replace")
        print(_ANSI_ESCAPE.sub("", text), end="", flush=True)
    return next_offset


def follow_campaign_log(
    pid: int,
    log_path: str | Path,
    *,
    from_start: bool = True,
    poll_interval_s: float = 0.5,
) -> str:
    """Stream a detached campaign's log without owning its lifecycle.

    Interrupting this function only detaches the viewer.  The training worker
    remains in its independent process group and continues in the background.
    """

    if poll_interval_s <= 0.0:
        raise ValueError("poll_interval_s must be positive.")
    selected_log = Path(log_path)
    offset = 0
    if not from_start and selected_log.is_file():
        offset = selected_log.stat().st_size

    try:
        while True:
            offset = _emit_log_delta(selected_log, offset)
            state = process_state(pid)
            if state != "running":
                offset = _emit_log_delta(selected_log, offset)
                print(f"\n[training worker PID={pid}: {state}]", flush=True)
                return state
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        state = process_state(pid)
        print(
            f"\n[stopped following log; training PID={pid} is {state}]",
            flush=True,
        )
        return state


def latest_launcher_record(campaign: TrainingCampaign) -> tuple[int | None, Path | None, str]:
    """Return the newest PID, log path, and process state for a campaign."""

    paths = sorted(campaign.launcher_dir.glob(f"*_{campaign.run_name}.pid"), reverse=True)
    if not paths:
        return None, None, "not found"
    pid_path = paths[0]
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None, pid_path.with_suffix(".log"), "invalid pid file"
    return pid, pid_path.with_suffix(".log"), process_state(pid)


def render_campaign_status(campaign: TrainingCampaign, *, log_tail_lines: int = 30) -> str:
    """Render a non-blocking campaign, checkpoint, and launcher snapshot."""

    pid, log_path, process_status = latest_launcher_record(campaign)
    lines = [
        f"Campaign: {campaign.run_name} | worker: "
        f"{pid if pid is not None else 'not found'} ({process_status})"
    ]

    run_dirs = [
        path
        for path in campaign.experiment.logs_root.glob(f"*_{campaign.run_name}")
        if path.is_dir()
    ]
    if run_dirs:
        latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)
        checkpoints = sorted(
            latest_run.glob("model_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        lines.append(
            f"Latest run: {latest_run.name} | checkpoint: {checkpoints[-1].name if checkpoints else 'none'}"
        )
    else:
        lines.append("Latest run: none")
    lines.append(f"TensorBoard: tensorboard --logdir {campaign.experiment.logs_root}")

    if log_path is not None and log_path.is_file() and log_tail_lines > 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-log_tail_lines:]
        lines.extend((f"\n[log tail: {log_path.name}]", _ANSI_ESCAPE.sub("", "\n".join(tail))))
    elif log_path is not None and not log_path.is_file():
        lines.append(f"Log has not been created yet: {log_path}")
    return "\n".join(lines)


def process_state(pid: int) -> str:
    """Return running, zombie, or exited for one process ID."""

    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1][0]
    except (FileNotFoundError, IndexError, PermissionError):
        return "exited"
    return "zombie" if state in {"Z", "X"} else "running"


def process_command(pid: int) -> str:
    """Return the null-separated process command as readable text."""

    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8",
            "replace",
        )
    except (FileNotFoundError, PermissionError):
        return ""


def experiment_environment(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the single subprocess environment used by train and eval."""

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("TERM", "xterm")
    if extra_env:
        env.update(extra_env)
    return env


def terminate_process_group(
    process_group: int,
    leader_pid: int,
    *,
    timeout_s: float = 10.0,
) -> None:
    """Terminate one verified process group, escalating after ``timeout_s``."""

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and process_state(leader_pid) == "running":
        time.sleep(0.2)
    if process_state(leader_pid) == "running":
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass

def configure_plots() -> None:
    import matplotlib.pyplot as plt

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

    env = experiment_environment(extra_env)

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
            terminate_process_group(process.pid, process.pid)
            process.wait()
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
    """Read the persisted reward policy for a run."""

    try:
        return load_run_manifest(run_dir).reward_profile
    except (FileNotFoundError, TypeError, ValueError):
        return "unknown"


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
    import pandas as pd

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
    print(f"RL policy root: {_display_path(str(spec.logs_root), spec.isaaclab_root)}")
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
        "agent.experiment_name": spec.rsl_experiment_name,
        "agent.policy.actor_hidden_dims": list(architecture.actor_hidden_dims),
        "agent.policy.critic_hidden_dims": list(architecture.critic_hidden_dims),
    }
    return [
        f"{name}={json.dumps(value, separators=(',', ':'))}"
        for name, value in values.items()
    ]


def _kinematic_limit_overrides(limits: TrajectoryKinematicLimits) -> list[str]:
    return [
        f"env.trajectory_max_speed_mps={limits.max_speed_mps}",
        f"env.trajectory_max_acceleration_mps2={limits.max_acceleration_mps2}",
        f"env.trajectory_max_orientation_rate_radps={limits.max_orientation_rate_radps}",
        f"env.trajectory_max_jerk_mps3={limits.max_jerk_mps3}",
        f"env.trajectory_retime_samples={limits.retime_samples}",
    ]


def build_train_command(spec: ExperimentSpec, request: TrainRequest) -> list[str]:
    from simulation.training.recipe import load_training_recipe

    recipe = load_training_recipe(request.training_recipe)
    if spec.mlp_architecture != recipe.mlp_architecture:
        raise ValueError(
            f"Experiment architecture {spec.mlp_architecture!r} does not match training recipe "
            f"{recipe.mlp_architecture!r}."
        )
    if request.reward_profile != recipe.reward_profile:
        raise ValueError(
            f"Requested reward {request.reward_profile!r} does not match training recipe "
            f"{recipe.reward_profile!r}."
        )
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
        "--training_recipe",
        str(request.training_recipe),
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
    if request.domain_randomization_features is not None:
        command.append("env.domain_randomization_feature_override_enabled=true")
        command.append(
            "env.domain_randomization.enabled_features="
            + json.dumps(list(request.domain_randomization_features), separators=(",", ":"))
        )
    if request.trajectory_curriculum is not None:
        command.extend(_trajectory_curriculum_overrides(request.trajectory_curriculum))
    command.extend(_kinematic_limit_overrides(recipe.kinematic_limits))
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
                training_recipe=request.training_recipe,
                seed=request.seed,
                num_envs=num_envs,
                run_name=f"gpu_bench_{request.reward_profile}_{num_envs}",
                headless=request.headless,
                extra_args=request.extra_args,
                max_iterations=1,
                rollout_steps_per_env=request.rollout_steps_per_env,
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

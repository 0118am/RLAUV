"""Foreground command execution and run selection for AUV training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

import psutil

from .recipe import (
    DEFAULT_TRAINING_RECIPE,
    PROJECT_ROOT,
    ExperimentSpec,
    TrainingRecipe,
    TrainRequest,
    load_training_recipe,
    run_input_paths,
)


@dataclass(frozen=True)
class TrainingCampaign:
    """Configuration for one notebook-managed training worker."""

    experiment: ExperimentSpec
    train: TrainRequest
    recipe: TrainingRecipe


def build_training_campaign(
    *,
    isaaclab_root: str | Path,
    recipe_path: str | Path = DEFAULT_TRAINING_RECIPE,
    rlpolicy_root: str | Path = PROJECT_ROOT / "simulation/rlpolicy",
    seed: int = 42,
    num_envs: int = 1024,
    run_name: str = "t60_precision_v11",
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
        training_recipe=selected_recipe_path,
        seed=seed,
        num_envs=num_envs,
        run_name=run_name,
        headless=headless,
        extra_args=("--logger", "tensorboard"),
    )
    return TrainingCampaign(experiment=experiment, train=train, recipe=recipe)


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
    *,
    timeout_s: float = 10.0,
) -> None:
    """Terminate one verified process group, escalating after ``timeout_s``."""

    members: list[psutil.Process] = []
    for process in psutil.process_iter(("pid",)):
        try:
            if os.getpgid(process.info["pid"]) == process_group:
                members.append(process)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    _, alive = psutil.wait_procs(members, timeout=timeout_s)
    if alive:
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
    return os.path.relpath(candidate, start=cwd)


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
    """Preview or run a subprocess whose lifetime is owned by the calling cell."""

    normalized = [str(part) for part in command]
    displayed_command = display_command(normalized, cwd=cwd)
    if label:
        print(f"[{label}]")
    print(displayed_command)
    if not execute:
        return None

    env = experiment_environment(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")

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
                # Isolate the launcher and all descendants so one cell interrupt
                # can terminate the complete training worker tree.
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
    except BaseException as error:
        if process is not None and process.poll() is None:
            terminate_process_group(process.pid)
            process.wait()
        if isinstance(error, KeyboardInterrupt):
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
    return int(match.group(1))


def list_runs(spec: ExperimentSpec) -> list[Path]:
    if not spec.logs_root.exists():
        return []
    return sorted(
        (
            path
            for path in spec.logs_root.iterdir()
            if run_input_paths(path).recipe.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def has_trained_checkpoint(run_dir: Path) -> bool:
    return any(checkpoint_iter(path.name) > 0 for path in run_dir.glob("model_*.pt"))


def reward_profile_for_run(run_dir: Path) -> str:
    """Read the reward policy from the run-local training recipe."""

    return load_training_recipe(run_input_paths(run_dir).recipe).reward_profile


def resolve_run(spec: ExperimentSpec, load_run: str) -> str:
    run_dir = spec.logs_root / load_run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not has_trained_checkpoint(run_dir):
        raise ValueError(f"Run {load_run!r} has no trained checkpoint.")
    return load_run


def checkpoints_in_run_dir(run_dir: Path) -> list[str]:
    """List eligible checkpoints from an already-resolved run directory."""

    checkpoints = sorted((path.name for path in run_dir.glob("model_*.pt")), key=checkpoint_iter)
    checkpoints = [name for name in checkpoints if checkpoint_iter(name) > 0]
    if not checkpoints:
        raise FileNotFoundError(f"No eligible checkpoint found in {run_dir}")
    return checkpoints


def select_checkpoints(available: Sequence[str], selection: str | Sequence[str]) -> list[str]:
    """Resolve one checkpoint selection against a single directory scan."""

    requested = [selection] if isinstance(selection, str) else list(selection)
    resolved: list[str] = []
    for item in requested:
        if item == "all":
            resolved.extend(available)
        elif item == "latest":
            resolved.append(available[-1])
        elif item in available:
            resolved.append(item)
        else:
            raise ValueError(f"Checkpoint {item!r} is unavailable. Choices: {list(available)}")
    return list(dict.fromkeys(resolved))


def runs_dataframe(spec: ExperimentSpec) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for run_dir in list_runs(spec):
        checkpoints = sorted((path.name for path in run_dir.glob("model_*.pt")), key=checkpoint_iter)
        actual_profile = reward_profile_for_run(run_dir)
        trained = any(checkpoint_iter(name) > 0 for name in checkpoints)
        status = "ready" if trained else "model_0-only"
        rows.append(
            {
                "run": run_dir.name,
                "modified": pd.to_datetime(run_dir.stat().st_mtime, unit="s"),
                "reward_profile": actual_profile,
                "num_checkpoints": len(checkpoints),
                "latest_checkpoint": checkpoints[-1] if checkpoints else "",
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def show_active_paths(spec: ExperimentSpec, load_run: str) -> None:
    run = resolve_run(spec, load_run)
    print("IsaacLab root: .")
    print(f"RL policy root: {_display_path(str(spec.logs_root), spec.isaaclab_root)}")
    print(f"Reward profile: {reward_profile_for_run(spec.logs_root / run)}")
    print(f"Active run: {run}")
    print(f"Run logs: {_display_path(str(spec.logs_root / run), spec.isaaclab_root)}")
    print(f"Run results: {_display_path(str(spec.results_root(run)), spec.isaaclab_root)}")


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
        "--training_recipe",
        str(request.training_recipe),
        "--experiment_name",
        str(spec.logs_root),
    ]
    if request.run_name:
        command.extend(["--run_name", request.run_name])
    if request.headless:
        command.append("--headless")
    command.extend(request.extra_args)
    if request.resume_load_run:
        # These are argparse flags of IsaacLab's RSL-RL launcher, rather than
        # Hydra fields.  In particular, the launcher's ``--resume`` default
        # otherwise overwrites ``agent.resume=true`` after Hydra resolves it.
        command.extend(("--resume", "--load_run", request.resume_load_run))
        if request.resume_checkpoint:
            command.extend(("--checkpoint", request.resume_checkpoint))
    return command

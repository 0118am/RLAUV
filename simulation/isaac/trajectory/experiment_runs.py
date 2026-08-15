"""Training run and checkpoint discovery."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import re

from simulation.isaac.trajectory.experiment_models import ExperimentSpec
from simulation.isaac.trajectory.experiment_process import _display_path
from simulation.isaac.trajectory.optional_dependencies import pd

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

    config_path = run_dir / "params" / "env.yaml"
    if not config_path.exists():
        return "unknown"
    match = re.search(
        r"^\s*tracking_reward_profile:\s*['\"]?([^\s#'\"]+)",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    stored_name = match.group(1) if match else "policy_0"
    return stored_name


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
    print(f"RL policy root: {_display_path(str(spec.logs_root), spec.isaaclab_root)}")
    print(f"Reward profile: {reward_profile_for_run(spec.logs_root / run)}")
    print(f"Active run: {run}")
    print(f"Run logs: {_display_path(str(spec.logs_root / run), spec.isaaclab_root)}")
    print(f"Run results: {_display_path(str(spec.results_root(run)), spec.isaaclab_root)}")


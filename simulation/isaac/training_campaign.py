"""Detached process lifecycle for one AUV training campaign."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from simulation.isaac.trajectory.experiment_commands import build_train_command
from simulation.isaac.trajectory.experiment_models import ExperimentSpec, TrainRequest
from simulation.isaac.training_profiles import (
    DEFAULT_DOMAIN_RANDOMIZATION_SPEC,
    DEFAULT_ENVIRONMENT_PROFILE,
    PROJECT_ROOT,
    default_trajectory_curriculum,
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

@dataclass(frozen=True)
class TrainingCampaign:
    """Configuration for one directly managed training worker."""

    experiment: ExperimentSpec
    train: TrainRequest

    @property
    def run_name(self) -> str:
        return self.train.run_name

    @property
    def launcher_dir(self) -> Path:
        return self.experiment.logs_root / "_launcher"


def build_default_campaign(
    *,
    isaaclab_root: str | Path,
    rlpolicy_root: str | Path = PROJECT_ROOT / "simulation/isaac/rlpolicy",
    mlp_architecture: str = "mlp_history_5",
    reward_profile: str = "policy_6",
    seed: int = 42,
    num_envs: int = 1024,
    run_name: str | None = None,
    headless: bool = True,
    max_iterations: int = 500,
    rollout_steps_per_env: int = 256,
    environment_profile: str | Path = DEFAULT_ENVIRONMENT_PROFILE,
    domain_randomization_spec: str | Path = DEFAULT_DOMAIN_RANDOMIZATION_SPEC,
) -> TrainingCampaign:
    """Build the repository's default T60 training campaign."""

    selected_run_name = run_name or f"t60_{reward_profile}"
    experiment = ExperimentSpec(
        isaaclab_root=Path(isaaclab_root).expanduser().resolve(),
        rlpolicy_root=Path(rlpolicy_root).expanduser().resolve(),
        mlp_architecture=mlp_architecture,
    )
    train = TrainRequest(
        reward_profile=reward_profile,
        seed=seed,
        num_envs=num_envs,
        run_name=selected_run_name,
        headless=headless,
        extra_args=("--logger", "tensorboard"),
        max_iterations=max_iterations,
        rollout_steps_per_env=rollout_steps_per_env,
        environment_profile=Path(environment_profile).expanduser().resolve(),
        domain_randomization_spec=Path(domain_randomization_spec).expanduser().resolve(),
        trajectory_curriculum=default_trajectory_curriculum(),
    )
    return TrainingCampaign(experiment=experiment, train=train)


def _process_state(pid: int) -> str:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1][0]
    except (FileNotFoundError, IndexError, PermissionError):
        return "exited"
    return "zombie" if state in {"Z", "X"} else "running"


def _process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except (FileNotFoundError, PermissionError):
        return ""


def _is_campaign_command(command: str, campaign: TrainingCampaign) -> bool:
    is_train = (
        "simulation/isaac/trajectory/train.py" in command
        and f"--run_name {campaign.run_name}" in command
    )
    is_eval = (
        "simulation/isaac/trajectory/evaluate.py" in command
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
        command = _process_command(pid)
        if _process_state(pid) != "running" or not _is_campaign_command(command, campaign):
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
        if pid in found or not _is_campaign_command(command, campaign) or _process_state(pid) != "running":
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

    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and any(_process_state(pid) == "running" for pid in process_groups.values()):
        time.sleep(0.2)
    for process_group, leader_pid in process_groups.items():
        if _process_state(leader_pid) == "running":
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass

    removed: list[Path] = []
    if clean_stale:
        for pid_path, pid in pid_records.items():
            if _process_state(pid) != "running":
                pid_path.unlink(missing_ok=True)
                removed.append(pid_path)
    return sorted(process_groups), sorted(removed)


def _validate_launch_paths(campaign: TrainingCampaign) -> None:
    launcher = campaign.experiment.isaaclab_root / "isaaclab.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"IsaacLab launcher not found: {launcher}")
    for path in (campaign.train.environment_profile, campaign.train.domain_randomization_spec):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Training configuration not found: {path}")


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
    launch_env = os.environ.copy()
    launch_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    launch_env.setdefault("TERM", "xterm")
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
    return pid, pid_path.with_suffix(".log"), _process_state(pid)


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

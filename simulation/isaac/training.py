"""Python API for configuring and managing AUV training campaigns.

The module is deliberately free of Isaac Sim imports.  It owns the experiment
recipe, supervisor JSON, detached process lifecycle, and status reporting;
``trajectory/train.py`` is the isolated Isaac Sim worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

from environment.profiles.domain_randomization import (
    domain_randomization_parameters_requiring_sources,
    load_domain_randomization_spec_json,
    write_domain_randomization_spec_json,
)
from environment.profiles.environment_profile import (
    load_environment_profile_json,
    write_environment_profile_json,
)

from simulation.isaac.trajectory.experiment import (
    CompetenceGateCriteria,
    ExperimentSpec,
    TrainRequest,
    TrajectoryCurriculumRequest,
    build_train_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT_PROFILE = (
    PROJECT_ROOT / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
)
DEFAULT_DOMAIN_RANDOMIZATION_SPEC = (
    PROJECT_ROOT / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
)
COMPETENCE_SUPERVISOR_SCRIPT = (
    "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
    "simulation/isaac/trajectory/competence_curriculum.py"
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROFILE_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TrainingProfilePaths:
    """Materialized environment and DR inputs selected by ``train.ipynb``."""

    environment: Path
    randomization: Path


def materialize_training_profiles(
    run_name: str,
    *,
    environment_base: str | Path = DEFAULT_ENVIRONMENT_PROFILE,
    randomization_base: str | Path = DEFAULT_DOMAIN_RANDOMIZATION_SPEC,
    output_root: str | Path = PROJECT_ROOT / "simulation/isaac/rlpolicy/_configs",
    hydrodynamics: dict[str, Any] | None = None,
    pool_boundary: dict[str, Any] | None = None,
    free_surface: dict[str, Any] | None = None,
    randomization: dict[str, Any] | None = None,
) -> TrainingProfilePaths:
    """Write immutable per-run inputs from notebook-selected numeric values.

    The notebook owns experiment choices; the source profiles remain reviewed
    baselines. Generated files live beside ignored policy outputs so changing
    one run never mutates the repository's physical source data.
    """

    slug = _PROFILE_NAME_SAFE.sub("-", run_name.strip()).strip("-._")
    if not slug:
        raise ValueError("run_name must contain at least one filename-safe character.")
    output_dir = Path(output_root).expanduser().resolve() / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    base_environment = load_environment_profile_json(environment_base)
    resolved_environment = replace(
        base_environment,
        name=f"{base_environment.name}--{slug}",
        description=f"{base_environment.description} Per-run values selected by train.ipynb ({slug}).",
        hydrodynamics=replace(base_environment.hydrodynamics, **(hydrodynamics or {})),
        pool_boundary=replace(base_environment.pool_boundary, **(pool_boundary or {})),
        free_surface=replace(base_environment.free_surface, **(free_surface or {})),
    )
    environment_path = output_dir / "environment.json"
    write_environment_profile_json(resolved_environment, environment_path)

    base_randomization = load_domain_randomization_spec_json(randomization_base)
    resolved_parameters = replace(base_randomization.parameters, **(randomization or {}))
    sources = dict(base_randomization.parameter_sources)
    for parameter in domain_randomization_parameters_requiring_sources(resolved_parameters):
        sources.setdefault(parameter, f"Numeric value selected in train.ipynb for run {slug}.")
    metadata = dict(base_randomization.metadata)
    metadata["configured_by"] = "train.ipynb"
    metadata["run_name"] = slug
    resolved_randomization = replace(
        base_randomization,
        name=f"{base_randomization.name}--{slug}",
        description=f"{base_randomization.description} Per-run values selected by train.ipynb ({slug}).",
        base_profile_name=resolved_environment.name,
        parameters=resolved_parameters,
        parameter_sources=sources,
        metadata=metadata,
    )
    randomization_path = output_dir / "domain_randomization.json"
    write_domain_randomization_spec_json(resolved_randomization, randomization_path)
    return TrainingProfilePaths(environment=environment_path, randomization=randomization_path)


def default_trajectory_curriculum() -> TrajectoryCurriculumRequest:
    """Return the version-controlled T60 speed and amplitude curriculum."""

    return TrajectoryCurriculumRequest(
        enabled=True,
        amplitude_x_range=(0.60, 0.78),
        amplitude_y_range=(0.55, 0.75),
        amplitude_z_range=(0.08, 0.20),
        period_range=(10.0, 20.0),
        speed_levels_mps=(0.1, 0.2, 0.3, 0.4),
        stage_steps=(9_750, 22_500, 40_500),
        stage_0_types=(8, 9, 10),
        stage_1_types=(8, 9, 10),
        stage_2_types=(8, 9, 10),
        stage_3_types=(8, 9, 10),
        amplitude_scales=(0.55, 0.75, 0.90, 1.0),
        vertical_amplitude_scales=(0.25, 0.50, 0.75, 1.0),
        period_min_by_stage=(20.0, 10.0, 10.0, 10.0),
        period_max_by_stage=(20.0, 10.0, 10.0, 10.0),
    )


@dataclass(frozen=True)
class TrainingCampaign:
    """Complete, serializable configuration for one managed training run."""

    experiment: ExperimentSpec
    train: TrainRequest
    criteria: CompetenceGateCriteria = CompetenceGateCriteria()
    use_competence_gate: bool = True
    segment_iterations: int = 25
    total_iterations: int = 500

    def __post_init__(self) -> None:
        if self.segment_iterations <= 0:
            raise ValueError("segment_iterations must be positive.")
        if self.total_iterations <= 0:
            raise ValueError("total_iterations must be positive.")
        if self.use_competence_gate and (
            self.train.trajectory_curriculum is None or not self.train.trajectory_curriculum.enabled
        ):
            raise ValueError("A competence-gated campaign requires an enabled trajectory curriculum.")

    @property
    def run_name(self) -> str:
        return self.train.run_name

    @property
    def launcher_dir(self) -> Path:
        return self.experiment.logs_root / "_launcher"

    @property
    def curriculum_dir(self) -> Path:
        return self.experiment.logs_root / "_curriculum"

    @property
    def state_path(self) -> Path:
        return self.curriculum_dir / f"{self.run_name}_supervisor_state.json"

    @property
    def config_path(self) -> Path:
        return self.curriculum_dir / f"{self.run_name}_campaign.json"

    def supervisor_command(self, *, restart: bool = False) -> list[str]:
        command = [
            "./isaaclab.sh",
            "-p",
            COMPETENCE_SUPERVISOR_SCRIPT,
            "--config",
            str(self.config_path),
        ]
        if restart:
            command.append("--restart")
        return command


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
    use_competence_gate: bool = True,
    segment_iterations: int = 25,
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
    return TrainingCampaign(
        experiment=experiment,
        train=train,
        use_competence_gate=use_competence_gate,
        segment_iterations=segment_iterations,
        total_iterations=max_iterations,
    )


def campaign_payload(campaign: TrainingCampaign) -> dict[str, Any]:
    """Serialize a campaign for the detached competence supervisor."""

    train_payload = asdict(campaign.train)
    for name in ("environment_profile", "domain_randomization_spec"):
        if train_payload[name] is not None:
            train_payload[name] = str(train_payload[name])
    return {
        "experiment": {
            "isaaclab_root": str(campaign.experiment.isaaclab_root),
            "rlpolicy_root": str(campaign.experiment.rlpolicy_root),
            "mlp_architecture": campaign.experiment.mlp_architecture,
            "task_name": campaign.experiment.task_name,
        },
        "train": train_payload,
        "criteria": asdict(campaign.criteria),
        "segment_iterations": campaign.segment_iterations,
        "total_iterations": campaign.total_iterations,
        "state_path": str(campaign.state_path),
    }


def write_campaign_config(campaign: TrainingCampaign) -> Path:
    """Atomically write the supervisor configuration and return its path."""

    path = campaign.config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(campaign_payload(campaign), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


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


def _read_state(campaign: TrainingCampaign) -> dict[str, Any]:
    if not campaign.state_path.is_file():
        return {}
    try:
        return json.loads(campaign.state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid state file"}


def _campaign_run_ids(campaign: TrainingCampaign) -> set[str]:
    run_ids = {campaign.run_name}
    latest_run = _read_state(campaign).get("latest_run")
    if latest_run:
        run_ids.add(str(latest_run))
    return run_ids


def _is_campaign_command(command: str, campaign: TrainingCampaign) -> bool:
    run_ids = _campaign_run_ids(campaign)
    is_train = (
        "simulation/isaac/trajectory/train.py" in command
        and f"--run_name {campaign.run_name}" in command
    )
    is_supervisor = (
        "simulation/isaac/trajectory/competence_curriculum.py" in command
        and str(campaign.config_path) in command
    )
    is_eval = "simulation/isaac/trajectory/evaluate.py" in command and any(
        f"--load_run {run_id}" in command for run_id in run_ids
    )
    return is_train or is_supervisor or is_eval


@dataclass(frozen=True)
class CampaignProcess:
    pid: int
    process_group: int | None
    command: str
    pid_path: Path | None = None


def campaign_processes(campaign: TrainingCampaign) -> tuple[list[CampaignProcess], dict[Path, int]]:
    """Find live worker processes and all launcher PID records for a campaign."""

    pid_records: dict[Path, int] = {}
    for pattern in (f"*_{campaign.run_name}.pid", f"*_{campaign.run_name}_gate.pid"):
        for pid_path in campaign.launcher_dir.glob(pattern):
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
    restart: bool = False,
    replace_running: bool = True,
) -> tuple[int, Path]:
    """Launch a competence supervisor or direct worker in a detached session."""

    _validate_launch_paths(campaign)
    running, _ = campaign_processes(campaign)
    if running and not replace_running:
        raise RuntimeError(f"Campaign {campaign.run_name!r} already has running processes.")
    if running:
        stop_campaign(campaign, clean_stale=True)

    if campaign.use_competence_gate:
        write_campaign_config(campaign)
        command = campaign.supervisor_command(restart=restart)
        launch_suffix = f"{campaign.run_name}_gate"
        label = "COMPETENCE CURRICULUM"
    else:
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

    suffix = "_gate.pid" if campaign.use_competence_gate else ".pid"
    paths = sorted(campaign.launcher_dir.glob(f"*_{campaign.run_name}{suffix}"), reverse=True)
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
    process_role = "supervisor" if campaign.use_competence_gate else "worker"
    lines = [
        f"Campaign: {campaign.run_name} | {process_role}: "
        f"{pid if pid is not None else 'not found'} ({process_status})"
    ]
    state = _read_state(campaign)
    if state:
        lines.append(
            "Gate: {status} | completed: {completed}/{total} | stage: {stage} | latest: {latest}".format(
                status=state.get("status", "unknown"),
                completed=state.get("completed_iterations", 0),
                total=campaign.total_iterations,
                stage=state.get("stage", "unknown"),
                latest=state.get("latest_checkpoint") or "none",
            )
        )

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

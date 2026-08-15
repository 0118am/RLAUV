"""Data contracts shared by trajectory training and evaluation tooling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from robot.control.trajectory import EVALUATION_TRAJECTORY_NAMES
from simulation.isaac.ppo.architectures import MlpArchitecture, get_mlp_architecture
from simulation.isaac.trajectory.evaluation_cases import (
    DEFAULT_CURRENT_TAU_S,
    DEFAULT_DYNAMICS_SCALE,
    DEFAULT_EVALUATION_DURATION_S,
    DEFAULT_RANDOM_CURVE_COUNT,
)

@dataclass(frozen=True)
class ExperimentSpec:
    """Filesystem contract shared by train and eval."""

    isaaclab_root: Path
    rlpolicy_root: Path
    # A named feed-forward profile is the only architecture selector.
    mlp_architecture: str = "mlp_history_5"
    task_name: str = "Isaac-AUV-Traj-Direct-v1"
    train_script: str = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/isaac/trajectory/train.py"
    )
    eval_script: str = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/isaac/trajectory/evaluate.py"
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
    def rsl_experiment_name(self) -> str:
        """Absolute RSL-RL experiment root inside this repository."""

        return str(self.logs_root)

    @property
    def logs_root(self) -> Path:
        return self.rlpolicy_root.expanduser().resolve() / self.architecture.experiment_name

    def results_root(self, run_name: str) -> Path:
        return self.logs_root / run_name / "evaluation"


@dataclass(frozen=True)
class TrajectoryCurriculumRequest:
    """Explicit trajectory curriculum for a training campaign."""

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
    """Parameters controlled by the Python training manager."""

    reward_profile: str
    seed: int = 42
    num_envs: int = 4096
    run_name: str = "trajectory"
    headless: bool = True
    extra_args: tuple[str, ...] = ("--logger", "tensorboard")
    max_iterations: int | None = None
    rollout_steps_per_env: int | None = None
    environment_profile: str | Path | None = None
    domain_randomization_spec: str | Path | None = None
    # ``None`` preserves the recipe's feature selection. An empty tuple is a
    # valid explicit request for deterministic reset/step physics while still
    # retaining the recipe identity in the run manifest.
    domain_randomization_features: tuple[str, ...] | None = None
    trajectory_curriculum: TrajectoryCurriculumRequest | None = None
    resume_load_run: str = ""
    resume_checkpoint: str = ""


@dataclass(frozen=True)
class EvalRequest:
    """Parameters controlled by the evaluation notebook."""

    reward_profile: str
    seed: int = 42
    checkpoint: str | Sequence[str] = "latest"
    trajectories: tuple[str, ...] = EVALUATION_TRAJECTORY_NAMES
    duration_s: float = DEFAULT_EVALUATION_DURATION_S
    headless: bool = True
    skip_existing: bool = True
    include_initial_checkpoint: bool = False
    align_initial_target: bool = True
    random_curve_count: int = DEFAULT_RANDOM_CURVE_COUNT
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
    environment_profile: str | Path | None = None
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
    eval_current_tau: float = DEFAULT_CURRENT_TAU_S
    eval_damping_scale: float = DEFAULT_DYNAMICS_SCALE
    eval_thruster_scale: float = DEFAULT_DYNAMICS_SCALE
    eval_thruster_tau_scale: float = DEFAULT_DYNAMICS_SCALE
    disturbance_name: str | None = None

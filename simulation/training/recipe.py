"""Versioned, strictly validated training recipes and run-local inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from common.schema import (
    FiniteJsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StrictBoolean,
    StrictFrozenModel,
)

from simulation.domain_randomization import (
    DOMAIN_RANDOMIZATION_PARAMETER_NAMES,
    DomainRandomizationProfile,
    DomainRandomizationSpec,
    load_domain_randomization_spec_json,
    write_domain_randomization_spec_json,
)
from environment.profile import (
    EnvironmentProfile,
    FreeSurfaceProfile,
    HydrodynamicsProfile,
    PoolBoundaryProfile,
    load_environment_profile_json,
    write_environment_profile_json,
)
from robot.control.trajectory import (
    AXIS_SINE,
    EVALUATION_TRAJECTORY_NAMES,
    LATERAL_WAVE,
    TrajectoryKinematicLimits,
    VERTICAL_WAVE,
)
from simulation.training.evaluation.config import (
    DEFAULT_CURRENT_TAU_S,
    DEFAULT_DYNAMICS_SCALE,
    DEFAULT_EVALUATION_DURATION_S,
    DEFAULT_RANDOM_CURVE_COUNT,
)
from simulation.training.ppo.networks import MlpArchitecture, get_mlp_architecture
from simulation.training.rewards import canonical_tracking_reward_policy_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RECIPE_SCHEMA_VERSION = 5
DEFAULT_TRAINING_RECIPE = PROJECT_ROOT / "simulation/training/recipes/t60_trajectory_precision_v11.json"


@dataclass(frozen=True)
class ExperimentSpec:
    """Filesystem contract shared by train and eval."""

    isaaclab_root: Path
    rlpolicy_root: Path
    # Supplied by the selected training recipe.
    mlp_architecture: str
    task_name: str = "Isaac-AUV-Traj-Direct-v1"
    train_script: str = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/training/train.py"
    )
    eval_script: str = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/training/evaluation/cli.py"
    )

    @property
    def architecture(self) -> MlpArchitecture:
        return get_mlp_architecture(self.mlp_architecture)

    @property
    def logs_root(self) -> Path:
        return self.rlpolicy_root.expanduser().resolve() / self.architecture.experiment_name

    def results_root(self, run_name: str) -> Path:
        return self.logs_root / run_name / "evaluation"


class AxisSineLevelRequest(StrictFrozenModel):
    """One peak-speed/amplitude pair for selected translation axes."""

    axes: tuple[Literal["surge", "sway", "heave"], ...]
    peak_speed_mps: PositiveFloat
    amplitude_scale: PositiveFloat


class TravelingSineLevelRequest(StrictFrozenModel):
    """One feasible speed/curvature tier for one traveling-wave plane."""

    trajectory: Literal["lateral_wave", "vertical_wave"]
    wave_count: PositiveInt
    path_speed_mps: PositiveFloat
    longitudinal_scale: PositiveFloat
    transverse_scales: tuple[PositiveFloat, ...]


class TrajectoryCurriculumStageRequest(StrictFrozenModel):
    """New command levels introduced at one policy step."""

    start_step: NonNegativeInt
    add_axis_sine_levels: tuple[AxisSineLevelRequest, ...]
    add_traveling_sine_levels: tuple[TravelingSineLevelRequest, ...]


class TrajectoryCurriculumRequest(StrictFrozenModel):
    """Explicit trajectory curriculum for a training campaign."""

    enabled: StrictBoolean
    amplitude_x_range: tuple[PositiveFloat, PositiveFloat]
    amplitude_y_range: tuple[PositiveFloat, PositiveFloat]
    amplitude_z_range: tuple[PositiveFloat, PositiveFloat]
    stages: tuple[TrajectoryCurriculumStageRequest, ...]


RuntimeTrajectoryCommand = tuple[int, int, int, float, tuple[float, float, float]]


def expand_trajectory_stage_commands(
    stage: TrajectoryCurriculumStageRequest,
) -> tuple[RuntimeTrajectoryCommand, ...]:
    """Expand one compact recipe stage into balanced runtime commands."""

    commands: list[RuntimeTrajectoryCommand] = []
    axis_ids = {"surge": 0, "sway": 1, "heave": 2}
    for level in stage.add_axis_sine_levels:
        scale = float(level.amplitude_scale)
        for axis_name in level.axes:
            commands.append(
                (
                    AXIS_SINE,
                    axis_ids[axis_name],
                    1,
                    float(level.peak_speed_mps),
                    (scale, scale, scale),
                )
            )
    for level in stage.add_traveling_sine_levels:
        longitudinal_scale = float(level.longitudinal_scale)
        trajectory_type = (
            LATERAL_WAVE if level.trajectory == "lateral_wave" else VERTICAL_WAVE
        )
        for transverse_scale_value in level.transverse_scales:
            transverse_scale = float(transverse_scale_value)
            amplitude_scales = (
                (longitudinal_scale, transverse_scale, 1.0)
                if trajectory_type == LATERAL_WAVE
                else (longitudinal_scale, 1.0, transverse_scale)
            )
            commands.append(
                (
                    trajectory_type,
                    0,
                    int(level.wave_count),
                    float(level.path_speed_mps),
                    amplitude_scales,
                )
            )
    return tuple(commands)


class InitialStateRequest(StrictFrozenModel):
    """Target-relative reset errors sampled independently for every episode."""

    position_radius_m: NonNegativeFloat
    attitude_error_max_rad: NonNegativeFloat
    linear_velocity_error_max_mps: NonNegativeFloat
    angular_velocity_error_max_radps: NonNegativeFloat


@dataclass(frozen=True)
class TrainRequest:
    """Parameters controlled by the Python training manager."""

    training_recipe: Path
    seed: int = 42
    num_envs: int = 4096
    run_name: str = "trajectory"
    headless: bool = True
    extra_args: tuple[str, ...] = ("--logger", "tensorboard")
    resume_load_run: str = ""
    resume_checkpoint: str = ""


@dataclass(frozen=True)
class EvalRequest:
    """Parameters controlled by the evaluation notebook."""

    seed: int = 42
    checkpoint: str | Sequence[str] = "latest"
    trajectories: tuple[str, ...] = EVALUATION_TRAJECTORY_NAMES
    duration_s: float = DEFAULT_EVALUATION_DURATION_S
    headless: bool = True
    skip_existing: bool = True
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


class TrainingRecipe(StrictFrozenModel):
    """Complete versioned behavior selection for one family of training runs."""

    name: str
    mlp_architecture: str
    reward_profile: str
    environment_base: str
    domain_randomization_base: str
    max_iterations: PositiveInt
    rollout_steps_per_env: PositiveInt
    episode_length_s: PositiveFloat
    trajectory_startup_duration_s: NonNegativeFloat
    use_boundaries: StrictBoolean
    initial_state: InitialStateRequest
    trajectory_curriculum: TrajectoryCurriculumRequest
    kinematic_limits: TrajectoryKinematicLimits
    environment_overrides: dict[str, dict[str, FiniteJsonValue]]
    randomization_overrides: dict[str, FiniteJsonValue]
    schema_version: Literal[TRAINING_RECIPE_SCHEMA_VERSION] = TRAINING_RECIPE_SCHEMA_VERSION
    description: str = ""

    @property
    def architecture(self) -> MlpArchitecture:
        return get_mlp_architecture(self.mlp_architecture)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError(f"Training recipe paths must be repository-relative, got {value!r}.")
        resolved = (PROJECT_ROOT / path).resolve()
        if PROJECT_ROOT not in resolved.parents:
            raise ValueError(f"Training recipe path escapes the repository: {value!r}.")
        return resolved

    @model_validator(mode="after")
    def validate_recipe(self) -> "TrainingRecipe":
        if not self.name.strip():
            raise ValueError("Training recipe name must be non-empty.")
        self.architecture
        canonical_tracking_reward_policy_name(self.reward_profile)
        stages = self.trajectory_curriculum.stages
        if not stages or stages[0].start_step != 0:
            raise ValueError("Trajectory curriculum must start with a stage at policy step 0.")
        if any(right.start_step <= left.start_step for left, right in zip(stages, stages[1:])):
            raise ValueError("Trajectory curriculum stage start steps must be strictly increasing.")
        for stage in stages:
            stage_speeds = [
                *(float(level.peak_speed_mps) for level in stage.add_axis_sine_levels),
                *(float(level.path_speed_mps) for level in stage.add_traveling_sine_levels),
            ]
            if stage_speeds and max(stage_speeds) > float(self.kinematic_limits.max_speed_mps):
                raise ValueError("Trajectory curriculum speed exceeds the kinematic speed limit.")
        if set(self.environment_overrides) - {"hydrodynamics", "pool_boundary", "free_surface"}:
            raise ValueError("environment_overrides accepts only hydrodynamics, pool_boundary, and free_surface.")
        environment_contracts = {
            "hydrodynamics": HydrodynamicsProfile.model_fields,
            "pool_boundary": PoolBoundaryProfile.model_fields,
            "free_surface": FreeSurfaceProfile.model_fields,
        }
        for section, values in self.environment_overrides.items():
            unknown = sorted(set(values) - set(environment_contracts[section]))
            if unknown:
                raise ValueError(
                    f"Unknown environment_overrides.{section} field(s): " + ", ".join(unknown)
                )
        randomization_unknown = sorted(
            set(self.randomization_overrides) - DOMAIN_RANDOMIZATION_PARAMETER_NAMES
        )
        if randomization_unknown:
            raise ValueError(
                "Unknown randomization_overrides field(s): " + ", ".join(randomization_unknown)
            )
        for path in (self.resolve_path(self.environment_base), self.resolve_path(self.domain_randomization_base)):
            if not path.is_file():
                raise FileNotFoundError(f"Training recipe input does not exist: {path}")
        return self

    def resolve_profiles(self) -> tuple[EnvironmentProfile, DomainRandomizationSpec]:
        """Resolve the environment and randomization inputs selected by this recipe."""

        source_environment = load_environment_profile_json(self.resolve_path(self.environment_base))
        hydrodynamics = HydrodynamicsProfile.model_validate(
            source_environment.hydrodynamics.model_dump(mode="python")
            | self.environment_overrides.get("hydrodynamics", {})
        )
        pool_boundary = PoolBoundaryProfile.model_validate(
            source_environment.pool_boundary.model_dump(mode="python")
            | self.environment_overrides.get("pool_boundary", {})
        )
        free_surface = FreeSurfaceProfile.model_validate(
            source_environment.free_surface.model_dump(mode="python")
            | self.environment_overrides.get("free_surface", {})
        )
        environment = EnvironmentProfile(
            name=f"{source_environment.name}--{self.name}",
            description=f"{source_environment.description} Training recipe: {self.name}.",
            hydrodynamics=hydrodynamics,
            pool_boundary=pool_boundary,
            free_surface=free_surface,
        )

        source_randomization = load_domain_randomization_spec_json(
            self.resolve_path(self.domain_randomization_base)
        )
        parameters = DomainRandomizationProfile.model_validate(
            source_randomization.parameters.model_dump(mode="python")
            | self.randomization_overrides
        )
        randomization = DomainRandomizationSpec(
            name=f"{source_randomization.name}--{self.name}",
            description=f"{source_randomization.description} Training recipe: {self.name}.",
            parameters=parameters,
            metadata={**source_randomization.metadata, "training_recipe": self.name},
        )
        return environment, randomization


@dataclass(frozen=True)
class RunInputPaths:
    recipe: Path
    environment: Path
    domain_randomization: Path


def run_input_paths(run_dir: str | Path) -> RunInputPaths:
    """Return the three run-local inputs written beside a checkpoint."""

    input_dir = Path(run_dir).expanduser().resolve() / "params" / "inputs"
    return RunInputPaths(
        recipe=input_dir / "training_recipe.json",
        environment=input_dir / "environment.json",
        domain_randomization=input_dir / "domain_randomization.json",
    )


def apply_trajectory_kinematic_limits(env_cfg: Any, limits: TrajectoryKinematicLimits) -> Any:
    """Apply the trajectory-generator limits selected by a recipe."""

    env_cfg.trajectory_max_speed_mps = limits.max_speed_mps
    env_cfg.trajectory_max_acceleration_mps2 = limits.max_acceleration_mps2
    env_cfg.trajectory_max_yaw_rate_radps = limits.max_yaw_rate_radps
    env_cfg.trajectory_max_jerk_mps3 = limits.max_jerk_mps3
    env_cfg.trajectory_retime_samples = limits.retime_samples
    return env_cfg


def apply_training_recipe(recipe: TrainingRecipe, env_cfg: Any, agent_cfg: Any) -> tuple[Any, Any]:
    """Apply every behavioral training setting from the single recipe source."""

    architecture = recipe.architecture
    curriculum = recipe.trajectory_curriculum

    env_cfg.mlp_architecture = architecture.name
    env_cfg.critic_privileged_fields_override = []
    env_cfg.tracking_reward_profile = recipe.reward_profile
    env_cfg.domain_randomization_feature_override_enabled = False
    env_cfg.episode_length_s = float(recipe.episode_length_s)
    env_cfg.trajectory_startup_duration_s = float(recipe.trajectory_startup_duration_s)
    env_cfg.use_boundaries = bool(recipe.use_boundaries)
    env_cfg.trajectory_initial_position_radius = float(
        recipe.initial_state.position_radius_m
    )
    env_cfg.trajectory_initial_attitude_error_max_rad = float(
        recipe.initial_state.attitude_error_max_rad
    )
    env_cfg.trajectory_initial_linear_velocity_error_max_mps = float(
        recipe.initial_state.linear_velocity_error_max_mps
    )
    env_cfg.trajectory_initial_angular_velocity_error_max_radps = float(
        recipe.initial_state.angular_velocity_error_max_radps
    )
    agent_cfg.policy.actor_hidden_dims = list(architecture.actor_hidden_dims)
    agent_cfg.policy.critic_hidden_dims = list(architecture.critic_hidden_dims)
    agent_cfg.policy.activation = architecture.activation
    agent_cfg.max_iterations = recipe.max_iterations
    agent_cfg.num_steps_per_env = recipe.rollout_steps_per_env

    env_cfg.trajectory_curriculum = curriculum.enabled
    env_cfg.trajectory_amp_x_range = list(curriculum.amplitude_x_range)
    env_cfg.trajectory_amp_y_range = list(curriculum.amplitude_y_range)
    env_cfg.trajectory_amp_z_range = list(curriculum.amplitude_z_range)
    env_cfg.trajectory_curriculum_stage_start_steps = [
        int(stage.start_step) for stage in curriculum.stages
    ]
    active_commands: list[RuntimeTrajectoryCommand] = []
    stage_commands: list[tuple[RuntimeTrajectoryCommand, ...]] = []
    for stage in curriculum.stages:
        active_commands.extend(expand_trajectory_stage_commands(stage))
        stage_commands.append(tuple(active_commands))
    env_cfg.trajectory_curriculum_stage_types = [
        [command[0] for command in commands] for commands in stage_commands
    ]
    env_cfg.trajectory_curriculum_stage_axes = [
        [command[1] for command in commands] for commands in stage_commands
    ]
    env_cfg.trajectory_curriculum_stage_wave_counts = [
        [command[2] for command in commands] for commands in stage_commands
    ]
    env_cfg.trajectory_curriculum_stage_speeds_mps = [
        [command[3] for command in commands] for commands in stage_commands
    ]
    env_cfg.trajectory_curriculum_stage_amplitude_scales = [
        [list(command[4]) for command in commands] for commands in stage_commands
    ]
    apply_trajectory_kinematic_limits(env_cfg, recipe.kinematic_limits)
    return env_cfg, agent_cfg


def load_training_recipe(path: str | Path = DEFAULT_TRAINING_RECIPE) -> TrainingRecipe:
    """Load one strict recipe and reject misspelled or obsolete fields."""

    selected = Path(path).expanduser().resolve()
    return TrainingRecipe.model_validate_json(selected.read_bytes())


def materialize_run_inputs(recipe: TrainingRecipe, run_dir: str | Path) -> RunInputPaths:
    """Write exact resolved inputs inside a newly-created training run."""

    environment, randomization = recipe.resolve_profiles()
    paths = run_input_paths(run_dir)
    input_dir = paths.recipe.parent
    input_dir.mkdir(parents=True, exist_ok=True)
    paths.recipe.write_text(recipe.model_dump_json(indent=2) + "\n", encoding="utf-8")
    write_environment_profile_json(environment, paths.environment)
    write_domain_randomization_spec_json(randomization, paths.domain_randomization)
    return paths

"""Versioned, strictly validated training recipes and run-local inputs."""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Mapping

from environment.profiles.domain_randomization import (
    DomainRandomizationSpec,
    domain_randomization_parameter_names,
    load_domain_randomization_spec_json,
    write_domain_randomization_spec_json,
)
from environment.profiles.environment_profile import (
    EnvironmentProfile,
    FreeSurfaceProfile,
    HydrodynamicsProfile,
    PoolBoundaryProfile,
    load_environment_profile_json,
    write_environment_profile_json,
)
from robot.control.trajectory import (
    EVALUATION_TRAJECTORY_NAMES,
    TRAJECTORY_TYPE_IDS,
    TrajectoryKinematicLimits,
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
TRAINING_RECIPE_SCHEMA_VERSION = 1
DEFAULT_TRAINING_RECIPE = PROJECT_ROOT / "simulation/training/recipes/t60_trajectory_policy_6_v1.json"


@dataclass(frozen=True)
class ExperimentSpec:
    """Filesystem contract shared by train and eval."""

    isaaclab_root: Path
    rlpolicy_root: Path
    # Supplied by a validated training recipe or a loaded run manifest.
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
    training_recipe: Path
    seed: int = 42
    num_envs: int = 4096
    run_name: str = "trajectory"
    headless: bool = True
    extra_args: tuple[str, ...] = ("--logger", "tensorboard")
    max_iterations: int | None = None
    rollout_steps_per_env: int | None = None
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

    reward_profile: str | None = None
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


@dataclass(frozen=True)
class TrainingRecipe:
    """Complete versioned behavior selection for one family of training runs."""

    name: str
    mlp_architecture: str
    reward_profile: str
    environment_base: str
    domain_randomization_base: str
    max_iterations: int
    rollout_steps_per_env: int
    trajectory_curriculum: TrajectoryCurriculumRequest
    kinematic_limits: TrajectoryKinematicLimits
    environment_overrides: Mapping[str, Mapping[str, Any]]
    randomization_overrides: Mapping[str, Any]
    schema_version: int = TRAINING_RECIPE_SCHEMA_VERSION
    description: str = ""

    @property
    def architecture(self):
        return get_mlp_architecture(self.mlp_architecture)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError(f"Training recipe paths must be repository-relative, got {value!r}.")
        resolved = (PROJECT_ROOT / path).resolve()
        if PROJECT_ROOT not in resolved.parents:
            raise ValueError(f"Training recipe path escapes the repository: {value!r}.")
        return resolved

    def validate(self) -> None:
        if self.schema_version != TRAINING_RECIPE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported training recipe schema {self.schema_version}; "
                f"expected {TRAINING_RECIPE_SCHEMA_VERSION}."
            )
        if not self.name.strip():
            raise ValueError("Training recipe name must be non-empty.")
        self.architecture
        if canonical_tracking_reward_policy_name(self.reward_profile) == "custom":
            raise ValueError("Versioned training recipes must select a named reward profile.")
        if self.max_iterations <= 0 or self.rollout_steps_per_env <= 0:
            raise ValueError("Training iteration and rollout counts must be positive.")
        self.kinematic_limits.validate()
        if not self.trajectory_curriculum.stage_0_types:
            raise ValueError("Training recipe must select at least one trajectory type.")
        known_types = set(TRAJECTORY_TYPE_IDS.values())
        for values in (
            self.trajectory_curriculum.stage_0_types,
            self.trajectory_curriculum.stage_1_types,
            self.trajectory_curriculum.stage_2_types,
            self.trajectory_curriculum.stage_3_types,
        ):
            if not set(values).issubset(known_types):
                raise ValueError(f"Unknown trajectory type IDs in training recipe: {values}.")
        if set(self.environment_overrides) - {"hydrodynamics", "pool_boundary", "free_surface"}:
            raise ValueError("environment_overrides accepts only hydrodynamics, pool_boundary, and free_surface.")
        for path in (self.resolve_path(self.environment_base), self.resolve_path(self.domain_randomization_base)):
            if not path.is_file():
                raise FileNotFoundError(f"Training recipe input does not exist: {path}")
        # Resolve the complete inputs during validation so unknown override keys
        # fail before Isaac Sim is launched.
        self.resolve_profiles()

    def resolve_profiles(self) -> tuple[EnvironmentProfile, DomainRandomizationSpec]:
        environment = load_environment_profile_json(self.resolve_path(self.environment_base))
        environment = replace(
            environment,
            name=f"{environment.name}--{self.name}",
            description=f"{environment.description} Training recipe: {self.name}.",
            hydrodynamics=replace(
                environment.hydrodynamics,
                **copy.deepcopy(dict(self.environment_overrides.get("hydrodynamics", {}))),
            ),
            pool_boundary=replace(
                environment.pool_boundary,
                **copy.deepcopy(dict(self.environment_overrides.get("pool_boundary", {}))),
            ),
            free_surface=replace(
                environment.free_surface,
                **copy.deepcopy(dict(self.environment_overrides.get("free_surface", {}))),
            ),
        )
        environment.validate()

        randomization = load_domain_randomization_spec_json(
            self.resolve_path(self.domain_randomization_base)
        )
        parameters = replace(
            randomization.parameters,
            **copy.deepcopy(dict(self.randomization_overrides)),
        )
        randomization = replace(
            randomization,
            name=f"{randomization.name}--{self.name}",
            description=f"{randomization.description} Training recipe: {self.name}.",
            base_profile_name=environment.name,
            parameters=parameters,
            metadata={**dict(randomization.metadata), "training_recipe": self.name},
        )
        randomization.validate()
        return environment, randomization

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trajectory_curriculum"] = asdict(self.trajectory_curriculum)
        data["kinematic_limits"] = asdict(self.kinematic_limits)
        return data


@dataclass(frozen=True)
class RunInputPaths:
    recipe: Path
    environment: Path
    domain_randomization: Path


def _strict_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant {value!r} is not allowed.")
            ),
        )
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a JSON object.")
    return data


def load_training_recipe(path: str | Path = DEFAULT_TRAINING_RECIPE) -> TrainingRecipe:
    """Load one strict recipe and reject misspelled or obsolete fields."""

    selected = Path(path).expanduser().resolve()
    data = _strict_json(selected)
    allowed = {
        "schema_version",
        "name",
        "description",
        "mlp_architecture",
        "reward_profile",
        "environment_base",
        "domain_randomization_base",
        "max_iterations",
        "rollout_steps_per_env",
        "trajectory_curriculum",
        "kinematic_limits",
        "environment_overrides",
        "randomization_overrides",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("Unknown training recipe field(s): " + ", ".join(unknown))
    required = allowed - {"schema_version", "description"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("Training recipe is missing field(s): " + ", ".join(missing))
    curriculum_data = data["trajectory_curriculum"]
    limits_data = data["kinematic_limits"]
    if not isinstance(curriculum_data, Mapping) or not isinstance(limits_data, Mapping):
        raise TypeError("trajectory_curriculum and kinematic_limits must be JSON objects.")
    nested_contracts = (
        ("trajectory_curriculum", curriculum_data, {item.name for item in fields(TrajectoryCurriculumRequest)}),
        ("kinematic_limits", limits_data, {item.name for item in fields(TrajectoryKinematicLimits)}),
    )
    for name, values, field_names in nested_contracts:
        nested_unknown = sorted(set(values) - field_names)
        if nested_unknown:
            raise ValueError(f"Unknown {name} field(s): " + ", ".join(nested_unknown))
    environment_overrides = data["environment_overrides"]
    randomization_overrides = data["randomization_overrides"]
    if not isinstance(environment_overrides, Mapping) or not isinstance(randomization_overrides, Mapping):
        raise TypeError("environment_overrides and randomization_overrides must be JSON objects.")
    environment_contracts = {
        "hydrodynamics": {item.name for item in fields(HydrodynamicsProfile)},
        "pool_boundary": {item.name for item in fields(PoolBoundaryProfile)},
        "free_surface": {item.name for item in fields(FreeSurfaceProfile)},
    }
    for section, values in environment_overrides.items():
        if section not in environment_contracts:
            raise ValueError(f"Unknown environment_overrides section: {section}")
        if not isinstance(values, Mapping):
            raise TypeError(f"environment_overrides.{section} must be a JSON object.")
        nested_unknown = sorted(set(values) - environment_contracts[section])
        if nested_unknown:
            raise ValueError(
                f"Unknown environment_overrides.{section} field(s): " + ", ".join(nested_unknown)
            )
    randomization_unknown = sorted(
        set(randomization_overrides) - domain_randomization_parameter_names()
    )
    if randomization_unknown:
        raise ValueError(
            "Unknown randomization_overrides field(s): " + ", ".join(randomization_unknown)
        )
    recipe = TrainingRecipe(
        schema_version=int(data.get("schema_version", TRAINING_RECIPE_SCHEMA_VERSION)),
        name=str(data["name"]),
        description=str(data.get("description", "")),
        mlp_architecture=str(data["mlp_architecture"]),
        reward_profile=str(data["reward_profile"]),
        environment_base=str(data["environment_base"]),
        domain_randomization_base=str(data["domain_randomization_base"]),
        max_iterations=int(data["max_iterations"]),
        rollout_steps_per_env=int(data["rollout_steps_per_env"]),
        trajectory_curriculum=TrajectoryCurriculumRequest(**copy.deepcopy(dict(curriculum_data))),
        kinematic_limits=TrajectoryKinematicLimits(**copy.deepcopy(dict(limits_data))),
        environment_overrides=copy.deepcopy(dict(environment_overrides)),
        randomization_overrides=copy.deepcopy(dict(randomization_overrides)),
    )
    recipe.validate()
    return recipe


def materialize_run_inputs(recipe: TrainingRecipe, run_dir: str | Path) -> RunInputPaths:
    """Write exact resolved inputs inside a newly-created training run."""

    recipe.validate()
    environment, randomization = recipe.resolve_profiles()
    input_dir = Path(run_dir).resolve() / "params" / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = input_dir / "training_recipe.json"
    environment_path = input_dir / "environment.json"
    randomization_path = input_dir / "domain_randomization.json"
    with recipe_path.open("w", encoding="utf-8") as stream:
        json.dump(recipe.to_dict(), stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    write_environment_profile_json(environment, environment_path)
    write_domain_randomization_spec_json(randomization, randomization_path)
    return RunInputPaths(recipe_path, environment_path, randomization_path)

"""Portable per-run contract used by training, evaluation, and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from robot.control.trajectory.observation_contract import ACTION_DIM
from simulation.training.ppo.networks import get_mlp_architecture
from simulation.training.rewards import canonical_tracking_reward_policy_name


RUN_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunManifest:
    recipe_name: str
    task_name: str
    mlp_architecture: str
    reward_profile: str
    observation_dim: int
    critic_observation_dim: int
    action_dim: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    physics_dt: float
    decimation: int
    seed: int
    num_envs: int
    rollout_steps_per_env: int
    max_iterations: int
    inputs: Mapping[str, str]
    schema_version: int = RUN_MANIFEST_SCHEMA_VERSION
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @property
    def run_dir(self) -> Path:
        if self.source_path is None:
            raise ValueError("RunManifest was not loaded from or written to a run directory.")
        return self.source_path.parent.parent

    def input_path(self, name: str) -> Path:
        if name not in self.inputs:
            raise KeyError(f"Run manifest does not define input {name!r}.")
        relative = Path(self.inputs[name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Run manifest input must be run-relative: {relative}")
        path = self.run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Run input is missing: {path}")
        return path

    def validate(self, *, check_inputs: bool = True) -> None:
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported run manifest schema {self.schema_version}.")
        architecture = get_mlp_architecture(self.mlp_architecture)
        canonical_tracking_reward_policy_name(self.reward_profile)
        expected = (
            architecture.observation_dim,
            architecture.critic_observation_dim,
            ACTION_DIM,
            architecture.actor_hidden_dims,
            architecture.critic_hidden_dims,
        )
        actual = (
            self.observation_dim,
            self.critic_observation_dim,
            self.action_dim,
            tuple(self.actor_hidden_dims),
            tuple(self.critic_hidden_dims),
        )
        if actual != expected:
            raise ValueError(f"Run manifest network contract does not match {self.mlp_architecture!r}.")
        if self.physics_dt <= 0.0 or self.decimation <= 0 or self.num_envs <= 0:
            raise ValueError("Run manifest contains invalid simulation dimensions.")
        required_inputs = {"recipe", "environment", "domain_randomization"}
        if set(self.inputs) != required_inputs:
            raise ValueError(f"Run manifest inputs must be exactly {sorted(required_inputs)}.")
        if check_inputs and self.source_path is not None:
            for name in required_inputs:
                self.input_path(name)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        data["actor_hidden_dims"] = list(self.actor_hidden_dims)
        data["critic_hidden_dims"] = list(self.critic_hidden_dims)
        data["inputs"] = dict(self.inputs)
        return data


def build_run_manifest(
    *, recipe, task_name: str, env_cfg: Any, agent_cfg: Any, run_dir: str | Path
) -> RunManifest:
    architecture = recipe.architecture
    configured_actor_dims = tuple(agent_cfg.policy.actor_hidden_dims)
    configured_critic_dims = tuple(agent_cfg.policy.critic_hidden_dims)
    if configured_actor_dims != architecture.actor_hidden_dims:
        raise ValueError(
            f"Agent actor hidden layers {configured_actor_dims} do not match recipe "
            f"{architecture.actor_hidden_dims}."
        )
    if configured_critic_dims != architecture.critic_hidden_dims:
        raise ValueError(
            f"Agent critic hidden layers {configured_critic_dims} do not match recipe "
            f"{architecture.critic_hidden_dims}."
        )
    path = Path(run_dir).resolve() / "params" / "run_manifest.json"
    manifest = RunManifest(
        recipe_name=recipe.name,
        task_name=task_name,
        mlp_architecture=architecture.name,
        reward_profile=recipe.reward_profile,
        observation_dim=architecture.observation_dim,
        critic_observation_dim=architecture.critic_observation_dim,
        action_dim=ACTION_DIM,
        actor_hidden_dims=architecture.actor_hidden_dims,
        critic_hidden_dims=architecture.critic_hidden_dims,
        activation=str(agent_cfg.policy.activation),
        physics_dt=float(env_cfg.sim.dt),
        decimation=int(env_cfg.decimation),
        seed=int(agent_cfg.seed),
        num_envs=int(env_cfg.scene.num_envs),
        rollout_steps_per_env=int(agent_cfg.num_steps_per_env),
        max_iterations=int(agent_cfg.max_iterations),
        inputs={
            "recipe": "params/inputs/training_recipe.json",
            "environment": "params/inputs/environment.json",
            "domain_randomization": "params/inputs/domain_randomization.json",
        },
        source_path=path,
    )
    manifest.validate()
    return manifest


def write_run_manifest(manifest: RunManifest) -> Path:
    if manifest.source_path is None:
        raise ValueError("Run manifest requires a destination source_path.")
    manifest.validate()
    manifest.source_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest.source_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest.to_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest.source_path


def load_run_manifest(value: str | Path) -> RunManifest:
    selected = Path(value).expanduser().resolve()
    path = selected / "params" / "run_manifest.json" if selected.is_dir() else selected
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a JSON object.")
    allowed = {field.name for field in RunManifest.__dataclass_fields__.values()} - {"source_path"}
    unknown = sorted(set(data) - allowed)
    missing = sorted(allowed - set(data))
    if unknown or missing:
        raise ValueError(f"Invalid run manifest fields; unknown={unknown}, missing={missing}.")
    manifest = RunManifest(
        **{key: data[key] for key in allowed if key not in {"actor_hidden_dims", "critic_hidden_dims"}},
        actor_hidden_dims=tuple(data["actor_hidden_dims"]),
        critic_hidden_dims=tuple(data["critic_hidden_dims"]),
        source_path=path,
    )
    manifest.validate()
    return manifest


def validate_manifest_selection(
    manifest: RunManifest,
    *,
    mlp_architecture: str | None = None,
    reward_profile: str | None = None,
) -> None:
    if mlp_architecture is not None and mlp_architecture != manifest.mlp_architecture:
        raise ValueError(
            f"Requested architecture {mlp_architecture!r} does not match run manifest "
            f"{manifest.mlp_architecture!r}."
        )
    if reward_profile is not None and reward_profile != manifest.reward_profile:
        raise ValueError(
            f"Requested reward profile {reward_profile!r} does not match run manifest "
            f"{manifest.reward_profile!r}."
        )

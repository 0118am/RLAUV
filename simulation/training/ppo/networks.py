"""Single registry of deployable Actor/Critic network designs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from rsl_rl.networks import EmpiricalNormalization, MLP

from robot.control.trajectory.observation_contract import (
    BASE_OBSERVATION_DIM,
    HISTORY_OBSERVATION_FIELD_DIMENSIONS,
)
from simulation.training.ppo.squashed_actor_critic import (
    ACTION_SQUASH_VERSION,
    ACTION_SQUASH_VERSION_STATE_KEY,
)


TRAJECTORY_CRITIC_PRIVILEGED_FIELDS = (
    "true_root_state",
    "water_current_b",
    "generalized_acceleration_b",
    "effective_linear_damping_ratio",
    "effective_quadratic_damping_ratio",
    "effective_fluid_added_mass_ratio",
    "effective_buoyant_volume_ratio",
    "realized_thruster_force",
    "thruster_force_scale",
    "common_thruster_force_scale",
    "thruster_parameters",
)

CRITIC_PRIVILEGED_FIELD_DIMENSIONS = {
    "true_root_state": 13,
    "water_current_b": 3,
    "generalized_acceleration_b": 6,
    "effective_linear_damping_ratio": 6,
    "effective_quadratic_damping_ratio": 6,
    "effective_fluid_added_mass_ratio": 6,
    "effective_buoyant_volume_ratio": 1,
    "realized_thruster_force": 8,
    "thruster_force_scale": 8,
    "common_thruster_force_scale": 1,
    "thruster_parameters": 2,
}


@dataclass(frozen=True)
class MlpArchitecture:
    """Complete feed-forward input, history, and layer-width contract."""

    name: str
    history_steps: int
    history_fields: tuple[str, ...]
    critic_privileged_fields: tuple[str, ...]
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    experiment_name: str

    @property
    def history_feature_dim(self) -> int:
        return sum(
            HISTORY_OBSERVATION_FIELD_DIMENSIONS[name]
            for name in self.history_fields
        )

    @property
    def observation_dim(self) -> int:
        return BASE_OBSERVATION_DIM + self.history_steps * self.history_feature_dim

    @property
    def critic_privileged_dim(self) -> int:
        return sum(
            CRITIC_PRIVILEGED_FIELD_DIMENSIONS[name]
            for name in self.critic_privileged_fields
        )

    @property
    def critic_observation_dim(self) -> int:
        return self.observation_dim + self.critic_privileged_dim


MLP_30D = MlpArchitecture(
    name="mlp_30d",
    history_steps=0,
    history_fields=(),
    critic_privileged_fields=TRAJECTORY_CRITIC_PRIVILEGED_FIELDS,
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    activation="elu",
    experiment_name="auv_traj_mlp",
)

MLP_HISTORY_8 = MlpArchitecture(
    name="mlp_history_8",
    # Eight prior 25 Hz samples cover 320 ms: the fixed 50 ms communication
    # delay, measured 50 ms sensor delay, and more than three 40 ms actuator
    # time constants.
    history_steps=8,
    history_fields=(
        "position_error_b",
        "linear_velocity_error_b",
        "attitude_error_quat",
        "angular_velocity_error_b",
        "previous_motor_command",
    ),
    critic_privileged_fields=TRAJECTORY_CRITIC_PRIVILEGED_FIELDS,
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    activation="elu",
    experiment_name="auv_traj_mlp_history_8",
)

MLP_ARCHITECTURES = {
    architecture.name: architecture
    for architecture in (MLP_30D, MLP_HISTORY_8)
}


def initialize_ppo_mlp(mlp: MLP, *, output_gain: float) -> None:
    """Apply the PPO orthogonal initialization and zero all linear biases."""

    linear_layers = [
        layer for layer in mlp if isinstance(layer, torch.nn.Linear)
    ]
    for layer in linear_layers[:-1]:
        torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        torch.nn.init.zeros_(layer.bias)

    output_layer = linear_layers[-1]
    torch.nn.init.orthogonal_(output_layer.weight, gain=output_gain)
    torch.nn.init.zeros_(output_layer.bias)


def get_mlp_architecture(name: str) -> MlpArchitecture:
    try:
        return MLP_ARCHITECTURES[name]
    except KeyError as error:
        available = ", ".join(MLP_ARCHITECTURES)
        raise ValueError(
            f"Unknown MLP architecture {name!r}. Available: {available}."
        ) from error


class TrajectoryEvaluationActor(torch.nn.Module):
    """Deterministic tanh-squashed actor without critic modules."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str,
        normalize_observations: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer(
            ACTION_SQUASH_VERSION_STATE_KEY,
            torch.tensor(ACTION_SQUASH_VERSION, dtype=torch.int64),
        )
        self.actor_obs_normalizer = (
            EmpiricalNormalization(observation_dim)
            if normalize_observations
            else torch.nn.Identity()
        )
        self.actor = MLP(observation_dim, action_dim, hidden_dims, activation)

    def latent_action_mean(
        self,
        observations: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Return the unconstrained Gaussian mean before the action transform."""

        policy_observation = (
            observations["policy"]
            if not isinstance(observations, torch.Tensor)
            else observations
        )
        return self.actor(self.actor_obs_normalizer(policy_observation))

    def action_and_latent_mean(
        self,
        observations: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded command and its pre-tanh mean in one Actor pass."""

        latent_mean = self.latent_action_mean(observations)
        return torch.tanh(latent_mean), latent_mean

    def forward(self, observations: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        actions, _ = self.action_and_latent_mean(observations)
        return actions


def load_evaluation_actor(
    checkpoint_path: str | Path,
    *,
    observation_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    activation: str,
    device: str | torch.device,
) -> TrajectoryEvaluationActor:
    """Load actor weights without moving training state to the target device."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = checkpoint["model_state_dict"]
    normalizer_state = {
        name: value
        for name, value in state_dict.items()
        if name.startswith("actor_obs_normalizer.")
    }
    actor = TrajectoryEvaluationActor(
        observation_dim,
        action_dim,
        hidden_dims,
        activation,
        normalize_observations=bool(normalizer_state),
    )
    deployable_state = {
        name: value
        for name, value in state_dict.items()
        if name == ACTION_SQUASH_VERSION_STATE_KEY
        or name.startswith("actor.")
        or name.startswith("actor_obs_normalizer.")
    }
    actor.load_state_dict(deployable_state)
    actor = actor.to(device)
    actor.eval()
    return actor

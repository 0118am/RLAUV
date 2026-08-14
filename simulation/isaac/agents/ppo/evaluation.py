"""Actor-only checkpoint loading for trajectory evaluation and deployment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from rsl_rl.networks import EmpiricalNormalization, MLP


class TrajectoryEvaluationActor(torch.nn.Module):
    """Deterministic feed-forward Actor without critic modules."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str,
        normalize_observations: bool = False,
    ) -> None:
        super().__init__()
        self.actor_obs_normalizer = (
            EmpiricalNormalization(observation_dim) if normalize_observations else torch.nn.Identity()
        )
        self.actor = MLP(observation_dim, action_dim, hidden_dims, activation)

    def forward(self, observations: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        policy_observation = (
            observations["policy"] if not isinstance(observations, torch.Tensor) else observations
        )
        return self.actor(self.actor_obs_normalizer(policy_observation))


def _state_with_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }


def load_evaluation_actor(
    checkpoint_path: str | Path,
    *,
    observation_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    activation: str,
    device: str | torch.device,
) -> TrajectoryEvaluationActor:
    """Load actor-side weights safely without moving training state to the target device."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = checkpoint["model_state_dict"]
    normalizer_state = _state_with_prefix(state_dict, "actor_obs_normalizer.")
    actor = TrajectoryEvaluationActor(
        observation_dim,
        action_dim,
        hidden_dims,
        activation,
        normalize_observations=bool(normalizer_state),
    )
    actor.actor.load_state_dict(_state_with_prefix(state_dict, "actor."))
    if normalizer_state:
        actor.actor_obs_normalizer.load_state_dict(normalizer_state)
    actor = actor.to(device)
    actor.eval()
    return actor

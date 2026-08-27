"""Architecture-owned actor loading shared by evaluation and export."""

from __future__ import annotations

from pathlib import Path

from robot.control.trajectory.observation_contract import ACTION_DIM
from simulation.training.ppo.networks import MlpArchitecture, load_evaluation_actor


def load_actor(
    checkpoint: str | Path,
    architecture: MlpArchitecture,
    *,
    device: str,
):
    """Load a checkpoint using its named recipe architecture."""

    return load_evaluation_actor(
        checkpoint,
        observation_dim=architecture.observation_dim,
        action_dim=ACTION_DIM,
        hidden_dims=list(architecture.actor_hidden_dims),
        activation=architecture.activation,
        device=device,
    )

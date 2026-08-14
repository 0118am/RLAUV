"""Tests for critic-free trajectory checkpoint evaluation."""

from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic

from simulation.isaac.agents.ppo.evaluation import load_evaluation_actor


def _full_policy(*, normalize_observations: bool = False):
    observations = TensorDict({"policy": torch.zeros(2, 30)}, batch_size=[2])
    return ActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=8,
        actor_obs_normalization=normalize_observations,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )


def test_mlp_evaluation_loads_actor_without_critic(tmp_path: Path) -> None:
    torch.manual_seed(7)
    full_policy = _full_policy()
    checkpoint_path = tmp_path / "model_1.pt"
    torch.save({"model_state_dict": full_policy.state_dict()}, checkpoint_path)
    evaluation_actor = load_evaluation_actor(
        checkpoint_path,
        observation_dim=30,
        action_dim=8,
        hidden_dims=[512, 256, 128],
        activation="elu",
        device="cpu",
    )
    observations = TensorDict({"policy": torch.randn(2, 30)}, batch_size=[2])

    full_policy.reset()
    expected = full_policy.act_inference(observations)
    actual = evaluation_actor(observations)

    assert torch.allclose(actual, expected)
    assert not hasattr(evaluation_actor, "critic")
    assert all(not name.startswith("critic") for name, _ in evaluation_actor.named_parameters())


def test_mlp_evaluation_preserves_actor_observation_normalization(tmp_path: Path) -> None:
    torch.manual_seed(17)
    full_policy = _full_policy(normalize_observations=True)
    normalization_samples = torch.randn(128, 30) * 3.0 + 7.0
    full_policy.actor_obs_normalizer.update(normalization_samples)
    checkpoint_path = tmp_path / "normalized_model.pt"
    torch.save({"model_state_dict": full_policy.state_dict()}, checkpoint_path)

    evaluation_actor = load_evaluation_actor(
        checkpoint_path,
        observation_dim=30,
        action_dim=8,
        hidden_dims=[512, 256, 128],
        activation="elu",
        device="cpu",
    )
    observations = TensorDict({"policy": torch.randn(2, 30) * 2.0 + 4.0}, batch_size=[2])

    full_policy.eval()
    expected = full_policy.act_inference(observations)
    actual = evaluation_actor(observations)

    assert torch.allclose(actual, expected)

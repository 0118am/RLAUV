"""Regression checks for the selected native RSL-RL PPO behavior."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def _native_ppo() -> tuple[PPO, TensorDict]:
    torch.manual_seed(7)
    num_envs = 32
    observation_dim = 4
    action_dim = 2
    observations = TensorDict(
        {"policy": torch.randn(num_envs, observation_dim)},
        batch_size=[num_envs],
    )
    policy = ActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=action_dim,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        activation="elu",
        init_noise_std=0.5,
        noise_std_type="log",
    )
    algorithm = PPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.994009,
        lam=0.9604,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=3.0e-4,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
    )
    algorithm.init_storage("rl", num_envs, 4, observations, [action_dim])
    return algorithm, observations


def _collect_rollout(algorithm: PPO, observations: TensorDict) -> None:
    for _ in range(4):
        algorithm.act(observations)
        algorithm.process_env_step(
            observations,
            torch.randn(observations.batch_size[0]),
            torch.zeros(observations.batch_size[0], dtype=torch.bool),
            {},
        )
    algorithm.compute_returns(observations)


def test_native_adaptive_ppo_reduces_shared_learning_rate_for_high_kl() -> None:
    algorithm, observations = _native_ppo()
    _collect_rollout(algorithm, observations)
    algorithm.storage.mu.add_(10.0)

    losses = algorithm.update()

    assert type(algorithm) is PPO
    assert algorithm.schedule == "adaptive"
    assert algorithm.desired_kl == 0.01
    assert algorithm.learning_rate < 3.0e-4
    assert len(algorithm.optimizer.param_groups) == 1
    assert algorithm.optimizer.param_groups[0]["lr"] == algorithm.learning_rate
    assert set(losses) == {"value_function", "surrogate", "entropy"}

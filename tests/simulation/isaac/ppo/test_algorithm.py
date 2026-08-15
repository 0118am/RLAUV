"""Unit checks for trajectory PPO's rollout-level KL controller."""

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic

from simulation.isaac.ppo.algorithm import RolloutAdaptivePPO


def _algorithm(*, desired_kl: float, kl_stop: float, kl_low: float) -> tuple[RolloutAdaptivePPO, TensorDict]:
    torch.manual_seed(7)
    num_envs, observation_dim, action_dim = 64, 4, 2
    observations = TensorDict({"policy": torch.randn(num_envs, observation_dim)}, batch_size=[num_envs])
    policy = ActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=action_dim,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        activation="elu",
        init_noise_std=0.5,
    )
    algorithm = RolloutAdaptivePPO(
        policy,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        desired_kl=desired_kl,
        rollout_kl_stop=kl_stop,
        rollout_kl_low=kl_low,
        rollout_lr_min=1.0e-4,
        rollout_lr_max=5.0e-4,
        device="cpu",
    )
    algorithm.init_storage("rl", num_envs, 8, observations, [action_dim])
    return algorithm, observations


def _collect_rollout(algorithm: RolloutAdaptivePPO, observations: TensorDict) -> TensorDict:
    for _ in range(8):
        algorithm.act(observations)
        next_observations = TensorDict(
            {"policy": torch.randn(observations.batch_size[0], observations["policy"].shape[-1])},
            batch_size=observations.batch_size,
        )
        algorithm.process_env_step(
            next_observations,
            torch.randn(observations.batch_size[0]),
            torch.zeros(observations.batch_size[0], dtype=torch.bool),
            {},
        )
        observations = next_observations
    algorithm.compute_returns(observations)
    return observations


def test_rollout_controller_early_stops_and_reduces_next_learning_rate() -> None:
    algorithm, observations = _algorithm(desired_kl=5.0e-12, kl_stop=1.0e-10, kl_low=1.0e-12)
    _collect_rollout(algorithm, observations)

    losses = algorithm.update()

    assert algorithm.last_rollout_early_stop
    assert algorithm.last_rollout_updates < 8
    assert losses["early_stop"] == 1.0
    assert losses["ppo_updates"] == algorithm.last_rollout_updates
    assert algorithm.learning_rate < 3.0e-4


def test_rollout_controller_raises_learning_rate_only_after_the_rollout() -> None:
    algorithm, observations = _algorithm(desired_kl=0.01, kl_stop=0.015, kl_low=0.005)
    _collect_rollout(algorithm, observations)

    losses = algorithm.update()

    assert not algorithm.last_rollout_early_stop
    assert algorithm.last_rollout_updates == 8
    assert losses["kl_probe"] < 0.005
    assert algorithm.learning_rate == 3.3e-4
    assert losses["early_stop"] == 0.0

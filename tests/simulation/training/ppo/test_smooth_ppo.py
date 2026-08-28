"""Focused regression checks for the selected smooth PPO behavior."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.networks import MLP

from simulation.training.ppo.networks import initialize_ppo_mlp
from simulation.training.ppo.squashed_actor_critic import (
    AUVTanhGaussianActorCritic,
)
from simulation.training.ppo.smooth_ppo import (
    AUVSmoothPPO,
    diagonal_gaussian_kl_divergence,
    normalized_action_curvature_loss,
    normalized_vertical_action_curvature_loss,
)


def test_ppo_mlp_initialization_is_orthogonal_with_zero_bias() -> None:
    mlp = MLP(5, 2, [7, 4], "elu")

    initialize_ppo_mlp(mlp, output_gain=0.01)

    linear_layers = [layer for layer in mlp if isinstance(layer, torch.nn.Linear)]
    for layer in linear_layers:
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    output_weight = linear_layers[-1].weight
    torch.testing.assert_close(
        output_weight @ output_weight.T,
        torch.eye(output_weight.shape[0]) * 0.01**2,
    )


def _smooth_ppo() -> tuple[AUVSmoothPPO, TensorDict]:
    torch.manual_seed(7)
    num_envs = 32
    observation_dim = 4
    action_dim = 8
    observations = TensorDict(
        {"policy": torch.randn(num_envs, observation_dim)},
        batch_size=[num_envs],
    )
    policy = AUVTanhGaussianActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=action_dim,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        activation="elu",
        init_noise_std=0.5,
        noise_std_type="log",
    )
    algorithm = AUVSmoothPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.994009,
        lam=0.9604,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=3.0e-5,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        critic_learning_rate=3.0e-4,
        action_curvature_loss_coef=2.5,
        vertical_action_curvature_loss_coef=0.5,
        action_curvature_policy_dt_s=0.04,
        action_curvature_scale_per_s2=625.0,
    )
    algorithm.init_storage("rl", num_envs, 4, observations, [action_dim])
    return algorithm, observations


def _collect_rollout(algorithm: AUVSmoothPPO, observations: TensorDict) -> None:
    for _ in range(4):
        algorithm.act(observations)
        algorithm.process_env_step(
            observations,
            torch.randn(observations.batch_size[0]),
            torch.zeros(observations.batch_size[0], dtype=torch.bool),
            {},
        )
    algorithm.compute_returns(observations)


def test_training_policy_samples_bounded_motor_commands() -> None:
    algorithm, observations = _smooth_ppo()

    actions = algorithm.act(observations)

    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)
    torch.testing.assert_close(
        actions,
        torch.tanh(algorithm.policy.last_latent_actions),
    )
    torch.testing.assert_close(
        algorithm.policy.act_inference(observations),
        algorithm.policy.bounded_action_mean,
    )


def test_squashed_log_prob_matches_torch_transformed_distribution() -> None:
    algorithm, observations = _smooth_ppo()
    policy = algorithm.policy
    policy.act(observations)
    latent_actions = policy.last_latent_actions
    transform = torch.distributions.TanhTransform(cache_size=1)
    actions = transform(latent_actions)
    expected = torch.distributions.TransformedDistribution(
        policy.distribution,
        [transform],
    ).log_prob(actions).sum(dim=-1)

    from_latent = policy.get_actions_log_prob_from_latent(latent_actions)
    from_actions = policy.get_actions_log_prob(actions)

    torch.testing.assert_close(from_latent, expected)
    torch.testing.assert_close(from_actions, expected, rtol=1.0e-5, atol=1.0e-5)


def test_squashed_log_prob_remains_finite_when_float_tanh_reaches_endpoint() -> None:
    algorithm, observations = _smooth_ppo()
    output_layer = [
        layer
        for layer in algorithm.policy.actor
        if isinstance(layer, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.fill_(30.0)

    actions = algorithm.policy.act(observations)
    log_prob = algorithm.policy.get_actions_log_prob_from_latent(
        algorithm.policy.last_latent_actions
    )

    assert torch.all(actions == 1.0)
    assert torch.all(torch.isfinite(log_prob))
    assert torch.all(torch.isfinite(algorithm.policy.entropy))


def test_rollout_stores_executed_and_pre_tanh_actions_consistently() -> None:
    algorithm, observations = _smooth_ppo()
    _collect_rollout(algorithm, observations)

    assert torch.all(algorithm.storage.actions > -1.0)
    assert torch.all(algorithm.storage.actions < 1.0)
    torch.testing.assert_close(
        algorithm.storage.actions,
        torch.tanh(algorithm.storage.pre_tanh_actions),
    )


def test_smooth_ppo_uses_separate_fixed_actor_and_critic_rates() -> None:
    algorithm, observations = _smooth_ppo()
    _collect_rollout(algorithm, observations)

    losses = algorithm.update()

    assert type(algorithm) is AUVSmoothPPO
    assert algorithm.schedule == "fixed"
    assert algorithm.desired_kl == 0.01
    assert algorithm.learning_rate == 3.0e-5
    assert algorithm.critic_learning_rate == 3.0e-4
    assert len(algorithm.optimizer.param_groups) == 2
    assert algorithm.optimizer.param_groups[0]["lr"] == algorithm.learning_rate
    assert (
        algorithm.optimizer.param_groups[1]["lr"]
        == algorithm.critic_learning_rate
    )
    optimized_parameters = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_parameters == {
        id(parameter) for parameter in algorithm.policy.parameters()
    }
    assert set(losses) == {
        "value_function",
        "surrogate",
        "entropy",
        "action_curvature",
        "vertical_action_curvature",
        "kl",
        "kl_max",
        "actor_update_fraction",
        "critic_update_fraction",
    }


def test_diagonal_gaussian_kl_matches_torch_distribution() -> None:
    old_mean = torch.tensor([[0.0, -0.5], [0.25, 0.75]])
    old_std = torch.tensor([[1.0, 0.5], [0.75, 1.25]])
    new_mean = torch.tensor([[0.5, 0.25], [-0.5, 1.0]])
    new_std = torch.tensor([[0.75, 1.5], [1.25, 0.5]])

    actual = diagonal_gaussian_kl_divergence(
        old_mean,
        old_std,
        new_mean,
        new_std,
    )
    expected = torch.distributions.kl_divergence(
        torch.distributions.Independent(
            torch.distributions.Normal(old_mean, old_std),
            1,
        ),
        torch.distributions.Independent(
            torch.distributions.Normal(new_mean, new_std),
            1,
        ),
    )

    torch.testing.assert_close(actual, expected)


def test_kl_early_stopping_only_ends_remaining_actor_updates() -> None:
    algorithm, observations = _smooth_ppo()
    algorithm.desired_kl = 1.0e-12
    _collect_rollout(algorithm, observations)

    losses = algorithm.update()

    assert losses["kl_max"] > 1.5 * algorithm.desired_kl
    assert losses["actor_update_fraction"] == 0.5
    assert losses["critic_update_fraction"] == 1.0
    actor_step = algorithm.optimizer.state[algorithm._actor_parameters[0]]["step"]
    critic_step = algorithm.optimizer.state[algorithm._critic_parameters[0]]["step"]
    assert actor_step.item() == 1
    assert critic_step.item() == 2


def test_normalized_action_curvature_is_zero_for_ramp_and_positive_for_reversal() -> None:
    previous_previous = torch.tensor([[-0.5, -0.5]])
    previous = torch.zeros((1, 2))
    ramp = torch.tensor([[0.5, 0.5]])
    reversal = torch.tensor([[-0.5, -0.5]])

    ramp_loss = normalized_action_curvature_loss(
        ramp,
        previous,
        previous_previous,
        policy_dt_s=0.04,
        acceleration_scale_per_s2=625.0,
    )
    reversal_loss = normalized_action_curvature_loss(
        reversal,
        previous,
        previous_previous,
        policy_dt_s=0.04,
        acceleration_scale_per_s2=625.0,
    )

    torch.testing.assert_close(ramp_loss, torch.tensor(0.0))
    torch.testing.assert_close(reversal_loss, torch.tensor(1.0))


def test_vertical_action_curvature_penalizes_each_t1_t4_channel() -> None:
    previous = torch.zeros((1, 8))
    previous_previous = torch.tensor(
        [[-0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]
    )
    ramp = -previous_previous
    reversal = previous_previous
    common_vertical_reversal = torch.tensor(
        [[-0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]]
    )
    horizontal_reversal = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5]]
    )

    ramp_loss = normalized_vertical_action_curvature_loss(
        ramp,
        previous,
        previous_previous,
        policy_dt_s=0.04,
        acceleration_scale_per_s2=625.0,
    )
    reversal_loss = normalized_vertical_action_curvature_loss(
        reversal,
        previous,
        previous_previous,
        policy_dt_s=0.04,
        acceleration_scale_per_s2=625.0,
    )
    common_vertical_loss = normalized_vertical_action_curvature_loss(
        common_vertical_reversal,
        previous,
        common_vertical_reversal,
        policy_dt_s=0.04,
        acceleration_scale_per_s2=625.0,
    )
    horizontal_loss = normalized_vertical_action_curvature_loss(
        horizontal_reversal,
        previous,
        horizontal_reversal,
        policy_dt_s=0.04,
        acceleration_scale_per_s2=625.0,
    )

    torch.testing.assert_close(ramp_loss, torch.tensor(0.0))
    torch.testing.assert_close(reversal_loss, torch.tensor(1.0))
    torch.testing.assert_close(common_vertical_loss, torch.tensor(1.0))
    torch.testing.assert_close(horizontal_loss, torch.tensor(0.0))

"""PPO with a deterministic Actor action-curvature loss.

The temporal term uses the unnormalized second-difference core of Grad-CAPS
with a fixed physical scale; it is not the displacement-normalized Grad-CAPS
objective.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


def diagonal_gaussian_kl_divergence(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> torch.Tensor:
    """Return KL(old || new), summed over independent action dimensions."""

    return torch.sum(
        torch.log(new_std / old_std)
        + (old_std.square() + (old_mean - new_mean).square())
        / (2.0 * new_std.square())
        - 0.5,
        dim=-1,
    )


def normalized_action_curvature_loss(
    action_mean: torch.Tensor,
    previous_action_mean: torch.Tensor,
    previous_previous_action_mean: torch.Tensor,
    *,
    policy_dt_s: float,
    acceleration_scale_per_s2: float,
) -> torch.Tensor:
    """Return the RMS normalized second difference of deterministic actions."""

    normalized_curvature = (
        action_mean - 2.0 * previous_action_mean + previous_previous_action_mean
    ) / (policy_dt_s * policy_dt_s * acceleration_scale_per_s2)
    return torch.linalg.vector_norm(normalized_curvature, dim=-1).mean() / math.sqrt(
        action_mean.shape[-1]
    )


class AUVSmoothPPO(PPO):
    """Current feed-forward RSL-RL PPO plus Actor mean action curvature."""

    def __init__(
        self,
        *args,
        action_curvature_loss_coef: float,
        action_curvature_policy_dt_s: float,
        action_curvature_scale_per_s2: float,
        critic_learning_rate: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_curvature_loss_coef = action_curvature_loss_coef
        self.action_curvature_policy_dt_s = action_curvature_policy_dt_s
        self.action_curvature_scale_per_s2 = action_curvature_scale_per_s2
        self.critic_learning_rate = critic_learning_rate

        actor_parameters = [
            *self.policy.actor.parameters(),
            self.policy.log_std,
        ]
        critic_parameters = list(self.policy.critic.parameters())
        self.optimizer = optim.Adam(
            (
                {"params": actor_parameters, "lr": self.learning_rate},
                {"params": critic_parameters, "lr": self.critic_learning_rate},
            )
        )
        self._actor_parameters = actor_parameters
        self._critic_parameters = critic_parameters

    def _indexed_mini_batch_generator(self):
        """Match the native feed-forward generator and retain flat rollout indices."""

        storage = self.storage
        batch_size = storage.num_envs * storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(
            self.num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        values = storage.values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        old_mean = storage.mu.flatten(0, 1)
        old_std = storage.sigma.flatten(0, 1)
        for _ in range(self.num_learning_epochs):
            for mini_batch_index in range(self.num_mini_batches):
                start = mini_batch_index * mini_batch_size
                stop = (mini_batch_index + 1) * mini_batch_size
                batch_indices = indices[start:stop]
                yield (
                    batch_indices,
                    observations[batch_indices],
                    actions[batch_indices],
                    values[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                    old_actions_log_prob[batch_indices],
                    old_mean[batch_indices],
                    old_std[batch_indices],
                )

    def _action_curvature_loss(
        self,
        batch_indices: torch.Tensor,
        action_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate valid same-episode action triples from the rollout."""

        storage = self.storage
        time_indices = torch.div(
            batch_indices,
            storage.num_envs,
            rounding_mode="floor",
        )
        environment_indices = torch.remainder(batch_indices, storage.num_envs)
        candidate_positions = torch.nonzero(time_indices >= 2, as_tuple=False).squeeze(-1)
        candidate_times = time_indices[candidate_positions]
        candidate_envs = environment_indices[candidate_positions]
        same_episode = torch.logical_and(
            storage.dones[candidate_times - 1, candidate_envs, 0] == 0,
            storage.dones[candidate_times - 2, candidate_envs, 0] == 0,
        )
        valid_positions = candidate_positions[same_episode]
        if valid_positions.numel() == 0:
            return action_mean.sum() * 0.0

        valid_times = time_indices[valid_positions]
        valid_envs = environment_indices[valid_positions]
        past_observations = TensorDict(
            {
                name: torch.cat(
                    (
                        storage.observations[name][valid_times - 1, valid_envs],
                        storage.observations[name][valid_times - 2, valid_envs],
                    ),
                    dim=0,
                )
                for name in self.policy.obs_groups["policy"]
            },
            batch_size=[2 * valid_positions.numel()],
            device=self.device,
        )
        past_action_mean = self.policy.act_inference(past_observations)
        previous_action_mean, previous_previous_action_mean = past_action_mean.chunk(2)
        return normalized_action_curvature_loss(
            action_mean[valid_positions],
            previous_action_mean,
            previous_previous_action_mean,
            policy_dt_s=self.action_curvature_policy_dt_s,
            acceleration_scale_per_s2=self.action_curvature_scale_per_s2,
        )

    def update(self) -> dict[str, float]:
        """Run the current PPO update with the added Actor curvature loss."""

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_action_curvature_loss = 0.0
        mean_kl = 0.0
        max_kl = 0.0
        number_of_kl_checks = 0
        number_of_actor_updates = 0
        number_of_critic_updates = 0
        planned_number_of_updates = self.num_learning_epochs * self.num_mini_batches
        kl_stop_threshold = 1.5 * self.desired_kl
        actor_updates_enabled = True

        for (
            batch_indices,
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mean_batch,
            old_std_batch,
        ) in self._indexed_mini_batch_generator():
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch - advantages_batch.mean()
                    ) / (advantages_batch.std() + 1.0e-8)

            value_batch = self.policy.evaluate(obs_batch)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = self.value_loss_coef * value_loss
            if actor_updates_enabled:
                self.policy.act(obs_batch)
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                action_mean = self.policy.action_mean
                action_std = self.policy.action_std
                entropy_batch = self.policy.entropy

                with torch.no_grad():
                    kl = diagonal_gaussian_kl_divergence(
                        old_mean_batch,
                        old_std_batch,
                        action_mean,
                        action_std,
                    )
                    kl_mean = kl.mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        kl_mean /= self.gpu_world_size

                kl_value = kl_mean.item()
                mean_kl += kl_value
                max_kl = max(max_kl, kl_value)
                number_of_kl_checks += 1
                if kl_mean > kl_stop_threshold:
                    actor_updates_enabled = False
                else:
                    ratio = torch.exp(
                        actions_log_prob_batch
                        - torch.squeeze(old_actions_log_prob_batch)
                    )
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(
                        advantages_batch
                    ) * torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                    surrogate_loss = torch.max(
                        surrogate,
                        surrogate_clipped,
                    ).mean()
                    action_curvature_loss = self._action_curvature_loss(
                        batch_indices,
                        action_mean,
                    )
                    weighted_action_curvature_loss = (
                        self.action_curvature_loss_coef * action_curvature_loss
                    )
                    loss = (
                        loss
                        + surrogate_loss
                        - self.entropy_coef * entropy_batch.mean()
                        + weighted_action_curvature_loss
                    )
                    mean_surrogate_loss += surrogate_loss.item()
                    mean_entropy += entropy_batch.mean().item()
                    mean_action_curvature_loss += (
                        weighted_action_curvature_loss.item()
                    )
                    number_of_actor_updates += 1

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if actor_updates_enabled:
                nn.utils.clip_grad_norm_(
                    self._actor_parameters,
                    self.max_grad_norm,
                )
            nn.utils.clip_grad_norm_(
                self._critic_parameters,
                self.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            number_of_critic_updates += 1

        mean_value_loss /= number_of_critic_updates
        mean_surrogate_loss /= number_of_actor_updates
        mean_entropy /= number_of_actor_updates
        mean_action_curvature_loss /= number_of_actor_updates
        mean_kl /= number_of_kl_checks
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "action_curvature": mean_action_curvature_loss,
            "kl": mean_kl,
            "kl_max": max_kl,
            "actor_update_fraction": (
                number_of_actor_updates / planned_number_of_updates
            ),
            "critic_update_fraction": (
                number_of_critic_updates / planned_number_of_updates
            ),
        }

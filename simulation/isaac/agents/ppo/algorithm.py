"""Rollout-level KL control for the RSL-RL PPO implementation used by AUV.

RSL-RL's built-in ``adaptive`` schedule changes the learning rate after every
mini-batch.  The trajectory run uses 32 mini-batches and five epochs, so that
controller can make 160 large learning-rate decisions per rollout.  This
variant holds the rate fixed while consuming one rollout, stops the remaining
updates once the policy has moved too far, and adjusts the rate only for the
next rollout.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rsl_rl.algorithms import PPO


class RolloutAdaptivePPO(PPO):
    """PPO with one conservative KL-driven learning-rate decision per rollout.

    The project exposes feed-forward MLP policies only.
    """

    def __init__(
        self,
        *args: Any,
        schedule: str = "rollout_adaptive",
        rollout_kl_stop: float = 0.015,
        rollout_kl_low: float = 0.005,
        rollout_lr_up_factor: float = 1.1,
        rollout_lr_down_factor: float = 1.2,
        rollout_lr_min: float = 1.0e-4,
        rollout_lr_max: float = 5.0e-4,
        **kwargs: Any,
    ) -> None:
        if schedule != "rollout_adaptive":
            raise ValueError(
                "RolloutAdaptivePPO requires schedule='rollout_adaptive'; "
                f"received {schedule!r}."
            )
        numeric_parameters = {
            "rollout_kl_stop": rollout_kl_stop,
            "rollout_kl_low": rollout_kl_low,
            "rollout_lr_up_factor": rollout_lr_up_factor,
            "rollout_lr_down_factor": rollout_lr_down_factor,
            "rollout_lr_min": rollout_lr_min,
            "rollout_lr_max": rollout_lr_max,
        }
        if not all(math.isfinite(float(value)) for value in numeric_parameters.values()):
            raise ValueError("Rollout KL and learning-rate parameters must be finite.")
        if rollout_lr_up_factor <= 1.0 or rollout_lr_down_factor <= 1.0:
            raise ValueError("The rollout learning-rate adjustment factors must be greater than one.")
        if not (0.0 < rollout_lr_min <= rollout_lr_max):
            raise ValueError("Expected 0 < rollout_lr_min <= rollout_lr_max.")

        # ``PPO.update`` only invokes its built-in controller for the literal
        # value "adaptive".  Retaining the explicit schedule label documents
        # the selected behavior while keeping the learning rate fixed here.
        super().__init__(*args, schedule=schedule, **kwargs)
        if self.policy.is_recurrent:
            raise ValueError("RolloutAdaptivePPO supports feed-forward MLP policies only.")
        if self.rnd or self.symmetry:
            raise ValueError("RolloutAdaptivePPO does not support RND or symmetry extensions.")
        if self.desired_kl is None or self.desired_kl <= 0.0:
            raise ValueError("RolloutAdaptivePPO requires a positive desired_kl.")
        if not math.isfinite(float(self.desired_kl)):
            raise ValueError("RolloutAdaptivePPO requires a finite desired_kl.")
        if not (0.0 < rollout_kl_low < self.desired_kl < rollout_kl_stop):
            raise ValueError("Expected 0 < rollout_kl_low < desired_kl < rollout_kl_stop.")
        initial_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        if not math.isfinite(initial_learning_rate) or not rollout_lr_min <= initial_learning_rate <= rollout_lr_max:
            raise ValueError("Initial learning rate must be finite and within rollout_lr_min/rollout_lr_max.")

        self.rollout_kl_stop = rollout_kl_stop
        self.rollout_kl_low = rollout_kl_low
        self.rollout_lr_up_factor = rollout_lr_up_factor
        self.rollout_lr_down_factor = rollout_lr_down_factor
        self.rollout_lr_min = rollout_lr_min
        self.rollout_lr_max = rollout_lr_max
        self.last_rollout_kl = float("nan")
        self.last_rollout_early_stop = False
        self.last_rollout_updates = 0

    def _mean_kl(self, batch: tuple[Any, ...]) -> float:
        """Return the exact Gaussian KL for an unmodified mini-batch."""

        (
            obs_batch,
            _,
            _,
            _,
            _,
            _,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) = batch
        with torch.inference_mode():
            actor_obs = self.policy.get_actor_obs(obs_batch)
            actor_obs = self.policy.actor_obs_normalizer(actor_obs)
            self.policy._update_distribution(actor_obs)
            mu_batch = self.policy.action_mean[: old_mu_batch.shape[0]]
            sigma_batch = self.policy.action_std[: old_sigma_batch.shape[0]]
            epsilon = torch.finfo(sigma_batch.dtype).tiny
            kl = torch.sum(
                torch.log(sigma_batch.clamp_min(epsilon))
                - torch.log(old_sigma_batch.clamp_min(epsilon))
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch.clamp_min(epsilon)))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.clamp(torch.mean(kl), min=0.0)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
        return kl_mean.item()

    def _set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _adjust_next_rollout_learning_rate(self, probe_kl: float) -> None:
        """Apply one bounded, conservative learning-rate adjustment."""

        learning_rate = min(
            self.rollout_lr_max,
            max(self.rollout_lr_min, float(self.optimizer.param_groups[0]["lr"])),
        )
        if self.gpu_global_rank == 0:
            if probe_kl > self.desired_kl:
                learning_rate = max(self.rollout_lr_min, learning_rate / self.rollout_lr_down_factor)
            elif probe_kl < self.rollout_kl_low:
                learning_rate = min(self.rollout_lr_max, learning_rate * self.rollout_lr_up_factor)

        if self.is_multi_gpu:
            lr_tensor = torch.tensor(learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            learning_rate = lr_tensor.item()
        self._set_learning_rate(learning_rate)

    def update(self) -> dict[str, float]:
        """Update PPO, early-stop on KL, then set the next rollout's rate."""

        if self.storage is None:
            raise RuntimeError("PPO storage must be initialized before calling update().")

        # A resumed optimizer restores its parameter-group learning rate after
        # construction.  Treat that restored value as this rollout's fixed
        # rate instead of reverting to the initial configuration value.
        restored_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self._set_learning_rate(
            min(self.rollout_lr_max, max(self.rollout_lr_min, restored_learning_rate))
        )

        probe_batch: tuple[Any, ...] | None = None
        updates_completed = 0
        early_stop = False
        value_loss_total = 0.0
        surrogate_loss_total = 0.0
        entropy_total = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                _,
                _,
            ) = batch

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            # One Actor forward supplies both the KL gate and PPO likelihood;
            # no duplicate forward or unused stochastic action is generated.
            actor_obs = self.policy.get_actor_obs(obs_batch)
            actor_obs = self.policy.actor_obs_normalizer(actor_obs)
            self.policy._update_distribution(actor_obs)
            mu_batch = self.policy.action_mean[: old_mu_batch.shape[0]]
            sigma_batch = self.policy.action_std[: old_sigma_batch.shape[0]]
            epsilon = torch.finfo(sigma_batch.dtype).tiny
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch.clamp_min(epsilon))
                    - torch.log(old_sigma_batch.clamp_min(epsilon))
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch.clamp_min(epsilon)))
                    - 0.5,
                    dim=-1,
                )
                kl_mean = torch.clamp(torch.mean(kl), min=0.0)
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size
            if updates_completed > 0 and float(kl_mean.item()) > self.rollout_kl_stop:
                early_stop = True
                break

            if probe_batch is None:
                probe_batch = batch
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            entropy_batch = self.policy.entropy

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            value_loss_total += value_loss.item()
            surrogate_loss_total += surrogate_loss.item()
            entropy_total += entropy_batch.mean().item()
            updates_completed += 1

        self.storage.clear()

        if probe_batch is None or updates_completed == 0:
            raise RuntimeError("PPO update produced no mini-batches for the KL probe.")

        loss_dict = {
            "value_function": value_loss_total / updates_completed,
            "surrogate": surrogate_loss_total / updates_completed,
            "entropy": entropy_total / updates_completed,
        }

        # Re-evaluate the same unmodified, large mini-batch after all accepted
        # optimizer steps.  It is a stable probe of final policy movement and
        # controls only the *next* rollout's learning rate.
        probe_kl = self._mean_kl(probe_batch)
        self._adjust_next_rollout_learning_rate(probe_kl)

        self.last_rollout_kl = probe_kl
        self.last_rollout_early_stop = early_stop
        self.last_rollout_updates = updates_completed
        loss_dict["kl_probe"] = probe_kl
        loss_dict["early_stop"] = float(self.last_rollout_early_stop)
        loss_dict["kl_limit_exceeded"] = float(probe_kl > self.rollout_kl_stop)
        loss_dict["ppo_updates"] = float(updates_completed)
        return loss_dict

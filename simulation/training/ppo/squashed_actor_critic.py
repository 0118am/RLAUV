"""Tanh-squashed Gaussian ActorCritic for normalized motor commands."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch.distributions import TanhTransform, TransformedDistribution

from rsl_rl.modules import ActorCritic


ACTION_SQUASH_VERSION_STATE_KEY = "action_squash_version"
ACTION_SQUASH_VERSION = 1


class AUVTanhGaussianActorCritic(ActorCritic):
    """RSL-RL ActorCritic whose externally visible actions are ``tanh(z)``.

    The neural network continues to parameterize an unconstrained latent
    Gaussian.  Sampling, deterministic inference, entropy, and action
    log-probabilities expose the transformed motor-command distribution.
    Latent Gaussian parameters remain available for the analytic PPO KL,
    which is invariant under the shared bijective tanh transform.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            ACTION_SQUASH_VERSION_STATE_KEY,
            torch.tensor(ACTION_SQUASH_VERSION, dtype=torch.int64),
        )
        self.action_transform = TanhTransform(cache_size=1)
        self.squashed_distribution: TransformedDistribution | None = None
        self._last_latent_actions: torch.Tensor | None = None

    @property
    def bounded_action_mean(self) -> torch.Tensor:
        """Return the deterministic motor command ``tanh(mu)``."""

        return self.action_transform(self.action_mean)

    @property
    def last_latent_actions(self) -> torch.Tensor:
        """Return the pre-tanh sample produced by the latest :meth:`act` call."""

        return self._last_latent_actions

    @property
    def entropy(self) -> torch.Tensor:
        """Return a reparameterized one-sample estimate of transformed entropy."""

        return -self.get_actions_log_prob_from_latent(self.last_latent_actions)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Sample a bounded motor command and retain its exact latent sample."""

        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        self._last_latent_actions = self.distribution.rsample()
        return self.action_transform(self._last_latent_actions)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Return the bounded deterministic motor command ``tanh(mu)``."""

        return self.action_transform(super().act_inference(obs))

    def _update_distribution(self, obs: TensorDict) -> None:
        """Build PyTorch's transformed distribution from RSL-RL's latent Normal."""

        super()._update_distribution(obs)
        self.squashed_distribution = TransformedDistribution(
            self.distribution,
            [self.action_transform],
        )

    def get_actions_log_prob_from_latent(
        self,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the squashed-action density from exact pre-tanh samples."""

        actions = self.action_transform(latent_actions)
        return self.squashed_distribution.log_prob(actions).sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate bounded action log-probabilities through the inverse transform."""

        return self.squashed_distribution.log_prob(actions).sum(dim=-1)

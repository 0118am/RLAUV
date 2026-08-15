"""Current-state policy observations and feed-forward history for ``AUVTrajEnv``."""

from __future__ import annotations

from collections.abc import Sequence
import gymnasium as gym
import numpy as np
import torch

from robot.control.trajectory.observation_contract import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    OBSERVATION_FIELD_SLICES,
    OBSERVATION_NORMALIZATION_SCALES,
)
from simulation.isaac.ppo.architectures import CRITIC_PRIVILEGED_FIELD_DIMENSIONS, get_mlp_architecture
from .critic_observations import AUVCriticObservationMixin


class AUVObservationMixin(AUVCriticObservationMixin):
    """Own current-state policy observations and feed-forward history."""

    def _configure_mlp_observation_space(self, cfg) -> None:
        """Derive actor/critic spaces from the selected feed-forward profile.

        History is assembled after the current 30-D sample is normalized, so
        every cached value has the same deployed representation as the current
        actor input.
        """

        architecture = get_mlp_architecture(cfg.mlp_architecture)
        # ``mlp_architecture`` is the only external selector.  Copy the
        # profile's fields into runtime config so the same values are recorded
        # in Hydra snapshots without offering a second manual edit point.
        cfg.mlp_history_steps = architecture.history_steps
        cfg.mlp_history_fields = list(architecture.history_fields)
        selected_critic_fields = tuple(
            getattr(cfg, "critic_privileged_fields_override", ()) or architecture.critic_privileged_fields
        )
        unknown_critic_fields = set(selected_critic_fields) - set(CRITIC_PRIVILEGED_FIELD_DIMENSIONS)
        if unknown_critic_fields:
            raise ValueError(
                "Unknown Critic privileged fields: " + ", ".join(sorted(unknown_critic_fields)) + "."
            )
        cfg.critic_privileged_fields = list(selected_critic_fields)
        history_steps = architecture.history_steps
        fields = architecture.history_fields
        slices = self._observation_group_slices()
        try:
            history_feature_dim = sum(
                slices[name].stop - slices[name].start for name in fields
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown MLP history field {error.args[0]!r} for {cfg.mlp_architecture!r}."
            ) from error

        observation_dim = BASE_OBSERVATION_DIM + history_steps * history_feature_dim
        cfg.action_space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        cfg.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(observation_dim,),
            dtype=np.float32,
        )
        cfg.state_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(
                observation_dim
                + sum(CRITIC_PRIVILEGED_FIELD_DIMENSIONS[name] for name in selected_critic_fields),
            ),
            dtype=np.float32,
        )

    def _init_observation_state(self) -> None:
        self._observation_normalization_scale = torch.tensor(
            OBSERVATION_NORMALIZATION_SCALES,
            dtype=torch.float32,
            device=self.device,
        )
        self._init_mlp_history()
        self._init_critic_observation_state()

    def _init_mlp_history(self) -> None:
        """Allocate the causal history selected by ``simulation.isaac.ppo.architectures``."""

        self._mlp_history_steps = int(self.cfg.mlp_history_steps)
        fields = tuple(self.cfg.mlp_history_fields)
        slices = self._observation_group_slices()
        indices = [
            torch.arange(slices[name].start, slices[name].stop, dtype=torch.long, device=self.device)
            for name in fields
        ]
        self._mlp_history_indices = (
            torch.cat(indices) if indices else torch.empty(0, dtype=torch.long, device=self.device)
        )
        self._mlp_history = torch.zeros(
            (self.num_envs, self._mlp_history_steps, len(self._mlp_history_indices)),
            dtype=torch.float32,
            device=self.device,
        )

    def _state_for_observation(self):
        """Return the simulator state used to form the policy observation."""

        return (
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )
    def _observation_group_slices(self) -> dict[str, slice]:
        return dict(OBSERVATION_FIELD_SLICES)

    def _normalize_trajectory_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply fixed physical scales to the current 30-D trajectory sample."""

        return obs / self._observation_normalization_scale

    def _stack_mlp_history(self, normalized_current_obs: torch.Tensor) -> torch.Tensor:
        """Append prior selected samples, newest first, then retain this sample.

        The current sample remains at indices ``[0:30]``.  This makes a
        feed-forward MLP causal while preserving the exact information that a
        real controller can cache between 50-Hz policy updates.
        """

        if self._mlp_history_steps <= 0:
            return normalized_current_obs

        actor_obs = torch.cat((normalized_current_obs, self._mlp_history.flatten(start_dim=1)), dim=-1)
        selected_current = normalized_current_obs.index_select(1, self._mlp_history_indices)
        if self._mlp_history_steps > 1:
            self._mlp_history[:, 1:].copy_(self._mlp_history[:, :-1].clone())
        self._mlp_history[:, 0].copy_(selected_current)
        return actor_obs

    def _reset_mlp_history(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        self._mlp_history[env_ids] = 0.0

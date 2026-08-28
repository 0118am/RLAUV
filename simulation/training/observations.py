"""Actor and critic observations for ``AUVTrajEnv``."""

from __future__ import annotations

from collections.abc import Sequence
import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils

from common.tensor_math import quat_apply_wxyz, quat_conjugate_wxyz

from robot.control.trajectory.observation_contract import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    OBSERVATION_FIELD_SLICES,
    OBSERVATION_NORMALIZATION_SCALES,
    TRAJECTORY_OBSERVATION,
)
from simulation.training.ppo.networks import (
    CRITIC_PRIVILEGED_FIELD_DIMENSIONS,
    get_mlp_architecture,
)


class AUVCriticObservationMixin:
    """Own privileged-state normalization and Critic observation assembly."""

    def _init_critic_observation_state(self) -> None:
        environment = self.environment_runtime
        robot = self.robot_runtime
        self._critic_privileged_fields = tuple(self.cfg.critic_privileged_fields)
        self._critic_position_center = environment.pool_center_local
        self._critic_position_scale = environment.pool_half_extents
        self._critic_acceleration_scale = torch.tensor(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
            dtype=torch.float32,
            device=self.device,
        )
        self._critic_nominal_linear_damping = self._hydro_diagonal(environment.nominal_linear_damping)
        self._critic_nominal_quadratic_damping = self._hydro_diagonal(environment.nominal_quadratic_damping)
        self._critic_nominal_fluid_added_mass = self._hydro_diagonal(
            environment.nominal_fluid_added_mass
        )
        self._critic_max_thruster_force = max(float(robot.thruster_wake_reference_force_n), 1.0)
        self._critic_nominal_tau = max(float(self.cfg.dyn_time_constant), 1.0e-6)
        self._critic_nominal_tether_length = max(float(self.cfg.tether_winch_max_length), 1.0)

    @staticmethod
    def _hydro_diagonal(coefficients: torch.Tensor) -> torch.Tensor:
        """Return the per-DOF diagonal from vector or matrix hydro storage."""

        if coefficients.ndim == 2:
            return coefficients
        return torch.diagonal(coefficients, dim1=-2, dim2=-1)

    @staticmethod
    def _relative_to_nominal(value: torch.Tensor, nominal: torch.Tensor) -> torch.Tensor:
        """Scale a coefficient by its signed nonzero nominal counterpart."""

        denominator = torch.where(torch.abs(nominal) > 1.0e-6, nominal, torch.ones_like(nominal))
        return value / denominator

    def _critic_privileged_terms(self) -> dict[str, torch.Tensor]:
        """Build the exact latent dynamics state used only by the value model.

        The terms mirror the force path in ``_compute_dynamics``: instantaneous
        current, effective pool/free-surface hydrodynamic coefficients, and
        realized actuator output. They must not be
        reused when assembling the deployable Actor observation.
        """

        root_quat_w = self._robot.data.root_quat_w
        local_position_w = self._robot.data.root_pos_w - self.scene.env_origins
        effective_hydrodynamics = self._effective_hydrodynamic_state_for_critic()
        water_current_w = effective_hydrodynamics.water_current_w
        environment = self.environment_runtime
        robot = self.robot_runtime

        return {
            "true_root_state": torch.cat(
                (
                    (local_position_w - self._critic_position_center)
                    / self._critic_position_scale,
                    math_utils.quat_unique(root_quat_w),
                    self._robot.data.root_lin_vel_b
                    / TRAJECTORY_OBSERVATION.field("linear_velocity_error_b").physical_scale,
                    self._robot.data.root_ang_vel_b
                    / TRAJECTORY_OBSERVATION.field("angular_velocity_b").physical_scale,
                ),
                dim=-1,
            ),
            "water_current_b": quat_apply_wxyz(quat_conjugate_wxyz(root_quat_w), water_current_w)
            / TRAJECTORY_OBSERVATION.field("linear_velocity_error_b").physical_scale,
            "generalized_acceleration_b": environment.generalized_acceleration_b
            / self._critic_acceleration_scale,
            "effective_linear_damping_ratio": self._relative_to_nominal(
                self._hydro_diagonal(effective_hydrodynamics.linear_damping),
                self._critic_nominal_linear_damping,
            ),
            "effective_quadratic_damping_ratio": self._relative_to_nominal(
                self._hydro_diagonal(effective_hydrodynamics.quadratic_damping),
                self._critic_nominal_quadratic_damping,
            ),
            "effective_fluid_added_mass_ratio": self._relative_to_nominal(
                self._hydro_diagonal(effective_hydrodynamics.fluid_added_mass),
                self._critic_nominal_fluid_added_mass,
            ),
            "effective_buoyant_volume_ratio": robot.volumes * effective_hydrodynamics.buoyancy_scale
            / max(float(self.cfg.volume), 1.0e-6),
            "realized_thruster_force": robot.realized_thruster_force_n / self._critic_max_thruster_force,
            "thruster_force_scale": robot.thruster_force_scale,
            "common_thruster_force_scale": robot.common_thruster_force_scale,
            "thruster_parameters": torch.cat(
                (
                    (robot.thruster_time_constant / self._critic_nominal_tau).unsqueeze(-1),
                    robot.thruster_wake_loss_coefficient.unsqueeze(-1),
                ),
                dim=-1,
            ),
            "tether_slack_ratio": robot.tether_slack_length / self._critic_nominal_tether_length,
        }

    def _build_critic_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Concatenate Actor input with profile-selected simulator truth for V."""

        terms = self._critic_privileged_terms()
        privileged_obs = torch.cat([terms[name] for name in self._critic_privileged_fields], dim=-1)
        return torch.cat((actor_obs, privileged_obs), dim=-1)


class AUVObservationMixin(AUVCriticObservationMixin):
    """Own current-state policy observations and feed-forward history."""

    def _configure_mlp_observation_space(self, cfg) -> None:
        """Derive actor/critic spaces from the selected feed-forward profile.

        History is assembled after the current 33-D sample is normalized, so
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
            cfg.critic_privileged_fields_override or architecture.critic_privileged_fields
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
        self._latest_normalized_observation = torch.zeros(
            (self.num_envs, BASE_OBSERVATION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self._init_mlp_history()
        self._init_critic_observation_state()

    def _init_mlp_history(self) -> None:
        """Allocate history selected by ``simulation.training.ppo.networks``."""

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
        """Return the exact delayed fused state available to the Actor."""

        measurement = self.robot_runtime.pose_sensor.measure()
        return (
            measurement.position_w,
            measurement.quaternion_wxyz,
            measurement.linear_velocity_b,
            measurement.angular_velocity_b,
        )

    def _observation_group_slices(self) -> dict[str, slice]:
        return dict(OBSERVATION_FIELD_SLICES)

    def _normalize_trajectory_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply fixed physical scales to the current 33-D trajectory sample."""

        return obs / self._observation_normalization_scale

    def _stack_mlp_history(self, normalized_current_obs: torch.Tensor) -> torch.Tensor:
        """Append prior selected samples, newest first, without changing history.

        The current sample remains at indices ``[0:33]``.  This makes a
        feed-forward MLP causal while preserving the exact information that a
        deployed policy runtime can cache between 25-Hz updates.
        """

        if self._mlp_history.shape[1] == 0:
            return normalized_current_obs
        return torch.cat(
            (normalized_current_obs, self._mlp_history.flatten(start_dim=1)),
            dim=-1,
        )

    def _commit_mlp_history(self, normalized_current_obs: torch.Tensor) -> None:
        """Advance history exactly once for the policy sample being acted on."""

        if self._mlp_history.shape[1] == 0:
            return
        if self._mlp_history.shape[1] > 1:
            self._mlp_history[:, 1:].copy_(self._mlp_history[:, :-1].clone())
        self._mlp_history[:, 0].copy_(
            normalized_current_obs.index_select(1, self._mlp_history_indices)
        )

    def _reset_mlp_history(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        self._mlp_history[env_ids] = 0.0
        self._latest_normalized_observation[env_ids] = 0.0

"""Policy observation transport and feed-forward history for ``AUVTrajEnv``."""

from __future__ import annotations

from collections.abc import Sequence
import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_conjugate

from ...agents.ppo.architectures import CRITIC_PRIVILEGED_FIELD_DIMENSIONS, get_mlp_architecture
from environment.profiles.random_sampling import sample_symmetric_bounded_normal
from .sensing.sensors import (
    ObservationDelayBuffer,
    ObservationFilterState,
    apply_observation_sensor_model,
    build_observation_group_parameter,
)


def _numeric_config_values(value) -> tuple[float, ...]:
    """Flatten scalar/sequence/grouped config values without touching CUDA."""

    if value is None:
        return ()
    if isinstance(value, dict):
        return tuple(number for item in value.values() for number in _numeric_config_values(item))
    if isinstance(value, (list, tuple)):
        return tuple(number for item in value for number in _numeric_config_values(item))
    return (float(value),)


def _observation_sensor_model_required(cfg) -> bool:
    """Return whether any configured or randomized observation transport is non-identity."""

    randomization = getattr(cfg, "domain_randomization", None)

    def values(config_name: str, range_name: str) -> tuple[float, ...]:
        configured = _numeric_config_values(getattr(cfg, config_name))
        randomized = _numeric_config_values(getattr(randomization, range_name, ()))
        return configured + randomized

    return any(
        (
            any(value > 0.0 for value in values("observation_delay_steps", "observation_delay_steps_range")),
            any(value > 0.0 for value in values("observation_noise_std", "observation_noise_std_range")),
            any(value != 0.0 for value in values("observation_bias_range", "observation_bias_range")),
            any(
                value > 1.0
                for value in values("observation_update_period_steps", "observation_update_period_steps_range")
            ),
            any(
                value > 0.0
                for value in values(
                    "observation_dropout_probability",
                    "observation_dropout_probability_range",
                )
            ),
            any(
                value < 1.0
                for value in values("observation_lowpass_alpha", "observation_lowpass_alpha_range")
            ),
            any(
                value > 0.0
                for value in values("observation_bias_drift_std", "observation_bias_drift_std_range")
            ),
        )
    )


class AUVObservationMixin:
    """Owns policy-observation transport and feed-forward history."""

    def _configure_mlp_observation_space(self, cfg) -> None:
        """Derive actor/critic spaces from the selected feed-forward profile.

        Sensor corruption is defined for one physical, current 30-D sample.
        History is assembled only after that sample is normalized, so every
        cached value has exactly the same deployed representation as the
        current actor input.
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

        observation_dim = int(cfg.observation_base_dim) + history_steps * history_feature_dim
        cfg.action_space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(8,),
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

    def _init_observation_sensor_model(self) -> None:
        base_dim = int(self.cfg.observation_base_dim)
        delay_range = getattr(self.cfg.domain_randomization, "observation_delay_steps_range", [0, 0])
        max_delay_steps = max(int(self.cfg.observation_delay_steps), int(delay_range[1]))
        self.observation_delay_buffer = ObservationDelayBuffer(
            self.num_envs,
            base_dim,
            max_delay_steps,
            self.device,
        )
        self.observation_filter_state = ObservationFilterState(
            self.num_envs,
            base_dim,
            self.device,
        )
        self.observation_delay_steps = torch.full(
            (self.num_envs,),
            int(self.cfg.observation_delay_steps),
            dtype=torch.long,
            device=self.device,
        )
        self.observation_update_period_steps = torch.full(
            (self.num_envs,),
            int(self.cfg.observation_update_period_steps),
            dtype=torch.long,
            device=self.device,
        )
        self.observation_noise_std = torch.zeros(
            (self.num_envs, base_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.observation_bias = torch.zeros_like(self.observation_noise_std)
        self.observation_dropout_probability = torch.zeros_like(self.observation_noise_std)
        self.observation_lowpass_alpha = torch.ones_like(self.observation_noise_std)
        self.observation_bias_drift_std = torch.zeros_like(self.observation_noise_std)
        # This decision depends only on immutable config/range metadata. Keeping
        # it on the host removes seven CUDA reductions and synchronizations from
        # every policy step when the sensor transport is an identity mapping.
        self._observation_sensor_model_enabled = _observation_sensor_model_required(self.cfg)
        self._observation_normalization_scale = torch.ones(
            base_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._observation_normalization_scale[0:3] = float(self.cfg.observation_position_scale_m)
        self._observation_normalization_scale[3:9] = float(
            self.cfg.observation_linear_velocity_scale_mps
        )
        self._observation_normalization_scale[13:19] = float(
            self.cfg.observation_angular_velocity_scale_radps
        )
        self._observation_normalization_scale[19:22] = float(
            self.cfg.observation_linear_acceleration_scale_mps2
        )
        self._init_mlp_history()
        self._set_fixed_observation_noise(self._robot._ALL_INDICES)

    def _init_mlp_history(self) -> None:
        """Allocate the causal history selected by ``agents.ppo.architectures``."""

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
        self._critic_privileged_fields = tuple(self.cfg.critic_privileged_fields)
        self._critic_position_scale = torch.tensor(
            [self.cfg.max_auv_x, self.cfg.max_auv_y, self.cfg.max_auv_z],
            dtype=torch.float32,
            device=self.device,
        ).clamp_min(1.0)
        self._critic_acceleration_scale = torch.tensor(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
            dtype=torch.float32,
            device=self.device,
        )
        self._critic_nominal_linear_damping = self._hydro_diagonal(self._nominal_linear_damping)
        self._critic_nominal_quadratic_damping = self._hydro_diagonal(self._nominal_quadratic_damping)
        self._critic_nominal_added_mass = self._hydro_diagonal(self._nominal_added_mass_diag)
        delay_range = getattr(self.cfg.domain_randomization, "thruster_command_delay_steps_range", [0, 0])
        rate_range = getattr(self.cfg.domain_randomization, "thruster_max_command_rate_range", [0.0, 0.0])
        self._critic_max_thruster_force = max(float(self._thruster_wake_reference_force_n), 1.0)
        self._critic_max_delay_steps = max(
            1,
            int(self.cfg.thruster_command_delay_steps),
            int(delay_range[1]),
        )
        self._critic_max_command_rate = max(
            1.0,
            float(self.cfg.thruster_max_command_rate),
            float(rate_range[1]),
        )
        self._critic_nominal_tau = max(float(self.cfg.dyn_time_constant), 1.0e-6)
        self._critic_nominal_voltage = max(float(self.cfg.battery_voltage_nominal), 1.0e-6)
        self._critic_nominal_tether_length = max(float(self.cfg.tether_winch_max_length), 1.0)

    def _state_for_observation(self):
        """Return the simulator state used to form the policy observation."""

        return (
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )
    def _observation_group_slices(self) -> dict[str, slice]:
        return {
            "position_error_b": slice(0, 3),
            "target_linear_velocity_b": slice(3, 6),
            "linear_velocity_error_b": slice(6, 9),
            "attitude_error_quat": slice(9, 13),
            "angular_velocity_b": slice(13, 16),
            "target_angular_velocity_b": slice(16, 19),
            "target_linear_acceleration_b": slice(19, 22),
            "actions": slice(22, 30),
            "applied_action": slice(22, 30),
        }

    def _observation_cfg_tensor(self, value, name: str) -> torch.Tensor:
        """Expand scalar, channel-vector, or semantic-group observation config per environment."""

        base_dim = int(self.cfg.observation_base_dim)
        reference = torch.zeros(
            (self.num_envs, base_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if isinstance(value, dict):
            return build_observation_group_parameter(
                value,
                self._observation_group_slices(),
                reference,
            )

        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1).repeat(self.num_envs, base_dim)
        if tensor.ndim == 1 and tensor.shape[0] == base_dim:
            return tensor.reshape(1, base_dim).repeat(self.num_envs, 1)
        raise ValueError(f"{name} must be a scalar or length-{base_dim} sequence.")

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
        current, effective pool/free-surface hydrodynamic coefficients, sampled
        rigid-body properties, and realized actuator output.  They must not be
        reused when assembling the deployable Actor observation.
        """

        root_quat_w = self._robot.data.root_quat_w
        local_position_w = self._robot.data.root_pos_w - self.scene.env_origins
        effective_hydrodynamics = self._effective_hydrodynamic_state_for_critic()
        water_current_w = effective_hydrodynamics.water_current_w

        return {
            "true_root_state": torch.cat(
                (
                    local_position_w / self._critic_position_scale,
                    math_utils.quat_unique(root_quat_w),
                    self._robot.data.root_lin_vel_b / float(self.cfg.observation_linear_velocity_scale_mps),
                    self._robot.data.root_ang_vel_b / float(self.cfg.observation_angular_velocity_scale_radps),
                ),
                dim=-1,
            ),
            "water_current_b": quat_apply(quat_conjugate(root_quat_w), water_current_w)
            / float(self.cfg.observation_linear_velocity_scale_mps),
            "filtered_relative_acceleration_b": self._filtered_nu_r_dot / self._critic_acceleration_scale,
            "effective_linear_damping_ratio": self._relative_to_nominal(
                self._hydro_diagonal(effective_hydrodynamics.linear_damping),
                self._critic_nominal_linear_damping,
            ),
            "effective_quadratic_damping_ratio": self._relative_to_nominal(
                self._hydro_diagonal(effective_hydrodynamics.quadratic_damping),
                self._critic_nominal_quadratic_damping,
            ),
            "effective_added_mass_ratio": self._relative_to_nominal(
                self._hydro_diagonal(effective_hydrodynamics.added_mass),
                self._critic_nominal_added_mass,
            ),
            "effective_buoyant_volume_ratio": self.volumes * effective_hydrodynamics.buoyancy_scale
            / max(float(self.cfg.volume), 1.0e-6),
            "mass_ratio": self.masses / max(float(self.cfg.mass), 1.0e-6),
            "principal_inertia_ratio": self.inertia_principal_moments
            / self._nominal_principal_inertia.reshape(1, 3).clamp_min(1.0e-6),
            "center_of_mass_offset_b": self.center_of_mass_offsets / 0.1,
            "com_to_cob_offset_b": self.com_to_cob_offsets / 0.1,
            "realized_thruster_force": self.realized_thruster_force_n / self._critic_max_thruster_force,
            "thruster_force_scale": self.thruster_force_scale,
            "thruster_parameters": torch.cat(
                (
                    (self.thruster_time_constant / self._critic_nominal_tau).unsqueeze(-1),
                    (
                        self.thruster_delay_steps.to(dtype=torch.float32) / self._critic_max_delay_steps
                    ).unsqueeze(-1),
                    self.thruster_max_command_rate / self._critic_max_command_rate,
                    self.thruster_command_resolution,
                    self.thruster_command_dropout_probability,
                    self.thruster_wake_loss_coefficient.unsqueeze(-1),
                    self.thruster_reaction_torque_coeff.unsqueeze(-1),
                ),
                dim=-1,
            ),
            "battery_state": torch.cat(
                (
                    self.battery_voltage / self._critic_nominal_voltage,
                    self.battery_voltage_drop_per_s,
                ),
                dim=-1,
            ),
            "tether_slack_ratio": self.tether_slack_length / self._critic_nominal_tether_length,
            # Truth of the *separately managed* external wrench used by
            # PhysX. It is available only to profiles that explicitly select
            # it for the Critic and never enters the deployable Actor input.
            "high_order_residual_wrench_b": self.physx_hydrodynamic_wrench_manager.last_wrench_b / 100.0,
            "physx_hydrodynamics_scale": self.physx_hydrodynamic_wrench_manager.last_scale,
        }

    def _build_critic_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Concatenate Actor input with profile-selected simulator truth for V."""

        terms = self._critic_privileged_terms()
        privileged_obs = torch.cat([terms[name] for name in self._critic_privileged_fields], dim=-1)
        return torch.cat((actor_obs, privileged_obs), dim=-1)

    def _set_fixed_observation_noise(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)

        fixed_noise = self._observation_cfg_tensor(self.cfg.observation_noise_std, "observation_noise_std")
        bias_range = self._observation_cfg_tensor(self.cfg.observation_bias_range, "observation_bias_range")
        dropout_probability = self._observation_cfg_tensor(
            self.cfg.observation_dropout_probability,
            "observation_dropout_probability",
        )
        lowpass_alpha = self._observation_cfg_tensor(self.cfg.observation_lowpass_alpha, "observation_lowpass_alpha")
        bias_drift_std = self._observation_cfg_tensor(
            self.cfg.observation_bias_drift_std,
            "observation_bias_drift_std",
        )
        self.observation_noise_std[env_ids_device] = fixed_noise[env_ids_device]
        self.observation_bias[env_ids_device] = sample_symmetric_bounded_normal(
            bias_range[env_ids_device],
            (len(env_ids_device), self.cfg.observation_base_dim),
            self.device,
        )
        self.observation_dropout_probability[env_ids_device] = torch.clamp(
            dropout_probability[env_ids_device],
            min=0.0,
            max=1.0,
        )
        self.observation_lowpass_alpha[env_ids_device] = torch.clamp(
            lowpass_alpha[env_ids_device],
            min=0.0,
            max=1.0,
        )
        self.observation_bias_drift_std[env_ids_device] = torch.clamp(
            bias_drift_std[env_ids_device],
            min=0.0,
        )

    def _apply_observation_sensor_model(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._observation_sensor_model_enabled:
            return obs
        return apply_observation_sensor_model(
            obs,
            self.observation_delay_buffer,
            self.observation_delay_steps,
            self.observation_noise_std,
            self.observation_bias,
            self.observation_filter_state,
            self.observation_update_period_steps,
            self.observation_dropout_probability,
            self.observation_lowpass_alpha,
            self.observation_bias_drift_std,
            self.physics_dt * self.cfg.decimation,
        )

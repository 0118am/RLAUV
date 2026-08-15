"""Simulator-truth observations available only to the asymmetric Critic."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_conjugate

from robot.control.trajectory.observation_contract import TRAJECTORY_OBSERVATION


class AUVCriticObservationMixin:
    """Own privileged-state normalization and Critic observation assembly."""

    def _init_critic_observation_state(self) -> None:
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
                    self._robot.data.root_lin_vel_b
                    / TRAJECTORY_OBSERVATION.field("linear_velocity_error_b").physical_scale,
                    self._robot.data.root_ang_vel_b
                    / TRAJECTORY_OBSERVATION.field("angular_velocity_b").physical_scale,
                ),
                dim=-1,
            ),
            "water_current_b": quat_apply(quat_conjugate(root_quat_w), water_current_w)
            / TRAJECTORY_OBSERVATION.field("linear_velocity_error_b").physical_scale,
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

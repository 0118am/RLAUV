"""Construction and PhysX synchronization of per-environment dynamics state."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from environment.hydrodynamics.models import HydrodynamicForceModels
from environment.hydrodynamics.pool_effects import rectangular_sloshing_mode_frequencies
from robot.dynamics.parameters import AUV
from robot.dynamics.rigid_body import physx_principal_inertia_and_com_quat_xyzw
from robot.propulsion.curves import (
    get_thruster_positions,
    measured_thruster_body_forces,
)
from robot.propulsion.dynamics import FirstOrderThrusterResponse, ThrusterCommandProcessor

from .hydrodynamic_state import (
    PhysxHydrodynamicWrenchCfg,
    PhysxHydrodynamicWrenchManager,
    _nominal_hydro_coeff_tensor,
    _repeat_hydro_coeff_for_envs,
)


class AUVDynamicsStateMixin:
    """Own allocation and PhysX synchronization of dynamics state buffers."""

    def _init_vehicle_state(self, action_dim: int) -> None:
        # Get thruster configurations
        self.thruster_com_offsets = get_thruster_positions(self.device)
        self.num_thrusters = self.thruster_com_offsets.shape[0]
        if action_dim != self.num_thrusters:
            raise ValueError(
                f"Expected {self.num_thrusters} actions for the thruster model, got {action_dim}."
            )
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Get specific information about the AUV
        self._gravity_w = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float32)
        self._gravity_magnitude = self._gravity_w.norm()
        self._current_free_surface_z = torch.full(
            (self.num_envs, 1),
            float(self.cfg.free_surface_z),
            dtype=torch.float32,
            device=self.device,
        )

        nominal_principal_moments, nominal_principal_axes = physx_principal_inertia_and_com_quat_xyzw(
            self.cfg.inertia_diag,
            self.device,
        )
        self._nominal_principal_inertia = nominal_principal_moments
        self._nominal_principal_axes_xyzw = nominal_principal_axes
        self.inertia_principal_moments = nominal_principal_moments.reshape(1, 3).repeat(
            self.num_envs, 1
        )
        self.inertia_principal_axes_xyzw = nominal_principal_axes.reshape(1, 4).repeat(
            self.num_envs, 1
        )
        self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        self.center_of_mass_offsets = torch.as_tensor(
            self.cfg.center_of_mass_offset,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3).repeat(self.num_envs, 1)
        self._apply_nominal_rigid_body_properties()

        self.com_to_cob_offsets = torch.as_tensor(
            self.cfg.com_to_cob_offset,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3).repeat(self.num_envs, 1)
        volume = torch.as_tensor(self.cfg.volume, dtype=torch.float32, device=self.device)
        self.volumes = volume.reshape(1, 1).repeat(self.num_envs, 1)
        self._init_payload_domain()

        # Initialize dynamics calculators
        self._init_dynamics_state()

    def _init_dynamics_state(self) -> None:
        self._init_dynamics_models()
        self._init_flow_lookup_state()
        self._init_pool_and_actuator_reference_state()
        self._init_hydrodynamic_model_state()
        self._init_randomized_runtime_state()

    def _init_dynamics_models(self) -> None:
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_response = FirstOrderThrusterResponse(
            num_envs=self.num_envs,
            num_thrusters=self.num_thrusters,
            time_constant_s=self.cfg.dyn_time_constant,
            device=self.device,
        )
        self._thruster_force_curve_coefficients = torch.as_tensor(
            AUV.thruster_force_curve_coefficients,
            dtype=torch.float32,
            device=self.device,
        )
        delay_range = getattr(self.cfg.domain_randomization, "thruster_command_delay_steps_range", [0, 0])
        max_delay_steps = max(int(self.cfg.thruster_command_delay_steps), int(delay_range[1]))
        self.thruster_command_processor = ThrusterCommandProcessor(
            num_envs=self.num_envs,
            num_thrusters=self.num_thrusters,
            max_delay_steps=max_delay_steps,
            device=self.device,
        )
        self._runtime_zeros_env_1 = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        self._runtime_zeros_env_3 = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._runtime_ones_env_1 = torch.ones_like(self._runtime_zeros_env_1)
        self._runtime_ones_env_6 = torch.ones(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self._runtime_ones_thrusters = torch.ones(
            (self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device
        )
        self._runtime_flat_surface_z = torch.full_like(
            self._runtime_zeros_env_1,
            float(self.cfg.free_surface_z),
        )
        self._runtime_empty_frequencies = torch.empty(0, dtype=torch.float32, device=self.device)

    def _init_flow_lookup_state(self) -> None:
        self._periodic_current_amplitude_w = torch.as_tensor(
            self.cfg.water_current_periodic_amplitude_w,
            dtype=torch.float32,
            device=self.device,
        )
        self._periodic_current_period_s = torch.as_tensor(
            self.cfg.water_current_periodic_period_s,
            dtype=torch.float32,
            device=self.device,
        )
        self._periodic_current_phase_rad = torch.as_tensor(
            self.cfg.water_current_periodic_phase_rad,
            dtype=torch.float32,
            device=self.device,
        )
        self._current_field_bounds = torch.as_tensor(
            self.cfg.water_current_field_bounds,
            dtype=torch.float32,
            device=self.device,
        )
        self._current_field_shape = tuple(int(value) for value in self.cfg.water_current_field_shape)
        self._current_field_values = torch.as_tensor(
            self.cfg.water_current_field_values,
            dtype=torch.float32,
            device=self.device,
        )
        if self.cfg.water_current_field_enabled:
            current_field_bounds = np.asarray(self.cfg.water_current_field_bounds, dtype=np.float32)
            current_field_shape = self._current_field_shape
            if current_field_bounds.shape != (6,) or not (
                current_field_bounds[0] < current_field_bounds[1]
                and current_field_bounds[2] < current_field_bounds[3]
                and current_field_bounds[4] < current_field_bounds[5]
            ):
                raise ValueError("water_current_field_bounds must be ordered [xmin, xmax, ymin, ymax, zmin, zmax].")
            if len(current_field_shape) != 3 or any(value <= 0 for value in current_field_shape):
                raise ValueError("water_current_field_shape must contain three positive integers.")
            expected_values_shape = (*current_field_shape, 3)
            flattened_values_shape = (int(np.prod(current_field_shape)), 3)
            values_shape = tuple(np.asarray(self.cfg.water_current_field_values).shape)
            if values_shape not in (expected_values_shape, flattened_values_shape):
                raise ValueError(
                    "water_current_field_values must match the configured field shape or its flattened form."
                )
        self._damping_speed_points = torch.as_tensor(
            self.cfg.damping_speed_points,
            dtype=torch.float32,
            device=self.device,
        )

        def damping_scale_points(values) -> torch.Tensor:
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1 and tensor.numel() > 0:
                return tensor.reshape(-1, 1).expand(-1, 6)
            return tensor

        self._linear_damping_speed_scales = damping_scale_points(self.cfg.linear_damping_speed_scales)
        self._quadratic_damping_speed_scales = damping_scale_points(self.cfg.quadratic_damping_speed_scales)
        if self.cfg.speed_dependent_damping_enabled:
            speed_points = np.asarray(self.cfg.damping_speed_points, dtype=np.float32)
            if speed_points.ndim != 1 or speed_points.size < 2 or np.any(np.diff(speed_points) <= 0.0):
                raise ValueError("damping_speed_points must be a strictly increasing sequence with at least two values.")
            for name, values in (
                ("linear_damping_speed_scales", self._linear_damping_speed_scales),
                ("quadratic_damping_speed_scales", self._quadratic_damping_speed_scales),
            ):
                if values.numel() and values.shape != (speed_points.size, 6):
                    raise ValueError(f"{name} must contain one or six scales per damping speed point.")

    def _init_pool_and_actuator_reference_state(self) -> None:
        self._pool_bounds = torch.as_tensor(self.cfg.pool_bounds, dtype=torch.float32, device=self.device)
        self._sloshing_pool_bounds = torch.as_tensor(
            self.cfg.free_surface_sloshing_pool_bounds,
            dtype=torch.float32,
            device=self.device,
        )
        self._sloshing_mode_numbers = torch.as_tensor(
            self.cfg.free_surface_sloshing_mode_numbers,
            dtype=torch.float32,
            device=self.device,
        )
        self._sloshing_amplitudes_m = torch.as_tensor(
            self.cfg.free_surface_sloshing_amplitudes_m,
            dtype=torch.float32,
            device=self.device,
        )
        self._sloshing_phases_rad = torch.as_tensor(
            self.cfg.free_surface_sloshing_phases_rad,
            dtype=torch.float32,
            device=self.device,
        )
        if self.cfg.free_surface_sloshing_enabled:
            self._sloshing_angular_frequencies_rad_s = rectangular_sloshing_mode_frequencies(
                self._sloshing_pool_bounds,
                self.cfg.free_surface_sloshing_water_depth,
                self._sloshing_mode_numbers,
                float(self._gravity_magnitude),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._sloshing_angular_frequencies_rad_s = self._runtime_empty_frequencies
        self._thruster_spin_directions = torch.as_tensor(
            self.cfg.thruster_spin_directions,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, self.num_thrusters).expand(self.num_envs, self.num_thrusters)
        endpoint_commands = torch.tensor(
            [[-1.0] * self.num_thrusters, [1.0] * self.num_thrusters],
            dtype=torch.float32,
            device=self.device,
        )
        endpoint_forces = measured_thruster_body_forces(
            endpoint_commands,
            self._thruster_force_curve_coefficients,
        )
        self._thruster_wake_reference_force_n = max(
            float(torch.linalg.vector_norm(endpoint_forces, dim=-1).max().item()),
            1.0e-6,
        )
        dropout_range = getattr(
            self.cfg.domain_randomization,
            "thruster_command_dropout_probability_range",
            [0.0, 0.0],
        )
        self._thruster_command_dropout_enabled = float(self.cfg.thruster_command_dropout_probability) > 0.0 or (
            self._domain_randomization_feature_enabled("actuators")
            and max(float(value) for value in dropout_range) > 0.0
        )

    def _init_hydrodynamic_model_state(self) -> None:
        self._added_mass_enabled = bool(np.any(np.asarray(self.cfg.added_mass_diag, dtype=np.float32) != 0.0))
        residual_factors = np.concatenate(
            [
                np.asarray(values, dtype=np.float32).reshape(-1)
                for values in (
                    self.cfg.high_order_residual_added_mass_factor,
                    self.cfg.high_order_residual_linear_damping_factor,
                    self.cfg.high_order_residual_quadratic_damping_factor,
                    self.cfg.high_order_residual_cubic_damping_factor,
                )
            ]
        )
        self._high_order_residual_enabled = bool(
            self.cfg.high_order_residual_enabled and np.any(residual_factors != 0.0)
        )
        self._thruster_reaction_torque_enabled = float(self.cfg.thruster_reaction_torque_coeff) != 0.0
        self._added_mass_accel_filter_alpha = min(
            max(float(self.cfg.added_mass_accel_filter_alpha), 0.0),
            1.0,
        )
        self._effective_hydrodynamic_state = None
        self._pending_critic_hydrodynamic_env_ids = None
        # Final per-thruster force after all battery, pool, inflow, and wake
        # effects.  It is a Critic-only state and is cleared at every reset.
        self.realized_thruster_force_n = torch.zeros(
            (self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device
        )
        self.realized_thruster_forces_b = torch.zeros(
            (self.num_envs, self.num_thrusters, 3), dtype=torch.float32, device=self.device
        )
        self._nominal_linear_damping = _nominal_hydro_coeff_tensor(
            self.cfg.linear_damping, self.device, "linear_damping"
        )
        self._nominal_quadratic_damping = _nominal_hydro_coeff_tensor(
            self.cfg.quadratic_damping, self.device, "quadratic_damping"
        )
        self._nominal_added_mass_diag = _nominal_hydro_coeff_tensor(
            self.cfg.added_mass_diag, self.device, "added_mass_diag"
        )
        self.high_order_residual_added_mass_factor = _nominal_hydro_coeff_tensor(
            self.cfg.high_order_residual_added_mass_factor,
            self.device,
            "high_order_residual_added_mass_factor",
        )
        self.high_order_residual_linear_damping_factor = _nominal_hydro_coeff_tensor(
            self.cfg.high_order_residual_linear_damping_factor,
            self.device,
            "high_order_residual_linear_damping_factor",
        )
        self.high_order_residual_quadratic_damping_factor = _nominal_hydro_coeff_tensor(
            self.cfg.high_order_residual_quadratic_damping_factor,
            self.device,
            "high_order_residual_quadratic_damping_factor",
        )
        self.high_order_residual_cubic_damping_factor = _nominal_hydro_coeff_tensor(
            self.cfg.high_order_residual_cubic_damping_factor,
            self.device,
            "high_order_residual_cubic_damping_factor",
        )
        self.physx_hydrodynamic_wrench_manager = PhysxHydrodynamicWrenchManager(
            self.force_calculation_functions,
            PhysxHydrodynamicWrenchCfg(
                enabled=bool(self._high_order_residual_enabled and self.cfg.physx_high_order_wrench_enabled),
                base_scale=float(self.cfg.physx_high_order_wrench_base_scale),
                modulation_amplitude=float(self.cfg.physx_high_order_wrench_modulation_amplitude),
                modulation_frequency_hz=float(self.cfg.physx_high_order_wrench_modulation_frequency_hz),
                modulation_phase_rad=float(self.cfg.physx_high_order_wrench_modulation_phase_rad),
            ),
            added_mass_factor=self.high_order_residual_added_mass_factor,
            linear_damping_factor=self.high_order_residual_linear_damping_factor,
            quadratic_damping_factor=self.high_order_residual_quadratic_damping_factor,
            cubic_damping_factor=self.high_order_residual_cubic_damping_factor,
        )

    def _init_randomized_runtime_state(self) -> None:
        self._nominal_water_current_w = torch.tensor(
            self.cfg.water_current_w, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.linear_damping = _repeat_hydro_coeff_for_envs(self._nominal_linear_damping, self.num_envs)
        self.quadratic_damping = _repeat_hydro_coeff_for_envs(self._nominal_quadratic_damping, self.num_envs)
        self.added_mass_diag = _repeat_hydro_coeff_for_envs(self._nominal_added_mass_diag, self.num_envs)
        self.added_mass_randomization_scale = torch.ones(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self.damping_speed_linear_randomization_scale = torch.ones(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self.damping_speed_quadratic_randomization_scale = torch.ones_like(
            self.damping_speed_linear_randomization_scale
        )
        self.water_current_w = self._nominal_water_current_w.repeat(self.num_envs, 1)
        self.water_current_mean_w = self._nominal_water_current_w.repeat(self.num_envs, 1)
        current_std = float(getattr(self.cfg, "evaluation_current_variation_std", 0.0))
        if bool(getattr(self.cfg, "evaluation_current_override", False)):
            horizontal_limit = float(torch.linalg.vector_norm(self._nominal_water_current_w[0, 0:2]).item()) + 3.0 * current_std
            vertical_limit = abs(float(self._nominal_water_current_w[0, 2].item())) + 1.5 * current_std
        else:
            horizontal_limit = 0.0
            vertical_limit = 0.0
        self.water_current_horizontal_max = torch.full(
            (self.num_envs,), horizontal_limit, dtype=torch.float32, device=self.device
        )
        self.water_current_vertical_max = torch.full(
            (self.num_envs,), vertical_limit, dtype=torch.float32, device=self.device
        )
        self.water_current_tau = torch.full(
            (self.num_envs,), float(getattr(self.cfg, "evaluation_current_tau", 12.0)),
            dtype=torch.float32,
            device=self.device,
        )
        initial_force_scale = (
            float(self.cfg.evaluation_thruster_force_scale)
            if bool(getattr(self.cfg, "evaluation_thruster_force_scale_override", False))
            else 1.0
        )
        self.thruster_force_scale = torch.full(
            (self.num_envs, self.num_thrusters), initial_force_scale, device=self.device
        )
        self.thruster_time_constant = torch.full(
            (self.num_envs,), self.cfg.dyn_time_constant, dtype=torch.float32, device=self.device
        )
        self.thruster_delay_steps = torch.full(
            (self.num_envs,), int(self.cfg.thruster_command_delay_steps), dtype=torch.long, device=self.device
        )
        self.thruster_max_command_rate = torch.full(
            (self.num_envs, 1), self.cfg.thruster_max_command_rate, dtype=torch.float32, device=self.device
        )
        self.thruster_command_resolution = torch.full(
            (self.num_envs, 1), self.cfg.thruster_command_resolution, dtype=torch.float32, device=self.device
        )
        self.thruster_command_dropout_probability = torch.full(
            (self.num_envs, 1), self.cfg.thruster_command_dropout_probability, dtype=torch.float32, device=self.device
        )
        self.thruster_wake_loss_coefficient = torch.full(
            (self.num_envs,),
            self.cfg.thruster_wake_loss_coefficient,
            dtype=torch.float32,
            device=self.device,
        )
        self.thruster_reaction_torque_coeff = torch.full(
            (self.num_envs,),
            self.cfg.thruster_reaction_torque_coeff,
            dtype=torch.float32,
            device=self.device,
        )
        self.battery_initial_voltage = torch.full(
            (self.num_envs, 1), self.cfg.battery_voltage, dtype=torch.float32, device=self.device
        )
        self.battery_voltage = torch.full(
            (self.num_envs, 1), self.cfg.battery_voltage, dtype=torch.float32, device=self.device
        )
        self.battery_voltage_drop_per_s = torch.full(
            (self.num_envs, 1), self.cfg.battery_voltage_drop_per_s, dtype=torch.float32, device=self.device
        )
        self._battery_voltage_scale = torch.ones(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        self.tether_slack_length = torch.full(
            (self.num_envs, 1),
            self.cfg.tether_slack_length,
            dtype=torch.float32,
            device=self.device,
        )
        self.thruster_response.set_time_constants(self.thruster_time_constant)
        self._previous_nu_r = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._filtered_nu_r_dot = torch.zeros_like(self._previous_nu_r)
        self._has_previous_nu_r = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _apply_nominal_rigid_body_properties(self) -> None:
        """Apply the Heavy mass, inertia, and COM to the live PhysX body."""

        all_env_ids = self._robot._ALL_INDICES
        self._apply_runtime_mass_properties(all_env_ids)
        self._apply_runtime_center_of_mass(all_env_ids)
        self._robot.data.default_mass = self._robot.root_physx_view.get_masses().clone()
        self._robot.data.default_inertia = self._robot.root_physx_view.get_inertias().clone()

    def _apply_runtime_mass_properties(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Write per-env mass and matching inertia tensor into PhysX."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        env_ids_cpu = env_ids_device.detach().cpu()

        physx_masses = self._robot.root_physx_view.get_masses().clone()
        selected_masses = self.masses[env_ids_device].to(device=physx_masses.device, dtype=physx_masses.dtype)
        if physx_masses.ndim == 1:
            physx_masses[env_ids_cpu] = selected_masses.reshape(-1)
        else:
            physx_masses[env_ids_cpu] = selected_masses.reshape(len(env_ids_cpu), -1)
        self._robot.root_physx_view.set_masses(physx_masses, env_ids_cpu)

        physx_inertias = self._robot.root_physx_view.get_inertias().clone()
        selected_moments = self.inertia_principal_moments[env_ids_device].to(
            device=physx_inertias.device,
            dtype=physx_inertias.dtype,
        )
        flat_inertias = torch.diag_embed(selected_moments).reshape(-1, 9)
        if physx_inertias.ndim == 3:
            physx_inertias[env_ids_cpu, 0, :] = flat_inertias
        else:
            physx_inertias[env_ids_cpu, :] = flat_inertias
        self._robot.root_physx_view.set_inertias(physx_inertias, env_ids_cpu)

    def _apply_runtime_center_of_mass(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Write the body-frame COM offset into PhysX."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        env_ids_cpu = env_ids_device.detach().cpu()

        physx_coms = self._robot.root_physx_view.get_coms().clone()
        com_positions = self.center_of_mass_offsets[env_ids_device].to(
            device=physx_coms.device,
            dtype=physx_coms.dtype,
        )
        principal_axes = self.inertia_principal_axes_xyzw[env_ids_device].to(
            device=physx_coms.device,
            dtype=physx_coms.dtype,
        )
        if physx_coms.ndim == 3:
            physx_coms[env_ids_cpu, 0, :3] = com_positions
            physx_coms[env_ids_cpu, 0, 3:7] = principal_axes
        else:
            physx_coms[env_ids_cpu, :3] = com_positions
            physx_coms[env_ids_cpu, 3:7] = principal_axes
        self._robot.root_physx_view.set_coms(physx_coms, env_ids_cpu)

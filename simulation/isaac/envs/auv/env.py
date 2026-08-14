"""AUV IsaacLab environment facade.

Only lifecycle and policy-facing orchestration live here. Focused mixins own
observations, task curricula, the Isaac-to-wrench bridge, and debug rendering.
Physical equations and vehicle data come from the top-level ``environment``
and ``robot`` domains.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from environment.hydrodynamics.models import HydrodynamicForceModels
from environment.profiles.domain_randomization import (
    apply_domain_randomization_spec,
    resolve_domain_randomization_spec,
)
from environment.profiles.features import normalize_domain_randomization_features
from environment.profiles.pool_profile import (
    apply_pool_dynamics_profile,
    resolve_pool_dynamics_profile,
)
from environment.water.pool_effects import rectangular_sloshing_mode_frequencies
from robot.dynamics.parameters import AUV
from robot.dynamics.rigid_body import physx_principal_inertia_and_com_quat_xyzw
from robot.propulsion.thrusters import (
    DynamicsFirstOrder,
    ThrusterCommandProcessor,
    get_thruster_positions,
    measured_thruster_body_forces,
    reduce_point_forces_to_wrench,
)

from .physx_hydrodynamics import (
    PhysxHydrodynamicWrenchCfg,
    PhysxHydrodynamicWrenchManager,
)
from .config import AUVTrajEnvCfg
from .dynamics import AUVDynamicsMixin, _nominal_hydro_coeff_tensor, _repeat_hydro_coeff_for_envs
from .observations import AUVObservationMixin
from ...agents.rewards import apply_tracking_reward_policy, get_tracking_reward_function
from .trajectory.mixin import AUVTrajectoryMixin
from .trajectory.guidance import root_state_at_tracking_target
from .trajectory.kinematics import LATERAL_SINE, SPATIAL_HELIX, VERTICAL_SINE
from .visualization import AUVVisualizationMixin


def _scale_evaluation_damping(values, scale: float, name: str) -> list:
    """Scale a diagonal 6-vector or full 6x6 damping matrix."""

    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape not in ((6,), (6, 6)):
        raise ValueError(f"{name} must be a 6-vector or 6x6 matrix, got shape {coefficients.shape}.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} must contain only finite values.")
    return (coefficients * scale).tolist()


def _apply_evaluation_physics_overlay(cfg) -> None:
    """Apply the evaluation-only physics overlay after profile and DR setup."""

    overlay = dict(getattr(cfg, "evaluation_physics_overlay", {}) or {})
    allowed = {
        "damping_scale",
        "thruster_tau_scale",
        "water_current_w",
        "smooth_current",
        "current_variation_std",
        "current_tau",
        "current_feature_only",
        "thruster_force_scale",
    }
    unknown = set(overlay) - allowed
    if unknown:
        raise ValueError(f"Unknown evaluation physics overlay keys: {', '.join(sorted(unknown))}.")
    damping_scale = float(overlay.get("damping_scale", 1.0))
    tau_scale = float(overlay.get("thruster_tau_scale", 1.0))
    if not np.isfinite(damping_scale) or not np.isfinite(tau_scale) or damping_scale <= 0.0 or tau_scale <= 0.0:
        raise ValueError("Evaluation physics overlay scales must be finite and positive.")
    cfg.linear_damping = _scale_evaluation_damping(
        cfg.linear_damping,
        damping_scale,
        "linear_damping",
    )
    cfg.quadratic_damping = _scale_evaluation_damping(
        cfg.quadratic_damping,
        damping_scale,
        "quadratic_damping",
    )
    cfg.dyn_time_constant = float(cfg.dyn_time_constant) * tau_scale
    current_override = "water_current_w" in overlay or bool(overlay.get("smooth_current", False))
    if current_override:
        current = overlay.get("water_current_w", cfg.water_current_w)
        current_values = [float(value) for value in current]
        if len(current_values) != 3 or not all(np.isfinite(value) for value in current_values):
            raise ValueError("evaluation_physics_overlay.water_current_w must contain three finite values.")
        variation_std = float(overlay.get("current_variation_std", 0.0))
        current_tau = float(overlay.get("current_tau", 12.0))
        if not np.isfinite(variation_std) or variation_std < 0.0:
            raise ValueError("evaluation_physics_overlay.current_variation_std must be finite and non-negative.")
        if not np.isfinite(current_tau) or current_tau <= 0.0:
            raise ValueError("evaluation_physics_overlay.current_tau must be finite and positive.")

        cfg.water_current_w = current_values
        cfg.evaluation_current_override = True
        cfg.evaluation_current_variation_std = variation_std
        cfg.evaluation_current_tau = current_tau
        smooth_current = bool(overlay.get("smooth_current", False)) or variation_std > 0.0
        if smooth_current:
            cfg.domain_randomization.use_custom_randomization = True
            cfg.domain_randomization.water_current_smooth = True
            stage_count = max(1, len(cfg.domain_randomization.water_current_max_by_stage))
            cfg.domain_randomization.water_current_variation_std_by_stage = [variation_std] * stage_count
            cfg.domain_randomization.water_current_tau_range = [current_tau, current_tau]
            cfg.eval_domain_randomization = True
            if bool(overlay.get("current_feature_only", False)):
                cfg.domain_randomization.enabled_features = ["current"]

    if "thruster_force_scale" in overlay:
        force_scale = float(overlay["thruster_force_scale"])
        if not np.isfinite(force_scale) or force_scale <= 0.0:
            raise ValueError("evaluation_physics_overlay.thruster_force_scale must be finite and positive.")
        cfg.evaluation_thruster_force_scale_override = True
        cfg.evaluation_thruster_force_scale = force_scale


class AUVTrajEnv(
    AUVObservationMixin,
    AUVTrajectoryMixin,
    AUVDynamicsMixin,
    AUVVisualizationMixin,
    DirectRLEnv,
):
    """Canonical IsaacLab trajectory environment for the AUV vehicle.

    The facade owns DirectRLEnv callbacks and shared simulation state.  Mixin
    order is intentional: each focused component supplies a disjoint callback
    group before IsaacLab's base implementation is reached.
    """

    cfg: AUVTrajEnvCfg

    def __init__(self, cfg: AUVTrajEnvCfg, render_mode: str | None = None, **kwargs):
        self._tracking_reward_policy = apply_tracking_reward_policy(cfg)
        self._tracking_reward_fn = get_tracking_reward_function(cfg.tracking_reward_profile)
        feature_override = None
        if bool(getattr(cfg, "domain_randomization_feature_override_enabled", False)):
            feature_override = list(getattr(cfg.domain_randomization, "enabled_features", ()))
            normalize_domain_randomization_features(feature_override)
        if not getattr(cfg, "pool_dynamics_profile", None):
            raise ValueError("pool_dynamics_profile is required; implicit hydrodynamic fallbacks are disabled.")
        resolved_pool_profile = resolve_pool_dynamics_profile(cfg.pool_dynamics_profile)
        # The selected profile is the only nominal coefficient source.
        apply_pool_dynamics_profile(
            cfg,
            resolved_pool_profile,
            include_legacy_domain_randomization=False,
        )
        cfg.pool_dynamics_profile_name = resolved_pool_profile.name
        if getattr(cfg, "domain_randomization_spec", None):
            randomization_spec = resolve_domain_randomization_spec(cfg.domain_randomization_spec)
            apply_domain_randomization_spec(
                cfg,
                randomization_spec,
                base_profile=resolved_pool_profile,
            )
        if feature_override is not None:
            cfg.domain_randomization.enabled_features = feature_override
        cfg.domain_randomization.enabled_features = list(
            normalize_domain_randomization_features(
                getattr(cfg.domain_randomization, "enabled_features", None)
            )
        )
        # The only supported ordering is deterministic profile -> DR recipe
        # -> evaluation overlay. This prevents a selected profile from silently
        # replacing CLI evaluation modifiers.
        _apply_evaluation_physics_overlay(cfg)
        # The training notebook selects a named MLP profile.  Resolve its
        # history-expanded observation space before IsaacLab allocates its
        # vector-environment buffers; sensor transport itself remains 30-D.
        self._configure_mlp_observation_space(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        # Keep the PPO-sampled command separately from the bounded command
        # accepted by the actuator chain.  This exposes any action/log-prob
        # versus execution mismatch without changing the current policy API.
        action_dim = int(self.single_action_space.shape[0])
        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_applied_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Moving-target command buffers shared by training, evaluation, and
        # debug visualization.
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_lin_acc_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_lin_jerk_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_derivative_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # A host-side generation marker makes repeated reward/observation/eval
        # requests in the same policy step a zero-cost lookup. Partial reset
        # updates remain explicit because their episode clocks restart at zero.
        self._tracking_target_common_step: int | None = None
        # Host-side superset of trajectory types currently present in the
        # population. It lets the hot path omit impossible branches without a
        # CUDA reduction or dynamic-size indexing operation.

        # Per-environment trajectory parameters sampled at reset.  traj_type is
        # 0=circle, 1=Lissajous, 2=single-axis sine, 3=legacy helix, 4=spiral,
        # 5=chirp, 6=racetrack, 7=random smooth Fourier curve,
        # 8=lateral sine, 9=vertical sine, 10=spatial helix.
        self._traj_center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._traj_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._traj_axis = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._traj_amp_x = torch.zeros(self.num_envs, device=self.device)
        self._traj_amp_y = torch.zeros(self.num_envs, device=self.device)
        self._traj_amp_z = torch.zeros(self.num_envs, device=self.device)
        self._traj_period = torch.ones(self.num_envs, device=self.device)
        self._traj_target_speed_mps = torch.zeros(self.num_envs, device=self.device)
        self._traj_phase_x = torch.zeros(self.num_envs, device=self.device)
        self._traj_phase_y = torch.zeros(self.num_envs, device=self.device)
        retime_nodes = int(self.cfg.trajectory_retime_samples) + 1
        self._traj_retime_phase = torch.zeros(self.num_envs, retime_nodes, device=self.device)
        self._traj_retime_elapsed_s = torch.zeros(self.num_envs, retime_nodes, device=self.device)
        self._traj_retime_phase_rate = torch.zeros(self.num_envs, retime_nodes, device=self.device)
        self._traj_retime_phase_acceleration = torch.zeros(self.num_envs, retime_nodes, device=self.device)
        self._traj_effective_period_s = torch.ones(self.num_envs, device=self.device)
        self._traj_retimed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._traj_curvature_m_inv = torch.zeros(self.num_envs, device=self.device)
        self._traj_target_orientation_rate_radps = torch.zeros(self.num_envs, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Get thruster configurations
        self.thruster_com_offsets = get_thruster_positions(self.device)
        self.num_thrusters = self.thruster_com_offsets.shape[0]
        if action_dim != self.num_thrusters:
            raise ValueError(
                f"Expected {self.num_thrusters} actions for the thruster model, got {action_dim}."
            )
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

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
        self._init_thruster_dynamics()
        self._init_observation_sensor_model()
        
        # Sample initial trajectory commands and reset runtime state.
        self._reset_idx(self._robot._ALL_INDICES)


    def _init_thruster_dynamics(self):
        # Fluid and motor models are vectorized over environments.  The damping
        # coefficients are body-frame Fossen diagonal entries for nu_r.
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(
            self.num_envs,
            self.num_thrusters,
            self.cfg.dyn_time_constant,
            self.device,
        )
        self._thruster_force_curve_coefficients = torch.as_tensor(
            AUV.thruster_force_curve_coefficients,
            dtype=torch.float32,
            device=self.device,
        )
        delay_range = getattr(self.cfg.domain_randomization, "thruster_command_delay_steps_range", [0, 0])
        max_delay_steps = max(int(self.cfg.thruster_command_delay_steps), int(delay_range[1]))
        self.thruster_command_processor = ThrusterCommandProcessor(
            self.num_envs,
            self.num_thrusters,
            max_delay_steps,
            self.device,
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
        self.tether_slack_length = torch.full(
            (self.num_envs, 1),
            self.cfg.tether_slack_length,
            dtype=torch.float32,
            device=self.device,
        )
        self.thruster_dynamics.tau = self.thruster_time_constant
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

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        # This is a visual pool floor, not a water-surface ceiling. The AUV
        # starts at z=-8 m and the configured pool lower boundary is z=-15 m.
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(color=(0.12, 0.16, 0.18), size=(30.0, 30.0)),
            translation=(0.0, 0.0, -15.0),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug: print("original actions vec: ", actions)
        if self._debug: print("concatenated actions shape: ", self._actions)

        self._update_smooth_water_current()
        self._previous_actions[:] = self._actions
        # Snapshot the command actually realized at the preceding policy step.
        # ``_compute_dynamics`` advances this state once per physics substep.
        self._previous_applied_actions[:] = self.thruster_command_processor.rate_limited_state
        self._raw_actions[:] = actions.to(self.device)
        self._actions[:] = self._raw_actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)

    def _log_action_transport_diagnostics(self) -> None:
        """Periodically expose PPO-command-to-thruster transport statistics."""

        interval = max(1, int(getattr(self.cfg, "domain_randomization_log_interval_steps", 250)))
        last_step = getattr(self, "_last_action_transport_log_step", None)
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_action_transport_log_step = self.common_step_counter

        applied_actions = self.thruster_command_processor.rate_limited_state
        raw_clipped = self._raw_actions.abs() > 1.0
        transport_delta = self._actions - applied_actions
        log = self.extras.setdefault("log", {})
        log["raw_action_clip_fraction"] = raw_clipped.to(dtype=torch.float32).mean()
        log["raw_action_vector_clip_fraction"] = raw_clipped.any(dim=1).to(dtype=torch.float32).mean()
        log["requested_action_saturation_fraction"] = (
            self._actions.abs() > 0.95
        ).to(dtype=torch.float32).mean()
        log["applied_action_saturation_fraction"] = (
            applied_actions.abs() > 0.95
        ).to(dtype=torch.float32).mean()
        log["requested_to_applied_action_rms"] = torch.sqrt(torch.mean(transport_delta**2))
        log["requested_to_applied_action_fraction"] = (
            transport_delta.abs() > 1.0e-4
        ).to(dtype=torch.float32).mean()
        log["requested_action_rate_rms"] = torch.sqrt(torch.mean((self._actions - self._previous_actions) ** 2))
        log["applied_action_rate_rms"] = torch.sqrt(
            torch.mean((applied_actions - self._previous_applied_actions) ** 2)
        )
        log["realized_thruster_force_abs_mean_n"] = self.realized_thruster_force_n.abs().mean()
        log["realized_thruster_force_abs_max_n"] = self.realized_thruster_force_n.abs().max()
        realized_thruster_wrench_b = reduce_point_forces_to_wrench(
            self.thruster_com_offsets,
            self.realized_thruster_forces_b,
        )
        log["realized_thruster_wrench_force_norm_n"] = torch.linalg.vector_norm(
            realized_thruster_wrench_b[:, :3], dim=1
        ).mean()
        log["realized_thruster_wrench_torque_norm_nm"] = torch.linalg.vector_norm(
            realized_thruster_wrench_b[:, 3:], dim=1
        ).mean()
        # The total external wrench also contains buoyancy and hydrodynamics;
        # name it explicitly so the roughly 110 N buoyancy term is never
        # mistaken for impossible thrust from eight T60s.
        log["applied_total_external_wrench_force_norm_n"] = torch.linalg.vector_norm(
            self._thrust[:, 0, :], dim=1
        ).mean()
        log["applied_total_external_wrench_torque_norm_nm"] = torch.linalg.vector_norm(
            self._moment[:, 0, :], dim=1
        ).mean()
        log["target_speed_mps"] = torch.linalg.vector_norm(self._target_lin_vel_w, dim=1).mean()
        log["requested_trajectory_speed_mps"] = self._traj_target_speed_mps.mean()
        log["trajectory_lateral_sine_fraction"] = (self._traj_type == LATERAL_SINE).float().mean()
        log["trajectory_vertical_sine_fraction"] = (self._traj_type == VERTICAL_SINE).float().mean()
        log["trajectory_spatial_helix_fraction"] = (self._traj_type == SPATIAL_HELIX).float().mean()
        for speed_mps in self.cfg.trajectory_speed_levels_mps:
            speed_label = f"{float(speed_mps):.1f}".replace(".", "p")
            log[f"trajectory_speed_{speed_label}_mps_fraction"] = torch.isclose(
                self._traj_target_speed_mps,
                torch.as_tensor(float(speed_mps), device=self.device),
            ).float().mean()
        log["target_acceleration_mps2"] = torch.linalg.vector_norm(self._target_lin_acc_w, dim=1).mean()

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._actions)
        self._robot.permanent_wrench_composer.set_forces_and_torques(forces=self._thrust, torques=self._moment)

    def _get_observations(self) -> dict:
        root_position_w, root_quat_w, root_linear_velocity_b, root_angular_velocity_b = (
            self._state_for_observation()
        )
        # Keep the target synchronized with the current episode time before
        # constructing the policy observation.
        self._update_tracking_targets()
        target_pos_error_b = quat_apply(
            quat_conjugate(root_quat_w),
            self._target_pos_w - root_position_w,
        )
        target_lin_vel_b = quat_apply(
            quat_conjugate(root_quat_w),
            self._target_lin_vel_w,
        )
        target_lin_acc_b = quat_apply(
            quat_conjugate(root_quat_w),
            self._target_lin_acc_w,
        )
        target_ang_vel_b = quat_apply(
            quat_conjugate(root_quat_w),
            self._target_ang_vel_w,
        )
        linear_velocity_error_b = target_lin_vel_b - root_linear_velocity_b
        attitude_error_quat = math_utils.quat_unique(
            math_utils.quat_mul(quat_conjugate(root_quat_w), self._target_quat_w)
        )
        raw_obs = torch.cat(
            [
                target_pos_error_b,
                target_lin_vel_b,
                linear_velocity_error_b,
                attitude_error_quat,
                root_angular_velocity_b,
                target_ang_vel_b,
                target_lin_acc_b,
                # Feed back the command after actuator delay/rate limiting,
                # not merely the latest requested PPO action.  This signal is
                # available from the real controller and lets an MLP infer
                # short actuator transients from its explicit history.
                self.thruster_command_processor.rate_limited_state,
            ],
            dim=-1,
        )
        measured_obs = self._apply_observation_sensor_model(raw_obs)
        normalized_obs = self._normalize_trajectory_observation(measured_obs)
        actor_obs = self._stack_mlp_history(normalized_obs)
        # RSL-RL maps this simulator-only group to V(o, z_priv).  It is never
        # read by the exported Actor or by trajectory evaluation.
        return {"policy": actor_obs, "critic": self._build_critic_observation(actor_obs)}

    def _get_rewards(self) -> torch.Tensor:
        self._update_tracking_targets()
        self._log_action_transport_diagnostics()
        target_lin_vel_b = quat_apply(
            quat_conjugate(self._robot.data.root_quat_w),
            self._target_lin_vel_w,
        )
        if self.cfg.rew_action_source == "applied":
            reward_actions = self.thruster_command_processor.rate_limited_state
            previous_reward_actions = self._previous_applied_actions
        else:
            reward_actions = self._actions
            previous_reward_actions = self._previous_actions
        reward_args = (
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_track_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_forward,
            self.cfg.rew_scale_motion_alignment,
            self.cfg.rew_scale_actions,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_pos_sigma,
            self.cfg.rew_ang_sigma,
            self.cfg.rew_track_vel_sigma,
            self.cfg.rew_ang_vel_sigma,
            self.cfg.rew_forward_min_speed,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._target_pos_w,
            self._target_quat_w,
            target_lin_vel_b,
            reward_actions,
            previous_reward_actions,
        )
        if bool(
            self._tracking_reward_policy is not None
            and self._tracking_reward_policy.requires_action_rate_limit
        ):
            policy_dt = self.physics_dt * self.cfg.decimation
            rate = self.thruster_max_command_rate.reshape(-1, 1)
            # The command processor treats non-positive rate as unlimited.
            # Use the full normalized range as the penalty reference in that
            # case instead of treating zero as a hidden finite scale.
            rate_limit = torch.where(rate > 0.0, rate * policy_dt, torch.ones_like(rate))
            reward = self._tracking_reward_fn(*reward_args, rate_limit)
        else:
            reward = self._tracking_reward_fn(*reward_args)
        # DirectRLEnv computes dones before rewards. Penalize safety/boundary
        # terminations while leaving ordinary fixed-horizon timeouts neutral.
        return reward - self.cfg.rew_scale_terminated * self.reset_terminated.to(dtype=reward.dtype)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.episode_length_before_reset is not None:
            time_out = time_out | (self.episode_length_buf >= int(self.cfg.episode_length_before_reset))

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.thruster_dynamics.reset(env_ids)
        self.thruster_command_processor.reset(env_ids)
        self.observation_delay_buffer.reset(env_ids)
        self.observation_filter_state.reset(env_ids)
        self._reset_mlp_history(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_applied_actions[env_ids] = 0.0
        self.realized_thruster_force_n[env_ids] = 0.0
        self.realized_thruster_forces_b[env_ids] = 0.0
        self.physx_hydrodynamic_wrench_manager.reset(env_ids)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        # Sample t=0 of the command before constructing the initial rigid-body
        # state so position, attitude, and velocity can match the reference.
        self._reset_trajectory(env_ids)

        if not self.cfg.eval_mode and not self.cfg.trajectory_match_initial_state:
            # Randomize the initial position around the trajectory center.
            self._default_root_state[env_ids, :3] += self._sample_from_sphere(
                len(env_ids), self.cfg.trajectory_initial_position_radius
            )

            # Randomize initial linear and rotational velocities
            self._default_root_state[env_ids, 7:13] = math_utils.sample_uniform(
                -self.cfg.init_vel_max,
                self.cfg.init_vel_max,
                (len(env_ids), 6),
                device=self.device,
            )

        # Apply domain randomization
        self._reset_domain(env_ids)
        self._previous_nu_r[env_ids] = 0.0
        self._filtered_nu_r_dot[env_ids] = 0.0
        self._has_previous_nu_r[env_ids] = False

        if self.cfg.trajectory_match_initial_state:
            self._default_root_state[env_ids, :] = root_state_at_tracking_target(
                self._default_root_state[env_ids, :],
                self._target_pos_w[env_ids, :],
                self._target_quat_w[env_ids, :],
                self._target_lin_vel_w[env_ids, :],
                self._target_ang_vel_w[env_ids, :],
            )
        elif not self.cfg.eval_mode:
            # Apply guidance (set to target position and orientation).
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            # Guidance is a curriculum trick: a small fraction of resets start
            # at the target pose for near-target stabilization experience.
            self._default_root_state[env_ids_to_guide, :3] = self._target_pos_w[env_ids_to_guide, :3]
            self._default_root_state[env_ids_to_guide, 3:7] = self._target_quat_w[env_ids_to_guide, 0:4]
        elif self.cfg.trajectory_eval_align_initial_target:
            self._default_root_state[env_ids, :3] = self._target_pos_w[env_ids, :3]
            self._default_root_state[env_ids, 3:7] = self._target_quat_w[env_ids, 0:4]

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)
        self._pending_critic_hydrodynamic_env_ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        )

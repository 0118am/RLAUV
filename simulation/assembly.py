"""The single IsaacLab/PhysX assembly for the environment-owned AUV task.

Only simulator-facing lifecycle and orchestration live here. Training mixins
provide observations, curricula, rewards, and debug rendering; environment
and robot modules provide the physical equations and vehicle data. This file
alone owns the Isaac/PhysX state and wrench bridge.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate

from environment.profiles.composition import resolve_runtime_composition
from environment.profiles.evaluation import apply_evaluation_physics_overlay
from environment.profiles.features import normalize_domain_randomization_features
from environment.randomization import reset_current, reset_hydrodynamics
from environment.runtime import BodyKinematics, EnvironmentRuntimeState
from robot.randomization import reset_actuators, reset_battery, reset_rigid_body
from robot.runtime_state import RobotRuntimeState
from simulation.training.observations import AUVObservationMixin
from simulation.training.rewards import apply_tracking_reward_policy, get_tracking_reward_function
from simulation.training.config import AUVTrajEnvCfg
from robot.control.trajectory.guidance import root_state_at_tracking_target
from robot.control.trajectory import LATERAL_SINE, SPATIAL_HELIX, VERTICAL_SINE
from simulation.training.trajectory import AUVTrajectoryMixin
from simulation.training.visualization import AUVVisualizationMixin

class AUVTrajEnv(
    AUVObservationMixin,
    AUVTrajectoryMixin,
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
        self._tracking_reward_fn, self._tracking_reward_variant = get_tracking_reward_function(
            cfg.tracking_reward_profile
        )
        feature_override = None
        if cfg.domain_randomization_feature_override_enabled:
            feature_override = list(cfg.domain_randomization.enabled_features)
            normalize_domain_randomization_features(feature_override)
        if not cfg.environment_profile:
            raise ValueError("environment_profile is required; implicit physics fallbacks are disabled.")
        composition = resolve_runtime_composition(
            cfg.environment_profile,
            cfg.domain_randomization_spec,
        )
        self._runtime_composition = composition
        composition.apply(cfg)
        if feature_override is not None:
            cfg.domain_randomization.enabled_features = feature_override
        cfg.domain_randomization.enabled_features = list(
            normalize_domain_randomization_features(cfg.domain_randomization.enabled_features)
        )
        # The only supported ordering is deterministic profile -> DR recipe
        # -> evaluation overlay. This prevents a selected profile from silently
        # replacing CLI evaluation modifiers.
        apply_evaluation_physics_overlay(cfg)
        self._configured_pool_center = tuple(
            0.5 * (float(cfg.pool_bounds[index]) + float(cfg.pool_bounds[index + 1]))
            for index in (0, 2, 4)
        )
        cfg.viewer.lookat = self._configured_pool_center
        # The Python training recipe selects a named MLP profile. Resolve its
        # history-expanded observation space before IsaacLab allocates its
        # vector-environment buffers.
        self._configure_mlp_observation_space(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = int(self.single_action_space.shape[0])
        self._init_action_state(action_dim)
        self._init_trajectory_state()
        gravity_w = torch.as_tensor(
            self.sim.cfg.gravity, dtype=torch.float32, device=self.device
        )
        self.environment_runtime = EnvironmentRuntimeState(
            cfg,
            num_envs=self.num_envs,
            device=self.device,
            gravity_w=gravity_w,
            pool_center_local=self._configured_pool_center,
        )
        self.robot_runtime = RobotRuntimeState(
            cfg,
            model=self._runtime_composition.robot.model,
            num_envs=self.num_envs,
            action_dim=action_dim,
            device=self.device,
        )
        self._init_observation_state()

        self.set_debug_vis(self.cfg.debug_vis)

        # Sample initial trajectory commands and reset runtime state.
        self._reset_idx(self._robot._ALL_INDICES)

    def _init_action_state(self, action_dim: int) -> None:
        """Allocate policy-action and applied-wrench lifecycle buffers."""

        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, action_dim, device=self.device
        )
        self._previous_applied_actions = torch.zeros_like(self._previous_actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros_like(self._thrust)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(
            pos=self._configured_pool_center
        )
        self._robot = RigidObject(self.cfg.robot_cfg)

        # This is the pool floor, not a water-surface ceiling. Its height comes
        # from the same environment profile used by boundary termination.
        floor_cfg = sim_utils.CuboidCfg(
            size=(30.0, 30.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.16, 0.18)
            ),
        )
        floor_cfg.func(
            "/World/ground",
            floor_cfg,
            translation=(0.0, 0.0, float(self.cfg.pool_bounds[4]) - 0.01),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        stage = self._get_disturbance_curriculum_stage()
        self.environment_runtime.advance_smooth_current(
            stage=stage,
            enabled=self._domain_randomization_feature_enabled("current"),
            policy_dt=self.physics_dt * self.cfg.decimation,
        )
        episode_time_s = (
            self.episode_length_buf.to(dtype=torch.float32).unsqueeze(-1)
            * self.physics_dt
            * self.cfg.decimation
        )
        self.robot_runtime.advance_battery(episode_time_s)
        self._previous_actions[:] = self._actions
        # Snapshot the command actually realized at the preceding policy step.
        # ``_compute_dynamics`` advances this state once per physics substep.
        self._previous_applied_actions[:] = (
            self.robot_runtime.thruster_command_processor.rate_limited_state
        )
        incoming_actions = actions.to(device=self.device, dtype=self._actions.dtype)
        self._raw_actions.copy_(incoming_actions)
        self._actions.copy_(incoming_actions)

    def _log_action_transport_diagnostics(self) -> None:
        """Periodically expose PPO-command-to-thruster transport statistics."""

        interval = max(1, int(getattr(self.cfg, "domain_randomization_log_interval_steps", 250)))
        last_step = getattr(self, "_last_action_transport_log_step", None)
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_action_transport_log_step = self.common_step_counter

        robot = self.robot_runtime
        applied_actions = robot.thruster_command_processor.rate_limited_state
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
        log["realized_thruster_force_abs_mean_n"] = robot.realized_thruster_force_n.abs().mean()
        log["realized_thruster_force_abs_max_n"] = robot.realized_thruster_force_n.abs().max()
        log["realized_thruster_wrench_force_norm_n"] = torch.linalg.vector_norm(
            robot.realized_thruster_wrench_b[:, :3], dim=1
        ).mean()
        log["realized_thruster_wrench_torque_norm_nm"] = torch.linalg.vector_norm(
            robot.realized_thruster_wrench_b[:, 3:], dim=1
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
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._actions)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._thrust,
            torques=self._moment,
        )

    def _body_kinematics(self) -> BodyKinematics:
        """Copy the current PhysX view into the runtime's read-only contract."""

        return BodyKinematics(
            root_position_w=self._robot.data.root_pos_w,
            root_position_local_w=self._robot.data.root_pos_w - self.scene.env_origins,
            root_quat_w=self._robot.data.root_quat_w,
            root_linear_velocity_w=self._robot.data.root_lin_vel_w,
            root_linear_velocity_b=self._robot.data.root_lin_vel_b,
            root_angular_velocity_b=self._robot.data.root_ang_vel_b,
            scene_origins_w=self.scene.env_origins,
        )

    def _effective_hydrodynamic_state_for_critic(self):
        effective = self.environment_runtime.effective_state
        if effective is None:
            effective = self.environment_runtime.calculate_effective_state(
                self._body_kinematics(),
                sim_time_s=self._sim_step_counter * self.physics_dt,
                additional_scale=self._additional_hydrodynamics_scale(),
            )
        return effective

    def _compute_dynamics(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Call environment and robot models, then form the single COM wrench."""

        physics_time_s = self._sim_step_counter * self.physics_dt
        kinematics = self._body_kinematics()
        raw_thruster_forces_b = self.robot_runtime.advance_thruster_forces(
            actions, physics_time_s=physics_time_s, physics_dt=self.physics_dt
        )
        additional_scale = self._additional_hydrodynamics_scale()
        effective_hydrodynamics = self.environment_runtime.calculate_effective_state(
            kinematics,
            sim_time_s=physics_time_s,
            additional_scale=additional_scale,
        )
        thruster_forces, thruster_torques = self.robot_runtime.compose_thruster_wrench(
            raw_thruster_forces_b,
            relative_velocity_b=effective_hydrodynamics.relative_velocity_b,
            environment_thruster_scale=effective_hydrodynamics.thruster_scale,
        )
        relative_acceleration_b = None
        if float(self.cfg.added_mass_inertia_scale) > 0.0:
            relative_acceleration_b = self.environment_runtime.update_relative_acceleration(
                effective_hydrodynamics.relative_velocity_b,
                physics_dt=self.physics_dt,
            ) * float(self.cfg.added_mass_inertia_scale)
        fluid_forces, fluid_torques = self.environment_runtime.compose_fluid_wrench(
            kinematics,
            effective_hydrodynamics,
            volumes=self.robot_runtime.volumes,
            com_to_cob_offsets=self.robot_runtime.com_to_cob_offsets,
            relative_acceleration_b=relative_acceleration_b,
        )
        tether_forces, tether_torques = self.robot_runtime.compose_tether_wrench(
            kinematics,
            water_current_w=effective_hydrodynamics.water_current_w,
            gravity_w=self.environment_runtime.gravity_w,
            physics_dt=self.physics_dt,
            additional_scale=additional_scale,
        )
        return (
            fluid_forces + thruster_forces + tether_forces,
            fluid_torques + thruster_torques + tether_torques,
        )

    def _get_observations(self) -> dict:
        root_position_w, root_quat_w, root_linear_velocity_b, root_angular_velocity_b = (
            self._state_for_observation()
        )
        # Keep the target synchronized with the current episode time before
        # constructing the policy observation.
        self._update_tracking_targets()
        root_quat_conjugate = quat_conjugate(root_quat_w)
        target_pos_error_b = quat_apply(
            root_quat_conjugate,
            self._target_pos_w - root_position_w,
        )
        target_lin_vel_b = quat_apply(
            root_quat_conjugate,
            self._target_lin_vel_w,
        )
        target_lin_acc_b = quat_apply(
            root_quat_conjugate,
            self._target_lin_acc_w,
        )
        target_ang_vel_b = quat_apply(
            root_quat_conjugate,
            self._target_ang_vel_w,
        )
        linear_velocity_error_b = target_lin_vel_b - root_linear_velocity_b
        attitude_error_quat = math_utils.quat_unique(
            math_utils.quat_mul(root_quat_conjugate, self._target_quat_w)
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
                self.robot_runtime.thruster_command_processor.rate_limited_state,
            ],
            dim=-1,
        )
        normalized_obs = self._normalize_trajectory_observation(raw_obs)
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
            reward_actions = self.robot_runtime.thruster_command_processor.rate_limited_state
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
        policy_dt = self.physics_dt * self.cfg.decimation
        rate = self.robot_runtime.thruster_max_command_rate.reshape(-1, 1)
        # A non-positive command rate means unlimited; use the normalized
        # command range as the inactive denominator for all reward variants.
        rate_limit = torch.where(rate > 0.0, rate * policy_dt, torch.ones_like(rate))
        reward = self._tracking_reward_fn(
            self._tracking_reward_variant,
            *reward_args,
            rate_limit,
        )
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
            local_position = self._robot.data.root_pos_w - self.scene.env_origins
            out_of_bounds = (
                (local_position[:, 0] < self.environment_runtime.pool_bounds[0])
                | (local_position[:, 0] > self.environment_runtime.pool_bounds[1])
                | (local_position[:, 1] < self.environment_runtime.pool_bounds[2])
                | (local_position[:, 1] > self.environment_runtime.pool_bounds[3])
                | (local_position[:, 2] < self.environment_runtime.pool_bounds[4])
                | (local_position[:, 2] > self.environment_runtime.pool_bounds[5])
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return out_of_bounds, time_out

    def _sample_initial_position_offset(self, count: int, radius: float) -> torch.Tensor:
        """Sample uniformly inside the configured reset-error sphere."""

        directions = torch.randn((count, 3), device=self.device)
        directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True).clamp_min(1.0e-8)
        radii = float(radius) * torch.rand((count, 1), device=self.device).pow(1.0 / 3.0)
        return radii * directions

    def _domain_randomization_enabled(self) -> bool:
        return bool(self.cfg.domain_randomization.use_custom_randomization) and (
            not self.cfg.eval_mode or bool(self.cfg.eval_domain_randomization)
        )

    def _domain_randomization_feature_enabled(self, feature: str) -> bool:
        selected = normalize_domain_randomization_features(
            self.cfg.domain_randomization.enabled_features
        )
        if feature not in ("rigid_body", "current", "hydrodynamics", "actuators", "battery"):
            raise ValueError(f"Unknown domain-randomization feature {feature!r}.")
        return self._domain_randomization_enabled() and feature in selected

    def _disturbance_curriculum_global_step(self) -> int:
        return int(self.common_step_counter)

    def _get_disturbance_curriculum_stage(self) -> int:
        stages = self.cfg.domain_randomization.water_current_max_by_stage
        forced_stage = int(self.cfg.eval_disturbance_stage)
        if self.cfg.eval_mode and forced_stage >= 0:
            return min(forced_stage, len(stages) - 1)
        if not self.cfg.domain_randomization.disturbance_curriculum:
            return len(stages) - 1
        stage = sum(
            self._disturbance_curriculum_global_step() >= boundary
            for boundary in self.cfg.domain_randomization.disturbance_curriculum_stage_steps
        )
        return min(stage, len(stages) - 1)

    def _additional_hydrodynamics_scale(self) -> float:
        if not self._domain_randomization_enabled():
            return 1.0
        scales = self.cfg.domain_randomization.additional_hydrodynamics_scale_by_stage
        if not scales:
            return 1.0
        return float(scales[self._get_disturbance_curriculum_stage()])

    def _reset_domain(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Reset explicit robot/environment state, then synchronize PhysX once."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        stage = self._get_disturbance_curriculum_stage()
        payload_scale = reset_rigid_body(
            self.robot_runtime,
            self.cfg,
            ids,
            stage,
            enabled=self._domain_randomization_feature_enabled("rigid_body"),
        )
        reset_current(
            self.environment_runtime,
            self.cfg,
            ids,
            stage,
            enabled=self._domain_randomization_feature_enabled("current"),
        )
        reset_hydrodynamics(
            self.environment_runtime,
            self.cfg,
            ids,
            stage,
            enabled=self._domain_randomization_feature_enabled("hydrodynamics"),
        )
        reset_actuators(
            self.robot_runtime,
            self.cfg,
            ids,
            stage,
            enabled=self._domain_randomization_feature_enabled("actuators"),
        )
        reset_battery(
            self.robot_runtime,
            self.cfg,
            ids,
            stage,
            enabled=self._domain_randomization_feature_enabled("battery"),
        )
        self.robot_runtime.tether_slack_length[ids] = self.cfg.tether_slack_length
        if payload_scale is not None:
            self.environment_runtime.apply_payload_hydrodynamic_scale(
                ids,
                linear_damping=payload_scale.linear_damping,
                quadratic_damping=payload_scale.quadratic_damping,
                added_mass=payload_scale.added_mass,
            )
        self._apply_runtime_mass_properties(ids)
        self._apply_runtime_center_of_mass(ids)
        self._log_domain_randomization_state()

    def _log_domain_randomization_state(self) -> None:
        interval = max(1, int(self.cfg.domain_randomization_log_interval_steps))
        last_step = getattr(self, "_last_domain_randomization_log_step", None)
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_domain_randomization_log_step = self.common_step_counter
        environment = self.environment_runtime
        robot = self.robot_runtime
        log = self.extras.setdefault("log", {})
        log["enabled"] = float(self._domain_randomization_enabled())
        for feature in ("rigid_body", "current", "hydrodynamics", "actuators", "battery"):
            log[f"feature_{feature}_enabled"] = float(
                self._domain_randomization_feature_enabled(feature)
            )
        log["curriculum_stage"] = float(self._get_disturbance_curriculum_stage())
        log["curriculum_global_step"] = float(self._disturbance_curriculum_global_step())
        log["additional_hydrodynamics_scale"] = self._additional_hydrodynamics_scale()

        def add_stats(name: str, values: torch.Tensor) -> None:
            flat = values.detach().to(dtype=torch.float32).reshape(-1)
            if flat.numel():
                log[f"{name}_mean"] = flat.mean()
                log[f"{name}_std"] = flat.std(unbiased=False)
                log[f"{name}_min"] = flat.min()
                log[f"{name}_max"] = flat.max()

        add_stats("mass_kg", robot.masses)
        add_stats("volume_m3", robot.volumes)
        add_stats("center_of_mass_offset_m", torch.linalg.vector_norm(robot.center_of_mass_offsets, dim=1))
        add_stats("com_to_cob_offset_m", torch.linalg.vector_norm(robot.com_to_cob_offsets, dim=1))
        add_stats("principal_inertia_kg_m2", robot.inertia_principal_moments)
        add_stats("added_mass_randomization_scale", environment.added_mass_randomization_scale)
        add_stats("added_mass_coefficient", environment.added_mass)
        if robot.payload_sample_count > 0:
            add_stats("payload_sample_index", robot.payload_sample_indices)
        add_stats("water_current_mps", torch.linalg.vector_norm(environment.water_current_w, dim=1))
        add_stats("thruster_force_scale", robot.thruster_force_scale)
        add_stats("thruster_time_constant_s", robot.thruster_time_constant)
        add_stats("thruster_delay_steps", robot.thruster_delay_steps)
        add_stats("battery_voltage_v", robot.battery_voltage)

    def _apply_runtime_mass_properties(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Write environment-resolved mass and inertia into PhysX."""

        env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_cpu = env_ids_device.detach().cpu()

        physx_masses = self._robot.root_physx_view.get_masses().clone()
        selected_masses = self.robot_runtime.masses[env_ids_device].to(
            device=physx_masses.device,
            dtype=physx_masses.dtype,
        )
        if physx_masses.ndim == 1:
            physx_masses[env_ids_cpu] = selected_masses.reshape(-1)
        else:
            physx_masses[env_ids_cpu] = selected_masses.reshape(len(env_ids_cpu), -1)
        self._robot.root_physx_view.set_masses(physx_masses, env_ids_cpu)

        physx_inertias = self._robot.root_physx_view.get_inertias().clone()
        selected_moments = self.robot_runtime.inertia_principal_moments[env_ids_device].to(
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
        """Write the environment-resolved COM and principal axes into PhysX."""

        env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_cpu = env_ids_device.detach().cpu()
        physx_coms = self._robot.root_physx_view.get_coms().clone()
        com_positions = self.robot_runtime.center_of_mass_offsets[env_ids_device].to(
            device=physx_coms.device,
            dtype=physx_coms.dtype,
        )
        principal_axes = self.robot_runtime.inertia_principal_axes_xyzw[env_ids_device].to(
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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.robot_runtime.reset_dynamic_buffers(env_ids_device)
        self.environment_runtime.reset_acceleration(env_ids_device)
        self.environment_runtime.effective_state = None
        self._reset_mlp_history(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_applied_actions[env_ids] = 0.0

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        self._pool_center_w[env_ids, :] = self._default_root_state[env_ids, :3]

        # Sample t=0 of the command before constructing the initial rigid-body
        # state so position, attitude, and velocity can match the reference.
        self._reset_trajectory(env_ids)

        if not self.cfg.eval_mode and not self.cfg.trajectory_match_initial_state:
            # Randomize the initial position around the trajectory center.
            self._default_root_state[env_ids, :3] += self._sample_initial_position_offset(
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


def register_environment() -> None:
    """Register the assembled task and its training configuration with Gym."""

    import gymnasium as gym

    from simulation.training.ppo.config import AUVTrajPPORunnerCfg

    gym.register(
        id="Isaac-AUV-Traj-Direct-v1",
        entry_point=AUVTrajEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": AUVTrajEnvCfg,
            "rsl_rl_cfg_entry_point": AUVTrajPPORunnerCfg,
        },
    )


__all__ = ["AUVTrajEnv", "AUVTrajEnvCfg", "register_environment"]

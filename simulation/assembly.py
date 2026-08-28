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

from common.tensor_math import (
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quaternion_error_magnitude,
)

from environment.randomization import reset_current, reset_hydrodynamics
from environment.runtime import BodyKinematics, EnvironmentRuntimeState
from robot.randomization import reset_actuators
from robot.runtime_state import RobotRuntimeState
from simulation.composition import resolve_runtime_composition
from simulation.dynamics import calculate_total_inertia_physx_wrench
from simulation.domain_randomization import (
    DOMAIN_RANDOMIZATION_FEATURES,
    disturbance_stage_count,
    normalize_domain_randomization_features,
)
from simulation.training.observations import AUVObservationMixin
from simulation.training.evaluation.config import apply_evaluation_physics_overlay
from simulation.training.rewards import apply_tracking_reward_policy, get_tracking_reward_function
from simulation.training.config import AUVTrajEnvCfg
from robot.control.trajectory.guidance import root_state_at_tracking_target
from robot.control.trajectory import (
    AXIS_SINE,
    LATERAL_WAVE,
    VERTICAL_WAVE,
)
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
        self._tracking_reward_fn = get_tracking_reward_function(cfg.tracking_reward_profile)
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
        # The only supported ordering is deterministic profile -> DR recipe
        # -> evaluation overlay. This prevents a selected profile from silently
        # replacing CLI evaluation modifiers.
        apply_evaluation_physics_overlay(cfg)
        cfg.domain_randomization.enabled_features = list(
            normalize_domain_randomization_features(cfg.domain_randomization.enabled_features)
        )
        self._domain_randomization_features = frozenset(
            cfg.domain_randomization.enabled_features
        )
        self._configured_pool_center = tuple(
            0.5 * (float(cfg.pool_bounds[index]) + float(cfg.pool_bounds[index + 1]))
            for index in (0, 2, 4)
        )
        cfg.viewer.lookat = self._configured_pool_center
        # The Python training recipe selects a named MLP profile. Resolve its
        # history-expanded observation space before IsaacLab allocates its
        # vector-environment buffers.
        self._configure_mlp_observation_space(cfg)
        self._last_action_transport_log_step = None
        self._last_domain_randomization_log_step = None
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

    def _init_action_state(self, action_dim: int) -> None:
        """Allocate policy-action and applied-wrench lifecycle buffers."""

        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, action_dim, device=self.device
        )
        self._previous_processed_commands = torch.zeros_like(self._previous_actions)
        self._previous_previous_processed_commands = torch.zeros_like(
            self._previous_actions
        )
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
        # Commit the observation that produced this action. Observation reads
        # themselves stay pure, so wrapper/runner initialization cannot create
        # synthetic duplicate history frames.
        self._commit_mlp_history(self._latest_normalized_observation)
        stage = self._get_disturbance_curriculum_stage()
        self.environment_runtime.advance_smooth_current(
            stage=stage,
            enabled=self._domain_randomization_feature_enabled("current"),
            policy_dt=self.physics_dt * self.cfg.decimation,
        )
        self._previous_actions[:] = self._actions
        # Snapshot the processed command from the preceding policy step. The
        # separate first-order response owns the physically realized force.
        self._previous_previous_processed_commands[:] = self._previous_processed_commands
        self._previous_processed_commands[:] = (
            self.robot_runtime.thruster_command_processor.processed_commands
        )
        incoming_actions = actions.to(device=self.device, dtype=self._actions.dtype)
        self._raw_actions.copy_(incoming_actions)
        self._actions.copy_(incoming_actions.clamp(min=-1.0, max=1.0))

    def _log_action_transport_diagnostics(self) -> None:
        """Periodically expose PPO-command-to-thruster transport statistics."""

        interval = int(self.cfg.domain_randomization_log_interval_steps)
        last_step = self._last_action_transport_log_step
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_action_transport_log_step = self.common_step_counter

        robot = self.robot_runtime
        processed_commands = robot.thruster_command_processor.processed_commands
        policy_dt_s = self.physics_dt * self.cfg.decimation
        raw_clipped = self._raw_actions.abs() > 1.0
        transport_delta = self._actions - processed_commands
        log = self.extras.setdefault("log", {})
        log["raw_action_clip_fraction"] = raw_clipped.to(dtype=torch.float32).mean()
        log["raw_action_vector_clip_fraction"] = raw_clipped.any(dim=1).to(dtype=torch.float32).mean()
        log["requested_action_saturation_fraction"] = (
            self._actions.abs() > 0.95
        ).to(dtype=torch.float32).mean()
        log["processed_command_saturation_fraction"] = (
            processed_commands.abs() > 0.95
        ).to(dtype=torch.float32).mean()
        log["requested_to_processed_command_rms"] = torch.sqrt(torch.mean(transport_delta**2))
        log["requested_to_processed_command_fraction"] = (
            transport_delta.abs() > 1.0e-4
        ).to(dtype=torch.float32).mean()
        log["requested_action_rate_rms_per_s"] = torch.sqrt(
            torch.mean(
                ((self._actions - self._previous_actions) / policy_dt_s).square()
            )
        )
        log["processed_command_rate_rms_per_s"] = torch.sqrt(
            torch.mean(
                (
                    (processed_commands - self._previous_processed_commands)
                    / policy_dt_s
                ).square()
            )
        )
        processed_acceleration_per_s2 = (
            processed_commands
            - 2.0 * self._previous_processed_commands
            + self._previous_previous_processed_commands
        ) / (policy_dt_s * policy_dt_s)
        log["processed_command_acceleration_rms_per_s2"] = torch.sqrt(
            torch.mean(processed_acceleration_per_s2.square())
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

    def _apply_action(self) -> None:
        self.robot_runtime.pose_sensor.record(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )
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
            actions, physics_time_s=physics_time_s
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
        velocity_b = torch.cat(
            (kinematics.root_linear_velocity_b, kinematics.root_angular_velocity_b), dim=-1
        )
        current_acceleration_b = self.environment_runtime.update_current_acceleration(
            velocity_b,
            effective_hydrodynamics.relative_velocity_b,
            physics_dt=self.physics_dt,
        )
        fluid_forces, fluid_torques = self.environment_runtime.compose_fluid_wrench(
            kinematics,
            effective_hydrodynamics,
            volumes=self.robot_runtime.volumes,
            com_to_cob_offsets=self.robot_runtime.com_to_cob_offsets,
        )
        tether_forces, tether_torques = self.robot_runtime.compose_tether_wrench(
            kinematics,
            water_current_w=effective_hydrodynamics.water_current_w,
            gravity_w=self.environment_runtime.gravity_w,
            physics_dt=self.physics_dt,
            additional_scale=additional_scale,
        )
        external_wrench_b = torch.cat(
            (
                fluid_forces + thruster_forces + tether_forces,
                fluid_torques + thruster_torques + tether_torques,
            ),
            dim=-1,
        )
        gravity_force_w = self.robot_runtime.masses * self.environment_runtime.gravity_w.reshape(1, 3)
        gravity_force_b = quat_apply_wxyz(
            quat_conjugate_wxyz(kinematics.root_quat_w), gravity_force_w
        )
        physx_wrench_b, generalized_acceleration_b = calculate_total_inertia_physx_wrench(
            external_wrench_b,
            velocity_b,
            gravity_force_b,
            self.robot_runtime.masses,
            self.robot_runtime.inertia_body_matrices,
            effective_hydrodynamics.fluid_added_mass,
            current_acceleration_b,
        )
        self.environment_runtime.generalized_acceleration_b[:] = generalized_acceleration_b
        return physx_wrench_b[:, :3], physx_wrench_b[:, 3:]

    def _get_observations(self) -> dict:
        root_position_w, root_quat_w, root_linear_velocity_b, root_angular_velocity_b = (
            self._state_for_observation()
        )
        # Keep the target synchronized with the current episode time before
        # constructing the policy observation.
        self._update_tracking_targets()
        root_quat_conjugate = quat_conjugate_wxyz(root_quat_w)
        target_pos_error_b = quat_apply_wxyz(
            root_quat_conjugate,
            self._target_pos_w - root_position_w,
        )
        target_lin_vel_b = quat_apply_wxyz(
            root_quat_conjugate,
            self._target_lin_vel_w,
        )
        target_lin_acc_b = quat_apply_wxyz(
            root_quat_conjugate,
            self._target_lin_acc_w,
        )
        target_ang_vel_b = quat_apply_wxyz(
            root_quat_conjugate,
            self._target_ang_vel_w,
        )
        linear_velocity_error_b = target_lin_vel_b - root_linear_velocity_b
        attitude_error_quat = math_utils.quat_unique(
            quat_multiply_wxyz(root_quat_conjugate, self._target_quat_w)
        )
        gravity_direction_w = (
            self.environment_runtime.gravity_w
            / self.environment_runtime.gravity_magnitude
        ).reshape(1, 3).expand_as(root_position_w)
        projected_gravity_b = quat_apply_wxyz(
            root_quat_conjugate,
            gravity_direction_w,
        )
        raw_obs = torch.cat(
            [
                target_pos_error_b,
                target_lin_vel_b,
                linear_velocity_error_b,
                attitude_error_quat,
                projected_gravity_b,
                root_angular_velocity_b,
                target_ang_vel_b,
                target_lin_acc_b,
                # Feed back the command after dropout, quantization, and
                # saturation. The distinct realized-force state remains in
                # the simulator-only Critic observation.
                self.robot_runtime.thruster_command_processor.processed_commands,
            ],
            dim=-1,
        )
        normalized_obs = self._normalize_trajectory_observation(raw_obs)
        self._latest_normalized_observation.copy_(normalized_obs)
        actor_obs = self._stack_mlp_history(normalized_obs)
        # RSL-RL maps this simulator-only group to V(o, z_priv).  It is never
        # read by the exported Actor or by trajectory evaluation.
        return {"policy": actor_obs, "critic": self._build_critic_observation(actor_obs)}

    def _get_rewards(self) -> torch.Tensor:
        self._update_tracking_targets()
        self._log_action_transport_diagnostics()
        target_lin_vel_b = quat_apply_wxyz(
            quat_conjugate_wxyz(self._robot.data.root_quat_w),
            self._target_lin_vel_w,
        )
        target_ang_vel_b = quat_apply_wxyz(
            quat_conjugate_wxyz(self._robot.data.root_quat_w),
            self._target_ang_vel_w,
        )
        reward_commands = self.robot_runtime.thruster_command_processor.processed_commands
        reward_args = (
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_attitude_recovery,
            self.cfg.rew_scale_attitude_precision,
            self.cfg.rew_scale_track_vel,
            self.cfg.rew_scale_angular_velocity_broad,
            self.cfg.rew_scale_angular_velocity_precision,
            self.cfg.rew_scale_actions,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_action_rate_scale_per_s,
            self.physics_dt * self.cfg.decimation,
            self.cfg.rew_pos_sigma,
            self.cfg.rew_attitude_recovery_transition,
            self.cfg.rew_attitude_recovery_zero,
            self.cfg.rew_attitude_precision_sigma,
            self.cfg.rew_track_vel_sigma,
            self.cfg.rew_angular_velocity_broad_sigma,
            self.cfg.rew_angular_velocity_precision_sigma,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._target_pos_w,
            self._target_quat_w,
            target_lin_vel_b,
            target_ang_vel_b,
            reward_commands,
            self._previous_processed_commands,
        )
        terms = self._tracking_reward_fn(*reward_args)
        reward = terms[0]
        terminated_penalty = (
            self.cfg.rew_scale_terminated
            * self.reset_terminated.to(dtype=reward.dtype)
        )
        total_reward = reward - terminated_penalty

        position_error = torch.linalg.vector_norm(
            self._target_pos_w - self._robot.data.root_pos_w,
            dim=1,
        )
        velocity_error = torch.linalg.vector_norm(
            target_lin_vel_b - self._robot.data.root_lin_vel_b,
            dim=1,
        )
        angular_velocity_error = torch.linalg.vector_norm(
            target_ang_vel_b - self._robot.data.root_ang_vel_b,
            dim=1,
        )
        attitude_error = quaternion_error_magnitude(
            self._target_quat_w,
            self._robot.data.root_quat_w,
        )
        attitude_error_deg = torch.rad2deg(attitude_error)
        log = self.extras.setdefault("log", {})
        log["tracking/position_rmse_m"] = torch.sqrt(torch.mean(position_error.square()))
        log["tracking/position_error_mean_m"] = position_error.mean()
        log["tracking/velocity_rmse_mps"] = torch.sqrt(torch.mean(velocity_error.square()))
        log["tracking/angular_velocity_rmse_radps"] = torch.sqrt(
            torch.mean(angular_velocity_error.square())
        )
        log["tracking/attitude_error_mean_deg"] = attitude_error_deg.mean()
        log["trajectory/target_speed_mean_mps"] = torch.linalg.vector_norm(
            self._target_lin_vel_w, dim=1
        ).mean()
        log["trajectory/target_acceleration_mean_mps2"] = torch.linalg.vector_norm(
            self._target_lin_acc_w, dim=1
        ).mean()
        log["trajectory/target_curvature_mean_m_inv"] = self._traj_curvature_m_inv.mean()
        log["trajectory/target_curvature_max_m_inv"] = self._traj_curvature_m_inv.max()
        axis_sine = self._traj_type == AXIS_SINE
        trajectory_masks = {
            "surge_sine": axis_sine & (self._traj_axis == 0),
            "sway_sine": axis_sine & (self._traj_axis == 1),
            "heave_sine": axis_sine & (self._traj_axis == 2),
            "lateral_wave": self._traj_type == LATERAL_WAVE,
            "vertical_wave": self._traj_type == VERTICAL_WAVE,
        }
        for trajectory_name, trajectory_mask_bool in trajectory_masks.items():
            trajectory_mask = trajectory_mask_bool.to(attitude_error.dtype)
            log[f"trajectory/{trajectory_name}_fraction"] = trajectory_mask.mean()
            log[f"tracking/{trajectory_name}_attitude_error_mean_deg"] = (
                torch.sum(attitude_error_deg * trajectory_mask)
                / trajectory_mask.sum().clamp_min(1.0)
            )
        log["curriculum/trajectory_stage"] = float(self._get_trajectory_curriculum_stage())
        log["curriculum/requested_speed_mean_mps"] = self._traj_target_speed_mps.mean()
        log["curriculum/requested_speed_max_mps"] = self._traj_target_speed_mps.max()
        log["curriculum/wave_count_mean"] = self._traj_wave_count.float().mean()
        log["curriculum/amplitude_scale_x_mean"] = self._traj_amplitude_scales[:, 0].mean()
        log["curriculum/amplitude_scale_y_mean"] = self._traj_amplitude_scales[:, 1].mean()
        log["curriculum/amplitude_scale_z_mean"] = self._traj_amplitude_scales[:, 2].mean()
        log["reward/position"] = terms[1].mean()
        attitude_recovery_rewards = terms[2]
        attitude_precision_rewards = terms[3]
        attitude_rewards = attitude_recovery_rewards + attitude_precision_rewards
        log["reward/attitude_recovery"] = attitude_recovery_rewards.sum(dim=1).mean()
        log["reward/attitude_precision"] = attitude_precision_rewards.sum(dim=1).mean()
        log["reward/attitude"] = attitude_rewards.sum(dim=1).mean()
        log["reward/roll"] = attitude_rewards[:, 0].mean()
        log["reward/pitch"] = attitude_rewards[:, 1].mean()
        log["reward/yaw"] = attitude_rewards[:, 2].mean()
        log["reward/velocity"] = terms[4].mean()
        log["reward/angular_velocity_broad"] = terms[5].mean()
        log["reward/angular_velocity_precision"] = terms[6].mean()
        log["reward/angular_velocity"] = (terms[5] + terms[6]).mean()
        log["reward/action_penalty"] = terms[7].mean()
        log["reward/action_rate_penalty"] = terms[8].mean()
        log["reward/running_total"] = reward.mean()
        log["reward/termination_penalty"] = terminated_penalty.mean()
        log["reward/total"] = total_reward.mean()
        log["reward/running_upper_bound"] = self._tracking_reward_policy.maximum_positive_reward
        log["reward/running_lower_bound"] = self._tracking_reward_policy.minimum_running_reward
        # DirectRLEnv computes dones before rewards. Penalize safety/boundary
        # terminations while leaving ordinary fixed-horizon timeouts neutral.
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.episode_length_before_reset is not None:
            time_out = time_out | (self.episode_length_buf >= int(self.cfg.episode_length_before_reset))

        if self.cfg.use_boundaries:
            local_position = self._robot.data.root_pos_w - self.scene.env_origins
            body_half_extents_w = self.environment_runtime.body_half_extents_world(
                self._robot.data.root_quat_w
            )
            lower_corner = local_position - body_half_extents_w
            upper_corner = local_position + body_half_extents_w
            out_of_bounds = (
                (lower_corner[:, 0] < self.environment_runtime.pool_bounds[0])
                | (upper_corner[:, 0] > self.environment_runtime.pool_bounds[1])
                | (lower_corner[:, 1] < self.environment_runtime.pool_bounds[2])
                | (upper_corner[:, 1] > self.environment_runtime.pool_bounds[3])
                | (lower_corner[:, 2] < self.environment_runtime.pool_bounds[4])
                | (upper_corner[:, 2] > self.environment_runtime.pool_bounds[5])
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

    def _sample_initial_attitude_error(self, count: int) -> torch.Tensor:
        """Sample an isotropic target-relative rotation inside the configured ball."""

        rotation_vector = self._sample_initial_position_offset(
            count,
            self.cfg.trajectory_initial_attitude_error_max_rad,
        )
        angle = torch.linalg.vector_norm(rotation_vector, dim=1, keepdim=True)
        axis = rotation_vector / angle.clamp_min(1.0e-8)
        half_angle = 0.5 * angle
        return torch.cat((torch.cos(half_angle), axis * torch.sin(half_angle)), dim=1)

    def _domain_randomization_enabled(self) -> bool:
        return bool(self.cfg.domain_randomization.use_custom_randomization) and (
            not self.cfg.eval_mode or bool(self.cfg.eval_domain_randomization)
        )

    def _domain_randomization_feature_enabled(self, feature: str) -> bool:
        return self._domain_randomization_enabled() and feature in self._domain_randomization_features

    def _disturbance_curriculum_global_step(self) -> int:
        return int(self.common_step_counter)

    def _get_disturbance_curriculum_stage(self) -> int:
        stage_count = disturbance_stage_count(self.cfg.domain_randomization)
        forced_stage = int(self.cfg.eval_disturbance_stage)
        if self.cfg.eval_mode and forced_stage >= 0:
            return forced_stage
        if not self.cfg.domain_randomization.disturbance_curriculum:
            return stage_count - 1
        return sum(
            self._disturbance_curriculum_global_step() >= boundary
            for boundary in self.cfg.domain_randomization.disturbance_curriculum_stage_steps
        )

    def _additional_hydrodynamics_scale(self) -> float:
        if not self._domain_randomization_enabled():
            return 1.0
        scales = self.cfg.domain_randomization.additional_hydrodynamics_scale_by_stage
        if scales is None:
            return 1.0
        return float(scales[self._get_disturbance_curriculum_stage()])

    def _reset_domain(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Reset explicit robot/environment state, then synchronize PhysX once."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        stage = self._get_disturbance_curriculum_stage()
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
        self.robot_runtime.tether_slack_length[ids] = self.cfg.tether_slack_length
        self._apply_runtime_mass_properties(ids)
        self._apply_runtime_center_of_mass(ids)
        self._log_domain_randomization_state()

    def _log_domain_randomization_state(self) -> None:
        interval = int(self.cfg.domain_randomization_log_interval_steps)
        last_step = self._last_domain_randomization_log_step
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_domain_randomization_log_step = self.common_step_counter
        environment = self.environment_runtime
        robot = self.robot_runtime
        log = self.extras.setdefault("log", {})
        prefix = "domain_randomization/"
        log[f"{prefix}enabled"] = float(self._domain_randomization_enabled())
        for feature in DOMAIN_RANDOMIZATION_FEATURES:
            log[f"{prefix}feature_{feature}_enabled"] = float(
                self._domain_randomization_feature_enabled(feature)
            )
        log[f"{prefix}curriculum_stage"] = float(
            self._get_disturbance_curriculum_stage()
        )
        log[f"{prefix}curriculum_global_step"] = float(
            self._disturbance_curriculum_global_step()
        )
        log[f"{prefix}additional_hydrodynamics_scale"] = (
            self._additional_hydrodynamics_scale()
        )

        def add_stats(name: str, values: torch.Tensor) -> None:
            flat = values.detach().to(dtype=torch.float32).reshape(-1)
            if flat.numel():
                log[f"{prefix}{name}_mean"] = flat.mean()
                log[f"{prefix}{name}_std"] = flat.std(unbiased=False)
                log[f"{prefix}{name}_min"] = flat.min()
                log[f"{prefix}{name}_max"] = flat.max()

        add_stats(
            "linear_damping_randomization_scale",
            environment.linear_damping_randomization_scale,
        )
        add_stats(
            "quadratic_damping_randomization_scale",
            environment.quadratic_damping_randomization_scale,
        )
        add_stats("linear_damping_coefficient", environment.linear_damping)
        add_stats("quadratic_damping_coefficient", environment.quadratic_damping)
        add_stats(
            "fluid_added_mass_randomization_scale",
            environment.fluid_added_mass_randomization_scale,
        )
        add_stats("fluid_added_mass_coefficient", environment.fluid_added_mass)
        add_stats("water_current_mps", torch.linalg.vector_norm(environment.water_current_w, dim=1))
        add_stats("thruster_force_scale", robot.thruster_force_scale)
        add_stats("common_thruster_force_scale", robot.common_thruster_force_scale)
        add_stats("thruster_time_constant_s", robot.thruster_time_constant)

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
        principal_axes_matrix = self.robot_runtime.inertia_principal_axes[env_ids_device].to(
            device=physx_coms.device,
            dtype=physx_coms.dtype,
        )
        principal_axes_wxyz = math_utils.quat_from_matrix(principal_axes_matrix)
        principal_axes_xyzw = principal_axes_wxyz[:, [1, 2, 3, 0]]
        if physx_coms.ndim == 3:
            physx_coms[env_ids_cpu, 0, :3] = com_positions
            physx_coms[env_ids_cpu, 0, 3:7] = principal_axes_xyzw
        else:
            physx_coms[env_ids_cpu, :3] = com_positions
            physx_coms[env_ids_cpu, 3:7] = principal_axes_xyzw
        self._robot.root_physx_view.set_coms(physx_coms, env_ids_cpu)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.robot_runtime.reset_dynamic_buffers(env_ids_device)
        self.environment_runtime.reset_kinematic_history(env_ids_device)
        self.environment_runtime.effective_state = None
        self._reset_mlp_history(env_ids_device)
        self._raw_actions[env_ids_device] = 0.0
        self._actions[env_ids_device] = 0.0
        self._previous_actions[env_ids_device] = 0.0
        self._previous_processed_commands[env_ids_device] = 0.0
        self._previous_previous_processed_commands[env_ids_device] = 0.0

        self._default_root_state[env_ids_device, :] = self._robot.data.default_root_state[
            env_ids_device
        ]
        self._default_root_state[env_ids_device, :3] += self.scene.env_origins[
            env_ids_device
        ]

        self._pool_center_w[env_ids_device, :] = self._default_root_state[
            env_ids_device, :3
        ]

        # Sample t=0 of the command before constructing the initial rigid-body
        # state relative to the target rather than to the pool centre.
        self._reset_trajectory(env_ids_device)

        # Apply domain randomization
        self._reset_domain(env_ids_device)

        if not self.cfg.eval_mode:
            count = int(env_ids_device.numel())
            self._default_root_state[env_ids_device, :] = root_state_at_tracking_target(
                self._default_root_state[env_ids_device, :],
                self._target_pos_w[env_ids_device, :],
                self._target_quat_w[env_ids_device, :],
                self._target_lin_vel_w[env_ids_device, :],
                self._target_ang_vel_w[env_ids_device, :],
            )
            self._default_root_state[
                env_ids_device, :3
            ] += self._sample_initial_position_offset(
                count,
                self.cfg.trajectory_initial_position_radius,
            )
            attitude_error = self._sample_initial_attitude_error(count)
            self._default_root_state[env_ids_device, 3:7] = math_utils.quat_unique(
                quat_multiply_wxyz(
                    self._target_quat_w[env_ids_device],
                    attitude_error,
                )
            )
            self._default_root_state[env_ids_device, 7:10] += (
                self._sample_initial_position_offset(
                    count,
                    self.cfg.trajectory_initial_linear_velocity_error_max_mps,
                )
            )
            self._default_root_state[env_ids_device, 10:13] += (
                self._sample_initial_position_offset(
                    count,
                    self.cfg.trajectory_initial_angular_velocity_error_max_radps,
                )
            )
        elif self.cfg.trajectory_eval_align_initial_target:
            self._default_root_state[env_ids_device, :] = root_state_at_tracking_target(
                self._default_root_state[env_ids_device, :],
                self._target_pos_w[env_ids_device, :],
                self._target_quat_w[env_ids_device, :],
                self._target_lin_vel_w[env_ids_device, :],
                self._target_ang_vel_w[env_ids_device, :],
            )

        self._robot.write_root_pose_to_sim(
            self._default_root_state[env_ids_device, :7], env_ids_device
        )
        self._robot.write_root_velocity_to_sim(
            self._default_root_state[env_ids_device, 7:], env_ids_device
        )
        initial_quaternion = self._default_root_state[env_ids_device, 3:7]
        initial_quaternion_conjugate = quat_conjugate_wxyz(initial_quaternion)
        self.robot_runtime.pose_sensor.reset(
            env_ids_device,
            self._default_root_state[env_ids_device, :3],
            initial_quaternion,
            quat_apply_wxyz(
                initial_quaternion_conjugate,
                self._default_root_state[env_ids_device, 7:10],
            ),
            quat_apply_wxyz(
                initial_quaternion_conjugate,
                self._default_root_state[env_ids_device, 10:13],
            ),
        )


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

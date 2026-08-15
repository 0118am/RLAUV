"""AUV IsaacLab environment facade.

Only lifecycle and policy-facing orchestration live here. Focused mixins own
observations, task curricula, the Isaac-to-wrench bridge, and debug rendering.
Physical equations and vehicle data come from the top-level ``environment``
and ``robot`` domains.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from environment.profiles.features import normalize_domain_randomization_features
from robot.propulsion.curves import reduce_point_forces_to_wrench

from .physics_adapter import AUVDynamicsMixin
from .config import AUVTrajEnvCfg
from .config_overlays import apply_evaluation_physics_overlay
from .composition import resolve_isaac_composition
from .domain_randomization import AUVDomainRandomizationMixin
from .observations import AUVObservationMixin
from simulation.isaac.rewards import apply_tracking_reward_policy, get_tracking_reward_function
from robot.control.trajectory.guidance import root_state_at_tracking_target
from robot.control.trajectory import LATERAL_SINE, SPATIAL_HELIX, VERTICAL_SINE
from simulation.isaac.trajectory.mixin import AUVTrajectoryMixin
from .visualization import AUVVisualizationMixin




class AUVTrajEnv(
    AUVObservationMixin,
    AUVTrajectoryMixin,
    AUVDomainRandomizationMixin,
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
        if cfg.domain_randomization_feature_override_enabled:
            feature_override = list(cfg.domain_randomization.enabled_features)
            normalize_domain_randomization_features(feature_override)
        if not cfg.environment_profile:
            raise ValueError("environment_profile is required; implicit physics fallbacks are disabled.")
        composition = resolve_isaac_composition(
            cfg.environment_profile,
            cfg.domain_randomization_spec,
        )
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
        # The Python training recipe selects a named MLP profile. Resolve its
        # history-expanded observation space before IsaacLab allocates its
        # vector-environment buffers.
        self._configure_mlp_observation_space(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        action_dim = int(self.single_action_space.shape[0])
        self._init_action_state(action_dim)
        self._init_trajectory_state()
        self._init_vehicle_state(action_dim)
        self._init_observation_state()

        self.set_debug_vis(self.cfg.debug_vis)
        if self._debug:
            print("mass: ", list(self._robot.root_physx_view._masses))
        
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
        # Episode time and sampled battery parameters are constant throughout
        # the decimation substeps, so update voltage once per policy action.
        self._update_battery_voltage_scale()
        self._previous_actions[:] = self._actions
        # Snapshot the command actually realized at the preceding policy step.
        # ``_compute_dynamics`` advances this state once per physics substep.
        self._previous_applied_actions[:] = self.thruster_command_processor.rate_limited_state
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
                self.thruster_command_processor.rate_limited_state,
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
        self.thruster_response.reset(env_ids)
        self.thruster_command_processor.reset(env_ids)
        self._reset_mlp_history(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_applied_actions[env_ids] = 0.0
        self.realized_thruster_force_n[env_ids] = 0.0
        self.realized_thruster_forces_b[env_ids] = 0.0

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

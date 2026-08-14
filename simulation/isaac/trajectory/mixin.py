"""Trajectory commands, curricula, and domain-randomization mixin.

These methods decide task distribution and curriculum state. They deliberately
do not duplicate reusable physics equations from sibling packages.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
import isaaclab.utils.math as math_utils

from environment.randomization import reset_current, reset_hydrodynamics
from robot.randomization import reset_actuators, reset_battery
from robot.randomization.rigid_body import (
    apply_payload_hydrodynamics,
    initialize_payload_domain,
    reset_rigid_body,
)
from environment.profiles.features import domain_randomization_feature_enabled
from robot.control.trajectory.guidance import (
    quaternion_align_body_x_with_velocity,
    quaternion_step_angular_velocity_body,
)
from robot.control.trajectory.kinematics import (
    LATERAL_SINE,
    RetimedTrajectoryTables,
    SPATIAL_HELIX,
    TrajectoryKinematicLimits,
    VERTICAL_SINE,
    build_retimed_tables,
    evaluate_retimed_reference,
    sample_retimed_phase,
    smooth_startup_time,
)


class AUVTrajectoryMixin:
    """Owns reset-time task sampling and trajectory/disturbance curricula."""

    def _trajectory_kinematic_limits(self) -> TrajectoryKinematicLimits:
        """Materialize the versioned simulator envelope from runtime config."""

        return TrajectoryKinematicLimits(
            max_speed_mps=float(self.cfg.trajectory_max_speed_mps),
            max_acceleration_mps2=float(self.cfg.trajectory_max_acceleration_mps2),
            max_orientation_rate_radps=float(self.cfg.trajectory_max_orientation_rate_radps),
            max_jerk_mps3=float(self.cfg.trajectory_max_jerk_mps3),
            retime_samples=int(self.cfg.trajectory_retime_samples),
        )

    def _get_trajectory_curriculum_stage(self) -> int:
        """Return the active trajectory curriculum stage from global policy steps."""

        if not self.cfg.trajectory_curriculum:
            return -1

        forced_stage = int(getattr(self.cfg, "curriculum_gate_stage", -1))
        if forced_stage >= 0:
            return min(forced_stage, len(self.cfg.trajectory_curriculum_amp_scales) - 1)

        stage = 0
        for step_boundary in self.cfg.trajectory_curriculum_stage_steps:
            if self.common_step_counter >= step_boundary:
                stage += 1

        return min(stage, len(self.cfg.trajectory_curriculum_amp_scales) - 1)

    def _get_trajectory_training_profile(self):
        """Return trajectory type/range settings for the current curriculum stage."""

        if not self.cfg.trajectory_curriculum:
            return (
                self.cfg.trajectory_train_types,
                self.cfg.trajectory_amp_x_range,
                self.cfg.trajectory_amp_y_range,
                self.cfg.trajectory_amp_z_range,
                self.cfg.trajectory_period_range,
            )

        stage = self._get_trajectory_curriculum_stage()
        stage_types = (
            self.cfg.trajectory_curriculum_stage_0_types,
            self.cfg.trajectory_curriculum_stage_1_types,
            self.cfg.trajectory_curriculum_stage_2_types,
            self.cfg.trajectory_curriculum_stage_3_types,
        )[stage]
        amp_scale = self.cfg.trajectory_curriculum_amp_scales[stage]
        z_amp_scale = self.cfg.trajectory_curriculum_z_amp_scales[stage]
        amp_x_range = [self.cfg.trajectory_amp_x_range[0] * amp_scale, self.cfg.trajectory_amp_x_range[1] * amp_scale]
        amp_y_range = [self.cfg.trajectory_amp_y_range[0] * amp_scale, self.cfg.trajectory_amp_y_range[1] * amp_scale]
        amp_z_range = [
            self.cfg.trajectory_amp_z_range[0] * z_amp_scale,
            self.cfg.trajectory_amp_z_range[1] * z_amp_scale,
        ]
        period_range = [
            self.cfg.trajectory_curriculum_period_min[stage],
            self.cfg.trajectory_curriculum_period_max[stage],
        ]

        return stage_types, amp_x_range, amp_y_range, amp_z_range, period_range

    def _reset_trajectory(self, env_ids: Sequence[int]):
        """Sample trajectory parameters for reset environments."""

        num_env_ids = len(env_ids)
        zeros = torch.zeros(num_env_ids, device=self.device)

        # The trajectory center is the environment origin at the nominal
        # starting depth, so the command stays local to each cloned env.
        self._traj_center_w[env_ids, :] = self._default_env_origins[env_ids, :]
        self._target_quat_w[env_ids, :] = math_utils.quat_from_euler_xyz(zeros, zeros, zeros)
        self._target_lin_acc_w[env_ids, :] = 0.0
        self._target_lin_jerk_w[env_ids, :] = 0.0
        self._target_ang_vel_w[env_ids, :] = 0.0
        self._target_derivative_step[env_ids] = -1
        self._traj_target_speed_mps[env_ids] = 0.0

        if self.cfg.trajectory_eval_mode:
            # Fixed eval trajectories use deterministic parameters so repeated
            # evaluations are comparable across checkpoints.
            self._traj_type[env_ids] = self.cfg.trajectory_eval_type
            self._traj_axis[env_ids] = 0
            if self.cfg.trajectory_eval_type == 7:
                amp_x_lower, amp_x_upper = self.cfg.trajectory_amp_x_range
                amp_y_lower, amp_y_upper = self.cfg.trajectory_amp_y_range
                amp_z_lower, amp_z_upper = self.cfg.trajectory_amp_z_range
                period_lower, period_upper = self.cfg.trajectory_period_range
                self._traj_amp_x[env_ids] = math_utils.sample_uniform(
                    amp_x_lower, amp_x_upper, (num_env_ids,), device=self.device
                )
                self._traj_amp_y[env_ids] = math_utils.sample_uniform(
                    amp_y_lower, amp_y_upper, (num_env_ids,), device=self.device
                )
                self._traj_amp_z[env_ids] = math_utils.sample_uniform(
                    amp_z_lower, amp_z_upper, (num_env_ids,), device=self.device
                )
                self._traj_period[env_ids] = math_utils.sample_uniform(
                    period_lower, period_upper, (num_env_ids,), device=self.device
                )
                self._traj_phase_x[env_ids] = math_utils.sample_uniform(
                    0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
                )
                self._traj_phase_y[env_ids] = math_utils.sample_uniform(
                    0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
                )
            else:
                self._traj_amp_x[env_ids] = self.cfg.trajectory_eval_amp_x
                self._traj_amp_y[env_ids] = self.cfg.trajectory_eval_amp_y
                self._traj_amp_z[env_ids] = self.cfg.trajectory_eval_amp_z
                self._traj_period[env_ids] = self.cfg.trajectory_eval_period
                self._traj_phase_x[env_ids] = 0.0
                self._traj_phase_y[env_ids] = 0.0
            if self.cfg.trajectory_eval_type in (LATERAL_SINE, VERTICAL_SINE, SPATIAL_HELIX):
                self._traj_target_speed_mps[env_ids] = float(self.cfg.trajectory_eval_speed_mps)
        else:
            # Random smooth trajectories form the RL training command
            # distribution.  The shapes share one compact parameterization so
            # the observation interface remains identical across all samples.
            train_types, amp_x_range, amp_y_range, amp_z_range, period_range = self._get_trajectory_training_profile()
            amp_x_lower, amp_x_upper = amp_x_range
            amp_y_lower, amp_y_upper = amp_y_range
            amp_z_lower, amp_z_upper = amp_z_range
            period_lower, period_upper = period_range
            train_types = torch.as_tensor(train_types, device=self.device, dtype=torch.long)
            speed_levels = torch.as_tensor(
                self.cfg.trajectory_speed_levels_mps, device=self.device, dtype=torch.float32
            )
            if speed_levels.ndim != 1 or speed_levels.numel() == 0 or bool(torch.any(speed_levels <= 0.0)):
                raise ValueError("trajectory_speed_levels_mps must be a non-empty list of positive speeds.")
            if bool(torch.any(speed_levels > float(self.cfg.trajectory_max_speed_mps))):
                raise ValueError("trajectory_speed_levels_mps exceeds trajectory_max_speed_mps.")
            controlled_type = (
                (train_types == LATERAL_SINE)
                | (train_types == VERTICAL_SINE)
                | (train_types == SPATIAL_HELIX)
            )
            if bool(torch.all(controlled_type)):
                # Draw from a shuffled Cartesian pool. A full 12-environment
                # reset therefore contains every 3-shape x 4-speed pair once;
                # smaller/asynchronous reset batches remain uniformly random.
                combination_count = train_types.numel() * speed_levels.numel()
                repeats = (num_env_ids + combination_count - 1) // combination_count
                combinations = torch.arange(combination_count, device=self.device).repeat(repeats)
                combinations = combinations[torch.randperm(combinations.numel(), device=self.device)[:num_env_ids]]
                train_type_indices = torch.div(combinations, speed_levels.numel(), rounding_mode="floor")
                speed_indices = torch.remainder(combinations, speed_levels.numel())
            else:
                train_type_indices = torch.randint(0, len(train_types), (num_env_ids,), device=self.device)
                speed_indices = torch.randint(0, speed_levels.numel(), (num_env_ids,), device=self.device)
            self._traj_type[env_ids] = train_types[train_type_indices]
            self._traj_axis[env_ids] = torch.randint(0, 3, (num_env_ids,), device=self.device)
            self._traj_amp_x[env_ids] = math_utils.sample_uniform(
                amp_x_lower, amp_x_upper, (num_env_ids,), device=self.device
            )
            self._traj_amp_y[env_ids] = math_utils.sample_uniform(
                amp_y_lower, amp_y_upper, (num_env_ids,), device=self.device
            )
            self._traj_amp_z[env_ids] = math_utils.sample_uniform(
                amp_z_lower, amp_z_upper, (num_env_ids,), device=self.device
            )
            self._traj_period[env_ids] = math_utils.sample_uniform(
                period_lower, period_upper, (num_env_ids,), device=self.device
            )
            self._traj_phase_x[env_ids] = math_utils.sample_uniform(
                0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
            )
            self._traj_phase_y[env_ids] = math_utils.sample_uniform(
                0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
            )

            speed_controlled = (
                (self._traj_type[env_ids] == LATERAL_SINE)
                | (self._traj_type[env_ids] == VERTICAL_SINE)
                | (self._traj_type[env_ids] == SPATIAL_HELIX)
            )
            self._traj_target_speed_mps[env_ids] = torch.where(
                speed_controlled,
                speed_levels[speed_indices],
                self._traj_target_speed_mps[env_ids],
            )

        trajectory_types = self._traj_type[env_ids]
        lateral = trajectory_types == LATERAL_SINE
        vertical = trajectory_types == VERTICAL_SINE
        helix = trajectory_types == SPATIAL_HELIX
        self._traj_amp_y[env_ids] = torch.where(
            lateral,
            torch.full_like(self._traj_amp_y[env_ids], float(self.cfg.trajectory_lateral_sine_amplitude_m)),
            self._traj_amp_y[env_ids],
        )
        self._traj_amp_z[env_ids] = torch.where(
            vertical,
            torch.full_like(self._traj_amp_z[env_ids], float(self.cfg.trajectory_vertical_sine_amplitude_m)),
            self._traj_amp_z[env_ids],
        )
        self._traj_amp_x[env_ids] = torch.where(
            helix,
            torch.full_like(self._traj_amp_x[env_ids], float(self.cfg.trajectory_spatial_helix_radius_x_m)),
            self._traj_amp_x[env_ids],
        )
        self._traj_amp_y[env_ids] = torch.where(
            helix,
            torch.full_like(self._traj_amp_y[env_ids], float(self.cfg.trajectory_spatial_helix_radius_y_m)),
            self._traj_amp_y[env_ids],
        )
        self._traj_amp_z[env_ids] = torch.where(
            helix,
            torch.full_like(self._traj_amp_z[env_ids], float(self.cfg.trajectory_spatial_helix_amplitude_z_m)),
            self._traj_amp_z[env_ids],
        )

        if self.cfg.trajectory_eval_mode and self.cfg.trajectory_eval_type == 7:
            sampled_amplitudes = torch.stack(
                (self._traj_amp_x[env_ids], self._traj_amp_y[env_ids], self._traj_amp_z[env_ids]), dim=-1
            )
            if bool(torch.any(sampled_amplitudes <= 0.0)):
                raise ValueError(
                    "random_smooth evaluation requires positive x/y/z amplitude ranges; "
                    "a static reference is not a trajectory evaluation."
                )

        tables = build_retimed_tables(
            self._traj_type[env_ids],
            self._traj_axis[env_ids],
            self._traj_amp_x[env_ids],
            self._traj_amp_y[env_ids],
            self._traj_amp_z[env_ids],
            self._traj_period[env_ids],
            self._traj_phase_x[env_ids],
            self._traj_phase_y[env_ids],
            self._traj_target_speed_mps[env_ids],
            radius_min=float(self.cfg.trajectory_eval_radius_min),
            radius_max=float(self.cfg.trajectory_eval_radius_max),
            chirp_rate=float(self.cfg.trajectory_eval_chirp_rate),
            harmonic_ratio=float(self.cfg.trajectory_random_smooth_harmonic_ratio),
            limits=self._trajectory_kinematic_limits(),
        )
        self._traj_retime_phase[env_ids] = tables.phase
        self._traj_retime_elapsed_s[env_ids] = tables.elapsed_s
        self._traj_retime_phase_rate[env_ids] = tables.phase_rate
        self._traj_retime_phase_acceleration[env_ids] = tables.phase_acceleration
        self._traj_period[env_ids] = tables.requested_period_s
        self._traj_effective_period_s[env_ids] = tables.effective_period_s
        self._traj_retimed[env_ids] = tables.retimed

        self._update_tracking_targets(env_ids)
        # At t=0 the startup ramp deliberately makes velocity zero, so its
        # direction is undefined. Use the unramped path tangent once at reset
        # to point body +X along the direction in which motion will begin.
        if self.cfg.trajectory_align_heading_with_velocity:
            initial_time = torch.zeros(num_env_ids, dtype=torch.float32, device=self.device)
            initial_phase, initial_phase_rate, initial_phase_acceleration = sample_retimed_phase(
                tables,
                initial_time,
            )
            _, initial_tangent_velocity_w, _, _ = evaluate_retimed_reference(
                self._traj_type[env_ids],
                self._traj_axis[env_ids],
                self._traj_amp_x[env_ids],
                self._traj_amp_y[env_ids],
                self._traj_amp_z[env_ids],
                self._traj_phase_x[env_ids],
                self._traj_phase_y[env_ids],
                initial_phase,
                initial_phase_rate,
                initial_phase_acceleration,
                radius_min=float(self.cfg.trajectory_eval_radius_min),
                radius_max=float(self.cfg.trajectory_eval_radius_max),
                harmonic_ratio=float(self.cfg.trajectory_random_smooth_harmonic_ratio),
            )
            self._target_quat_w[env_ids, :] = quaternion_align_body_x_with_velocity(
                initial_tangent_velocity_w,
                self._target_quat_w[env_ids, :],
                self.cfg.trajectory_heading_min_speed,
            )
            self._previous_target_quat_w[env_ids, :] = self._target_quat_w[env_ids, :]

    def _update_tracking_targets(self, env_ids: Sequence[int] | None = None):
        """Update target pose/velocity from the stored trajectory parameters."""

        full_update = env_ids is None
        if full_update:
            current_generation = int(self.common_step_counter)
            if self._tracking_target_common_step == current_generation:
                return
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # Phase tables are built once at reset and sampled by policy-step time.
        # The same re-parameterized reference supplies pose, velocity and
        # acceleration, avoiding the former per-callback velocity difference.
        t = self.episode_length_buf[env_ids].to(dtype=torch.float32) * self.cfg.sim.dt * self.cfg.decimation
        tables = RetimedTrajectoryTables(
            phase=self._traj_retime_phase[env_ids],
            elapsed_s=self._traj_retime_elapsed_s[env_ids],
            phase_rate=self._traj_retime_phase_rate[env_ids],
            phase_acceleration=self._traj_retime_phase_acceleration[env_ids],
            requested_period_s=self._traj_period[env_ids],
            effective_period_s=self._traj_effective_period_s[env_ids],
            retimed=self._traj_retimed[env_ids],
        )
        trajectory_time, startup_speed_scale, startup_speed_scale_rate = smooth_startup_time(
            t,
            float(self.cfg.trajectory_startup_duration_s),
        )
        phase, nominal_phase_rate, nominal_phase_acceleration = sample_retimed_phase(tables, trajectory_time)
        # Chain rule for q(trajectory_time(t)). This keeps commanded position,
        # velocity, and acceleration mutually consistent throughout startup.
        phase_rate = nominal_phase_rate * startup_speed_scale
        phase_acceleration = (
            nominal_phase_acceleration * startup_speed_scale.square()
            + nominal_phase_rate * startup_speed_scale_rate
        )
        offsets_w, target_lin_vel_w, target_lin_acc_w, curvature = evaluate_retimed_reference(
            self._traj_type[env_ids],
            self._traj_axis[env_ids],
            self._traj_amp_x[env_ids],
            self._traj_amp_y[env_ids],
            self._traj_amp_z[env_ids],
            self._traj_phase_x[env_ids],
            self._traj_phase_y[env_ids],
            phase,
            phase_rate,
            phase_acceleration,
            radius_min=float(self.cfg.trajectory_eval_radius_min),
            radius_max=float(self.cfg.trajectory_eval_radius_max),
            harmonic_ratio=float(self.cfg.trajectory_random_smooth_harmonic_ratio),
        )
        previous_target_lin_acc_w = self._target_lin_acc_w[env_ids, :].clone()
        self._target_pos_w[env_ids, :] = self._traj_center_w[env_ids, :] + offsets_w
        self._target_lin_vel_w[env_ids, :] = target_lin_vel_w
        self._target_lin_acc_w[env_ids, :] = target_lin_acc_w
        self._traj_curvature_m_inv[env_ids] = curvature
        if self.cfg.trajectory_align_heading_with_velocity:
            heading_velocity_w = target_lin_vel_w.clone()
            heading_velocity_w[self._traj_type[env_ids] == 2] = 0.0
            self._target_quat_w[env_ids, :] = quaternion_align_body_x_with_velocity(
                heading_velocity_w,
                self._target_quat_w[env_ids, :],
                self.cfg.trajectory_heading_min_speed,
            )

        current_step = self.episode_length_buf[env_ids].to(dtype=torch.long)
        previous_step = self._target_derivative_step[env_ids]
        has_previous = previous_step >= 0
        elapsed_steps = torch.clamp(current_step - previous_step, min=1).to(dtype=torch.float32)
        dt_s = elapsed_steps * self.cfg.sim.dt * self.cfg.decimation
        linear_jerk = (target_lin_acc_w - previous_target_lin_acc_w) / dt_s.unsqueeze(-1)
        target_ang_vel_previous_b = quaternion_step_angular_velocity_body(
            self._previous_target_quat_w[env_ids, :],
            self._target_quat_w[env_ids, :],
            dt_s,
        )
        angular_velocity = math_utils.quat_apply(
            self._previous_target_quat_w[env_ids, :],
            target_ang_vel_previous_b,
        )
        self._target_lin_jerk_w[env_ids, :] = torch.where(
            has_previous.unsqueeze(-1), linear_jerk, torch.zeros_like(linear_jerk)
        )
        self._target_ang_vel_w[env_ids, :] = torch.where(
            has_previous.unsqueeze(-1), angular_velocity, torch.zeros_like(angular_velocity)
        )
        self._traj_target_orientation_rate_radps[env_ids] = torch.linalg.vector_norm(
            self._target_ang_vel_w[env_ids, :], dim=-1
        )
        self._previous_target_quat_w[env_ids, :] = self._target_quat_w[env_ids, :]
        self._target_derivative_step[env_ids] = current_step
        if full_update:
            self._tracking_target_common_step = current_generation

    def get_tracking_targets(self):
        """Return synchronized trajectory targets for eval/logging code."""

        self._update_tracking_targets()
        return self._target_pos_w, self._target_lin_vel_w, self._target_quat_w

    def get_tracking_kinematics(self) -> dict[str, torch.Tensor]:
        """Return synchronized reference diagnostics for evaluation logging."""

        self._update_tracking_targets()
        return {
            "target_acceleration_w": self._target_lin_acc_w,
            "target_jerk_w": self._target_lin_jerk_w,
            "target_curvature_m_inv": self._traj_curvature_m_inv,
            "target_orientation_rate_radps": self._traj_target_orientation_rate_radps,
            "requested_period_s": self._traj_period,
            "requested_speed_mps": self._traj_target_speed_mps,
            "effective_period_s": self._traj_effective_period_s,
            "retimed": self._traj_retimed,
        }

    def _init_payload_domain(self) -> None:
        """Prepare a categorical ensemble of physically correlated payloads."""

        initialize_payload_domain(self)

    def _reset_domain(self, env_ids: Sequence[int]):
        env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        payload_enabled = reset_rigid_body(
            self,
            env_ids_device,
            enabled=self._domain_randomization_feature_enabled("rigid_body"),
        )
        self._reset_disturbance_domain(env_ids_device)
        if payload_enabled:
            apply_payload_hydrodynamics(self, env_ids_device)
        self._log_domain_randomization_state()

    def _domain_randomization_enabled(self) -> bool:
        return bool(self.cfg.domain_randomization.use_custom_randomization) and (
            not self.cfg.eval_mode or bool(getattr(self.cfg, "eval_domain_randomization", False))
        )

    def _domain_randomization_feature_enabled(self, feature: str) -> bool:
        """Return whether one independently managed DR feature is active."""

        return domain_randomization_feature_enabled(self, feature)

    def _log_domain_randomization_state(self) -> None:
        """Expose effective sampled domains to RSL-RL/TensorBoard.

        Statistics cover the full active vectorized environment after each
        reset.  This makes it possible to audit the distribution that actually
        reached PhysX instead of relying only on configured bounds.
        """

        interval = max(1, int(getattr(self.cfg, "domain_randomization_log_interval_steps", 250)))
        last_step = getattr(self, "_last_domain_randomization_log_step", None)
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_domain_randomization_log_step = self.common_step_counter

        log = self.extras.setdefault("log", {})
        # Keep terminal and TensorBoard fields compact. The surrounding
        # episode/log context already identifies these as randomized-domain
        # diagnostics, so a repeated ``DomainRandomization/`` namespace only
        # makes the rollout summary harder to scan.
        log["enabled"] = float(self._domain_randomization_enabled())
        for feature in (
            "rigid_body",
            "current",
            "hydrodynamics",
            "actuators",
            "battery",
        ):
            log[f"feature_{feature}_enabled"] = float(
                self._domain_randomization_feature_enabled(feature)
            )
        if hasattr(self.cfg.domain_randomization, "water_current_max_by_stage"):
            log["curriculum_stage"] = float(
                self._get_disturbance_curriculum_stage()
            )
            log["curriculum_global_step"] = float(
                self._disturbance_curriculum_global_step()
            )
            log["additional_hydrodynamics_scale"] = float(
                self._additional_hydrodynamics_scale()
            )

        def add_stats(name: str, values: torch.Tensor) -> None:
            flat = values.detach().to(dtype=torch.float32).reshape(-1)
            if flat.numel() == 0:
                return
            log[f"{name}_mean"] = flat.mean()
            log[f"{name}_std"] = flat.std(unbiased=False)
            log[f"{name}_min"] = flat.min()
            log[f"{name}_max"] = flat.max()

        add_stats("mass_kg", self.masses)
        add_stats("volume_m3", self.volumes)
        add_stats("center_of_mass_offset_m", torch.linalg.vector_norm(self.center_of_mass_offsets, dim=1))
        add_stats("com_to_cob_offset_m", torch.linalg.vector_norm(self.com_to_cob_offsets, dim=1))
        add_stats("principal_inertia_kg_m2", self.inertia_principal_moments)
        add_stats("added_mass_randomization_scale", self.added_mass_randomization_scale)
        add_stats("added_mass_coefficient", self.added_mass_diag)
        if self._payload_sample_count > 0:
            add_stats("payload_sample_index", self.payload_sample_indices)
        add_stats("water_current_mps", torch.linalg.vector_norm(self.water_current_w, dim=1))
        add_stats("thruster_force_scale", self.thruster_force_scale)
        add_stats("thruster_time_constant_s", self.thruster_time_constant)
        add_stats("thruster_delay_steps", self.thruster_delay_steps)
        add_stats("battery_voltage_v", self.battery_voltage)

    def _disturbance_curriculum_global_step(self) -> int:
        """Return the monotonic DR step count used by resumed campaigns."""

        offset = int(getattr(self.cfg, "disturbance_curriculum_global_step_offset", 0))
        if offset < 0:
            raise ValueError("disturbance_curriculum_global_step_offset must be non-negative.")
        return offset + int(self.common_step_counter)

    def _get_disturbance_curriculum_stage(self) -> int:
        forced_eval_stage = int(getattr(self.cfg, "eval_disturbance_stage", -1))
        if self.cfg.eval_mode and forced_eval_stage >= 0:
            return min(forced_eval_stage, len(self.cfg.domain_randomization.water_current_max_by_stage) - 1)
        if not getattr(self.cfg.domain_randomization, "disturbance_curriculum", False):
            return len(self.cfg.domain_randomization.water_current_max_by_stage) - 1

        stage = 0
        for step_boundary in self.cfg.domain_randomization.disturbance_curriculum_stage_steps:
            if self._disturbance_curriculum_global_step() >= step_boundary:
                stage += 1
        return min(stage, len(self.cfg.domain_randomization.water_current_max_by_stage) - 1)

    def _reset_disturbance_domain(self, env_ids: Sequence[int]) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        stage = self._get_disturbance_curriculum_stage()
        reset_current(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("current"),
        )
        reset_hydrodynamics(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("hydrodynamics"),
        )
        reset_actuators(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("actuators"),
        )
        reset_battery(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("battery"),
        )
        self.tether_slack_length[env_ids_device] = self.cfg.tether_slack_length

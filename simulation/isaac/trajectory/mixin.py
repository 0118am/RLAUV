"""Trajectory commands and curriculum mixin.

These methods decide task distribution and curriculum state. They deliberately
do not duplicate reusable physics equations from sibling packages.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
import isaaclab.utils.math as math_utils

from robot.control.trajectory.guidance import (
    quaternion_align_body_x_with_velocity,
    quaternion_step_angular_velocity_body,
)
from robot.control.trajectory import (
    AXIS_SINE,
    LATERAL_SINE,
    RANDOM_SMOOTH,
    RetimedTrajectoryTables,
    SPATIAL_HELIX,
    TrajectoryKinematicLimits,
    VERTICAL_SINE,
    build_retimed_tables,
    evaluate_retimed_reference,
    sample_retimed_phase,
    smooth_startup_time,
)
from .sampling_mixin import AUVTrajectorySamplingMixin


class AUVTrajectoryMixin(AUVTrajectorySamplingMixin):
    """Own reset-time task sampling and moving-reference generation."""

    def _init_trajectory_state(self) -> None:
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
        # 0=circle, 1=Lissajous, 2=single-axis sine, 3=wavy loop, 4=spiral,
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

    def _initialize_trajectory_reset(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        zeros = torch.zeros(count, device=self.device)
        self._traj_center_w[env_ids] = self._default_env_origins[env_ids]
        self._target_quat_w[env_ids] = math_utils.quat_from_euler_xyz(zeros, zeros, zeros)
        self._target_lin_acc_w[env_ids] = 0.0
        self._target_lin_jerk_w[env_ids] = 0.0
        self._target_ang_vel_w[env_ids] = 0.0
        self._target_derivative_step[env_ids] = -1
        self._traj_target_speed_mps[env_ids] = 0.0



    def _apply_speed_controlled_geometry(self, env_ids: torch.Tensor) -> None:
        trajectory_types = self._traj_type[env_ids]
        overrides = (
            (self._traj_amp_y, trajectory_types == LATERAL_SINE, self.cfg.trajectory_lateral_sine_amplitude_m),
            (self._traj_amp_z, trajectory_types == VERTICAL_SINE, self.cfg.trajectory_vertical_sine_amplitude_m),
            (self._traj_amp_x, trajectory_types == SPATIAL_HELIX, self.cfg.trajectory_spatial_helix_radius_x_m),
            (self._traj_amp_y, trajectory_types == SPATIAL_HELIX, self.cfg.trajectory_spatial_helix_radius_y_m),
            (self._traj_amp_z, trajectory_types == SPATIAL_HELIX, self.cfg.trajectory_spatial_helix_amplitude_z_m),
        )
        for target, selected, value in overrides:
            target[env_ids] = torch.where(
                selected,
                torch.full_like(target[env_ids], float(value)),
                target[env_ids],
            )

    def _build_trajectory_tables(
        self,
        env_ids: torch.Tensor,
    ) -> RetimedTrajectoryTables:
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
        return tables

    def _align_initial_trajectory_heading(
        self,
        env_ids: torch.Tensor,
        tables: RetimedTrajectoryTables,
    ) -> None:
        if not self.cfg.trajectory_align_heading_with_velocity:
            return
        initial_time = torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device)
        phase, phase_rate, phase_acceleration = sample_retimed_phase(tables, initial_time)
        _, tangent_velocity_w, _, _ = evaluate_retimed_reference(
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
        self._target_quat_w[env_ids] = quaternion_align_body_x_with_velocity(
            tangent_velocity_w,
            self._target_quat_w[env_ids],
            self.cfg.trajectory_heading_min_speed,
        )
        self._previous_target_quat_w[env_ids] = self._target_quat_w[env_ids]

    def _reset_trajectory(self, env_ids: Sequence[int]) -> None:
        """Sample trajectory parameters and rebuild phase tables."""

        selected = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._initialize_trajectory_reset(selected)
        if self.cfg.trajectory_eval_mode:
            self._sample_evaluation_trajectory(selected)
        else:
            self._sample_training_trajectory(selected)
        self._apply_speed_controlled_geometry(selected)
        if self.cfg.trajectory_eval_mode and self.cfg.trajectory_eval_type == RANDOM_SMOOTH:
            amplitudes = torch.stack(
                (
                    self._traj_amp_x[selected],
                    self._traj_amp_y[selected],
                    self._traj_amp_z[selected],
                ),
                dim=-1,
            )
            if bool(torch.any(amplitudes <= 0.0)):
                raise ValueError(
                    "random_smooth evaluation requires positive x/y/z amplitude ranges."
                )
        tables = self._build_trajectory_tables(selected)
        self._update_tracking_targets(selected)
        self._align_initial_trajectory_heading(selected, tables)
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
            heading_velocity_w[self._traj_type[env_ids] == AXIS_SINE] = 0.0
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

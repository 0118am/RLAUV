"""Reset-time sampling for training and evaluation trajectories."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

from robot.control.trajectory import (
    LATERAL_SINE, RANDOM_SMOOTH, SPATIAL_HELIX, VERTICAL_SINE,
)


class AUVTrajectorySamplingMixin:
    def _sample_evaluation_trajectory(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        self._traj_type[env_ids] = self.cfg.trajectory_eval_type
        self._traj_axis[env_ids] = 0
        if self.cfg.trajectory_eval_type == RANDOM_SMOOTH:
            ranges = (
                (self._traj_amp_x, self.cfg.trajectory_amp_x_range),
                (self._traj_amp_y, self.cfg.trajectory_amp_y_range),
                (self._traj_amp_z, self.cfg.trajectory_amp_z_range),
                (self._traj_period, self.cfg.trajectory_period_range),
            )
            for target, bounds in ranges:
                target[env_ids] = math_utils.sample_uniform(
                    bounds[0], bounds[1], (count,), device=self.device
                )
            self._traj_phase_x[env_ids] = math_utils.sample_uniform(
                0.0, 2.0 * torch.pi, (count,), device=self.device
            )
            self._traj_phase_y[env_ids] = math_utils.sample_uniform(
                0.0, 2.0 * torch.pi, (count,), device=self.device
            )
        else:
            self._traj_amp_x[env_ids] = self.cfg.trajectory_eval_amp_x
            self._traj_amp_y[env_ids] = self.cfg.trajectory_eval_amp_y
            self._traj_amp_z[env_ids] = self.cfg.trajectory_eval_amp_z
            self._traj_period[env_ids] = self.cfg.trajectory_eval_period
            self._traj_phase_x[env_ids] = 0.0
            self._traj_phase_y[env_ids] = 0.0
        if self.cfg.trajectory_eval_type in (LATERAL_SINE, VERTICAL_SINE, SPATIAL_HELIX):
            self._traj_target_speed_mps[env_ids] = float(
                self.cfg.trajectory_eval_speed_mps
            )


    def _sample_training_trajectory(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        train_types, amp_x_range, amp_y_range, amp_z_range, period_range = (
            self._get_trajectory_training_profile()
        )
        train_types = torch.as_tensor(train_types, device=self.device, dtype=torch.long)
        speed_levels = torch.as_tensor(
            self.cfg.trajectory_speed_levels_mps,
            device=self.device,
            dtype=torch.float32,
        )
        if speed_levels.ndim != 1 or speed_levels.numel() == 0 or bool(
            torch.any(speed_levels <= 0.0)
        ):
            raise ValueError(
                "trajectory_speed_levels_mps must be a non-empty list of positive speeds."
            )
        if bool(torch.any(speed_levels > float(self.cfg.trajectory_max_speed_mps))):
            raise ValueError("trajectory_speed_levels_mps exceeds trajectory_max_speed_mps.")

        controlled_types = (
            (train_types == LATERAL_SINE)
            | (train_types == VERTICAL_SINE)
            | (train_types == SPATIAL_HELIX)
        )
        if bool(torch.all(controlled_types)):
            combination_count = train_types.numel() * speed_levels.numel()
            repeats = (count + combination_count - 1) // combination_count
            combinations = torch.arange(combination_count, device=self.device).repeat(repeats)
            combinations = combinations[
                torch.randperm(combinations.numel(), device=self.device)[:count]
            ]
            type_indices = torch.div(
                combinations,
                speed_levels.numel(),
                rounding_mode="floor",
            )
            speed_indices = torch.remainder(combinations, speed_levels.numel())
        else:
            type_indices = torch.randint(0, len(train_types), (count,), device=self.device)
            speed_indices = torch.randint(
                0,
                speed_levels.numel(),
                (count,),
                device=self.device,
            )
        self._traj_type[env_ids] = train_types[type_indices]
        self._traj_axis[env_ids] = torch.randint(0, 3, (count,), device=self.device)
        for target, bounds in (
            (self._traj_amp_x, amp_x_range),
            (self._traj_amp_y, amp_y_range),
            (self._traj_amp_z, amp_z_range),
            (self._traj_period, period_range),
        ):
            target[env_ids] = math_utils.sample_uniform(
                bounds[0], bounds[1], (count,), device=self.device
            )
        self._traj_phase_x[env_ids] = math_utils.sample_uniform(
            0.0, 2.0 * torch.pi, (count,), device=self.device
        )
        self._traj_phase_y[env_ids] = math_utils.sample_uniform(
            0.0, 2.0 * torch.pi, (count,), device=self.device
        )
        selected_types = self._traj_type[env_ids]
        speed_controlled = (
            (selected_types == LATERAL_SINE)
            | (selected_types == VERTICAL_SINE)
            | (selected_types == SPATIAL_HELIX)
        )
        self._traj_target_speed_mps[env_ids] = torch.where(
            speed_controlled,
            speed_levels[speed_indices],
            self._traj_target_speed_mps[env_ids],
        )

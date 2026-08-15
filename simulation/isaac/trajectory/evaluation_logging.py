"""Chunked tensor-to-CSV logging for GPU trajectory evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


EVALUATION_LOG_SCHEMA_VERSION = 2


def indexed_columns(prefix: str, count: int, suffix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}{index}{suffix}" for index in range(count))


def evaluation_log_columns(action_dim: int) -> tuple[str, ...]:
    """Return the fixed numeric feature order used by :class:`ChunkedTensorLog`."""

    return (
        "water_current_x", "water_current_y", "water_current_z",
        "desired_x", "desired_y", "desired_z",
        "true_x", "true_y", "true_z",
        "desired_vx", "desired_vy", "desired_vz",
        "target_speed_mps", "requested_speed_mps", "target_acceleration_mps2",
        "target_jerk_mps3", "target_curvature_m_inv", "target_orientation_rate_radps",
        "requested_period_s", "effective_period_s", "trajectory_retimed",
        "true_vx", "true_vy", "true_vz",
        "position_w_x_m", "position_w_y_m", "position_w_z_m",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "angular_velocity_b_x_radps", "angular_velocity_b_y_radps", "angular_velocity_b_z_radps",
        *indexed_columns("action_", action_dim),
        *indexed_columns("raw_policy_action_", action_dim),
        *indexed_columns("raw_policy_action_clipped_", action_dim),
        "position_error", "velocity_error", "attitude_error",
        "command_heading_error_rad", "motion_sideslip_error_rad",
        "action_norm", "action_rms", "action_rate_rms",
        "raw_policy_action_norm", "raw_policy_action_rms", "raw_policy_action_clip_fraction",
        "reward",
        *indexed_columns("applied_action_", action_dim),
        *indexed_columns("requested_to_applied_action_delta_", action_dim),
        *indexed_columns("realized_thruster_force_", action_dim, "_n"),
        "requested_to_applied_action_rms", "applied_action_rate_rms",
        "realized_thruster_force_abs_mean_n", "realized_thruster_force_abs_max_n",
        "realized_wrench_force_x_n", "realized_wrench_force_y_n", "realized_wrench_force_z_n",
        "realized_wrench_torque_x_nm", "realized_wrench_torque_y_nm", "realized_wrench_torque_z_nm",
        "safety_terminated",
    )


class ChunkedTensorLog:
    """Transfer one dense tensor per chunk instead of synchronizing per scalar."""

    def __init__(
        self,
        columns: tuple[str, ...],
        *,
        num_envs: int,
        chunk_steps: int = 128,
    ) -> None:
        if num_envs <= 0 or chunk_steps <= 0:
            raise ValueError("num_envs and chunk_steps must be positive.")
        self.columns = columns
        self.num_envs = int(num_envs)
        self.chunk_steps = int(chunk_steps)
        self._payloads: list[torch.Tensor] = []
        self._steps: list[int] = []
        self._times: list[float] = []
        self._frames: list[pd.DataFrame] = []

    def append(
        self,
        *,
        step: int,
        time_s: float,
        active_mask: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if values.shape != (self.num_envs, len(self.columns)):
            raise ValueError(
                f"Expected log tensor {(self.num_envs, len(self.columns))}, got {tuple(values.shape)}."
            )
        if active_mask.shape != (self.num_envs,):
            raise ValueError(f"Expected active mask {(self.num_envs,)}, got {tuple(active_mask.shape)}.")
        # torch.cat owns fresh storage, so later simulator in-place updates
        # cannot mutate buffered history. The final column is the row mask.
        self._payloads.append(
            torch.cat((values, active_mask.to(dtype=values.dtype).unsqueeze(-1)), dim=-1)
        )
        self._steps.append(int(step))
        self._times.append(float(time_s))
        if len(self._payloads) >= self.chunk_steps:
            self.flush()

    def flush(self) -> None:
        if not self._payloads:
            return
        payload = torch.stack(self._payloads).detach().cpu().numpy()
        active = payload[..., -1].astype(bool).reshape(-1)
        features = payload[..., :-1].reshape(-1, len(self.columns))[active]
        step_grid = np.broadcast_to(
            np.asarray(self._steps, dtype=np.int64)[:, None],
            (len(self._steps), self.num_envs),
        ).reshape(-1)[active]
        time_grid = np.broadcast_to(
            np.asarray(self._times, dtype=np.float64)[:, None],
            (len(self._times), self.num_envs),
        ).reshape(-1)[active]
        env_grid = np.broadcast_to(
            np.arange(self.num_envs, dtype=np.int64)[None, :],
            (len(self._steps), self.num_envs),
        ).reshape(-1)[active]
        frame = pd.DataFrame(features, columns=self.columns)
        frame.insert(0, "time", time_grid)
        frame.insert(0, "episode_step", step_grid)
        frame.insert(0, "env_id", env_grid)
        self._frames.append(frame)
        self._payloads.clear()
        self._steps.clear()
        self._times.clear()

    def finish(self) -> pd.DataFrame:
        self.flush()
        if not self._frames:
            return pd.DataFrame(columns=("env_id", "episode_step", "time", *self.columns))
        return pd.concat(self._frames, ignore_index=True)

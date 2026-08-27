"""Pure-data checks for the schema-v7 evaluation summaries."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from simulation.training.evaluation.metrics import (
    _actuator_metrics,
    _boundary_metrics,
    _tracking_metrics,
)
from simulation.training.evaluation.runtime import (
    EVALUATION_LOG_SCHEMA_VERSION,
    _capture_tracking_snapshot,
    _post_step_actuator_values,
    evaluation_log_columns,
)


def _log() -> pd.DataFrame:
    data = {
        "time": [5.0, 6.0],
        "desired_x": [0.0, 1.0],
        "desired_y": [0.0, 0.0],
        "desired_z": [0.0, 0.0],
        "true_x": [-0.1, 0.9],
        "true_y": [0.0, 0.0],
        "true_z": [0.0, 0.0],
        "desired_vx": [0.1, 0.1],
        "desired_vy": [0.0, 0.0],
        "desired_vz": [0.0, 0.0],
        "true_vx": [0.1, 0.1],
        "true_vy": [0.0, 0.0],
        "true_vz": [0.0, 0.0],
        "position_error": [0.1, 0.1],
        "velocity_error": [0.0, 0.0],
        "attitude_error": [0.0, 0.0],
        "nose_to_target_heading_angle_rad": [0.0, 0.0],
        "nose_to_motion_heading_angle_rad": [0.0, 0.0],
        "target_curvature_m_inv": [0.0, 1.0],
        "reward": [0.9, 0.9],
        "action_rms": [0.0, 0.0],
        "action_rate_rms_per_s": [0.0, 0.0],
        "raw_policy_action_clip_fraction": [0.0, 0.0],
        "requested_to_processed_command_rms": [0.0, 0.0],
        "processed_command_rate_rms_per_s": [0.0, 0.0],
        "processed_command_acceleration_rms_per_s2": [250.0, 500.0],
        "realized_thruster_force_abs_mean_n": [1.0, 1.0],
        "realized_thruster_force_abs_max_n": [2.0, 2.0],
        "thruster_wrench_b_force_x_n": [3.0, 3.0],
        "thruster_wrench_b_force_y_n": [4.0, 4.0],
        "thruster_wrench_b_force_z_n": [0.0, 0.0],
        "physx_applied_wrench_b_force_x_n": [110.0, 110.0],
        "physx_applied_wrench_b_force_y_n": [0.0, 0.0],
        "physx_applied_wrench_b_force_z_n": [0.0, 0.0],
        "position_local_x_m": [0.0, 1.0],
        "position_local_y_m": [0.0, 0.0],
        "position_local_z_m": [0.0, 0.0],
        "quat_w": [1.0, 1.0],
        "quat_x": [0.0, 0.0],
        "quat_y": [0.0, 0.0],
        "quat_z": [0.0, 0.0],
    }
    for index in range(8):
        data[f"processed_command_{index}"] = [0.0, 0.0]
    return pd.DataFrame(data)


def test_schema_v7_has_level_heading_and_physical_command_derivatives() -> None:
    columns = evaluation_log_columns(8)
    assert EVALUATION_LOG_SCHEMA_VERSION == 7
    assert len(columns) == len(set(columns)) == 111
    assert "action_rate_rms_per_s" in columns
    assert "processed_command_rate_rms_per_s" in columns
    assert "processed_command_acceleration_rms_per_s2" in columns
    assert "target_yaw_rate_radps" in columns
    assert "nose_to_target_heading_angle_rad" in columns
    assert "thruster_wrench_b_force_x_n" in columns
    assert "physx_applied_wrench_b_force_x_n" in columns


def test_target_heading_metric_uses_commanded_yaw_during_vertical_motion() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    half_yaw = torch.tensor(torch.pi / 4.0)
    target_quaternion = torch.tensor(
        [[torch.cos(half_yaw), 0.0, 0.0, torch.sin(half_yaw)]]
    )
    vertical_velocity = torch.tensor([[0.0, 0.0, 0.2]])
    zeros_1 = torch.zeros(1)
    zeros_3 = torch.zeros(1, 3)
    robot_data = SimpleNamespace(
        root_pos_w=zeros_3,
        root_quat_w=identity,
        root_lin_vel_b=vertical_velocity,
        root_ang_vel_b=zeros_3,
    )
    kinematics = {
        "target_acceleration_w": zeros_3,
        "target_jerk_w": zeros_3,
        "target_curvature_m_inv": zeros_1,
        "target_yaw_rate_radps": zeros_1,
        "requested_period_s": torch.ones(1),
        "requested_speed_mps": torch.full((1,), 0.2),
        "effective_period_s": torch.ones(1),
        "retimed": torch.zeros(1, dtype=torch.bool),
    }
    unwrapped = SimpleNamespace(
        get_tracking_targets=lambda: (zeros_3, vertical_velocity, target_quaternion),
        get_tracking_kinematics=lambda: kinematics,
        _robot=SimpleNamespace(data=robot_data),
        scene=SimpleNamespace(env_origins=zeros_3),
    )

    snapshot = _capture_tracking_snapshot(SimpleNamespace(unwrapped=unwrapped))

    torch.testing.assert_close(
        snapshot.nose_to_target_heading_angle,
        torch.tensor([torch.pi / 2.0]),
    )
    assert torch.isnan(snapshot.nose_to_motion_heading_angle).all()


def test_post_step_actuator_log_uses_physical_command_derivatives() -> None:
    processed = torch.tensor([[0.6, -0.4], [0.2, 0.8]])
    previous = torch.tensor([[0.2, -0.1], [0.1, 0.3]])
    previous_previous = torch.tensor([[0.0, 0.2], [-0.1, 0.0]])
    robot = SimpleNamespace(
        thruster_command_processor=SimpleNamespace(processed_commands=processed),
        realized_thruster_force_n=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        realized_thruster_wrench_b=torch.zeros(2, 6),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            robot_runtime=robot,
            _thrust=torch.zeros(2, 1, 3),
            _moment=torch.zeros(2, 1, 3),
        )
    )

    values, next_previous, next_previous_previous = _post_step_actuator_values(
        env,
        requested_actions=torch.zeros_like(processed),
        previous_processed_commands=previous,
        previous_previous_processed_commands=previous_previous,
        policy_dt_s=0.04,
    )

    expected_rate = (processed - previous) / 0.04
    expected_acceleration = (processed - 2.0 * previous + previous_previous) / 0.04**2
    expected_rate_rms = torch.sqrt(torch.mean(expected_rate.square(), dim=1))
    expected_acceleration_rms = torch.sqrt(
        torch.mean(expected_acceleration.square(), dim=1)
    )
    rate_column = 3 * processed.shape[1] + 1
    acceleration_column = 3 * processed.shape[1] + 2
    torch.testing.assert_close(values[:, rate_column], expected_rate_rms)
    torch.testing.assert_close(values[:, acceleration_column], expected_acceleration_rms)
    torch.testing.assert_close(next_previous, processed)
    torch.testing.assert_close(next_previous_previous, previous)


def test_tracking_metrics_report_axis_bias_and_steady_window() -> None:
    cfg = SimpleNamespace(
        trajectory_startup_duration_s=4.0,
    )
    domain_samples = pd.DataFrame(
        {
            "sampled_thruster_time_constant_s": [0.08],
            "pose_sensor_delay_s": [0.05],
        }
    )
    metrics = _tracking_metrics(_log(), cfg, domain_samples)
    assert np.isclose(metrics["position_rmse"], 0.1)
    assert np.isclose(metrics["position_bias_x_m"], 0.1)
    assert metrics["cross_track_position_error_rmse_m"] == 0.0
    assert np.isclose(metrics["steady_position_rmse"], 0.1)
    assert np.isclose(metrics["steady_state_start_s"], 4.29)


def test_actuator_and_boundary_metrics_separate_thruster_from_physx_wrench() -> None:
    cfg = SimpleNamespace(
        rew_action_deadband=0.1,
        body_bounds_size_m=(0.5, 0.4, 0.2),
        pool_bounds=(-1.0, 2.0, -1.0, 1.0, -1.0, 1.0),
    )
    actuator = _actuator_metrics(_log(), cfg)
    boundary = _boundary_metrics(_log(), cfg)
    assert actuator["mean_thruster_wrench_force_norm_n"] == 5.0
    assert actuator["mean_physx_applied_wrench_force_norm_n"] == 110.0
    assert actuator["processed_command_deadband_fraction"] == 1.0
    assert actuator["mean_processed_command_acceleration_rms_per_s2"] == 375.0
    assert np.isclose(boundary["minimum_vehicle_boundary_clearance_m"], 0.75)

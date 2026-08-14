"""Shared AUV environment/workflow regression cases collected by domain tests.

This module is deliberately not named ``test_*.py``. Domain collectors under
Domain-aligned test modules expose every case to pytest without maintaining a handwritten
``__main__`` execution list.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from environment.water import current_fields, pool_effects
from environment.hydrodynamics import models as hydro
from environment.identification import fitters as calibration
from environment.profiles import pool_profile as profiles
from robot.dynamics import parameters as model_params
from robot.dynamics import rigid_body as rigid_body_properties
from robot.dynamics import tether
from robot.propulsion import thrusters
from simulation.isaac.envs.auv.sensing import sensors
from simulation.isaac.envs.auv.validation import replay as replay_validation
from environment.calibration import audit_pool_profile as audit_cli
from environment.calibration import build_pool_profile_from_calibration as profile_builder_cli
from environment.calibration import fit_pool_environment_logs as environment_fit_cli
from environment.calibration import fit_pool_hydrodynamics_logs as hydrodynamics_fit_cli
from environment.calibration import fit_pool_static_logs as static_fit_cli
from environment.calibration import fit_pool_tether_logs as tether_fit_cli
from environment.calibration import fit_pool_thruster_logs as thruster_fit_cli
from simulation.isaac.workflows.replay import run_pool_action_replay_checked as checked_replay_cli
from simulation.isaac.workflows.replay import summarize_pool_replay_validation as replay_summary_cli
from simulation.isaac.workflows.replay import validate_pool_replay as replay_validation_cli


def _assert_raises(error_type, callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_replay_state(time_s):
    time_s = torch.as_tensor(time_s, dtype=torch.float64)
    position = torch.stack(
        (
            torch.sin(0.4 * time_s),
            0.7 * torch.cos(0.3 * time_s),
            0.1 * time_s + 0.05 * torch.sin(0.2 * time_s),
        ),
        dim=-1,
    )
    linear_velocity = torch.stack(
        (
            0.4 * torch.cos(0.4 * time_s),
            -0.21 * torch.sin(0.3 * time_s),
            0.1 + 0.01 * torch.cos(0.2 * time_s),
        ),
        dim=-1,
    )
    yaw = 0.15 * time_s + 0.05 * torch.sin(0.25 * time_s)
    quaternion = torch.stack(
        (
            torch.cos(0.5 * yaw),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.sin(0.5 * yaw),
        ),
        dim=-1,
    )
    angular_velocity = torch.stack(
        (
            torch.zeros_like(time_s),
            torch.zeros_like(time_s),
            0.15 + 0.0125 * torch.cos(0.25 * time_s),
        ),
        dim=-1,
    )
    actions = torch.stack((0.4 * torch.sin(0.6 * time_s), 0.3 * torch.cos(0.5 * time_s)), dim=-1)
    return position, quaternion, linear_velocity, angular_velocity, actions


def _synthetic_replay_pair(simulation_time_offset_s=0.2):
    measured_time = torch.arange(0.0, 10.01, 0.1, dtype=torch.float64)
    measured = replay_validation.ReplayTrajectory(measured_time, *_synthetic_replay_state(measured_time))

    simulated_time = torch.arange(0.0, 10.41, 0.1, dtype=torch.float64)
    physical_time = simulated_time - float(simulation_time_offset_s)
    position, quaternion, linear_velocity, angular_velocity, actions = _synthetic_replay_state(physical_time)
    frame_yaw = torch.tensor(0.4, dtype=torch.float64)
    frame_quaternion = torch.tensor(
        [[torch.cos(0.5 * frame_yaw), 0.0, 0.0, torch.sin(0.5 * frame_yaw)]],
        dtype=torch.float64,
    )
    inverse_frame = replay_validation._quat_conjugate(frame_quaternion)
    inverse_frame_rows = inverse_frame.repeat(simulated_time.numel(), 1)
    translation = torch.tensor([1.2, -0.4, 0.3], dtype=torch.float64)
    simulated = replay_validation.ReplayTrajectory(
        simulated_time,
        replay_validation._quat_apply(inverse_frame_rows, position - translation),
        replay_validation._quat_multiply(inverse_frame_rows, quaternion),
        replay_validation._quat_apply(inverse_frame_rows, linear_velocity),
        angular_velocity,
        actions,
    )
    return measured, simulated


def test_relative_damping_dissipates_relative_motion():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.04, -0.02, 0.01],
            [-0.3, 0.2, -0.1, -0.03, 0.05, -0.02],
        ]
    )
    linear = torch.tensor([0.00526, 0.00526, 0.00526, 0.00032, 0.00032, 0.00032])
    quadratic = torch.tensor([39.196, 68.272, 135.402, 0.277, 1.387, 0.770])
    damping = model.calculate_relative_damping_wrench(nu_r, linear, quadratic)
    assert torch.all(torch.sum(nu_r * damping, dim=-1) <= 0.0)


def test_z_up_pool_convention_keeps_buoyancy_and_restoring_signs_consistent():
    """Exercise the z-up pool contract without requiring an Isaac process."""

    gravity = torch.tensor([0.0, 0.0, -9.81])
    model = hydro.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    neutral_volume = torch.tensor([[10.0 / 1000.0]])
    cob = torch.tensor([[0.0, 0.0, 0.02]])
    buoyancy, torque = model.calculate_buoyancy_forces(identity, gravity, 1000.0, neutral_volume, cob)
    assert torch.allclose(buoyancy, torch.tensor([[0.0, 0.0, 98.1]]), atol=1.0e-5)
    assert torch.allclose(buoyancy + 10.0 * gravity.reshape(1, 3), torch.zeros((1, 3)), atol=1.0e-5)
    assert torch.allclose(torque, torch.zeros_like(torque), atol=1.0e-6)

    # A positive roll has a restoring (negative roll) buoyancy moment for a
    # COB above the COM in a z-up world.
    rolled = torch.tensor([[0.99875027, 0.04997917, 0.0, 0.0]])
    _, roll_torque = model.calculate_buoyancy_forces(rolled, gravity, 1000.0, neutral_volume, cob)
    assert roll_torque[0, 0] < 0.0

    _, _, surface_buoyancy, _ = pool_effects.calculate_free_surface_scales(
        torch.tensor([[0.0, 0.0, -1.0]]), -1.0, 0.5, 1.4, 1.2, 1.15, 0.95, 0.9
    )
    assert torch.allclose(surface_buoyancy, torch.tensor([[0.95]]))


def test_measured_thruster_vector_curve_matches_pwm_endpoints_and_reduces_point_forces():
    commands = torch.stack((-torch.ones(8, dtype=torch.float64), torch.ones(8, dtype=torch.float64)))
    pwm = thrusters.normalized_command_to_pwm_us(commands)
    forces = thrusters.measured_thruster_body_forces(commands)

    expected_negative = torch.tensor(
        [
            [0.371767550000, -0.670443252500, -7.831306000000],
            [-0.041407625000, -0.513367312500, -6.307440562500],
            [0.371767550000, 0.670443252500, -7.831306000000],
            [-0.041407625000, 0.513367312500, -6.307440562500],
            [-4.566026062500, -1.889365191875, -0.357713116250],
            [-4.566026062500, 1.889365191875, -0.357713116250],
            [5.888204875000, -2.108183577500, -1.785394187500],
            [5.888204875000, 2.108183577500, -1.785394187500],
        ],
        dtype=torch.float64,
    )
    expected_positive = torch.tensor(
        [
            [0.952406656250, 0.158601187500, 3.870241375000],
            [0.457870437500, 0.909703375000, 7.300617712500],
            [0.952406656250, -0.158601187500, 3.870241375000],
            [0.457870437500, -0.909703375000, 7.300617712500],
            [3.058405437500, -0.653428125000, -1.535113387500],
            [3.058405437500, 0.653428125000, -1.535113387500],
            [-1.386683375000, -0.835819250000, -2.739376937500],
            [-1.386683375000, 0.835819250000, -2.739376937500],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(pwm[0], torch.full((8,), 1300.0, dtype=torch.float64))
    assert torch.equal(pwm[1], torch.full((8,), 1700.0, dtype=torch.float64))
    assert forces.shape == (2, 8, 3)
    assert torch.allclose(forces[0], expected_negative, atol=1.0e-12)
    assert torch.allclose(forces[1], expected_positive, atol=1.0e-12)

    positions = thrusters.get_thruster_positions(torch.device("cpu"), torch.float64)
    wrench = thrusters.reduce_point_forces_to_wrench(positions, forces)
    manual = torch.cat(
        (forces.sum(dim=-2), torch.cross(positions.unsqueeze(0), forces, dim=-1).sum(dim=-2)),
        dim=-1,
    )
    assert torch.allclose(wrench, manual, atol=1.0e-12)


def test_full_matrix_linear_damping_dissipates_relative_motion():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.04, -0.02, 0.01],
            [-0.3, 0.2, -0.1, -0.03, 0.05, -0.02],
        ]
    )
    base = torch.tensor(
        [
            [0.8, 0.1, 0.0, 0.0, 0.0, 0.02],
            [0.1, 1.0, 0.03, 0.0, 0.0, 0.0],
            [0.0, 0.03, 1.2, 0.0, 0.04, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.01, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.3, 0.02],
            [0.02, 0.0, 0.0, 0.0, 0.02, 0.25],
        ]
    )
    linear = base.T @ base + 0.01 * torch.eye(6)
    quadratic = torch.zeros(6)
    damping = model.calculate_relative_damping_wrench(nu_r, linear, quadratic)
    assert torch.all(torch.sum(nu_r * damping, dim=-1) <= 0.0)


def test_high_order_residual_hydrodynamics_is_passive_coupled_and_tensor_safe():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [[0.4, -0.2, 0.1, 0.03, -0.04, 0.02], [-0.3, 0.1, -0.2, 0.02, 0.01, -0.05]],
        dtype=torch.float64,
    )
    acceleration = torch.zeros_like(nu_r)
    linear_factor = torch.tensor(
        [
            [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.6, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.25],
        ],
        dtype=torch.float64,
    )
    residual = model.calculate_high_order_residual_wrench(
        nu_r,
        acceleration,
        added_mass_factor=torch.zeros(6, dtype=torch.float64),
        linear_damping_factor=linear_factor,
        quadratic_damping_factor=torch.zeros(6, dtype=torch.float64),
        cubic_damping_factor=torch.zeros(6, dtype=torch.float64),
    )

    expected = -(nu_r @ (linear_factor @ linear_factor.T).T)
    assert torch.allclose(residual, expected, atol=1.0e-12)
    assert torch.all(torch.sum(nu_r * residual, dim=-1) <= 1.0e-12)
    assert residual.dtype == nu_r.dtype
    assert residual.device == nu_r.device


def test_high_order_residual_fossen_integration_and_zero_factors_are_neutral():
    model = hydro.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    linear_velocity = torch.tensor([[0.4, 0.0, 0.0]])
    angular_velocity = torch.zeros((1, 3))
    common_args = dict(
        root_quats_w=quat,
        root_linvels_b=linear_velocity,
        root_angvels_b=angular_velocity,
        gravity_w=torch.zeros(3),
        fluid_density=1000.0,
        volumes=torch.zeros((1, 1)),
        com_to_cob_offsets=torch.zeros((1, 3)),
        linear_damping=torch.zeros(6),
        quadratic_damping=torch.zeros(6),
        water_current_w=torch.zeros((1, 3)),
        added_mass_diag=torch.zeros(6),
        relative_acceleration_b=torch.zeros((1, 6)),
    )
    neutral_force, neutral_torque = model.calculate_fossen_fluid_forces(
        **common_args,
        high_order_residual_enabled=True,
        high_order_residual_added_mass_factor=torch.zeros(6),
        high_order_residual_linear_damping_factor=torch.zeros(6),
        high_order_residual_quadratic_damping_factor=torch.zeros(6),
        high_order_residual_cubic_damping_factor=torch.zeros(6),
    )
    force, torque = model.calculate_fossen_fluid_forces(
        **common_args,
        high_order_residual_enabled=True,
        high_order_residual_added_mass_factor=torch.zeros(6),
        high_order_residual_linear_damping_factor=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        high_order_residual_quadratic_damping_factor=torch.zeros(6),
        high_order_residual_cubic_damping_factor=torch.zeros(6),
    )

    assert torch.allclose(neutral_force, torch.zeros_like(neutral_force))
    assert torch.allclose(neutral_torque, torch.zeros_like(neutral_torque))
    assert torch.allclose(force, torch.tensor([[-1.6, 0.0, 0.0]]))
    assert torch.allclose(torque, torch.zeros_like(torque))


def test_high_order_residual_damping_fit_requires_residual_data_and_emits_psd_factors():
    torch.manual_seed(5)
    nu_r = torch.randn(512, 6, dtype=torch.float64)
    factor = torch.diag(torch.tensor([0.8, 0.6, 0.5, 0.2, 0.3, 0.4], dtype=torch.float64))
    damping = factor @ factor.T
    residual_wrench = -(nu_r @ damping.T)

    fit = calibration.fit_high_order_hydrodynamic_residual_damping(nu_r, residual_wrench)
    updates = fit.to_cfg_updates()
    fitted_damping = fit.linear_damping_factor @ fit.linear_damping_factor.T

    assert torch.allclose(fitted_damping, damping, atol=1.0e-8)
    assert torch.max(fit.residual_rms) < 1.0e-8
    assert updates["high_order_residual_enabled"] is True
    assert len(updates["high_order_residual_linear_damping_factor"]) == 6


def test_speed_dependent_damping_scale_interpolates_shared_curve():
    nu_r = torch.tensor([[0.0, 0.5, 1.0, 1.5, -2.0, 3.0]])

    scale = hydro.calculate_speed_dependent_damping_scale(
        nu_r,
        speed_points=[0.0, 2.0],
        scale_points=[1.0, 3.0],
    )

    expected = torch.tensor([[1.0, 1.5, 2.0, 2.5, 3.0, 3.0]])
    assert torch.allclose(scale, expected)


def test_speed_dependent_damping_scale_interpolates_per_dof_curves():
    nu_r = torch.tensor([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5]])

    scale = hydro.calculate_speed_dependent_damping_scale(
        nu_r,
        speed_points=[0.0, 1.0],
        scale_points=[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ],
    )

    expected = torch.tensor([[1.5, 3.0, 4.5, 6.0, 7.5, 9.0]])
    assert torch.allclose(scale, expected)


def test_calibration_fits_diagonal_linear_quadratic_damping_from_synthetic_log():
    speeds = torch.tensor([-1.0, -0.7, -0.4, -0.2, 0.2, 0.4, 0.7, 1.0])
    nu_r = speeds.reshape(-1, 1).repeat(1, 6)
    linear = torch.tensor([1.0, 1.5, 2.0, 0.2, 0.3, 0.4])
    quadratic = torch.tensor([4.0, 5.0, 6.0, 0.7, 0.8, 0.9])
    applied_wrench = linear.reshape(1, 6) * nu_r + quadratic.reshape(1, 6) * torch.abs(nu_r) * nu_r

    fit = calibration.fit_diagonal_linear_quadratic_damping(
        time_s=torch.arange(len(speeds), dtype=torch.float32),
        nu_r=nu_r,
        applied_wrench=applied_wrench,
        effective_mass=torch.ones(6),
        relative_acceleration=torch.zeros_like(nu_r),
    )

    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-5)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-5)
    assert fit.to_cfg_updates()["linear_damping"] == fit.linear_damping.tolist()


def test_calibration_fits_full_matrix_linear_quadratic_damping_from_synthetic_log():
    torch.manual_seed(2)
    nu_r = torch.randn(80, 6)
    linear = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.1],
            [0.1, 1.4, 0.2, 0.0, 0.0, 0.3],
            [0.0, 0.1, 2.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.1, 0.0],
            [0.0, 0.0, 0.2, 0.1, 0.6, 0.0],
            [0.1, 0.3, 0.0, 0.0, 0.0, 0.8],
        ]
    )
    quadratic = torch.tensor(
        [
            [2.0, 0.1, 0.0, 0.0, 0.0, 0.2],
            [0.2, 2.5, 0.1, 0.0, 0.0, 0.4],
            [0.0, 0.2, 3.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.1, 0.0],
            [0.0, 0.0, 0.3, 0.1, 0.7, 0.0],
            [0.2, 0.4, 0.0, 0.0, 0.0, 0.9],
        ]
    )
    applied_wrench = nu_r @ linear.T + (torch.abs(nu_r) * nu_r) @ quadratic.T

    fit = calibration.fit_full_matrix_linear_quadratic_damping(
        time_s=torch.arange(nu_r.shape[0], dtype=torch.float32),
        nu_r=nu_r,
        applied_wrench=applied_wrench,
        effective_mass=torch.ones(6),
        relative_acceleration=torch.zeros_like(nu_r),
    )

    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-4)
    assert fit.sample_count == nu_r.shape[0]
    assert fit.to_cfg_updates()["quadratic_damping"] == fit.quadratic_damping.tolist()


def test_calibration_fits_diagonal_added_mass_and_damping_from_synthetic_log():
    torch.manual_seed(3)
    time_s = torch.arange(80, dtype=torch.float32)
    acceleration = torch.randn(80, 6)
    nu_r = torch.randn(80, 6)
    rigid_inertia = torch.tensor([11.5, 11.5, 11.5, 0.3, 0.4, 0.5])
    added_mass = torch.tensor([1.2, 1.4, 1.6, 0.05, 0.06, 0.07])
    linear = torch.tensor([0.8, 0.9, 1.0, 0.10, 0.11, 0.12])
    quadratic = torch.tensor([2.0, 2.2, 2.4, 0.20, 0.22, 0.24])
    applied_wrench = (
        (rigid_inertia + added_mass).reshape(1, 6) * acceleration
        + linear.reshape(1, 6) * nu_r
        + quadratic.reshape(1, 6) * torch.abs(nu_r) * nu_r
    )

    fit = calibration.fit_diagonal_added_mass_linear_quadratic_damping(
        time_s,
        nu_r,
        applied_wrench,
        rigid_body_inertia=rigid_inertia,
        relative_acceleration=acceleration,
    )

    assert torch.allclose(fit.added_mass, added_mass, atol=1.0e-4)
    assert torch.allclose(fit.effective_inertia, rigid_inertia + added_mass, atol=1.0e-4)
    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-4)
    assert fit.to_cfg_updates()["added_mass_diag"] == fit.added_mass.tolist()


def test_calibration_fits_full_matrix_added_mass_and_damping_from_synthetic_log():
    torch.manual_seed(4)
    time_s = torch.arange(140, dtype=torch.float32)
    acceleration = torch.randn(140, 6)
    nu_r = torch.randn(140, 6)
    rigid_inertia = torch.diag(torch.tensor([11.5, 11.5, 11.5, 0.3, 0.4, 0.5]))
    added_mass = torch.tensor(
        [
            [1.2, 0.1, 0.0, 0.0, 0.0, 0.05],
            [0.1, 1.4, 0.08, 0.0, 0.0, 0.04],
            [0.0, 0.08, 1.6, 0.0, 0.06, 0.0],
            [0.0, 0.0, 0.0, 0.05, 0.01, 0.0],
            [0.0, 0.0, 0.06, 0.01, 0.06, 0.02],
            [0.05, 0.04, 0.0, 0.0, 0.02, 0.07],
        ]
    )
    linear = torch.tensor(
        [
            [0.8, 0.05, 0.0, 0.0, 0.0, 0.02],
            [0.03, 0.9, 0.04, 0.0, 0.0, 0.02],
            [0.0, 0.04, 1.0, 0.0, 0.03, 0.0],
            [0.0, 0.0, 0.0, 0.10, 0.01, 0.0],
            [0.0, 0.0, 0.03, 0.01, 0.11, 0.0],
            [0.02, 0.02, 0.0, 0.0, 0.0, 0.12],
        ]
    )
    quadratic = torch.tensor(
        [
            [2.0, 0.1, 0.0, 0.0, 0.0, 0.03],
            [0.1, 2.2, 0.08, 0.0, 0.0, 0.04],
            [0.0, 0.08, 2.4, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.20, 0.02, 0.0],
            [0.0, 0.0, 0.05, 0.02, 0.22, 0.01],
            [0.03, 0.04, 0.0, 0.0, 0.01, 0.24],
        ]
    )
    effective = rigid_inertia + added_mass
    applied_wrench = acceleration @ effective.T + nu_r @ linear.T + (torch.abs(nu_r) * nu_r) @ quadratic.T

    fit = calibration.fit_full_matrix_added_mass_linear_quadratic_damping(
        time_s,
        nu_r,
        applied_wrench,
        rigid_body_inertia=rigid_inertia,
        relative_acceleration=acceleration,
    )

    assert torch.allclose(fit.added_mass, added_mass, atol=1.0e-4)
    assert torch.allclose(fit.effective_inertia, effective, atol=1.0e-4)
    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-4)
    assert fit.symmetrized_added_mass is True
    assert fit.to_cfg_updates()["added_mass_diag"] == fit.added_mass.tolist()


def test_hydrodynamics_calibration_log_pipeline_fits_full_physical_matrices():
    torch.manual_seed(17)
    sample_count = 180
    time_s = torch.arange(sample_count, dtype=torch.float32) * 0.05
    acceleration = torch.randn(sample_count, 6)
    nu_r = torch.randn(sample_count, 6)
    profile = profiles.NOMINAL_POOL_DYNAMICS_PROFILE
    rigid_mass = torch.zeros(6, 6)
    rigid_mass[0:3, 0:3] = torch.eye(3) * profile.rigid_body.mass
    rigid_mass[3:6, 3:6] = rigid_body_properties.inertia_matrix_tensor(
        profile.rigid_body.inertia_diag,
        torch.device("cpu"),
    )
    added_base = torch.tensor(
        [
            [1.2, 0.1, 0.0, 0.0, 0.0, 0.04],
            [0.1, 1.4, 0.08, 0.0, 0.0, 0.03],
            [0.0, 0.08, 1.6, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.08, 0.01, 0.0],
            [0.0, 0.0, 0.05, 0.01, 0.10, 0.02],
            [0.04, 0.03, 0.0, 0.0, 0.02, 0.12],
        ]
    )
    added_mass = 0.5 * (added_base + added_base.T)
    linear_seed = torch.tensor(
        [
            [0.8, 0.05, 0.0, 0.0, 0.0, 0.02],
            [0.03, 0.9, 0.04, 0.0, 0.0, 0.02],
            [0.0, 0.04, 1.0, 0.0, 0.03, 0.0],
            [0.0, 0.0, 0.0, 0.10, 0.01, 0.0],
            [0.0, 0.0, 0.03, 0.01, 0.11, 0.0],
            [0.02, 0.02, 0.0, 0.0, 0.0, 0.12],
        ]
    )
    linear_damping = linear_seed.T @ linear_seed + 0.05 * torch.eye(6)
    quadratic_damping = torch.diag(torch.tensor([2.0, 2.2, 2.4, 0.2, 0.22, 0.24]))
    wrench = (
        acceleration @ (rigid_mass + added_mass).T
        + nu_r @ linear_damping.T
        + (torch.abs(nu_r) * nu_r) @ quadratic_damping.T
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        header = [
            "time_s",
            *hydrodynamics_fit_cli.NU_COLUMNS,
            *hydrodynamics_fit_cli.WRENCH_COLUMNS,
            *hydrodynamics_fit_cli.ACCEL_COLUMNS,
        ]
        _write_csv(
            root / hydrodynamics_fit_cli.MOTION_LOG_FILENAME,
            header,
            [
                [
                    float(time_s[index]),
                    *nu_r[index].tolist(),
                    *wrench[index].tolist(),
                    *acceleration[index].tolist(),
                ]
                for index in range(sample_count)
            ],
        )

        result = hydrodynamics_fit_cli.fit_hydrodynamics_calibration_logs(root, fit_mode="full")
        output_path = root / "hydrodynamics_updates.json"
        report_path = root / "hydrodynamics_report.json"
        exit_code = hydrodynamics_fit_cli.main(
            [
                str(root),
                "--fit-mode",
                "full",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        merged = profiles.merge_pool_dynamics_cfg_updates(cfg_updates=output_updates)

    assert torch.allclose(torch.tensor(result.cfg_updates["added_mass_diag"]), added_mass, atol=2.0e-4)
    assert torch.allclose(torch.tensor(result.cfg_updates["linear_damping"]), linear_damping, atol=2.0e-4)
    assert torch.allclose(torch.tensor(result.cfg_updates["quadratic_damping"]), quadratic_damping, atol=2.0e-4)
    assert result.diagnostics["design_rank"] == 18
    assert result.diagnostics["sampled_passivity"]["is_passive"] is True
    assert result.diagnostics["added_mass_projection"]["projected_min_eigenvalue"] >= -1.0e-6
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert merged.hydrodynamics.added_mass == result.cfg_updates["added_mass_diag"]
    assert report["source_files"] == [hydrodynamics_fit_cli.MOTION_LOG_FILENAME]


def test_hydrodynamics_pipeline_fits_high_order_cfd_residual_wrench_mode():
    torch.manual_seed(6)
    sample_count = 256
    nu_r = torch.randn(sample_count, 6)
    factor = torch.diag(torch.tensor([0.7, 0.6, 0.5, 0.2, 0.25, 0.3]))
    damping = factor @ factor.T
    cfd_residual_wrench = -(nu_r @ damping.T)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / hydrodynamics_fit_cli.MOTION_LOG_FILENAME,
            ["time_s", *hydrodynamics_fit_cli.NU_COLUMNS, *hydrodynamics_fit_cli.WRENCH_COLUMNS],
            [
                [0.05 * index, *nu_r[index].tolist(), *cfd_residual_wrench[index].tolist()]
                for index in range(sample_count)
            ],
        )
        result = hydrodynamics_fit_cli.fit_hydrodynamics_calibration_logs(
            root,
            fit_mode="high-order-residual",
        )

    fitted_factor = torch.tensor(result.cfg_updates["high_order_residual_linear_damping_factor"])
    assert torch.allclose(fitted_factor @ fitted_factor.T, damping, atol=1.0e-5)
    assert result.cfg_updates["high_order_residual_enabled"] is True
    assert result.diagnostics["wrench_interpretation"].startswith("CFD/experiment fluid-wrench residual")
    assert max(result.diagnostics["fit"]["residual_rms_by_dof"]) < 1.0e-5


def test_hydrodynamics_pipeline_fits_identifiable_speed_dependent_damping():
    speed_samples = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.7, -0.8, 0.9, -1.0])
    nu_r = speed_samples.reshape(-1, 1).repeat(1, 6)
    acceleration = torch.zeros_like(nu_r)
    nominal_linear = torch.tensor([1.0, 1.5, 2.0, 0.2, 0.3, 0.4])
    nominal_quadratic = torch.tensor([4.0, 5.0, 6.0, 0.7, 0.8, 0.9])
    low_scale = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    high_scale = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0, 0.9])
    wrench = torch.zeros_like(nu_r)
    for index, speed in enumerate(torch.abs(speed_samples)):
        scale = low_scale if speed < 0.55 else high_scale
        wrench[index] = (
            nominal_linear * scale * nu_r[index]
            + nominal_quadratic * scale * speed * nu_r[index]
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / hydrodynamics_fit_cli.MOTION_LOG_FILENAME,
            [
                "time_s",
                *hydrodynamics_fit_cli.NU_COLUMNS,
                *hydrodynamics_fit_cli.WRENCH_COLUMNS,
                *hydrodynamics_fit_cli.ACCEL_COLUMNS,
            ],
            [
                [
                    float(index),
                    *nu_r[index].tolist(),
                    *wrench[index].tolist(),
                    *acceleration[index].tolist(),
                ]
                for index in range(speed_samples.numel())
            ],
        )
        result = hydrodynamics_fit_cli.fit_hydrodynamics_calibration_logs(
            root,
            fit_mode="diagonal",
            damping_speed_points=[0.25, 0.85],
        )
        output_path = root / "speed_damping_updates.json"
        exit_code = hydrodynamics_fit_cli.main(
            [
                str(root),
                "--fit-mode",
                "diagonal",
                "--damping-speed-points",
                "0.25",
                "0.85",
                "--output",
                str(output_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)

    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == result.domain_randomization_updates
    assert output_domain["use_custom_randomization"] is True
    assert result.cfg_updates["speed_dependent_damping_enabled"] is True
    assert result.cfg_updates["damping_speed_points"] == [0.25, 0.85]
    assert "damping_speed_linear_scale_range" in result.domain_randomization_updates
    assert "damping_speed_quadratic_scale_range" in result.domain_randomization_updates
    design_rank = torch.tensor(result.diagnostics["speed_dependent_damping"]["design_rank_by_bin_dof"])
    required_rank = torch.tensor(result.diagnostics["speed_dependent_damping"]["required_rank_by_bin_dof"])
    assert torch.equal(design_rank, required_rank)
    assert torch.tensor(result.cfg_updates["linear_damping_speed_scales"]).shape == (2, 6)
    assert torch.tensor(result.cfg_updates["quadratic_damping_speed_scales"]).shape == (2, 6)


def test_calibration_projects_added_mass_to_symmetric_psd():
    added_mass = torch.diag(torch.tensor([1.2, 1.0, -0.15, 0.2, 0.3, 0.4]))
    added_mass[0, 1] = 0.2
    added_mass[1, 0] = -0.1

    projection = calibration.project_added_mass_to_physical(added_mass, min_eigenvalue=0.05)
    projected = projection.projected_matrix

    assert projection.original_min_eigenvalue < 0.0
    assert projection.projected_min_eigenvalue >= 0.05 - 1.0e-5
    assert projection.symmetrized_input is True
    assert torch.allclose(projected, projected.T, atol=1.0e-6)
    assert torch.all(torch.linalg.eigvalsh(projected) >= 0.05 - 1.0e-5)
    assert projection.to_cfg_value() == projected.tolist()


def test_calibration_projects_linear_damping_to_dissipative_preserving_skew():
    torch.manual_seed(5)
    symmetric = torch.diag(torch.tensor([1.0, 0.7, -0.2, 0.1, 0.2, 0.3]))
    symmetric[0, 1] = 0.1
    symmetric[1, 0] = 0.1
    skew = torch.zeros(6, 6)
    skew[0, 5] = 0.4
    skew[5, 0] = -0.4
    damping = symmetric + skew

    projection = calibration.project_linear_damping_to_dissipative(damping, preserve_skew=True)
    projected = projection.projected_matrix

    assert projection.original_min_eigenvalue < 0.0
    assert projection.projected_min_eigenvalue >= -1.0e-6
    assert projection.preserved_skew is True
    assert torch.allclose(0.5 * (projected - projected.T), skew, atol=1.0e-6)
    assert torch.all(torch.linalg.eigvalsh(0.5 * (projected + projected.T)) >= -1.0e-6)

    nu_r = torch.randn(64, 6)
    dissipated_power = calibration.calculate_damping_dissipated_power(nu_r, linear_damping=projected)
    assert torch.all(dissipated_power >= -1.0e-5)
    assert calibration.damping_is_dissipative_for_samples(nu_r, linear_damping=projected)


def test_calibration_checks_sampled_quadratic_damping_power():
    nu_r = torch.tensor(
        [
            [0.5, -0.2, 0.1, 0.03, -0.04, 0.02],
            [-0.4, 0.3, -0.2, -0.02, 0.05, -0.01],
        ]
    )
    quadratic = torch.tensor([2.0, 2.5, 3.0, 0.2, 0.25, 0.3])
    bad_quadratic = torch.tensor([-2.0, 2.5, 3.0, 0.2, 0.25, 0.3])

    assert torch.all(calibration.calculate_damping_dissipated_power(nu_r, quadratic_damping=quadratic) > 0.0)
    assert calibration.damping_is_dissipative_for_samples(nu_r, quadratic_damping=quadratic)
    assert not calibration.damping_is_dissipative_for_samples(nu_r, quadratic_damping=bad_quadratic)


def test_calibration_fits_speed_dependent_damping_scales_from_synthetic_log():
    speed_samples = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.7, -0.8, 0.9, -1.0])
    nu_r = speed_samples.reshape(-1, 1).repeat(1, 6)
    nominal_linear = torch.tensor([1.0, 1.5, 2.0, 0.2, 0.3, 0.4])
    nominal_quadratic = torch.tensor([4.0, 5.0, 6.0, 0.7, 0.8, 0.9])
    speed_points = [0.25, 0.85]
    linear_scales = torch.tensor(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [1.6, 1.5, 1.4, 1.3, 1.2, 1.1],
        ]
    )
    quadratic_scales = torch.tensor(
        [
            [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            [1.4, 1.3, 1.2, 1.1, 1.0, 0.9],
        ]
    )

    applied_wrench = torch.zeros_like(nu_r)
    for sample_index, speed in enumerate(torch.abs(speed_samples)):
        bin_index = 0 if speed < 0.55 else 1
        applied_wrench[sample_index] = (
            nominal_linear * linear_scales[bin_index] * nu_r[sample_index]
            + nominal_quadratic * quadratic_scales[bin_index] * speed * nu_r[sample_index]
        )

    fit = calibration.fit_speed_dependent_damping_scales(
        time_s=torch.arange(len(speed_samples), dtype=torch.float32),
        nu_r=nu_r,
        applied_wrench=applied_wrench,
        effective_mass=torch.ones(6),
        speed_points=speed_points,
        nominal_linear_damping=nominal_linear,
        nominal_quadratic_damping=nominal_quadratic,
        relative_acceleration=torch.zeros_like(nu_r),
    )

    assert torch.allclose(fit.linear_scales, linear_scales, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_scales, quadratic_scales, atol=1.0e-4)
    assert torch.all(fit.linear_scale_std < 2.0e-5)
    assert torch.all(fit.quadratic_scale_std < 2.0e-5)
    updates = fit.to_cfg_updates(speed_points)
    randomization_updates = fit.to_domain_randomization_updates()
    assert updates["speed_dependent_damping_enabled"] is True
    assert updates["damping_speed_points"] == speed_points
    assert randomization_updates["use_custom_randomization"] is True
    assert randomization_updates["damping_speed_linear_scale_range"][0] <= 1.0
    assert randomization_updates["damping_speed_quadratic_scale_range"][1] >= 1.0
    assert torch.all(fit.design_rank == 2)

    _assert_raises(
        ValueError,
        calibration.fit_speed_dependent_damping_scales,
        time_s=torch.arange(4, dtype=torch.float32),
        nu_r=torch.tensor([[0.1] * 6, [-0.1] * 6, [0.8] * 6, [-0.8] * 6]),
        applied_wrench=torch.ones(4, 6),
        effective_mass=torch.ones(6),
        speed_points=[0.1, 0.8],
        nominal_linear_damping=torch.ones(6),
        nominal_quadratic_damping=torch.ones(6),
        relative_acceleration=torch.zeros(4, 6),
        require_identifiable=True,
    )


def test_calibration_fits_water_current_process_from_synthetic_log():
    alpha = 0.8
    time_s = torch.arange(8, dtype=torch.float32)
    powers = alpha ** torch.arange(len(time_s), dtype=torch.float32)
    mean_current = torch.tensor([0.1, -0.02, 0.01])
    residual = torch.stack((powers, -0.5 * powers, 0.25 * powers), dim=-1)
    current = mean_current.reshape(1, 3) + residual

    fit = calibration.fit_water_current_process(time_s, current, mean_current_w=mean_current)

    assert torch.allclose(fit.mean_current_w, mean_current)
    assert abs(fit.estimated_alpha - alpha) < 1.0e-6
    assert abs(fit.tau_s - (-1.0 / torch.log(torch.tensor(alpha)).item())) < 1.0e-5
    assert fit.sample_count == len(time_s)
    assert fit.to_cfg_updates()["water_current_w"] == mean_current.tolist()
    updates = fit.to_domain_randomization_updates(stage_count=2)
    assert updates["use_custom_randomization"] is True
    assert updates["water_current_smooth"] is True
    assert updates["water_current_tau_range"][0] == updates["water_current_tau_range"][1]
    assert len(updates["water_current_max_by_stage"]) == 2


def test_calibration_fits_buoyancy_volume_from_force_samples():
    rho = 997.0
    volume = 0.0123
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    force_w = -rho * volume * gravity_w
    samples = force_w.reshape(1, 3).repeat(5, 1)

    fit = calibration.fit_buoyancy_volume_from_forces(samples, water_density=rho, gravity_w=gravity_w)
    updates = fit.to_cfg_updates()

    assert abs(fit.volume - volume) < 1.0e-8
    assert torch.allclose(fit.mean_buoyancy_force_w, force_w, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-6
    assert fit.sample_count == 5
    assert updates["volume"] == fit.volume
    assert updates["water_rho"] == rho


def test_calibration_fits_com_to_cob_from_buoyancy_wrenches():
    offset = torch.tensor([0.08, -0.04, 0.025])
    forces_b = torch.tensor(
        [
            [0.0, 0.0, 120.0],
            [0.0, 120.0, 0.0],
            [120.0, 0.0, 0.0],
            [60.0, 80.0, 100.0],
        ]
    )
    torques_b = torch.cross(offset.reshape(1, 3).repeat(forces_b.shape[0], 1), forces_b, dim=-1)

    fit = calibration.fit_com_to_cob_offset_from_buoyancy_wrenches(forces_b, torques_b)
    updates = fit.to_cfg_updates()

    assert torch.allclose(fit.com_to_cob_offset, offset, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-5
    assert fit.sample_count == forces_b.shape[0]
    assert fit.design_rank == 3
    assert updates["com_to_cob_offset"] == fit.com_to_cob_offset.tolist()


def test_calibration_fits_com_to_cob_from_static_orientation_torques():
    offset = torch.tensor([0.04, -0.03, 0.02])
    rho = 997.0
    volume = 0.0115
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    half_sqrt = torch.sqrt(torch.tensor(0.5))
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, half_sqrt, 0.0, 0.0],
            [half_sqrt, 0.0, half_sqrt, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
        ]
    )
    force_w = (-rho * volume * gravity_w).reshape(1, 3).repeat(quats.shape[0], 1)
    force_b = hydro.quat_apply_wxyz(hydro.quat_conjugate_wxyz(quats), force_w)
    torques_b = torch.cross(offset.reshape(1, 3).repeat(quats.shape[0], 1), force_b, dim=-1)

    fit = calibration.fit_com_to_cob_offset_from_static_torques(
        root_quats_w=quats,
        buoyancy_torque_b_samples=torques_b,
        volume=volume,
        water_density=rho,
        gravity_w=gravity_w,
    )

    assert torch.allclose(fit.com_to_cob_offset, offset, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-5
    assert fit.sample_count == quats.shape[0]
    assert fit.design_rank == 3


def test_calibration_fits_mass_from_scale_readings():
    readings = torch.tensor([11.48, 11.52, 11.50, 11.50])

    fit = calibration.fit_mass_from_scale_readings(readings)
    updates = fit.to_cfg_updates()

    assert abs(fit.mass - 11.5) < 1.0e-6
    assert fit.residual_rms > 0.0
    assert fit.sample_count == readings.numel()
    assert updates["mass"] == fit.mass


def test_calibration_fits_inertia_tensor_from_axis_moments():
    inertia = torch.tensor(
        [
            [0.32, 0.018, -0.012],
            [0.018, 0.41, 0.015],
            [-0.012, 0.015, 0.53],
        ]
    )
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    moments = torch.sum((axes @ inertia) * axes, dim=-1)

    fit = calibration.fit_inertia_tensor_from_axis_moments(axes, moments)
    updates = fit.to_cfg_updates()

    assert torch.allclose(fit.inertia_tensor, inertia, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-6
    assert fit.sample_count == axes.shape[0]
    assert fit.design_rank == 6
    assert fit.min_eigenvalue_after_projection > 0.0
    assert updates["inertia_diag"] == fit.inertia_tensor.tolist()


def test_calibration_fits_inertia_tensor_from_compound_pendulum_periods():
    inertia = torch.tensor(
        [
            [0.28, 0.012, 0.006],
            [0.012, 0.37, -0.009],
            [0.006, -0.009, 0.49],
        ]
    )
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    moments = torch.sum((axes @ inertia) * axes, dim=-1)
    mass = 11.5
    gravity = 9.81
    distances = torch.tensor([0.35, 0.36, 0.37, 0.38, 0.39, 0.40])
    periods = 2.0 * torch.pi * torch.sqrt((moments + mass * distances * distances) / (mass * gravity * distances))

    recovered_moments = calibration.compound_pendulum_moments_from_periods(periods, mass, distances, gravity)
    fit = calibration.fit_inertia_tensor_from_compound_pendulum(
        axes,
        period_s_samples=periods,
        mass=mass,
        pivot_to_com_distance_samples=distances,
        gravity_mps2=gravity,
    )

    assert torch.allclose(recovered_moments, moments, atol=1.0e-6)
    assert torch.allclose(fit.inertia_tensor, inertia, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-6
    assert fit.design_rank == 6


def test_static_calibration_log_pipeline_builds_rigid_body_updates():
    rho = 997.0
    gravity_z = -9.81
    volume = 0.0118
    offset = torch.tensor([0.04, -0.03, 0.02])
    inertia = torch.tensor(
        [
            [0.31, 0.012, -0.006],
            [0.012, 0.42, 0.009],
            [-0.006, 0.009, 0.51],
        ]
    )
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    moments = torch.sum((axes @ inertia) * axes, dim=-1)
    half_sqrt = torch.sqrt(torch.tensor(0.5))
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, half_sqrt, 0.0, 0.0],
            [half_sqrt, 0.0, half_sqrt, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
        ]
    )
    force_w = torch.tensor([[0.0, 0.0, -rho * volume * gravity_z]]).repeat(quats.shape[0], 1)
    force_b = hydro.quat_apply_wxyz(hydro.quat_conjugate_wxyz(quats), force_w)
    torques_b = torch.cross(offset.reshape(1, 3).repeat(quats.shape[0], 1), force_b, dim=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "rigid_body_mass_readings.csv",
            ["sample_id", "mass_kg", "configuration"],
            [["m1", 11.48, "pool"], ["m2", 11.52, "pool"]],
        )
        _write_csv(
            root / "rigid_body_buoyancy_forces.csv",
            [
                "sample_id",
                "buoyancy_force_w_x_n",
                "buoyancy_force_w_y_n",
                "buoyancy_force_w_z_n",
                "water_density_kg_m3",
                "gravity_w_z_mps2",
            ],
            [[f"b{index}", 0.0, 0.0, -rho * volume * gravity_z, rho, gravity_z] for index in range(3)],
        )
        _write_csv(
            root / "rigid_body_axis_moments.csv",
            ["sample_id", "axis_b_x", "axis_b_y", "axis_b_z", "moment_kg_m2"],
            [
                [f"i{index}", *axes[index].tolist(), float(moments[index])]
                for index in range(axes.shape[0])
            ],
        )
        _write_csv(
            root / "rigid_body_static_buoyancy_torques.csv",
            [
                "sample_id",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "buoyancy_torque_b_x_nm",
                "buoyancy_torque_b_y_nm",
                "buoyancy_torque_b_z_nm",
                "volume_m3",
                "water_density_kg_m3",
            ],
            [
                [f"c{index}", *quats[index].tolist(), *torques_b[index].tolist(), volume, rho]
                for index in range(quats.shape[0])
            ],
        )

        result = static_fit_cli.fit_static_calibration_logs(root, gravity_z=gravity_z)
        output_path = root / "static_updates.json"
        report_path = root / "static_report.json"
        exit_code = static_fit_cli.main(
            [str(root), "--output", str(output_path), "--report", str(report_path)]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))

    assert abs(result.cfg_updates["mass"] - 11.5) < 1.0e-6
    assert abs(result.cfg_updates["volume"] - volume) < 1.0e-7
    assert result.cfg_updates["water_rho"] == rho
    assert torch.allclose(torch.tensor(result.cfg_updates["com_to_cob_offset"]), offset, atol=1.0e-6)
    assert torch.allclose(torch.tensor(result.cfg_updates["inertia_diag"]), inertia, atol=1.0e-6)
    assert result.diagnostics["center_of_buoyancy"]["design_rank"] == 3
    assert result.diagnostics["inertia"]["design_rank"] == 6
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert report["source_files"] == list(result.source_files)


def test_buoyancy_uses_world_gravity_then_body_frame():
    model = hydro.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    rho = 997.0
    volume = torch.tensor([[model_params.AUV.displaced_volume_m3]])
    cob = torch.tensor([[0.0, 0.0, 0.01]])

    force_b, torque_b = model.calculate_buoyancy_forces(q_identity, gravity_w, rho, volume, cob)
    expected_force_b = torch.tensor([[0.0, 0.0, rho * volume.item() * 9.81]])
    assert torch.allclose(force_b, expected_force_b, atol=1.0e-5)
    assert torch.allclose(torque_b, torch.cross(cob, expected_force_b, dim=-1), atol=1.0e-5)


def test_added_mass_coriolis_is_power_preserving():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.04, -0.05, 0.02],
            [-0.2, 0.1, 0.4, -0.03, 0.02, -0.06],
        ]
    )
    added_mass_diag = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    c_nu = model.calculate_added_mass_coriolis_wrench(nu_r, added_mass_diag)
    assert torch.allclose(torch.sum(nu_r * c_nu, dim=-1), torch.zeros(2), atol=1.0e-7)


def test_full_matrix_added_mass_coriolis_is_power_preserving():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.04, -0.05, 0.02],
            [-0.2, 0.1, 0.4, -0.03, 0.02, -0.06],
        ]
    )
    base = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.04],
            [0.2, 1.1, 0.1, 0.0, 0.0, 0.03],
            [0.0, 0.1, 1.4, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.02, 0.0],
            [0.0, 0.0, 0.05, 0.02, 0.3, 0.01],
            [0.04, 0.03, 0.0, 0.0, 0.01, 0.35],
        ]
    )
    added_mass = 0.5 * (base + base.T)
    c_nu = model.calculate_added_mass_coriolis_wrench(nu_r, added_mass)
    assert torch.allclose(torch.sum(nu_r * c_nu, dim=-1), torch.zeros(2), atol=1.0e-7)


def test_added_mass_inertia_wrench_is_negative_mass_times_relative_acceleration():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r_dot = torch.tensor(
        [
            [0.2, -0.1, 0.05, 0.01, -0.02, 0.03],
            [-0.3, 0.2, -0.04, -0.01, 0.03, -0.02],
        ]
    )
    added_mass_diag = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    wrench = model.calculate_added_mass_inertia_wrench(nu_r_dot, added_mass_diag)
    expected = -added_mass_diag.reshape(1, 6) * nu_r_dot
    assert torch.allclose(wrench, expected)


def test_fossen_fluid_forces_include_added_mass_inertia():
    model = hydro.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zeros_3 = torch.zeros(1, 3)
    zeros_6 = torch.zeros(6)
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    volume = torch.zeros(1, 1)
    cob = torch.zeros(1, 3)
    added_mass_diag = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    nu_r_dot = torch.tensor([[0.2, -0.1, 0.05, 0.01, -0.02, 0.03]])

    force, torque = model.calculate_fossen_fluid_forces(
        q_identity,
        zeros_3,
        zeros_3,
        gravity_w,
        997.0,
        volume,
        cob,
        zeros_6,
        zeros_6,
        zeros_3,
        added_mass_diag,
        nu_r_dot,
    )
    expected = -added_mass_diag.reshape(1, 6) * nu_r_dot
    assert torch.allclose(torch.cat((force, torque), dim=-1), expected)


def test_auv_thruster_geometry_uses_canonical_t1_to_t8_order():
    positions = thrusters.get_thruster_positions(torch.device("cpu"), torch.float64)
    expected = torch.tensor(
        [
            [0.13400, -0.16000, -0.17098],
            [-0.15000, -0.16000, -0.17098],
            [0.13400, 0.16000, -0.17098],
            [-0.15000, 0.16000, -0.17098],
            [-0.15039, 0.10360, -0.06312],
            [-0.15039, -0.10360, -0.06312],
            [0.13439, 0.10360, -0.06312],
            [0.13439, -0.10360, -0.06312],
        ],
        dtype=torch.float64,
    )

    assert model_params.AUV.thruster_labels == ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
    assert positions.shape == (8, 3)
    assert torch.equal(positions, expected)
    assert torch.all(positions[:4, 2] == -0.17098)
    assert torch.all(positions[4:, 2] == -0.06312)
    assert torch.all(positions[[0, 2, 6, 7], 0] > 0.0)
    assert torch.all(positions[[1, 3, 4, 5], 0] < 0.0)
    assert torch.all(positions[[2, 3, 4, 6], 1] > 0.0)
    assert torch.all(positions[[0, 1, 5, 7], 1] < 0.0)


def test_validation_inertia_sign_spd_and_physx_principal_axes_reconstruction():
    model = model_params.AUV
    expected = torch.tensor(
        [
            [0.115628684, -0.000010883, -0.001001989],
            [-0.000010883, 0.201210129, -0.000004539],
            [-0.001001989, -0.000004539, 0.259427119],
        ],
        dtype=torch.float64,
    )
    inertia = rigid_body_properties.inertia_matrix_tensor(
        model.inertia_tensor_body_kg_m2,
        torch.device("cpu"),
        torch.float64,
    )
    assert torch.equal(inertia, expected)
    assert torch.allclose(inertia, inertia.T, atol=1.0e-12)
    assert torch.linalg.eigvalsh(inertia).min() > 0.0
    rigid_mass = rigid_body_properties.rigid_body_mass_matrix(
        model.mass_kg,
        model.inertia_tensor_body_kg_m2,
        torch.device("cpu"),
        torch.float64,
        model.center_of_mass_offset_m,
    )
    assert torch.linalg.eigvalsh(rigid_mass).min() > 0.0
    assert torch.allclose(rigid_mass[:3, 3:], torch.zeros((3, 3), dtype=torch.float64))
    assert torch.allclose(rigid_mass[3:, 3:], inertia)

    moments, axes = rigid_body_properties.principal_inertia_and_axes(
        model.inertia_tensor_body_kg_m2,
        torch.device("cpu"),
        torch.float64,
    )
    reconstructed = axes @ torch.diag(moments) @ axes.T
    assert torch.linalg.det(axes) > 0.0
    assert torch.max(torch.abs(reconstructed - inertia)) < 1.0e-8
    assert torch.allclose(
        moments,
        torch.tensor([0.115621701, 0.201210130, 0.259434101], dtype=torch.float64),
        atol=2.0e-9,
    )


def test_validation_com_material_volume_displacement_and_zero_input_hydrostatics():
    model = model_params.AUV
    assert model.center_of_mass_from_coordinate_system_1_m == (-0.001306, 0.000061, 0.002385)
    assert model.center_of_mass_offset_m == (0.0, 0.0, 0.0)
    assert model.center_of_buoyancy_from_com_m == (0.003498, -0.000060, 0.018494)
    assert model.solid_material_volume_m3 == 0.008690716111
    assert model.displaced_volume_m3 == 0.011304505834
    assert model.solid_material_volume_m3 != model.displaced_volume_m3

    force_b, torque_b = hydro.HydrodynamicForceModels(1, torch.device("cpu")).calculate_buoyancy_forces(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([0.0, 0.0, -9.80665]),
        model.water_density_kg_m3,
        torch.tensor([[model.displaced_volume_m3]]),
        torch.tensor([model.center_of_buoyancy_from_com_m]),
    )
    expected_buoyancy = model.water_density_kg_m3 * 9.80665 * model.displaced_volume_m3
    assert abs(float(force_b[0, 2]) - expected_buoyancy) < 1.0e-5
    assert abs(expected_buoyancy - model.mass_kg * 9.80665 - 0.03438048699610258) < 1.0e-8
    assert torch.allclose(
        torque_b,
        torch.cross(torch.tensor([model.center_of_buoyancy_from_com_m]), force_b, dim=-1),
    )


def test_validation_thruster_mirror_symmetry_and_no_double_com_offset():
    positions = thrusters.get_thruster_positions(torch.device("cpu"), torch.float64)
    expected = torch.tensor(model_params.AUV.thruster_positions_body_m, dtype=torch.float64)
    assert torch.equal(positions, expected)

    commands = torch.full((1, 8), 0.7, dtype=torch.float64)
    forces = thrusters.measured_thruster_body_forces(commands)[0]
    for right, left in ((0, 2), (1, 3), (5, 4), (7, 6)):
        assert torch.allclose(forces[right, [0, 2]], forces[left, [0, 2]], atol=1.0e-12)
        assert torch.allclose(forces[right, 1], -forces[left, 1], atol=1.0e-12)


def test_measured_thruster_force_jacobian_matches_finite_difference():
    commands = torch.tensor(
        [[-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]],
        dtype=torch.float64,
    )
    analytic = thrusters.measured_thruster_force_jacobian(commands)
    epsilon = 1.0e-6
    for index in range(8):
        perturb = torch.zeros_like(commands)
        perturb[0, index] = epsilon
        finite_difference = (
            thrusters.measured_thruster_body_forces(commands + perturb)
            - thrusters.measured_thruster_body_forces(commands - perturb)
        ) / (2.0 * epsilon)
        assert torch.allclose(
            analytic[0, index],
            finite_difference[0, index],
            atol=1.0e-7,
            rtol=1.0e-6,
        )


def test_auv_model_parameters_are_consistent():
    model = model_params.AUV
    assert model.mass_kg == 11.301
    assert model.displaced_volume_m3 == 0.011304505834
    assert model.surface_area_m2 == 2.514359189
    assert model.visual_bounds_size_m == (0.561500000, 0.401999756, 0.190621773)
    assert model.thruster_pwm_center_us == 1500.0
    assert model.thruster_pwm_half_range_us == 200.0
    assert model.thruster_pwm_deadband_us == 25.0
    assert model.inertia_tensor_body_kg_m2[0][2] == -0.001001989
    coefficients = torch.as_tensor(model.thruster_force_curve_coefficients)
    assert coefficients.shape == (8, 4, 3)
    assert torch.equal(coefficients[0, 0], torch.tensor([1.04794e-5, -2.24161e-5, -4.84864e-5]))
    assert torch.equal(coefficients[7, 3], torch.tensor([4.26677e-3, 7.72990e-3, -2.07979e-2]))


def test_inertia_tensor_helper_accepts_diagonal_matrix_and_flat_values():
    diag = [1.0, 2.0, 3.0]
    matrix = [[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 3.0]]
    flat = [1.0, 0.1, 0.0, 0.1, 2.0, 0.2, 0.0, 0.2, 3.0]

    diag_matrix = rigid_body_properties.inertia_matrix_tensor(diag, torch.device("cpu"))
    full_matrix = rigid_body_properties.inertia_matrix_tensor(matrix, torch.device("cpu"))
    flat_matrix = rigid_body_properties.inertia_matrix_tensor(flat, torch.device("cpu"))

    assert torch.allclose(diag_matrix, torch.diag(torch.tensor(diag)))
    assert torch.allclose(full_matrix, torch.tensor(matrix))
    assert torch.allclose(flat_matrix, torch.tensor(matrix))
    assert torch.allclose(rigid_body_properties.inertia_diag_tensor(matrix, torch.device("cpu")), torch.tensor(diag))


def test_rigid_body_profile_accepts_full_symmetric_inertia_tensor():
    profile = profiles.PoolDynamicsProfile(
        rigid_body=profiles.RigidBodyProfile(
            inertia_diag=[
                [0.3, 0.01, 0.0],
                [0.01, 0.4, 0.02],
                [0.0, 0.02, 0.5],
            ]
        )
    )

    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)

    assert updates["inertia_diag"] == [
        [0.3, 0.01, 0.0],
        [0.01, 0.4, 0.02],
        [0.0, 0.02, 0.5],
    ]


def test_rigid_body_profile_rejects_nonsymmetric_inertia_tensor():
    bad_profile = profiles.PoolDynamicsProfile(
        rigid_body=profiles.RigidBodyProfile(
            inertia_diag=[
                [0.3, 0.01, 0.0],
                [0.02, 0.4, 0.02],
                [0.0, 0.02, 0.5],
            ]
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_nominal_pool_dynamics_profile_matches_vehicle_defaults():
    profile = profiles.NOMINAL_POOL_DYNAMICS_PROFILE
    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)

    model = model_params.AUV
    assert updates["mass"] == model.mass_kg
    assert updates["volume"] == model.displaced_volume_m3
    assert updates["water_rho"] == model.water_density_kg_m3
    assert updates["center_of_mass_offset"] == list(model.center_of_mass_offset_m)
    assert updates["com_to_cob_offset"] == list(model.center_of_buoyancy_from_com_m)
    assert updates["dyn_time_constant"] == 0.0
    assert updates["thruster_max_command_rate"] == 0.0
    assert updates["linear_damping"] == [0.0] * 6
    assert updates["quadratic_damping"] == [0.0] * 6
    assert updates["added_mass_diag"] == [0.0] * 6
    assert updates["high_order_residual_enabled"] is False
    assert updates["high_order_residual_cubic_damping_factor"] == [0.0] * 6
    assert profiles.pool_dynamics_domain_randomization_updates(profile) == {}


class _DummyDomainRandomization:
    pass


class _DummyCfg:
    def __init__(self):
        self.domain_randomization = _DummyDomainRandomization()


def test_pool_dynamics_profile_applies_measured_parameters_to_cfg():
    full_linear_damping = [
        [1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.1, 1.2, 0.0, 0.0, 0.0, 0.02],
        [0.0, 0.0, 1.4, 0.0, 0.03, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.03, 0.0, 0.25, 0.0],
        [0.0, 0.02, 0.0, 0.0, 0.0, 0.3],
    ]
    profile = profiles.PoolDynamicsProfile(
        name="measured-pool",
        hydrodynamics=profiles.HydrodynamicsProfile(
            linear_damping=full_linear_damping,
            quadratic_damping=[10.0, 12.0, 14.0, 0.4, 0.5, 0.6],
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 0.5, 1.0],
            linear_damping_speed_scales=[
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.1, 1.0, 1.2, 1.0, 1.1, 1.0],
                [1.2, 1.1, 1.4, 1.0, 1.2, 1.1],
            ],
            quadratic_damping_speed_scales=[1.0, 1.15, 1.3],
            added_mass=[1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
            water_current_w=[0.02, -0.01, 0.0],
            water_current_field_enabled=True,
            water_current_field_bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            water_current_field_shape=[2, 1, 1],
            water_current_field_values=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        ),
        thrusters=profiles.ThrusterProfile(
            command_delay_steps=2,
            max_command_rate=4.0,
            command_resolution=0.02,
            command_dropout_probability=0.05,
            wake_interaction_enabled=True,
            wake_loss_coefficient=0.2,
            wake_length=0.5,
            reaction_torque_coeff=0.02,
        ),
        pool_boundary=profiles.PoolBoundaryProfile(
            enabled=True,
            bounds=[-3.0, 3.0, -2.0, 2.0, 0.3, 2.2],
            effect_distance=0.4,
        ),
        free_surface=profiles.FreeSurfaceProfile(
            enabled=True,
            surface_z=0.3,
            effect_distance=0.5,
            heave_damping_scale=1.6,
            roll_pitch_damping_scale=1.25,
            added_mass_scale=1.2,
            buoyancy_scale=0.9,
            thrust_scale=0.75,
        ),
        tether=profiles.TetherProfile(
            enabled=True,
            slack_length=1.5,
            stiffness=15.0,
            winch_enabled=True,
            winch_target_length=2.4,
            winch_reel_speed=0.3,
            winch_min_length=1.0,
            winch_max_length=3.0,
            num_segments=4,
            segment_diameter=0.006,
            segment_density=1200.0,
            segment_buoyancy_density=997.0,
        ),
        observation=profiles.ObservationProfile(
            noise_std=[0.01] * 17,
            bias_range=0.02,
            delay_steps=1,
            update_period_steps=2,
            dropout_probability={"linear_velocity_b": 0.1},
            lowpass_alpha={"linear_velocity_b": 0.4, "angular_velocity_b": 0.6},
            bias_drift_std={"position_error_b": 0.001},
        ),
        domain_randomization=profiles.DomainRandomizationProfile(
            use_custom_randomization=True,
            mass_range=[11.0, 12.0],
            thruster_command_resolution_range=[0.0, 0.02],
            thruster_command_dropout_probability_range=[0.0, 0.1],
            thruster_wake_loss_coefficient_scale_range=[0.8, 1.2],
            thruster_reaction_torque_coeff_scale_range=[0.9, 1.1],
            damping_speed_linear_scale_range=[0.95, 1.05],
            damping_speed_quadratic_scale_range=[0.9, 1.15],
            observation_delay_steps_range=[0, 2],
            observation_update_period_steps_range=[1, 3],
            observation_dropout_probability_range=[0.0, 0.2],
            observation_lowpass_alpha_range=[0.4, 1.0],
            observation_bias_drift_std_range=[0.0, 0.002],
            disturbance_curriculum=True,
            disturbance_curriculum_stage_steps=[10, 20],
            water_current_smooth=True,
            water_current_tau_range=[4.0, 8.0],
            water_current_max_by_stage=[0.02, 0.06, 0.10],
            water_current_vertical_max_by_stage=[0.005, 0.01, 0.02],
            water_current_variation_std_by_stage=[0.001, 0.003, 0.006],
        ),
    )

    cfg = profiles.apply_pool_dynamics_profile(_DummyCfg(), profile)

    assert cfg.linear_damping == full_linear_damping
    assert cfg.added_mass_diag == [1.0, 1.1, 1.2, 0.1, 0.2, 0.3]
    assert cfg.speed_dependent_damping_enabled is True
    assert cfg.damping_speed_points == [0.0, 0.5, 1.0]
    assert cfg.linear_damping_speed_scales[1] == [1.1, 1.0, 1.2, 1.0, 1.1, 1.0]
    assert cfg.quadratic_damping_speed_scales == [1.0, 1.15, 1.3]
    assert cfg.water_current_field_enabled is True
    assert cfg.water_current_field_shape == [2, 1, 1]
    assert cfg.water_current_field_values == [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    assert cfg.thruster_command_delay_steps == 2
    assert cfg.thruster_command_resolution == 0.02
    assert cfg.thruster_command_dropout_probability == 0.05
    assert cfg.thruster_wake_interaction_enabled is True
    assert cfg.thruster_wake_loss_coefficient == 0.2
    assert cfg.thruster_wake_length == 0.5
    assert cfg.pool_boundary_effects_enabled is True
    assert cfg.free_surface_effects_enabled is True
    assert cfg.free_surface_z == 0.3
    assert cfg.free_surface_heave_damping_scale == 1.6
    assert cfg.free_surface_thrust_scale == 0.75
    assert cfg.tether_enabled is True
    assert cfg.tether_num_segments == 4
    assert cfg.tether_winch_enabled is True
    assert cfg.tether_winch_target_length == 2.4
    assert cfg.tether_winch_reel_speed == 0.3
    assert cfg.tether_winch_min_length == 1.0
    assert cfg.tether_winch_max_length == 3.0
    assert cfg.tether_segment_diameter == 0.006
    assert cfg.tether_segment_density == 1200.0
    assert cfg.tether_segment_buoyancy_density == 997.0
    assert cfg.observation_noise_std == [0.01] * 17
    assert cfg.observation_update_period_steps == 2
    assert cfg.observation_dropout_probability == {"linear_velocity_b": 0.1}
    assert cfg.observation_lowpass_alpha == {"linear_velocity_b": 0.4, "angular_velocity_b": 0.6}
    assert cfg.observation_bias_drift_std == {"position_error_b": 0.001}
    assert cfg.domain_randomization.use_custom_randomization is True
    assert cfg.domain_randomization.mass_range == [11.0, 12.0]
    assert cfg.domain_randomization.thruster_command_resolution_range == [0.0, 0.02]
    assert cfg.domain_randomization.thruster_command_dropout_probability_range == [0.0, 0.1]
    assert cfg.domain_randomization.thruster_wake_loss_coefficient_scale_range == [0.8, 1.2]
    assert cfg.domain_randomization.thruster_reaction_torque_coeff_scale_range == [0.9, 1.1]
    assert cfg.domain_randomization.damping_speed_linear_scale_range == [0.95, 1.05]
    assert cfg.domain_randomization.damping_speed_quadratic_scale_range == [0.9, 1.15]
    assert cfg.domain_randomization.observation_delay_steps_range == [0, 2]
    assert cfg.domain_randomization.observation_update_period_steps_range == [1, 3]
    assert cfg.domain_randomization.observation_dropout_probability_range == [0.0, 0.2]
    assert cfg.domain_randomization.observation_lowpass_alpha_range == [0.4, 1.0]
    assert cfg.domain_randomization.observation_bias_drift_std_range == [0.0, 0.002]
    assert cfg.domain_randomization.disturbance_curriculum is True
    assert cfg.domain_randomization.disturbance_curriculum_stage_steps == [10, 20]
    assert cfg.domain_randomization.water_current_smooth is True
    assert cfg.domain_randomization.water_current_tau_range == [4.0, 8.0]
    assert cfg.domain_randomization.water_current_max_by_stage == [0.02, 0.06, 0.10]
    assert cfg.domain_randomization.water_current_vertical_max_by_stage == [0.005, 0.01, 0.02]
    assert cfg.domain_randomization.water_current_variation_std_by_stage == [0.001, 0.003, 0.006]


def test_pool_dynamics_profile_rejects_bad_damping_speed_curve():
    bad_profile = profiles.PoolDynamicsProfile(
        hydrodynamics=profiles.HydrodynamicsProfile(
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 1.0],
            linear_damping_speed_scales=[1.0],
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_rejects_bad_water_current_randomization_parameters():
    bad_profile = profiles.PoolDynamicsProfile(
        domain_randomization=profiles.DomainRandomizationProfile(
            water_current_tau_range=[0.0, 2.0],
            water_current_max_by_stage=[0.02, 0.04],
            water_current_vertical_max_by_stage=[0.01],
            disturbance_curriculum_stage_steps=[10],
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_accepts_grouped_observation_parameters():
    profile = profiles.PoolDynamicsProfile(
        observation=profiles.ObservationProfile(
            noise_std={"linear_velocity_b": 0.01, "angular_velocity_b": [0.02, 0.02, 0.03]},
            bias_range={"position_error_b": 0.05},
            dropout_probability={"linear_velocity_b": 0.1},
            lowpass_alpha={"angular_velocity_b": 0.5},
            bias_drift_std={"position_error_b": 0.001},
        )
    )

    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)

    assert updates["observation_noise_std"] == {
        "linear_velocity_b": 0.01,
        "angular_velocity_b": [0.02, 0.02, 0.03],
    }
    assert updates["observation_bias_range"] == {"position_error_b": 0.05}
    assert updates["observation_dropout_probability"] == {"linear_velocity_b": 0.1}
    assert updates["observation_lowpass_alpha"] == {"angular_velocity_b": 0.5}
    assert updates["observation_bias_drift_std"] == {"position_error_b": 0.001}


def test_pool_dynamics_profile_round_trips_dict_and_json():
    profile = profiles.PoolDynamicsProfile(
        name="round-trip-pool",
        hydrodynamics=profiles.HydrodynamicsProfile(
            water_current_w=[0.01, 0.02, 0.0],
            added_mass=[1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
        ),
        thrusters=profiles.ThrusterProfile(command_delay_steps=1),
    )

    data = profiles.pool_dynamics_profile_to_dict(profile)
    restored = profiles.pool_dynamics_profile_from_dict(data)
    assert profiles.pool_dynamics_profile_to_dict(restored) == data

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pool_profile.json"
        profiles.write_pool_dynamics_profile_json(profile, path)
        loaded = profiles.load_pool_dynamics_profile_json(path)
    assert profiles.pool_dynamics_profile_to_dict(loaded) == data


def test_pool_dynamics_profile_merges_flat_calibration_updates():
    profile = profiles.merge_pool_dynamics_cfg_updates(
        cfg_updates=[
            {
                "mass": 11.7,
                "volume": 0.0118,
                "inertia_diag": [
                    [0.32, 0.01, 0.0],
                    [0.01, 0.41, 0.02],
                    [0.0, 0.02, 0.53],
                ],
                "com_to_cob_offset": [0.0, 0.0, 0.02],
            },
            {
                "added_mass_diag": [1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
                "linear_damping": [1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
                "thruster_command_delay_steps": 2,
            },
            {"mass_range": [11.5, 11.9]},
        ],
        domain_randomization_updates={"volume_range": [0.0116, 0.0120]},
        name="measured-pool",
        description="Merged from calibration updates.",
    )
    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)
    randomization_updates = profiles.pool_dynamics_domain_randomization_updates(profile)

    assert profile.name == "measured-pool"
    assert profile.description == "Merged from calibration updates."
    assert profile.rigid_body.mass == 11.7
    assert profile.rigid_body.inertia_diag[0][1] == 0.01
    assert profile.hydrodynamics.added_mass == [1.0, 1.1, 1.2, 0.1, 0.2, 0.3]
    assert profile.thrusters.command_delay_steps == 2
    assert updates["added_mass_diag"] == profile.hydrodynamics.added_mass
    assert updates["thruster_command_delay_steps"] == 2
    assert randomization_updates["mass_range"] == [11.5, 11.9]
    assert randomization_updates["volume_range"] == [0.0116, 0.0120]


def test_pool_profile_builder_cli_merges_update_json_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        updates_path = root / "calibration_updates.json"
        randomization_path = root / "randomization_updates.json"
        output_path = root / "measured_profile.json"
        updates_path.write_text(
            json.dumps(
                {
                    "cfg_updates": {
                        "mass": 11.6,
                        "pool_boundary_effects_enabled": True,
                        "pool_bounds": [-3.0, 3.0, -2.0, 2.0, 0.2, 2.0],
                        "pool_boundary_damping_scale": 1.4,
                    },
                    "domain_randomization_updates": {"mass_range": [11.4, 11.8]},
                }
            ),
            encoding="utf-8",
        )
        randomization_path.write_text(
            json.dumps({"water_current_max_by_stage": [0.02], "water_current_vertical_max_by_stage": [0.005]}),
            encoding="utf-8",
        )

        profile = profile_builder_cli.build_profile_from_files(
            base_profile_path=None,
            update_paths=[updates_path],
            domain_randomization_update_paths=[randomization_path],
            name="cli-measured-pool",
            strict=True,
        )
        exit_code = profile_builder_cli.main(
            [
                "--updates",
                str(updates_path),
                "--domain-randomization-updates",
                str(randomization_path),
                "--name",
                "cli-measured-pool",
                "--output",
                str(output_path),
            ]
        )
        loaded = profiles.load_pool_dynamics_profile_json(output_path)

    assert profile.pool_boundary.enabled is True
    assert profile.pool_boundary.bounds == [-3.0, 3.0, -2.0, 2.0, 0.2, 2.0]
    assert profile.domain_randomization.mass_range == [11.4, 11.8]
    assert profile.domain_randomization.water_current_max_by_stage == [0.02]
    assert exit_code == 0
    assert loaded.name == "cli-measured-pool"
    assert loaded.rigid_body.mass == 11.6


def test_pool_dynamics_profile_rejects_unknown_json_fields():
    _assert_raises(
        ValueError,
        profiles.pool_dynamics_profile_from_dict,
        {"name": "bad-profile", "thrusters": {"not_a_thruster_parameter": 1.0}},
    )


def test_pool_dynamics_profile_audit_flags_nominal_high_fidelity_gaps():
    report = profiles.audit_pool_dynamics_profile(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            sloshing_expected=True,
            tether_expected=True,
            spatial_current_expected=True,
        ),
    )
    warning_sections = {finding.section for finding in report.findings if finding.severity == "warning"}

    assert "hydrodynamics.added_mass" in warning_sections
    assert "thrusters.lookup_table" not in warning_sections
    assert "pool_boundary.enabled" in warning_sections
    assert "free_surface.enabled" in warning_sections
    assert "free_surface.sloshing" in warning_sections
    assert "tether.enabled" in warning_sections
    assert "domain_randomization" in warning_sections
    assert report.counts_by_severity["critical"] == 0
    assert report.readiness_score < 1.0
    assert report.to_dict()["profile_name"] == profiles.NOMINAL_POOL_DYNAMICS_PROFILE.name


def test_pool_profile_calibration_tasks_include_experiment_metadata():
    tasks = profiles.pool_profile_calibration_tasks(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            tether_expected=True,
            spatial_current_expected=True,
        ),
    )
    by_section = {task.section: task for task in tasks}

    assert "rigid_body.static_properties" in by_section
    assert "hydrodynamics.added_mass" in by_section
    assert "thrusters.lookup_table" not in by_section
    assert "fit_thruster_wake_loss_coefficient" in by_section["thrusters.wake_interaction"].calibration_functions
    assert "fit_thruster_reaction_torque_coefficient" in by_section[
        "thrusters.reaction_torque"
    ].calibration_functions
    assert "fit_rectangular_pool_sloshing_modes" in by_section["free_surface.sloshing"].calibration_functions
    assert by_section["rigid_body.static_properties"].priority == "P0"
    assert "fit_buoyancy_volume_from_forces" in by_section["rigid_body.static_properties"].calibration_functions
    assert "volume" in by_section["rigid_body.static_properties"].update_keys
    assert by_section["hydrodynamics.added_mass"].severity == "warning"
    assert "fit_full_matrix_added_mass_linear_quadratic_damping" in by_section[
        "hydrodynamics.added_mass"
    ].calibration_functions
    assert all(task.to_dict()["section"] == task.section for task in tasks)


def test_pool_profile_calibration_update_template_groups_missing_fields():
    template = profiles.pool_profile_calibration_update_template(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            spatial_current_expected=True,
        ),
    )
    payload = template["update_payload"]

    assert template["template_type"] == "pool_calibration_update_template"
    assert "cfg_updates" not in template
    assert "domain_randomization_updates" not in template
    assert payload["cfg_updates"]["mass"] is None
    assert payload["cfg_updates"]["added_mass_diag"] is None
    assert payload["cfg_updates"]["water_current_field_enabled"] is None
    assert payload["domain_randomization_updates"]["mass_range"] is None
    assert payload["domain_randomization_updates"]["water_current_max_by_stage"] is None
    assert template["unmapped_update_keys"] == []
    assert any(task["section"] == "hydrodynamics.water_current_field" for task in template["tasks"])


def test_pool_profile_calibration_log_schemas_describe_required_csv_inputs():
    schemas = profiles.pool_profile_calibration_log_schemas(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            spatial_current_expected=True,
            tether_expected=True,
        ),
    )
    by_filename = {schema.filename: schema for schema in schemas}

    assert "rigid_body_mass_readings.csv" in by_filename
    assert "hydrodynamics_motion_wrench_log.csv" in by_filename
    assert "thruster_static_stand.csv" not in by_filename
    assert "thruster_wake_interaction.csv" in by_filename
    assert "thruster_reaction_torque.csv" in by_filename
    assert "water_current_field_samples.csv" in by_filename
    assert "free_surface_wave_gauge.csv" in by_filename
    assert by_filename["rigid_body_mass_readings.csv"].csv_header == ("sample_id", "mass_kg", "configuration")
    assert "fit_mass_from_scale_readings" in by_filename["rigid_body_mass_readings.csv"].calibration_functions
    assert "nu_r_u_mps" in by_filename["hydrodynamics_motion_wrench_log.csv"].csv_header
    assert any(
        column.name == "nu_r_dot_u_mps2" and column.required is False
        for column in by_filename["hydrodynamics_motion_wrench_log.csv"].columns
    )
    assert "measured_target_thrust_scale" in by_filename["thruster_wake_interaction.csv"].csv_header


def test_pool_calibration_log_validator_detects_bad_values_and_missing_files():
    schemas = profiles.pool_profile_calibration_log_schemas(profiles.NOMINAL_POOL_DYNAMICS_PROFILE)
    mass_schema = next(schema for schema in schemas if schema.filename == "rigid_body_mass_readings.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mass_path = root / mass_schema.filename
        mass_path.write_text("sample_id,mass_kg,configuration\nscale-1,11.5,pool\n", encoding="utf-8")
        valid_report = profiles.validate_pool_calibration_log_directory(root, (mass_schema,))

        mass_path.write_text("sample_id,mass_kg,configuration\nscale-1,nan,pool\n", encoding="utf-8")
        invalid_report = profiles.validate_pool_calibration_log_directory(root, (mass_schema,))

        mass_path.unlink()
        missing_report = profiles.validate_pool_calibration_log_directory(root, (mass_schema,))

    assert valid_report.is_valid
    assert valid_report.row_counts[mass_schema.filename] == 1
    assert valid_report.to_dict()["error_count"] == 0
    assert audit_cli.exit_code_for_log_validation(valid_report) == 0
    assert not invalid_report.is_valid
    assert any(issue.column == "mass_kg" and "valid float" in issue.message for issue in invalid_report.issues)
    assert audit_cli.exit_code_for_log_validation(invalid_report) == 2
    assert not missing_report.is_valid
    assert any("missing" in issue.message for issue in missing_report.issues)


def test_pool_dynamics_profile_audit_accepts_configured_pool_profile_without_warnings():
    full_linear = [[0.0 for _ in range(6)] for _ in range(6)]
    full_added_mass = [[0.0 for _ in range(6)] for _ in range(6)]
    for index in range(6):
        full_linear[index][index] = 1.0 + 0.1 * index
        full_added_mass[index][index] = 0.2 + 0.01 * index

    profile = profiles.PoolDynamicsProfile(
        hydrodynamics=profiles.HydrodynamicsProfile(
            linear_damping=full_linear,
            quadratic_damping=full_linear,
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 1.0],
            linear_damping_speed_scales=[[1.0] * 6, [1.1] * 6],
            added_mass=full_added_mass,
            water_current_field_enabled=True,
            water_current_field_shape=[1, 1, 1],
            water_current_field_values=[[0.01, 0.0, 0.0]],
        ),
        thrusters=profiles.ThrusterProfile(
            command_delay_steps=1,
            max_command_rate=5.0,
            command_resolution=0.01,
            command_dropout_probability=0.01,
        ),
        battery=profiles.BatteryProfile(voltage_drop_per_s=0.01),
        pool_boundary=profiles.PoolBoundaryProfile(enabled=True),
        free_surface=profiles.FreeSurfaceProfile(enabled=True),
        tether=profiles.TetherProfile(enabled=True, num_segments=3),
        observation=profiles.ObservationProfile(noise_std=0.01, delay_steps=1),
        domain_randomization=profiles.DomainRandomizationProfile(
            water_current_max_by_stage=[0.05],
            water_current_vertical_max_by_stage=[0.01],
            water_current_variation_std_by_stage=[0.005],
        ),
    )

    report = profiles.audit_pool_dynamics_profile(
        profile,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            tether_expected=True,
            spatial_current_expected=True,
        ),
    )

    assert report.counts_by_severity["warning"] == 0
    assert not report.has_blocking_findings()
    assert report.readiness_score > 0.8


def test_pool_profile_audit_cli_loads_profile_json_and_sets_exit_code():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nominal_pool_profile.json"
        profiles.write_pool_dynamics_profile_json(profiles.NOMINAL_POOL_DYNAMICS_PROFILE, path)
        options = profiles.PoolProfileAuditOptions()
        report = audit_cli.load_and_audit_profile(path, options)
        text = audit_cli.format_audit_report(report)
        tasks = audit_cli.load_calibration_tasks(path, options)
        checklist_text = audit_cli.format_calibration_tasks(report.profile_name, tasks)
        template = audit_cli.load_calibration_update_template(path, options)
        log_schemas = audit_cli.load_calibration_log_schemas(path, options)
        log_template_dir = Path(tmpdir) / "log_templates"
        audit_cli.write_calibration_log_templates(log_template_dir, log_schemas)
        mass_csv = log_template_dir / "rigid_body_mass_readings.csv"
        schemas_json = log_template_dir / "schemas.json"
        mass_csv_text = mass_csv.read_text(encoding="utf-8")
        schemas_json_data = json.loads(schemas_json.read_text(encoding="utf-8"))
        mass_schema = next(schema for schema in log_schemas if schema.filename == mass_csv.name)
        mass_csv.write_text(mass_csv_text + "scale-1,11.5,pool\n", encoding="utf-8")
        log_validation = profiles.validate_pool_calibration_log_directory(log_template_dir, (mass_schema,))
        log_validation_text = audit_cli.format_calibration_log_validation_report(log_validation)

    assert "Profile: auv-nominal-pool" in text
    assert "Readiness score:" in text
    assert report.to_dict()["counts_by_severity"]["warning"] > 0
    assert audit_cli.exit_code_for_report(report, fail_on_warning=True, fail_on_critical=False) == 2
    assert audit_cli.exit_code_for_report(report, fail_on_warning=False, fail_on_critical=True) == 0
    assert "Calibration tasks:" in checklist_text
    assert "rigid_body.static_properties" in checklist_text
    assert template["update_payload"]["cfg_updates"]["mass"] is None
    assert mass_csv_text.startswith("sample_id,mass_kg,configuration")
    assert schemas_json_data[0]["filename"]
    assert log_validation.is_valid
    assert "Valid: yes" in log_validation_text


def test_t60_measured_mapping_preserves_batch_dtype_and_deadband():
    commands = torch.tensor(
        [
            [-0.125] * 8,
            [0.0] * 8,
            [0.125] * 8,
            [0.5] * 8,
        ],
        dtype=torch.float64,
    )
    pwm = thrusters.normalized_command_to_pwm_us(commands)
    forces = thrusters.measured_thruster_body_forces(commands)
    assert torch.equal(pwm[0], torch.full((8,), 1475.0, dtype=torch.float64))
    assert torch.equal(pwm[1], torch.full((8,), 1500.0, dtype=torch.float64))
    assert torch.equal(pwm[2], torch.full((8,), 1525.0, dtype=torch.float64))
    assert torch.equal(forces[:3], torch.zeros((3, 8, 3), dtype=torch.float64))
    assert torch.any(forces[3] != 0.0)
    assert forces.dtype == commands.dtype

    absolute_pwm = torch.tensor(
        [[1200.0] * 8, [1300.0] * 8, [1475.0] * 8, [1525.0] * 8, [1700.0] * 8, [1800.0] * 8],
        dtype=torch.float64,
    )
    absolute_forces = thrusters.thruster_body_forces_from_pwm_us(absolute_pwm)
    assert torch.equal(absolute_forces[0], absolute_forces[1])
    assert torch.equal(absolute_forces[2:4], torch.zeros((2, 8, 3), dtype=torch.float64))
    assert torch.equal(absolute_forces[4], absolute_forces[5])




def test_t60_force_dynamics_filters_three_component_forces():
    dynamics = thrusters.DynamicsFirstOrder(
        numEnvs=1,
        num_thrusters_per_env=1,
        tau=0.1,
        device=torch.device("cpu"),
    )
    command = torch.tensor([[[10.0, -5.0, 2.0]]])
    dynamics.update(command, torch.tensor([0.0]))
    realized = dynamics.update(command, torch.tensor([0.1]))

    expected = (1.0 - torch.exp(torch.tensor(-1.0))) * command
    assert realized.shape == (1, 1, 3)
    assert torch.allclose(realized, expected)


def test_calibration_fits_thruster_first_order_response_from_step_log():
    time_s = torch.arange(0.0, 2.0, 0.01)
    delay = 0.2
    tau = 0.4
    steady = 10.0
    progress = torch.where(
        time_s <= delay,
        torch.zeros_like(time_s),
        1.0 - torch.exp(-(time_s - delay) / tau),
    )
    thrust = steady * progress

    fit = calibration.fit_thruster_first_order_response(
        time_s,
        thrust,
        command_step_time_s=0.0,
        initial_thrust=0.0,
        steady_state_thrust=steady,
        delay_candidate_count=128,
    )
    updates = fit.to_cfg_updates(physics_dt_s=0.05)

    assert abs(fit.time_constant_s - tau) < 5.0e-3
    assert abs(fit.response_delay_s - delay) < 5.0e-3
    assert fit.residual_rms < 1.0e-3
    assert updates["dyn_time_constant"] == fit.time_constant_s
    assert updates["thruster_command_delay_steps"] == 4


def test_calibration_fits_thruster_voltage_exponent():
    voltage = torch.tensor([12.0, 14.0, 16.0, 18.0])
    exponent = 2.3
    scale = (voltage / 16.0) ** exponent

    fit = calibration.fit_thruster_voltage_exponent(voltage, scale, nominal_voltage=16.0)
    updates = fit.to_cfg_updates()

    assert abs(fit.thrust_exponent - exponent) < 1.0e-6
    assert fit.sample_count == 3
    assert fit.residual_rms < 1.0e-6
    assert updates["battery_voltage_nominal"] == 16.0
    assert updates["battery_voltage_thrust_exponent"] == fit.thrust_exponent


def test_calibration_fits_linear_battery_voltage_sag():
    time_s = torch.tensor([10.0, 11.0, 12.0, 13.0])
    voltage = torch.tensor([16.0, 15.9, 15.8, 15.7])

    fit = calibration.fit_battery_voltage_sag(time_s, voltage)
    updates = fit.to_cfg_updates()

    assert abs(fit.initial_voltage - 16.0) < 1.0e-6
    assert abs(fit.min_observed_voltage - 15.7) < 1.0e-6
    assert abs(fit.voltage_drop_per_s - 0.1) < 1.0e-6
    assert fit.residual_rms < 1.0e-6
    assert fit.sample_count == 4
    assert fit.time_origin_s == 10.0
    assert updates["battery_voltage"] == fit.initial_voltage
    assert updates["battery_min_voltage"] == fit.min_observed_voltage


def test_calibration_fits_thruster_wake_loss_and_reaction_torque():
    axial = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.2, 0.4])
    radial = torch.tensor([0.0, 0.05, 0.1, 0.15, 0.0, 0.5])
    source = torch.full_like(axial, 10.0)
    reference = torch.full_like(axial, 10.0)
    wake_length = 1.0
    wake_radius = 0.2
    expansion = 0.1
    loss_coefficient = 0.35
    radius_at_target = wake_radius + expansion * axial
    profile = torch.exp(-((radial / radius_at_target) ** 2)) * (1.0 - axial / wake_length)
    in_wake = (axial > 0.0) & (axial <= wake_length) & (radial <= radius_at_target)
    profile = torch.where(in_wake, profile, torch.zeros_like(profile))
    measured_scale = torch.clamp(1.0 - loss_coefficient * profile, min=0.6, max=1.0)

    wake_fit = calibration.fit_thruster_wake_loss_coefficient(
        source,
        reference,
        axial,
        radial,
        measured_scale,
        wake_length=wake_length,
        wake_radius=wake_radius,
        expansion_rate=expansion,
        min_scale=0.6,
    )
    thrust = torch.tensor([-10.0, -5.0, 5.0, 10.0])
    spin = torch.tensor([1.0, -1.0, 1.0, -1.0])
    torque_coefficient = 0.025
    torque = -torque_coefficient * spin * thrust
    torque_fit = calibration.fit_thruster_reaction_torque_coefficient(thrust, torque, spin)

    assert abs(wake_fit.loss_coefficient - loss_coefficient) < 1.0e-6
    assert wake_fit.loss_coefficient_std < 1.0e-6
    assert wake_fit.residual_rms < 1.0e-7
    assert wake_fit.informative_sample_count == 4
    assert wake_fit.to_cfg_updates()["thruster_wake_interaction_enabled"] is True
    assert torch.allclose(
        torch.tensor(wake_fit.to_domain_randomization_updates()["thruster_wake_loss_coefficient_scale_range"]),
        torch.tensor([1.0, 1.0]),
        atol=1.0e-6,
    )
    assert abs(torque_fit.torque_coefficient_m - torque_coefficient) < 1.0e-7
    assert torque_fit.torque_coefficient_std_m < 1.0e-7
    assert torque_fit.residual_rms_nm < 1.0e-7
    assert torque_fit.to_cfg_updates()["thruster_reaction_torque_coeff"] == torque_fit.torque_coefficient_m
    assert torch.allclose(
        torch.tensor(torque_fit.to_domain_randomization_updates()["thruster_reaction_torque_coeff_scale_range"]),
        torch.tensor([1.0, 1.0]),
        atol=1.0e-6,
    )


def test_thruster_calibration_log_pipeline_builds_dynamic_response_updates():
    time_s = torch.arange(0.0, 4.0, 0.02)
    step_time = 0.5
    response_delay = 0.14
    tau = 0.35
    steady_thrust = 6.0
    pwm_commands = torch.where(
        time_s < step_time,
        torch.full_like(time_s, 1500.0),
        torch.full_like(time_s, 1700.0),
    )
    response_start = step_time + response_delay
    thrust = torch.where(
        time_s <= response_start,
        torch.zeros_like(time_s),
        steady_thrust * (1.0 - torch.exp(-(time_s - response_start) / tau)),
    )
    battery_time = torch.tensor([0.0, 1.0, 2.0, 3.0])
    battery_voltage = torch.tensor([16.0, 15.9, 15.8, 15.7])
    voltage_exponent = 2.3
    battery_thrust_scale = (battery_voltage / 16.0) ** voltage_exponent

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "thruster_step_response.csv",
            ["time_s", "pwm_us", "measured_thrust_n", "voltage_v"],
            [
                [float(time_s[index]), float(pwm_commands[index]), float(thrust[index]), 16.0]
                for index in range(time_s.numel())
            ],
        )
        _write_csv(
            root / "battery_voltage_thrust_samples.csv",
            ["time_s", "voltage_v", "thrust_scale"],
            [
                [float(battery_time[index]), float(battery_voltage[index]), float(battery_thrust_scale[index])]
                for index in range(battery_time.numel())
            ],
        )

        result = thruster_fit_cli.fit_thruster_calibration_logs(root, physics_dt_s=0.02)
        output_path = root / "thruster_updates.json"
        report_path = root / "thruster_report.json"
        exit_code = thruster_fit_cli.main(
            [
                str(root),
                "--physics-dt",
                "0.02",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))

    legacy_static_keys = {
        "thruster_static_mapping_backend",
        "thruster_pwm_lookup_points_us",
        "thruster_pwm_lookup_thrust_points_n",
    }
    assert legacy_static_keys.isdisjoint(result.cfg_updates)
    assert "static_lookup" not in result.diagnostics
    assert result.source_files == ("thruster_step_response.csv", "battery_voltage_thrust_samples.csv")
    assert abs(result.cfg_updates["dyn_time_constant"] - tau) < 0.02
    assert abs(result.diagnostics["first_order_response"]["response_delay_s"] - response_delay) < 0.03
    assert result.cfg_updates["thruster_command_delay_steps"] in (6, 7, 8)
    assert abs(result.cfg_updates["battery_voltage"] - 16.0) < 1.0e-6
    assert abs(result.cfg_updates["battery_voltage_drop_per_s"] - 0.1) < 1.0e-6
    assert abs(result.cfg_updates["battery_voltage_thrust_exponent"] - voltage_exponent) < 1.0e-5
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert report["source_files"] == list(result.source_files)


def test_thruster_pipeline_fits_wake_and_reaction_torque_logs():
    axial = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.2, 0.4])
    radial = torch.tensor([0.0, 0.05, 0.1, 0.15, 0.0, 0.5])
    wake_length = 1.0
    wake_radius = 0.2
    expansion = 0.1
    min_scale = 0.6
    loss_coefficient = 0.35
    radius_at_target = wake_radius + expansion * axial
    profile = torch.exp(-((radial / radius_at_target) ** 2)) * (1.0 - axial / wake_length)
    profile = torch.where(
        (axial > 0.0) & (axial <= wake_length) & (radial <= radius_at_target),
        profile,
        torch.zeros_like(profile),
    )
    measured_scale = torch.clamp(1.0 - loss_coefficient * profile, min=min_scale, max=1.0)
    spin_directions = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]
    torque_coefficient = 0.025

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "thruster_wake_interaction.csv",
            [
                "source_thruster_index",
                "target_thruster_index",
                "source_thrust_n",
                "source_reference_thrust_n",
                "axial_distance_m",
                "radial_distance_m",
                "measured_target_thrust_scale",
            ],
            [
                [0, 1, 10.0, 10.0, float(axial[index]), float(radial[index]), float(measured_scale[index])]
                for index in range(axial.numel())
            ],
        )
        _write_csv(
            root / "thruster_reaction_torque.csv",
            ["thruster_index", "spin_direction", "thrust_n", "reaction_torque_nm"],
            [
                [index, spin, 5.0 + index, -torque_coefficient * spin * (5.0 + index)]
                for index, spin in enumerate(spin_directions)
            ],
        )
        result = thruster_fit_cli.fit_thruster_calibration_logs(
            root,
            wake_length=wake_length,
            wake_radius=wake_radius,
            wake_expansion_rate=expansion,
            wake_min_scale=min_scale,
        )
        output_path = root / "thruster_interaction_updates.json"
        exit_code = thruster_fit_cli.main(
            [
                str(root),
                "--wake-length",
                str(wake_length),
                "--wake-radius",
                str(wake_radius),
                "--wake-expansion-rate",
                str(expansion),
                "--wake-min-scale",
                str(min_scale),
                "--output",
                str(output_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        merged = profiles.merge_pool_dynamics_cfg_updates(
            cfg_updates=output_updates,
            domain_randomization_updates=output_domain,
        )

    assert exit_code == 0
    assert abs(result.cfg_updates["thruster_wake_loss_coefficient"] - loss_coefficient) < 1.0e-6
    assert abs(result.cfg_updates["thruster_reaction_torque_coeff"] - torque_coefficient) < 1.0e-7
    assert result.cfg_updates["thruster_spin_directions"] == spin_directions
    assert output_updates == result.cfg_updates
    assert output_domain == result.domain_randomization_updates
    assert output_domain["use_custom_randomization"] is True
    assert "thruster_wake_loss_coefficient_scale_range" in output_domain
    assert "thruster_reaction_torque_coeff_scale_range" in output_domain
    assert merged.thrusters.wake_interaction_enabled is True
    assert merged.thrusters.spin_directions == spin_directions
    assert merged.domain_randomization.use_custom_randomization is True
    assert merged.domain_randomization.thruster_wake_loss_coefficient_scale_range[0] <= 1.0


def test_thruster_command_processor_applies_step_delay():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=2,
        device=torch.device("cpu"),
    )
    delay_steps = torch.tensor([2])
    max_rate = torch.tensor([0.0])

    out_1 = processor.update(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_2 = processor.update(torch.tensor([[0.5, 0.5]]), delay_steps, max_rate, 0.1)
    out_3 = processor.update(torch.tensor([[0.0, 0.0]]), delay_steps, max_rate, 0.1)

    assert torch.allclose(out_1, torch.zeros(1, 2))
    assert torch.allclose(out_2, torch.zeros(1, 2))
    assert torch.allclose(out_3, torch.tensor([[1.0, -1.0]]))


def test_thruster_command_processor_applies_rate_limit():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )
    delay_steps = torch.tensor([0])
    max_rate = torch.tensor([2.0])

    out_1 = processor.update(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_2 = processor.update(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_3 = processor.update(torch.tensor([[-1.0, 1.0]]), delay_steps, max_rate, 0.1)

    assert torch.allclose(out_1, torch.tensor([[0.2, -0.2]]))
    assert torch.allclose(out_2, torch.tensor([[0.4, -0.4]]))
    assert torch.allclose(out_3, torch.tensor([[0.2, -0.2]]))


def test_thruster_command_processor_broadcasts_per_env_rate_limit():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=2,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )
    commands = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    delay_steps = torch.tensor([0, 0])
    max_rate = torch.tensor([[1.0], [3.0]])

    out = processor.update(commands, delay_steps, max_rate, 0.1)

    assert torch.allclose(out, torch.tensor([[0.1, -0.1], [0.3, -0.3]]))


def test_thruster_command_processor_quantizes_commands():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )

    out = processor.update(
        torch.tensor([[0.26, -0.24]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        command_resolution=torch.tensor([0.1]),
    )

    assert torch.allclose(out, torch.tensor([[0.3, -0.2]]))


def test_thruster_command_processor_dropout_holds_previous_command():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )

    out_1 = processor.update(
        torch.tensor([[0.8, -0.8]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        dropout_probability=torch.tensor([0.0]),
    )
    out_2 = processor.update(
        torch.tensor([[-0.5, 0.5]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        dropout_probability=torch.tensor([1.0]),
    )

    assert torch.allclose(out_1, torch.tensor([[0.8, -0.8]]))
    assert torch.allclose(out_2, out_1)


def test_voltage_thrust_scale_uses_nominal_voltage_ratio():
    voltage = torch.tensor([[16.0], [14.0]])
    scale = thrusters.calculate_voltage_thrust_scale(voltage, nominal_voltage=16.0, exponent=2.0)
    expected = torch.tensor([[1.0], [(14.0 / 16.0) ** 2]])
    assert torch.allclose(scale, expected)


def test_axial_inflow_thrust_scale_reduces_positive_inflow_only():
    axial_inflow = torch.tensor([[-1.0, 0.0, 0.5, 2.0]])
    scale = thrusters.calculate_axial_inflow_thrust_scale(
        axial_inflow,
        loss_coefficient=0.5,
        reference_speed=1.0,
        min_scale=0.4,
    )
    expected = torch.tensor([[1.0, 1.0, 0.875, 0.4]])
    assert torch.allclose(scale, expected)


def test_thruster_wake_interaction_reduces_downstream_thruster_only():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]])
    axes = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    thrust = torch.tensor([[10.0, 0.0, 0.0]])

    scale = thrusters.calculate_thruster_wake_interaction_scale(
        positions,
        axes,
        thrust,
        wake_length=1.0,
        wake_radius=0.1,
        loss_coefficient=0.5,
        expansion_rate=0.0,
        min_scale=0.2,
        reference_thrust=10.0,
    )

    expected = torch.tensor([[1.0, 0.6, 1.0]])
    assert torch.allclose(scale, expected, atol=1.0e-6)

    batched_scale = thrusters.calculate_thruster_wake_interaction_scale(
        positions.repeat(2, 1, 1),
        axes.repeat(2, 1, 1),
        thrust.repeat(2, 1),
        wake_length=1.0,
        wake_radius=0.1,
        loss_coefficient=torch.tensor([0.5, 0.25]),
        expansion_rate=0.0,
        min_scale=0.2,
        reference_thrust=10.0,
    )
    expected_batched = torch.tensor([[1.0, 0.6, 1.0], [1.0, 0.8, 1.0]])
    assert torch.allclose(batched_scale, expected_batched, atol=1.0e-6)


def test_thruster_reaction_torques_follow_spin_direction_and_signed_thrust():
    thrust = torch.tensor([[10.0, -5.0]])
    axes = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    torques = thrusters.calculate_reaction_torques(
        thrust,
        axes,
        torque_coeff=0.1,
        spin_directions=[1.0, -1.0],
    )

    expected = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 0.0, -0.5]]])
    assert torch.allclose(torques, expected)

    batched_torques = thrusters.calculate_reaction_torques(
        thrust.repeat(2, 1),
        axes.repeat(2, 1, 1),
        torque_coeff=torch.tensor([0.1, 0.2]),
        spin_directions=[1.0, -1.0],
    )
    expected_batched = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, 0.0, -0.5]],
            [[-2.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ]
    )
    assert torch.allclose(batched_torques, expected_batched)


def test_observation_delay_buffer_returns_current_until_history_is_available():
    buffer = sensors.ObservationDelayBuffer(num_envs=1, obs_dim=2, max_delay_steps=2, device=torch.device("cpu"))

    out_1 = buffer.update(torch.tensor([[1.0, 2.0]]), delay_steps=torch.tensor([2]))
    out_2 = buffer.update(torch.tensor([[3.0, 4.0]]), delay_steps=torch.tensor([2]))
    out_3 = buffer.update(torch.tensor([[5.0, 6.0]]), delay_steps=torch.tensor([2]))

    assert torch.allclose(out_1, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_2, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_3, torch.tensor([[1.0, 2.0]]))


def test_observation_sensor_model_adds_bias_without_noise():
    buffer = sensors.ObservationDelayBuffer(num_envs=2, obs_dim=3, max_delay_steps=0, device=torch.device("cpu"))
    obs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    bias = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])

    measured = sensors.apply_observation_sensor_model(
        obs,
        buffer,
        delay_steps=torch.tensor([0, 0]),
        noise_std=0.0,
        bias=bias,
    )

    assert torch.allclose(measured, obs + bias)


def test_observation_filter_state_holds_between_update_periods():
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=2, device=torch.device("cpu"))
    zeros = torch.zeros(1, 2)

    out_1 = state.update(torch.tensor([[1.0, 2.0]]), zeros, 0.0, update_period_steps=torch.tensor([2]))
    out_2 = state.update(torch.tensor([[3.0, 4.0]]), zeros, 0.0, update_period_steps=torch.tensor([2]))
    out_3 = state.update(torch.tensor([[5.0, 6.0]]), zeros, 0.0, update_period_steps=torch.tensor([2]))

    assert torch.allclose(out_1, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_2, out_1)
    assert torch.allclose(out_3, torch.tensor([[5.0, 6.0]]))


def test_observation_filter_state_dropout_holds_previous_measurement():
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=2, device=torch.device("cpu"))
    zeros = torch.zeros(1, 2)

    out_1 = state.update(torch.tensor([[1.0, 2.0]]), zeros, 0.0, dropout_probability=0.0)
    out_2 = state.update(torch.tensor([[3.0, 4.0]]), zeros, 0.0, dropout_probability=1.0)

    assert torch.allclose(out_1, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_2, out_1)


def test_observation_filter_state_lowpass_filters_updates():
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=1, device=torch.device("cpu"))

    out_1 = state.update(torch.tensor([[0.0]]), 0.0, 0.0, lowpass_alpha=0.5)
    out_2 = state.update(torch.tensor([[2.0]]), 0.0, 0.0, lowpass_alpha=0.5)
    out_3 = state.update(torch.tensor([[2.0]]), 0.0, 0.0, lowpass_alpha=0.5)

    assert torch.allclose(out_1, torch.tensor([[0.0]]))
    assert torch.allclose(out_2, torch.tensor([[1.0]]))
    assert torch.allclose(out_3, torch.tensor([[1.5]]))


def test_observation_filter_state_bias_drift_changes_measurement():
    torch.manual_seed(0)
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=2, device=torch.device("cpu"))

    measured = state.update(
        torch.zeros(1, 2),
        fixed_bias=0.0,
        noise_std=0.0,
        bias_drift_std=torch.ones(1, 2),
        dt=0.25,
    )

    assert not torch.allclose(measured, torch.zeros(1, 2))


def test_observation_group_parameter_builds_semantic_vector():
    reference = torch.zeros(2, 7)
    groups = {
        "position_error_b": slice(0, 3),
        "linear_velocity_b": slice(3, 6),
        "depth": 6,
    }

    parameter = sensors.build_observation_group_parameter(
        {
            "position_error_b": [0.1, 0.2, 0.3],
            "linear_velocity_b": 0.4,
            "depth": [1.0, 2.0],
        },
        groups,
        reference,
    )

    expected = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 1.0],
            [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 2.0],
        ]
    )
    assert torch.allclose(parameter, expected)


def test_pool_replay_validation_recovers_time_and_rigid_frame_alignment():
    measured, simulated = _synthetic_replay_pair(0.2)
    thresholds = replay_validation.ReplayMetricThresholds(
        max_position_rmse_m=1.0e-6,
        max_attitude_rmse_deg=1.0e-5,
        max_linear_velocity_rmse_mps=1.0e-6,
        max_angular_velocity_rmse_radps=1.0e-6,
        max_action_rmse=1.0e-6,
        min_overlap_duration_s=9.9,
    )
    result = replay_validation.validate_pool_replay(
        measured,
        simulated,
        max_time_offset_s=0.4,
        time_offset_resolution_s=0.1,
        frame_alignment="initial_pose",
        thresholds=thresholds,
    )

    assert abs(result.aligned.simulation_time_offset_s - 0.2) < 1.0e-9
    assert result.metrics["position"]["rmse"] < 1.0e-8
    assert result.metrics["attitude"]["rmse_deg"] < 1.0e-6
    assert result.metrics["actions"]["rmse"] < 1.0e-8
    assert result.passed is True


def test_pool_replay_validation_gate_rejects_shape_error():
    measured, simulated = _synthetic_replay_pair(0.2)
    distorted = replay_validation.ReplayTrajectory(
        simulated.time_s,
        simulated.position_w
        + torch.stack(
            (
                0.08 * torch.sin(0.7 * simulated.time_s),
                torch.zeros_like(simulated.time_s),
                torch.zeros_like(simulated.time_s),
            ),
            dim=-1,
        ),
        simulated.quaternion_wxyz,
        simulated.linear_velocity_w,
        simulated.angular_velocity_b,
        simulated.actions,
    )
    result = replay_validation.validate_pool_replay(
        measured,
        distorted,
        max_time_offset_s=0.4,
        time_offset_resolution_s=0.1,
        thresholds=replay_validation.ReplayMetricThresholds(max_position_rmse_m=0.01),
    )
    assert result.metrics["position"]["rmse"] > 0.01
    assert result.passed is False


def test_pool_replay_without_thresholds_is_metrics_only_not_a_pass():
    measured, simulated = _synthetic_replay_pair(0.0)
    result = replay_validation.validate_pool_replay(
        measured,
        simulated,
        max_time_offset_s=0.0,
        frame_alignment="none",
        min_overlap_samples=2,
    )

    assert result.gates == ()
    assert result.passed is False
    assert result.report_dict()["status"] == "metrics_only"


def test_pool_replay_static_tie_prefers_full_overlap_and_zero_offset():
    time_s = torch.arange(0.0, 2.1, 0.1, dtype=torch.float64)
    zeros_3 = torch.zeros(time_s.numel(), 3, dtype=torch.float64)
    identity = torch.zeros(time_s.numel(), 4, dtype=torch.float64)
    identity[:, 0] = 1.0
    trajectory = replay_validation.ReplayTrajectory(time_s, zeros_3, identity, zeros_3, zeros_3)

    result = replay_validation.validate_pool_replay(
        trajectory,
        trajectory,
        max_time_offset_s=0.5,
        time_offset_resolution_s=0.1,
        frame_alignment="none",
        thresholds=replay_validation.ReplayMetricThresholds(max_position_rmse_m=0.0),
        min_overlap_samples=2,
    )

    assert result.aligned.simulation_time_offset_s == 0.0
    assert result.metrics["sample_count"] == time_s.numel()


def test_pool_replay_validation_cli_writes_report_and_aligned_csv():
    measured, simulated = _synthetic_replay_pair(0.2)

    def write_replay(path, trajectory, env_id=None):
        header = [
            "time_s",
            "position_w_x_m",
            "position_w_y_m",
            "position_w_z_m",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
            "linear_velocity_w_x_mps",
            "linear_velocity_w_y_mps",
            "linear_velocity_w_z_mps",
            "angular_velocity_b_x_radps",
            "angular_velocity_b_y_radps",
            "angular_velocity_b_z_radps",
            "action_0",
            "action_1",
        ]
        if env_id is not None:
            header.insert(0, "env_id")
        rows = []
        for index in range(trajectory.time_s.numel()):
            row = [
                trajectory.time_s[index].item(),
                *trajectory.position_w[index].tolist(),
                *trajectory.quaternion_wxyz[index].tolist(),
                *trajectory.linear_velocity_w[index].tolist(),
                *trajectory.angular_velocity_b[index].tolist(),
                *trajectory.actions[index].tolist(),
            ]
            if env_id is not None:
                row.insert(0, env_id)
            rows.append(row)
        _write_csv(path, header, rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        measured_path = root / "measured.csv"
        simulated_path = root / "simulated.csv"
        report_path = root / "report.json"
        aligned_path = root / "aligned.csv"
        write_replay(measured_path, measured)
        write_replay(simulated_path, simulated, env_id=0)
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = replay_validation_cli.main(
                [
                    str(measured_path),
                    str(simulated_path),
                    "--output",
                    str(report_path),
                    "--aligned-output",
                    str(aligned_path),
                    "--simulated-env-id",
                    "0",
                    "--time-offset-resolution",
                    "0.1",
                    "--max-time-offset",
                    "0.4",
                    "--max-position-rmse",
                    "0.000001",
                    "--max-action-rmse",
                    "0.000001",
                ]
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        aligned_lines = aligned_path.read_text(encoding="utf-8").splitlines()

    assert exit_code == 0
    assert report["passed"] is True
    assert report["split"] == "held_out"
    assert report["evidence_scope"] == "held-out replay validation"
    assert len(aligned_lines) == report["metrics"]["sample_count"] + 1


def test_pool_replay_campaign_aggregates_only_independent_held_out_reports():
    measured, simulated = _synthetic_replay_pair(0.2)
    result = replay_validation.validate_pool_replay(
        measured,
        simulated,
        max_time_offset_s=0.4,
        time_offset_resolution_s=0.1,
        thresholds=replay_validation.ReplayMetricThresholds(
            max_position_rmse_m=1.0e-6,
            max_action_rmse=1.0e-6,
        ),
    )
    first = result.report_dict()
    first.update({"experiment_id": "held-out-yaw", "split": "held_out"})
    second = json.loads(json.dumps(first))
    second["experiment_id"] = "held-out-heave"
    fit_report = json.loads(json.dumps(first))
    fit_report.update({"experiment_id": "fit-surge", "split": "fit"})

    summary = replay_validation.aggregate_replay_validation_reports(
        [first, second, fit_report],
        min_held_out_cases=2,
    )
    assert summary["passed"] is True
    assert summary["held_out_case_count"] == 2
    assert summary["require_action_gate"] is True
    assert summary["excluded_non_held_out_experiments"] == ["fit-surge"]
    assert summary["aggregate_metrics"]["position_rmse_m"]["sample_weighted_rmse"] < 1.0e-8
    ungated = json.loads(json.dumps(first))
    ungated.update({"experiment_id": "ungated", "gates": []})
    ungated_summary = replay_validation.aggregate_replay_validation_reports(
        [ungated],
        min_held_out_cases=1,
    )
    assert ungated_summary["passed"] is False

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        first_path = root / "first.json"
        second_path = root / "second.json"
        output_path = root / "campaign.json"
        first_path.write_text(json.dumps(first), encoding="utf-8")
        second_path.write_text(json.dumps(second), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = replay_summary_cli.main(
                [
                    str(first_path),
                    str(second_path),
                    "--output",
                    str(output_path),
                    "--min-held-out-cases",
                    "2",
                ]
            )
        written = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert written["passed"] is True


def test_checked_isaac_replay_validates_output_artifact_and_builds_command():
    fixture = REPO_ROOT / "tests/data/pool_replay_smoke.csv"
    artifact = checked_replay_cli.validate_isaac_replay_output(fixture)
    assert artifact["sample_count"] == 11
    assert artifact["duration_s"] == 1.0
    assert artifact["action_count"] == 8
    _assert_raises(
        RuntimeError,
        checked_replay_cli.validate_isaac_replay_output,
        fixture,
        expected_duration_s=2.0,
    )

    args = checked_replay_cli.build_arg_parser().parse_args(
        [
            str(fixture),
            "--output",
            "/tmp/checked-replay.csv",
            "--device",
            "cpu",
        ]
    )
    command = checked_replay_cli.build_isaac_replay_command(args)
    assert command[0] == "/home/jining_yang/IsaacLab/isaaclab.sh"
    assert "play_pool_action_replay.py" in command[2]
    assert command[-1] == "--headless"

    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "missing.csv"
        _assert_raises(RuntimeError, checked_replay_cli.validate_isaac_replay_output, missing)


def test_trilinear_water_current_field_interpolates_regular_grid():
    values = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                values.append([float(ix), float(iy), float(iz)])

    currents = current_fields.calculate_trilinear_current_field(
        positions=torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]),
        bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        grid_shape=[2, 2, 2],
        grid_values=values,
    )

    expected = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
    assert torch.allclose(currents, expected)


def test_periodic_water_current_uses_axis_amplitude_period_and_phase():
    current = current_fields.calculate_periodic_water_current(
        torch.tensor([0.0, 1.0]),
        amplitude_w=torch.tensor([0.04, 0.02, 0.01]),
        period_s=torch.tensor([4.0, 4.0, 2.0]),
        phase_rad=torch.tensor([0.0, torch.pi / 2.0, 0.0]),
    )

    expected = torch.tensor(
        [
            [0.0, 0.02, 0.0],
            [0.04, 0.0, 0.0],
        ]
    )
    assert torch.allclose(current, expected, atol=1.0e-6)


def test_added_mass_normal_latent_maps_to_positive_mean_one_scales():
    generator = torch.Generator().manual_seed(42)
    normal_latent = torch.randn(100_000, generator=generator)
    scales = hydro.mean_one_lognormal_scale(normal_latent, 0.12)

    assert torch.all(scales > 0.0)
    assert abs(float(scales.mean()) - 1.0) < 0.002
    assert torch.equal(
        hydro.mean_one_lognormal_scale(normal_latent[:6], 0.0),
        torch.ones(6),
    )


def test_added_mass_axis_scaling_preserves_symmetric_positive_definite_matrix():
    nominal = torch.tensor(
        [
            [10.77, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.4, 24.86, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.3, 28.525, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.103, 0.002, 0.0],
            [0.0, 0.0, 0.0, 0.002, 0.120, 0.001],
            [0.0, 0.0, 0.0, 0.0, 0.001, 0.120],
        ]
    )
    scale = torch.tensor([0.85, 1.10, 1.25, 0.90, 1.05, 1.15])

    randomized = hydro.scale_hydrodynamic_coefficients(nominal, scale)

    assert torch.allclose(randomized, randomized.T)
    assert torch.all(torch.linalg.eigvalsh(randomized) > 0.0)
    assert torch.allclose(torch.diag(randomized), torch.diag(nominal) * scale)


def test_calibration_builds_water_current_field_grid_from_samples():
    positions = []
    currents = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                position = [float(ix), float(iy), float(iz)]
                positions.append(position)
                currents.append([position[0] + 0.1, position[1] - 0.2, position[2] + 0.3])

    fit = calibration.fit_water_current_field_grid(
        positions,
        currents,
        grid_shape=[2, 2, 2],
        bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        k_neighbors=4,
    )
    updates = fit.to_cfg_updates()
    interpolated = current_fields.calculate_trilinear_current_field(
        positions=torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]),
        bounds=updates["water_current_field_bounds"],
        grid_shape=updates["water_current_field_shape"],
        grid_values=updates["water_current_field_values"],
    )

    assert updates["water_current_field_enabled"] is True
    assert updates["water_current_field_shape"] == [2, 2, 2]
    assert fit.sample_count == 8
    assert torch.allclose(interpolated, torch.tensor([[0.6, 0.3, 0.8], [1.1, -0.2, 0.3]]))


def test_calibration_fits_pool_boundary_effect_scales_from_synthetic_samples():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [1.75, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.8, 0.0],
        ]
    )
    bounds = [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]
    effect_distance = 0.5
    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        positions,
        bounds,
        effect_distance,
        damping_scale_at_boundary=1.8,
        added_mass_scale_at_boundary=1.25,
        thrust_scale_at_boundary=0.7,
    )

    fit = calibration.fit_pool_boundary_effect_scales(
        positions,
        bounds,
        effect_distance,
        damping_scale_samples=damping,
        added_mass_scale_samples=added_mass,
        thrust_scale_samples=thrust,
    )
    updates = fit.to_cfg_updates()

    assert abs(fit.damping_scale_at_boundary - 1.8) < 1.0e-6
    assert abs(fit.added_mass_scale_at_boundary - 1.25) < 1.0e-6
    assert abs(fit.thrust_scale_at_boundary - 0.7) < 1.0e-6
    assert torch.equal(fit.sample_count, torch.tensor([4, 4, 4]))
    assert torch.allclose(fit.residual_rms, torch.zeros(3), atol=1.0e-6)
    assert updates["pool_boundary_effects_enabled"] is True
    assert updates["pool_boundary_damping_scale"] == fit.damping_scale_at_boundary


def test_calibration_fits_free_surface_effect_scales_from_synthetic_samples():
    positions = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.85],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ]
    )
    damping, added_mass, buoyancy, thrust = pool_effects.calculate_free_surface_scales(
        positions,
        surface_z=1.0,
        effect_distance=0.5,
        heave_damping_scale_at_surface=1.6,
        roll_pitch_damping_scale_at_surface=1.25,
        added_mass_scale_at_surface=1.35,
        buoyancy_scale_at_surface=0.82,
        thrust_scale_at_surface=0.65,
        depth_axis_sign=-1.0,
    )

    fit = calibration.fit_free_surface_effect_scales(
        positions,
        surface_z=1.0,
        effect_distance=0.5,
        heave_damping_scale_samples=damping[:, 2],
        roll_pitch_damping_scale_samples=damping[:, 3:5],
        added_mass_scale_samples=added_mass[:, 2:5],
        buoyancy_scale_samples=buoyancy,
        thrust_scale_samples=thrust,
    )
    updates = fit.to_cfg_updates()

    assert abs(fit.heave_damping_scale - 1.6) < 1.0e-6
    assert abs(fit.roll_pitch_damping_scale - 1.25) < 1.0e-6
    assert abs(fit.added_mass_scale - 1.35) < 1.0e-6
    assert abs(fit.buoyancy_scale - 0.82) < 1.0e-6
    assert abs(fit.thrust_scale - 0.65) < 1.0e-6
    assert torch.equal(fit.sample_count, torch.tensor([4, 8, 12, 4, 4]))
    assert torch.allclose(fit.residual_rms, torch.zeros(5), atol=1.0e-6)
    assert updates["free_surface_effects_enabled"] is True
    assert updates["free_surface_added_mass_scale"] == fit.added_mass_scale


def test_rectangular_pool_sloshing_obeys_dispersion_and_bottom_boundary():
    bounds = [0.0, 10.0, 0.0, 5.0]
    water_depth = 2.0
    amplitude = 0.1
    frequency = pool_effects.rectangular_sloshing_mode_frequencies(
        bounds,
        water_depth,
        [[1, 0]],
    )[0]
    wave_number = torch.tensor(torch.pi / 10.0)
    expected_frequency = torch.sqrt(9.81 * wave_number * torch.tanh(wave_number * water_depth))
    assert torch.allclose(frequency, expected_frequency)

    quarter_period = torch.pi / (2.0 * frequency)
    positions = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [5.0, 0.0, 1.0],
        ]
    )
    initial = pool_effects.calculate_rectangular_pool_sloshing_state(
        positions,
        0.0,
        base_surface_z=1.0,
        pool_bounds=bounds,
        water_depth=water_depth,
        mode_numbers=[[1, 0]],
        amplitudes_m=[amplitude],
        phases_rad=[0.0],
        depth_axis_sign=1.0,
    )
    quarter = pool_effects.calculate_rectangular_pool_sloshing_state(
        positions,
        quarter_period,
        base_surface_z=1.0,
        pool_bounds=bounds,
        water_depth=water_depth,
        mode_numbers=[[1, 0]],
        amplitudes_m=[amplitude],
        phases_rad=[0.0],
        depth_axis_sign=1.0,
    )
    assert torch.allclose(initial.surface_z[0], torch.tensor([0.9]), atol=1.0e-6)
    assert torch.allclose(initial.orbital_velocity_w, torch.zeros_like(initial.orbital_velocity_w), atol=1.0e-6)
    assert abs(float(quarter.orbital_velocity_w[0, 2]) - amplitude * float(frequency)) < 1.0e-5
    assert abs(float(quarter.orbital_velocity_w[1, 2])) < 1.0e-6
    assert abs(float(quarter.orbital_velocity_w[2, 0])) > 1.0e-3
    assert torch.all(torch.isfinite(quarter.orbital_velocity_w))


def test_sloshing_surface_drives_local_surface_scales():
    positions = torch.tensor([[0.0, 0.0, 1.2], [10.0, 0.0, 1.2]])
    state = pool_effects.calculate_rectangular_pool_sloshing_state(
        positions,
        0.0,
        base_surface_z=1.0,
        pool_bounds=[0.0, 10.0, 0.0, 5.0],
        water_depth=2.0,
        mode_numbers=[[1, 0]],
        amplitudes_m=[0.2],
        phases_rad=[0.0],
        depth_axis_sign=1.0,
    )
    damping, _, _, _ = pool_effects.calculate_free_surface_scales(
        positions,
        surface_z=state.surface_z,
        effect_distance=0.5,
        heave_damping_scale_at_surface=2.0,
        roll_pitch_damping_scale_at_surface=1.5,
        added_mass_scale_at_surface=1.2,
        buoyancy_scale_at_surface=0.8,
        thrust_scale_at_surface=0.7,
    )
    assert torch.allclose(state.surface_z[:, 0], torch.tensor([0.8, 1.2]), atol=1.0e-6)
    assert damping[1, 2] > damping[0, 2] > 1.0


def test_calibration_recovers_rectangular_pool_sloshing_modes():
    bounds = [-4.0, 4.0, -3.0, 3.0]
    modes = [[1, 0], [0, 1]]
    amplitudes = torch.tensor([0.08, 0.035])
    phases = torch.tensor([0.3, -0.7])
    time_s = torch.linspace(0.0, 24.0, 481)
    gauge_xy = torch.tensor(
        [[-3.1, -2.2], [0.7, -1.4], [2.6, 1.8], [-1.2, 2.1]],
        dtype=torch.float32,
    )
    gauge_indices = torch.arange(time_s.numel()) % gauge_xy.shape[0]
    sample_xy = gauge_xy[gauge_indices]
    positions = torch.cat((sample_xy, torch.ones(time_s.numel(), 1)), dim=-1)
    state = pool_effects.calculate_rectangular_pool_sloshing_state(
        positions,
        time_s,
        base_surface_z=1.0,
        pool_bounds=bounds,
        water_depth=4.0,
        mode_numbers=modes,
        amplitudes_m=amplitudes,
        phases_rad=phases,
        depth_axis_sign=1.0,
    )
    fit = calibration.fit_rectangular_pool_sloshing_modes(
        time_s,
        sample_xy,
        state.surface_z[:, 0],
        base_surface_z=1.0,
        pool_bounds=bounds,
        water_depth=4.0,
        mode_numbers=modes,
        depth_axis_sign=1.0,
    )
    phase_error = torch.atan2(torch.sin(fit.phases_rad - phases), torch.cos(fit.phases_rad - phases))
    assert torch.allclose(fit.amplitudes_m, amplitudes, atol=1.0e-5)
    assert torch.allclose(phase_error, torch.zeros_like(phase_error), atol=1.0e-5)
    assert fit.residual_rms < 1.0e-6
    assert fit.design_rank == 4
    updates = fit.to_cfg_updates()
    assert updates["free_surface_sloshing_enabled"] is True
    assert updates["free_surface_sloshing_mode_numbers"] == modes


def test_free_surface_profile_maps_and_validates_sloshing_configuration():
    profile = profiles.PoolDynamicsProfile(
        free_surface=profiles.FreeSurfaceProfile(
            enabled=True,
            sloshing_enabled=True,
            sloshing_pool_bounds=[-4.0, 4.0, -3.0, 3.0],
            sloshing_water_depth=4.0,
            sloshing_mode_numbers=[[1, 0], [0, 1]],
            sloshing_amplitudes_m=[0.08, 0.035],
            sloshing_phases_rad=[0.3, -0.7],
        )
    )
    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)
    assert updates["free_surface_sloshing_enabled"] is True
    assert updates["free_surface_sloshing_amplitudes_m"] == [0.08, 0.035]
    restored = profiles.pool_dynamics_profile_from_dict(profiles.pool_dynamics_profile_to_dict(profile))
    assert restored.free_surface.sloshing_mode_numbers == [[1, 0], [0, 1]]
    _assert_raises(
        ValueError,
        profiles.FreeSurfaceProfile(
            sloshing_mode_numbers=[[0, 0]],
            sloshing_amplitudes_m=[0.1],
        ).validate,
    )


def test_environment_pipeline_fits_free_surface_wave_gauge_log():
    bounds = [-4.0, 4.0, -3.0, 3.0]
    modes = [[1, 0], [0, 1]]
    time_s = torch.linspace(0.0, 18.0, 361)
    gauge_xy = torch.tensor([[-3.2, -2.1], [0.8, -1.5], [2.7, 1.7], [-1.1, 2.2]])
    sample_xy = gauge_xy[torch.arange(time_s.numel()) % gauge_xy.shape[0]]
    positions = torch.cat((sample_xy, torch.ones(time_s.numel(), 1)), dim=-1)
    state = pool_effects.calculate_rectangular_pool_sloshing_state(
        positions,
        time_s,
        base_surface_z=1.0,
        pool_bounds=bounds,
        water_depth=4.0,
        mode_numbers=modes,
        amplitudes_m=[0.07, 0.03],
        phases_rad=[0.25, -0.4],
    )
    rows = [
        [time_s[index].item(), sample_xy[index, 0].item(), sample_xy[index, 1].item(), state.surface_z[index, 0].item()]
        for index in range(time_s.numel())
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "free_surface_wave_gauge.csv",
            ["time_s", "gauge_x_m", "gauge_y_m", "surface_z_m"],
            rows,
        )
        result = environment_fit_cli.fit_environment_calibration_logs(
            root,
            pool_bounds=[*bounds, 1.0, 5.0],
            surface_z=1.0,
            sloshing_modes=modes,
            sloshing_water_depth=4.0,
        )
    assert result.cfg_updates["free_surface_sloshing_enabled"] is True
    assert torch.allclose(
        torch.tensor(result.cfg_updates["free_surface_sloshing_amplitudes_m"]),
        torch.tensor([0.07, 0.03]),
        atol=1.0e-5,
    )
    assert result.diagnostics["free_surface_sloshing"]["design_rank"] == 4
    merged = profiles.merge_pool_dynamics_cfg_updates(cfg_updates=result.cfg_updates)
    assert merged.free_surface.sloshing_enabled is True


def test_environment_calibration_log_pipeline_builds_current_and_proximity_updates():
    alpha = 0.8
    time_s = torch.arange(8, dtype=torch.float32)
    powers = alpha ** torch.arange(len(time_s), dtype=torch.float32)
    mean_current = torch.tensor([0.1, -0.02, 0.01])
    current = mean_current.reshape(1, 3) + torch.stack((powers, -0.5 * powers, 0.25 * powers), dim=-1)
    field_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    field_currents = torch.stack(
        (field_positions[:, 0], field_positions[:, 1], torch.zeros(field_positions.shape[0])),
        dim=-1,
    )
    bounds = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
    boundary_positions = torch.tensor([[1.0, 5.0, 5.0], [0.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    boundary_damping, boundary_added_mass, boundary_thrust = pool_effects.calculate_pool_boundary_scales(
        boundary_positions,
        bounds=bounds,
        effect_distance=2.0,
        damping_scale_at_boundary=1.5,
        added_mass_scale_at_boundary=1.25,
        thrust_scale_at_boundary=0.8,
    )
    surface_positions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.75], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    surface_damping, surface_added_mass, surface_buoyancy, surface_thrust = (
        pool_effects.calculate_free_surface_scales(
            surface_positions,
            surface_z=1.0,
            effect_distance=0.5,
            heave_damping_scale_at_surface=1.6,
            roll_pitch_damping_scale_at_surface=1.25,
            added_mass_scale_at_surface=1.35,
            buoyancy_scale_at_surface=0.82,
            thrust_scale_at_surface=0.65,
            depth_axis_sign=-1.0,
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "water_current_timeseries.csv",
            ["time_s", "current_w_x_mps", "current_w_y_mps", "current_w_z_mps"],
            [[float(time_s[index]), *current[index].tolist()] for index in range(time_s.numel())],
        )
        _write_csv(
            root / "water_current_field_samples.csv",
            ["pos_x_m", "pos_y_m", "pos_z_m", "current_w_x_mps", "current_w_y_mps", "current_w_z_mps"],
            [
                [*field_positions[index].tolist(), *field_currents[index].tolist()]
                for index in range(field_positions.shape[0])
            ],
        )
        _write_csv(
            root / "pool_boundary_effect_samples.csv",
            ["pos_x_m", "pos_y_m", "pos_z_m", "damping_scale", "added_mass_scale", "thrust_scale"],
            [
                [
                    *boundary_positions[index].tolist(),
                    float(boundary_damping[index, 0]),
                    float(boundary_added_mass[index, 0]),
                    float(boundary_thrust[index, 0]),
                ]
                for index in range(boundary_positions.shape[0])
            ],
        )
        _write_csv(
            root / "free_surface_effect_samples.csv",
            [
                "pos_z_m",
                "heave_damping_scale",
                "roll_pitch_damping_scale",
                "added_mass_scale",
                "buoyancy_scale",
                "thrust_scale",
            ],
            [
                [
                    float(surface_positions[index, 2]),
                    float(surface_damping[index, 2]),
                    float(surface_damping[index, 3]),
                    float(surface_added_mass[index, 2]),
                    float(surface_buoyancy[index, 0]),
                    float(surface_thrust[index, 0]),
                ]
                for index in range(surface_positions.shape[0])
            ],
        )

        result = environment_fit_cli.fit_environment_calibration_logs(
            root,
            current_stage_count=2,
            current_grid_shape=(2, 2, 1),
            current_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            pool_bounds=bounds,
            boundary_effect_distance=2.0,
            surface_z=1.0,
            surface_effect_distance=0.5,
        )
        output_path = root / "environment_updates.json"
        report_path = root / "environment_report.json"
        exit_code = environment_fit_cli.main(
            [
                str(root),
                "--current-stages",
                "2",
                "--current-grid-shape",
                "2",
                "2",
                "1",
                "--current-bounds",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "--pool-bounds",
                *[str(value) for value in bounds],
                "--boundary-effect-distance",
                "2.0",
                "--surface-z",
                "1.0",
                "--surface-effect-distance",
                "0.5",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        profile = profiles.merge_pool_dynamics_cfg_updates(
            cfg_updates=output_updates,
            domain_randomization_updates=output_domain,
        )

    assert torch.allclose(torch.tensor(result.cfg_updates["water_current_w"]), torch.mean(current, dim=0))
    assert result.cfg_updates["water_current_field_shape"] == [2, 2, 1]
    assert len(result.cfg_updates["water_current_field_values"]) == 4
    reconstructed_field = (
        torch.tensor(result.cfg_updates["water_current_field_values"])
        + torch.tensor(result.cfg_updates["water_current_w"])
    )
    assert torch.allclose(reconstructed_field, field_currents, atol=1.0e-5)
    assert result.diagnostics["water_current_field"]["value_semantics"].startswith("spatial residual")
    assert abs(result.cfg_updates["pool_boundary_damping_scale"] - 1.5) < 1.0e-6
    assert abs(result.cfg_updates["pool_boundary_added_mass_scale"] - 1.25) < 1.0e-6
    assert abs(result.cfg_updates["pool_boundary_thrust_scale"] - 0.8) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_heave_damping_scale"] - 1.6) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_roll_pitch_damping_scale"] - 1.25) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_added_mass_scale"] - 1.35) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_buoyancy_scale"] - 0.82) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_thrust_scale"] - 0.65) < 1.0e-6
    assert len(result.domain_randomization_updates["water_current_max_by_stage"]) == 2
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == result.domain_randomization_updates
    assert output_domain["use_custom_randomization"] is True
    assert profile.pool_boundary.enabled is True
    assert profile.free_surface.enabled is True
    assert report["source_files"] == list(result.source_files)


def test_calibration_fits_tether_spring_damper_from_synthetic_samples():
    length = torch.tensor([1.8, 2.0, 2.2, 2.5, 3.0, 2.4])
    velocity_along_tether = torch.tensor([0.0, 0.0, 0.0, -0.2, -0.4, 0.3])
    slack = 2.0
    stiffness = 20.0
    damping = 5.0
    tension = stiffness * torch.clamp(length - slack, min=0.0) + damping * torch.clamp(
        -velocity_along_tether,
        min=0.0,
    )

    fit = calibration.fit_tether_spring_damper(
        length,
        tension,
        velocity_along_tether,
        slack_length_candidates=[1.8, 2.0, 2.2],
    )
    updates = fit.to_cfg_updates()

    assert abs(fit.slack_length - slack) < 1.0e-6
    assert abs(fit.stiffness - stiffness) < 3.0e-5
    assert abs(fit.damping - damping) < 3.0e-5
    assert fit.residual_rms < 1.0e-5
    assert updates["tether_enabled"] is True
    assert updates["tether_slack_length"] == fit.slack_length
    assert updates["tether_stiffness"] == fit.stiffness
    assert updates["tether_damping"] == fit.damping


def test_calibration_fits_tether_drag_coefficient_from_synthetic_samples():
    relative_velocity = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    drag_coeff = 0.25
    speed = torch.linalg.norm(relative_velocity, dim=-1, keepdim=True)
    drag_force = -drag_coeff * speed * relative_velocity

    fit = calibration.fit_tether_drag_coefficient(relative_velocity, drag_force)
    updates = fit.to_cfg_updates()

    assert abs(fit.drag_coeff - drag_coeff) < 1.0e-6
    assert fit.residual_rms < 1.0e-7
    assert fit.sample_count == 3
    assert updates["tether_enabled"] is True
    assert updates["tether_drag_coeff"] == fit.drag_coeff


def test_tether_calibration_log_pipeline_builds_multisegment_updates():
    length = torch.tensor([1.8, 2.0, 2.2, 2.5, 3.0, 2.4])
    velocity_along_tether = torch.tensor([0.0, 0.0, 0.0, -0.2, -0.4, 0.3])
    slack = 2.0
    stiffness = 20.0
    damping = 5.0
    tension = stiffness * torch.clamp(length - slack, min=0.0) + damping * torch.clamp(
        -velocity_along_tether,
        min=0.0,
    )
    relative_velocity = torch.tensor([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.5]])
    drag_coeff = 0.25
    drag_force = -drag_coeff * torch.linalg.norm(relative_velocity, dim=-1, keepdim=True) * relative_velocity

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "tether_tension_samples.csv",
            ["length_m", "tension_n", "velocity_along_tether_mps"],
            [
                [float(length[index]), float(tension[index]), float(velocity_along_tether[index])]
                for index in range(length.numel())
            ],
        )
        _write_csv(
            root / "tether_drag_samples.csv",
            [
                "relative_velocity_x_mps",
                "relative_velocity_y_mps",
                "relative_velocity_z_mps",
                "drag_force_x_n",
                "drag_force_y_n",
                "drag_force_z_n",
            ],
            [
                [*relative_velocity[index].tolist(), *drag_force[index].tolist()]
                for index in range(relative_velocity.shape[0])
            ],
        )

        result = tether_fit_cli.fit_tether_calibration_logs(
            root,
            anchor_pos_w=(1.0, 2.0, 3.0),
            attach_offset_b=(-0.25, 0.0, 0.0),
            num_segments=4,
            segment_diameter=0.006,
            segment_density=1200.0,
            segment_buoyancy_density=997.0,
            slack_length_candidates=(1.8, 2.0, 2.2),
            winch_enabled=True,
            winch_target_length=2.4,
            winch_reel_speed=0.3,
            winch_min_length=1.0,
            winch_max_length=3.0,
        )
        output_path = root / "tether_updates.json"
        report_path = root / "tether_report.json"
        exit_code = tether_fit_cli.main(
            [
                str(root),
                "--anchor-pos-w",
                "1",
                "2",
                "3",
                "--attach-offset-b",
                "-0.25",
                "0",
                "0",
                "--num-segments",
                "4",
                "--segment-diameter",
                "0.006",
                "--segment-density",
                "1200",
                "--slack-candidates",
                "1.8",
                "2.0",
                "2.2",
                "--enable-winch",
                "--winch-target-length",
                "2.4",
                "--winch-reel-speed",
                "0.3",
                "--winch-min-length",
                "1.0",
                "--winch-max-length",
                "3.0",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        merged = profiles.merge_pool_dynamics_cfg_updates(cfg_updates=output_updates)

    assert abs(result.cfg_updates["tether_slack_length"] - slack) < 1.0e-6
    assert abs(result.cfg_updates["tether_stiffness"] - stiffness) < 3.0e-5
    assert abs(result.cfg_updates["tether_damping"] - damping) < 3.0e-5
    assert abs(result.cfg_updates["tether_drag_coeff"] - drag_coeff) < 1.0e-6
    assert result.cfg_updates["tether_num_segments"] == 4
    assert result.cfg_updates["tether_segment_diameter"] == 0.006
    assert result.cfg_updates["tether_winch_enabled"] is True
    assert result.cfg_updates["tether_winch_target_length"] == 2.4
    assert result.cfg_updates["tether_winch_reel_speed"] == 0.3
    assert result.cfg_updates["tether_winch_min_length"] == 1.0
    assert result.cfg_updates["tether_winch_max_length"] == 3.0
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert merged.tether.enabled is True
    assert merged.tether.anchor_pos_w == [1.0, 2.0, 3.0]
    assert merged.tether.winch_enabled is True
    assert report["source_files"] == list(result.source_files)


def test_pool_boundary_scales_are_one_away_from_walls():
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    bounds = [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]

    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        positions,
        bounds,
        effect_distance=0.5,
        damping_scale_at_boundary=1.5,
        added_mass_scale_at_boundary=1.2,
        thrust_scale_at_boundary=0.8,
    )

    assert torch.allclose(damping, torch.ones(1, 1))
    assert torch.allclose(added_mass, torch.ones(1, 1))
    assert torch.allclose(thrust, torch.ones(1, 1))


def test_pool_boundary_scales_increase_near_boundary():
    positions = torch.tensor([[1.75, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bounds = [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]

    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        positions,
        bounds,
        effect_distance=0.5,
        damping_scale_at_boundary=1.5,
        added_mass_scale_at_boundary=1.2,
        thrust_scale_at_boundary=0.8,
    )

    assert torch.allclose(damping, torch.tensor([[1.25], [1.5]]))
    assert torch.allclose(added_mass, torch.tensor([[1.1], [1.2]]))
    assert torch.allclose(thrust, torch.tensor([[0.9], [0.8]]))


def test_free_surface_scales_affect_heave_roll_pitch_near_surface():
    positions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.25], [0.0, 0.0, 1.75], [0.0, 0.0, 0.0]]
    )

    damping, added_mass, buoyancy, thrust = pool_effects.calculate_free_surface_scales(
        positions,
        surface_z=1.0,
        effect_distance=0.5,
        heave_damping_scale_at_surface=1.4,
        roll_pitch_damping_scale_at_surface=1.2,
        added_mass_scale_at_surface=1.3,
        buoyancy_scale_at_surface=0.8,
        thrust_scale_at_surface=0.6,
    )

    assert torch.allclose(damping[0], torch.tensor([1.0, 1.0, 1.4, 1.2, 1.2, 1.0]))
    assert torch.allclose(added_mass[0], torch.tensor([1.0, 1.0, 1.3, 1.3, 1.3, 1.0]))
    assert torch.allclose(buoyancy[0], torch.tensor([0.8]))
    assert torch.allclose(thrust[0], torch.tensor([0.6]))
    assert torch.all(damping[1, [2, 3, 4]] > torch.ones(3))
    assert torch.allclose(damping[1, [0, 1, 5]], torch.ones(3))
    assert torch.allclose(damping[2], torch.ones(6))
    assert torch.allclose(buoyancy[2], torch.ones(1))
    # A vehicle above the surface must not regain the full underwater model.
    assert torch.allclose(buoyancy[3], torch.tensor([0.8]))
    assert torch.allclose(thrust[3], torch.tensor([0.6]))


def test_tether_wrench_is_zero_inside_slack_without_drag():
    force_b, torque_b = tether.calculate_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[1.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=2.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    assert torch.allclose(force_b, torch.zeros(1, 3))
    assert torch.allclose(torque_b, torch.zeros(1, 3))


def test_tether_wrench_pulls_toward_anchor_after_slack():
    force_b, torque_b = tether.calculate_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[3.0, 0.0, 0.0],
        attach_offset_b=[0.0, 1.0, 0.0],
        slack_length=1.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    expected_force = torch.tensor([[3.0, -1.0, 0.0]]) / torch.sqrt(torch.tensor(10.0)) * (torch.sqrt(torch.tensor(10.0)) - 1.0) * 10.0
    expected_torque = torch.cross(torch.tensor([[0.0, 1.0, 0.0]]), expected_force, dim=-1)
    assert torch.allclose(force_b, expected_force)
    assert torch.allclose(torque_b, expected_torque)


def test_tether_wrench_accepts_per_env_slack_lengths():
    force_b, _ = tether.calculate_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(2, 3),
        water_current_w=torch.zeros(2, 3),
        anchor_pos_w=[2.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=torch.tensor([[3.0], [1.0]]),
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    assert torch.allclose(force_b[0], torch.zeros(3))
    assert torch.allclose(force_b[1], torch.tensor([10.0, 0.0, 0.0]))


def test_tether_winch_slack_length_updates_with_rate_limit_and_bounds():
    current = torch.tensor([[1.0], [3.0]])

    updated = tether.update_rate_limited_winch_slack_length(
        current,
        target_slack_length=torch.tensor([[2.0], [1.0]]),
        reel_speed_mps=0.5,
        dt_s=1.0,
        min_length=0.75,
        max_length=3.5,
    )
    clamped = tether.update_rate_limited_winch_slack_length(
        current,
        target_slack_length=10.0,
        reel_speed_mps=10.0,
        dt_s=1.0,
        min_length=0.75,
        max_length=3.5,
    )

    assert torch.allclose(updated, torch.tensor([[1.5], [2.5]]))
    assert torch.allclose(clamped, torch.tensor([[3.5], [3.5]]))


def test_multisegment_tether_matches_single_segment_without_distributed_loads():
    common_args = dict(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[3.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=1.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )
    single_force, single_torque = tether.calculate_tether_wrench(**common_args)
    multi_force, multi_torque = tether.calculate_multisegment_tether_wrench(
        **common_args,
        num_segments=4,
        segment_diameter=0.01,
        segment_density=1000.0,
        segment_buoyancy_density=1000.0,
        gravity_w=[0.0, 0.0, -9.81],
    )

    assert torch.allclose(multi_force, single_force)
    assert torch.allclose(multi_torque, single_torque)


def test_multisegment_tether_adds_negative_buoyancy_load():
    force_b, torque_b = tether.calculate_multisegment_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[2.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=2.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        num_segments=4,
        segment_diameter=0.1,
        segment_density=1100.0,
        segment_buoyancy_density=1000.0,
        gravity_w=[0.0, 0.0, -10.0],
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    expected_weight = 0.5 * (1100.0 - 1000.0) * torch.pi * (0.05**2) * 2.0 * -10.0
    assert torch.allclose(force_b, torch.tensor([[0.0, 0.0, expected_weight]]), atol=1.0e-5)
    assert torch.allclose(torque_b, torch.zeros(1, 3), atol=1.0e-6)


def test_multisegment_tether_drag_opposes_relative_motion():
    force_b, _ = tether.calculate_multisegment_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.tensor([[2.0, 0.0, 0.0]]),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[0.0, 2.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=2.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.25,
        num_segments=4,
        segment_diameter=0.01,
        segment_density=1000.0,
        segment_buoyancy_density=1000.0,
        gravity_w=[0.0, 0.0, -9.81],
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    assert force_b[0, 0] < 0.0
    assert torch.allclose(force_b[:, 1:], torch.zeros(1, 2), atol=1.0e-6)

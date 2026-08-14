"""Shared AUV environment/workflow regression cases collected by domain tests.

This module is deliberately not named ``test_*.py``. Domain collectors under
Domain-aligned test modules expose every case to pytest without maintaining a handwritten
``__main__`` execution list.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from environment.hydrodynamics import current_fields, pool_effects
from environment.hydrodynamics import models as hydro
from environment.profiles import pool_profile as profiles
from environment.profiles.domain_randomization import load_domain_randomization_spec_json
from robot.dynamics import parameters as model_params
from robot.dynamics import rigid_body as rigid_body_properties
from robot.dynamics import tether
from robot.propulsion import thrusters


def _assert_raises(error_type, callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


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


def test_pool_dynamics_profile_rejects_bad_damping_speed_curve():
    bad_profile = profiles.PoolDynamicsProfile(
        hydrodynamics=profiles.HydrodynamicsProfile(
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 1.0],
            linear_damping_speed_scales=[1.0],
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_domain_randomization_profile_rejects_bad_water_current_parameters():
    bad_profile = profiles.DomainRandomizationProfile(
        water_current_tau_range=[0.0, 2.0],
        water_current_max_by_stage=[0.02, 0.04],
        water_current_vertical_max_by_stage=[0.01],
        disturbance_curriculum_stage_steps=[10],
    )

    _assert_raises(ValueError, bad_profile.validate)


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






def test_pool_dynamics_profile_rejects_unknown_json_fields():
    _assert_raises(
        ValueError,
        profiles.pool_dynamics_profile_from_dict,
        {"name": "bad-profile", "thrusters": {"not_a_thruster_parameter": 1.0}},
    )
















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
    response = thrusters.FirstOrderThrusterResponse(
        num_envs=1,
        num_thrusters=1,
        time_constant_s=0.1,
        device=torch.device("cpu"),
    )
    command = torch.tensor([[[10.0, -5.0, 2.0]]])
    response.advance(command, torch.tensor([0.0]))
    realized = response.advance(command, torch.tensor([0.1]))

    expected = (1.0 - torch.exp(torch.tensor(-1.0))) * command
    assert realized.shape == (1, 1, 3)
    assert torch.allclose(realized, expected)














def test_thruster_command_processor_applies_step_delay():
    processor = thrusters.ThrusterCommandProcessor(
        num_envs=1,
        num_thrusters=2,
        max_delay_steps=2,
        device=torch.device("cpu"),
    )
    delay_steps = torch.tensor([2])
    max_rate = torch.tensor([0.0])

    out_1 = processor.process(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_2 = processor.process(torch.tensor([[0.5, 0.5]]), delay_steps, max_rate, 0.1)
    out_3 = processor.process(torch.tensor([[0.0, 0.0]]), delay_steps, max_rate, 0.1)

    assert torch.allclose(out_1, torch.zeros(1, 2))
    assert torch.allclose(out_2, torch.zeros(1, 2))
    assert torch.allclose(out_3, torch.tensor([[1.0, -1.0]]))


def test_thruster_command_processor_applies_rate_limit():
    processor = thrusters.ThrusterCommandProcessor(
        num_envs=1,
        num_thrusters=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )
    delay_steps = torch.tensor([0])
    max_rate = torch.tensor([2.0])

    out_1 = processor.process(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_2 = processor.process(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_3 = processor.process(torch.tensor([[-1.0, 1.0]]), delay_steps, max_rate, 0.1)

    assert torch.allclose(out_1, torch.tensor([[0.2, -0.2]]))
    assert torch.allclose(out_2, torch.tensor([[0.4, -0.4]]))
    assert torch.allclose(out_3, torch.tensor([[0.2, -0.2]]))


def test_thruster_command_processor_broadcasts_per_env_rate_limit():
    processor = thrusters.ThrusterCommandProcessor(
        num_envs=2,
        num_thrusters=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )
    commands = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    delay_steps = torch.tensor([0, 0])
    max_rate = torch.tensor([[1.0], [3.0]])

    out = processor.process(commands, delay_steps, max_rate, 0.1)

    assert torch.allclose(out, torch.tensor([[0.1, -0.1], [0.3, -0.3]]))


def test_thruster_command_processor_quantizes_commands():
    processor = thrusters.ThrusterCommandProcessor(
        num_envs=1,
        num_thrusters=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )

    out = processor.process(
        torch.tensor([[0.26, -0.24]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        command_resolution=torch.tensor([0.1]),
    )

    assert torch.allclose(out, torch.tensor([[0.3, -0.2]]))


def test_thruster_command_processor_dropout_holds_previous_command():
    processor = thrusters.ThrusterCommandProcessor(
        num_envs=1,
        num_thrusters=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )

    out_1 = processor.process(
        torch.tensor([[0.8, -0.8]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        dropout_probability=torch.tensor([0.0]),
    )
    out_2 = processor.process(
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

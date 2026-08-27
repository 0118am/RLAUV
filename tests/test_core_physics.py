"""Minimal numerical regression checks for the experiment's shared physics."""

from __future__ import annotations

import pytest
import torch

from environment.hydrodynamics import current_fields, models, pool_effects
from simulation.dynamics import calculate_total_inertia_physx_wrench
from robot.dynamics import tether
from robot.propulsion import curves
from robot.propulsion.dynamics import FirstOrderThrusterResponse, ThrusterCommandProcessor
from robot.runtime import T60_RUNTIME
from robot.sensors import DelayedPoseSensor


def test_hydrodynamic_terms_obey_energy_contracts() -> None:
    model = models.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    relative_velocity = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.04, -0.02, 0.01],
            [-0.3, 0.2, -0.1, -0.03, 0.05, -0.02],
        ]
    )
    linear_damping = torch.tensor([0.00526, 0.00526, 0.00526, 0.00032, 0.00032, 0.00032])
    quadratic_damping = torch.tensor([39.196, 68.272, 135.402, 0.277, 1.387, 0.770])
    damping = model.calculate_relative_damping_wrench(
        relative_velocity,
        linear_damping,
        quadratic_damping,
    )
    assert torch.all(torch.sum(relative_velocity * damping, dim=-1) <= 0.0)

    fluid_added_mass = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    coriolis = model.calculate_fluid_added_mass_coriolis_wrench(
        relative_velocity, fluid_added_mass
    )
    assert torch.allclose(
        torch.sum(relative_velocity * coriolis, dim=-1),
        torch.zeros(2),
        atol=1.0e-7,
    )


def test_full_hydrodynamic_matrices_use_relative_velocity_and_cross_dof_terms() -> None:
    model = models.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    body_linear_velocity = torch.tensor([[0.30, -0.20, 0.10]])
    body_angular_velocity = torch.tensor([[0.04, -0.03, 0.02]])
    current_w = torch.tensor([[0.10, -0.05, 0.02]])
    relative_velocity = model.calculate_relative_velocity(
        identity,
        body_linear_velocity,
        body_angular_velocity,
        current_w,
    )
    assert torch.allclose(
        relative_velocity,
        torch.tensor([[0.20, -0.15, 0.08, 0.04, -0.03, 0.02]]),
    )

    linear = torch.eye(6).unsqueeze(0)
    linear[0, 0, 2] = 2.5
    quadratic = (2.0 * torch.eye(6)).unsqueeze(0)
    quadratic[0, 4, 0] = -0.75
    damping = model.calculate_relative_damping_wrench(
        relative_velocity,
        linear,
        quadratic,
    )
    expected_damping = -(
        torch.bmm(linear, relative_velocity.unsqueeze(-1)).squeeze(-1)
        + torch.bmm(
            quadratic,
            (relative_velocity.abs() * relative_velocity).unsqueeze(-1),
        ).squeeze(-1)
    )
    assert torch.allclose(damping, expected_damping)



def test_total_inertia_solve_maps_exactly_to_physx_rigid_wrench() -> None:
    dtype = torch.float64
    external = torch.tensor([[3.0, -2.0, 4.0, 0.3, -0.2, 0.1]], dtype=dtype)
    velocity = torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.1, 0.2]], dtype=dtype)
    gravity = torch.tensor([[0.0, 0.0, -110.0]], dtype=dtype)
    mass = torch.tensor([[11.0]], dtype=dtype)
    inertia = torch.tensor(
        [[[0.12, -0.001, 0.002], [-0.001, 0.20, 0.0], [0.002, 0.0, 0.26]]],
        dtype=dtype,
    )
    fluid_added_mass = torch.diag(
        torch.tensor([14.0, 20.0, 35.0, 0.14, 0.42, 0.16], dtype=dtype)
    ).unsqueeze(0)
    fluid_added_mass[0, 0, 2] = fluid_added_mass[0, 2, 0] = 0.4
    fluid_added_mass[0, 1, 3] = fluid_added_mass[0, 3, 1] = 0.03
    current_acceleration = torch.tensor(
        [[0.02, -0.01, 0.005, 0.0, 0.0, 0.0]], dtype=dtype
    )

    physx_wrench, acceleration = calculate_total_inertia_physx_wrench(
        external,
        velocity,
        gravity,
        mass,
        inertia,
        fluid_added_mass,
        current_acceleration,
    )

    rigid_mass = torch.zeros((1, 6, 6), dtype=dtype)
    rigid_mass[:, :3, :3] = mass.reshape(1, 1, 1) * torch.eye(3, dtype=dtype)
    rigid_mass[:, 3:, 3:] = inertia
    omega = velocity[:, 3:]
    rigid_coriolis = torch.cat(
        (
            torch.cross(omega, mass * velocity[:, :3], dim=-1),
            torch.cross(omega, torch.bmm(inertia, omega.unsqueeze(-1)).squeeze(-1), dim=-1),
        ),
        dim=-1,
    )
    gravity_wrench = torch.cat((gravity, torch.zeros_like(gravity)), dim=-1)

    total_equation_left = torch.bmm(
        rigid_mass + fluid_added_mass, acceleration.unsqueeze(-1)
    ).squeeze(-1)
    total_equation_right = (
        external
        + gravity_wrench
        - rigid_coriolis
        + torch.bmm(fluid_added_mass, current_acceleration.unsqueeze(-1)).squeeze(-1)
    )
    physx_equation_left = (
        torch.bmm(rigid_mass, acceleration.unsqueeze(-1)).squeeze(-1) + rigid_coriolis
    )
    physx_equation_right = physx_wrench + gravity_wrench

    assert torch.allclose(total_equation_left, total_equation_right, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(physx_equation_left, physx_equation_right, atol=1.0e-12, rtol=0.0)

    no_fluid_added_mass_wrench, _ = calculate_total_inertia_physx_wrench(
        external,
        velocity,
        gravity,
        mass,
        inertia,
        torch.zeros_like(fluid_added_mass),
        torch.zeros_like(current_acceleration),
    )
    assert torch.allclose(
        no_fluid_added_mass_wrench, external, atol=1.0e-12, rtol=0.0
    )


def test_z_up_buoyancy_and_surface_conventions_are_consistent() -> None:
    model = models.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    gravity = torch.tensor([0.0, 0.0, -9.81])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    volume = torch.tensor([[0.01]])
    cob = torch.tensor([[0.0, 0.0, 0.02]])

    buoyancy, torque = model.calculate_buoyancy_forces(identity, gravity, 1000.0, volume, cob)
    assert torch.allclose(buoyancy, torch.tensor([[0.0, 0.0, 98.1]]), atol=1.0e-5)
    assert torch.allclose(buoyancy + 10.0 * gravity, torch.zeros((1, 3)), atol=1.0e-5)
    assert torch.allclose(torque, torch.zeros_like(torque))

    _, _, surface_buoyancy, _ = pool_effects.calculate_free_surface_scales(
        torch.tensor([[0.0, 0.0, 0.75]]),
        0.75,
        0.5,
        1.4,
        1.2,
        1.15,
        0.95,
        0.9,
    )
    assert torch.allclose(surface_buoyancy, torch.tensor([[0.95]]))


def test_pool_wall_clearance_uses_body_envelope_and_excludes_free_surface() -> None:
    bounds = [0.0, 20.0, 0.0, 20.0, 0.0, 100.0]
    half_extents = torch.tensor([[0.3, 0.2, 0.1]])
    near_open_surface = torch.tensor([[10.0, 10.0, 99.8]])
    scales = pool_effects.calculate_pool_boundary_scales(
        near_open_surface,
        half_extents,
        bounds,
        1.0,
        1.5,
        1.2,
        0.8,
    )
    assert all(torch.equal(scale, torch.ones_like(scale)) for scale in scales)

    near_bottom = torch.tensor([[10.0, 10.0, 0.2]])
    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        near_bottom,
        half_extents,
        bounds,
        1.0,
        1.5,
        1.2,
        0.8,
    )
    assert damping.item() == pytest.approx(1.45)
    assert added_mass.item() == pytest.approx(1.18)
    assert thrust.item() == pytest.approx(0.82)


def test_measured_positive_buoyancy_is_applied_without_nominal_scaling() -> None:
    vehicle = T60_RUNTIME.model
    fluid_density = 1000.0
    model = models.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    gravity = torch.tensor([0.0, 0.0, -9.81])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    volume = torch.tensor([[vehicle.displaced_volume_m3]])
    cob = torch.tensor([vehicle.center_of_buoyancy_from_com_m])

    buoyancy, _ = model.calculate_buoyancy_forces(
        identity,
        gravity,
        fluid_density,
        volume,
        cob,
    )
    net_force = buoyancy + vehicle.mass_kg * gravity

    buoyant_mass_surplus = (
        fluid_density * vehicle.displaced_volume_m3 - vehicle.mass_kg
    )
    assert abs(buoyant_mass_surplus - 0.24) < 1.0e-12
    assert torch.allclose(net_force, torch.tensor([[0.0, 0.0, 2.3544]]), atol=1.0e-5)


def test_thruster_curve_preserves_deadband_and_wrench_geometry() -> None:
    commands = torch.tensor([[-0.1] * 8, [0.1] * 8, [1.0] * 8], dtype=torch.float64)
    forces = curves.measured_thruster_body_forces(commands)
    assert torch.equal(forces[:2], torch.zeros((2, 8, 3), dtype=torch.float64))
    assert torch.any(forces[2] != 0.0)

    positions = curves.get_thruster_positions(torch.device("cpu"), torch.float64)
    wrench = curves.reduce_point_forces_to_wrench(positions, forces)
    expected = torch.cat(
        (
            forces.sum(dim=-2),
            torch.cross(positions.unsqueeze(0), forces, dim=-1).sum(dim=-2),
        ),
        dim=-1,
    )
    assert torch.allclose(wrench, expected)


def test_thruster_positions_preserve_verified_com_relative_channel_order() -> None:
    expected_m = torch.tensor(
        [
            [0.13400, -0.16000, -0.17098],   # T1 vertical/front-right
            [-0.15000, -0.16000, -0.17098],  # T2 vertical/rear-right
            [0.13400, 0.16000, -0.17098],    # T3 vertical/front-left
            [-0.15000, 0.16000, -0.17098],   # T4 vertical/rear-left
            [-0.15039, 0.10360, -0.06312],   # T5 horizontal/rear-left
            [-0.15039, -0.10360, -0.06312],  # T6 horizontal/rear-right
            [0.13439, 0.10360, -0.06312],    # T7 horizontal/front-left
            [0.13439, -0.10360, -0.06312],   # T8 horizontal/front-right
        ],
        dtype=torch.float64,
    )

    actual_m = curves.get_thruster_positions(torch.device("cpu"), torch.float64)
    assert torch.equal(actual_m, expected_m)


def test_pwm_branches_select_complete_installed_flu_vectors_without_sign_flip() -> None:
    pwm = torch.full((1, 8), 1700.0, dtype=torch.float64)
    forces = curves.thruster_body_forces_from_pwm_us(pwm)[0]
    coefficients = torch.tensor(
        T60_RUNTIME.model.thruster_force_curve_coefficients,
        dtype=torch.float64,
    )
    q = 1700.0 - T60_RUNTIME.model.thruster_pwm_center_us - T60_RUNTIME.model.thruster_pwm_deadband_us

    # Positive physical PWM selects (a_positive, b_positive) directly for all
    # thrusters. The installed orientation is already part of each FLU vector.
    expected_positive_flu = coefficients[:, 0, :] * q**2 + coefficients[:, 1, :] * q
    assert torch.allclose(forces, expected_positive_flu)
    assert torch.all(forces[:4, 2] < 0.0)
    assert torch.all(forces[4:6, 0] < 0.0)  # T5/T6: forward rotation, installed toward F-
    assert torch.all(forces[6:8, 0] > 0.0)  # T7/T8: forward, force toward +F

    # Mirrored installations retain equal F/Z and mirrored L components.
    assert torch.allclose(forces[4, [0, 2]], forces[5, [0, 2]])
    assert torch.allclose(forces[4, 1], -forces[5, 1])
    assert torch.allclose(forces[6, [0, 2]], forces[7, [0, 2]])
    assert torch.allclose(forces[6, 1], -forces[7, 1])

    negative_pwm = torch.full((1, 8), 1300.0, dtype=torch.float64)
    negative_forces = curves.thruster_body_forces_from_pwm_us(negative_pwm)[0]
    expected_negative_flu = coefficients[:, 2, :] * q**2 + coefficients[:, 3, :] * q
    assert torch.allclose(negative_forces, expected_negative_flu)
    assert torch.all(negative_forces[:4, 2] > 0.0)


def test_installed_curve_jacobian_matches_all_three_flu_components() -> None:
    commands = torch.full((1, 8), 0.8, dtype=torch.float64)
    epsilon = 1.0e-5
    numerical = (
        curves.measured_thruster_body_forces(commands + epsilon)
        - curves.measured_thruster_body_forces(commands - epsilon)
    ) / (2.0 * epsilon)
    analytical = curves.measured_thruster_force_jacobian(commands)
    assert torch.allclose(analytical, numerical, rtol=1.0e-6, atol=1.0e-6)


def test_nominal_thruster_response_is_80_ms_without_command_delay() -> None:
    assert T60_RUNTIME.thruster_time_constant_s == 0.08

    processor = ThrusterCommandProcessor(1, 8, torch.device("cpu"))
    commands = torch.ones((1, 8))
    processed = processor.process(commands)
    assert torch.equal(processed, commands)

    saturator = ThrusterCommandProcessor(1, 8, torch.device("cpu"))
    out_of_range = torch.tensor([[-2.0, 2.0] * 4])
    assert torch.equal(
        saturator.process(out_of_range),
        torch.tensor([[-1.0, 1.0] * 4]),
    )

    response = FirstOrderThrusterResponse(
        1,
        8,
        T60_RUNTIME.thruster_time_constant_s,
        torch.device("cpu"),
    )
    target = torch.ones((1, 8, 3))
    assert torch.equal(response.advance(target, 0.0), torch.zeros_like(target))
    realized = response.advance(target, T60_RUNTIME.thruster_time_constant_s)
    assert torch.allclose(realized, torch.full_like(target, 1.0 - torch.exp(torch.tensor(-1.0))))

    instantaneous = FirstOrderThrusterResponse(1, 8, 0.0, torch.device("cpu"))
    assert torch.equal(instantaneous.advance(target, 0.0), target)


def test_pose_sensor_applies_exact_50_ms_delay_on_the_100_hz_truth_grid() -> None:
    sensor = DelayedPoseSensor(1, 5, torch.device("cpu"))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zeros = torch.zeros((1, 3))
    sensor.reset(torch.tensor([0]), zeros, identity, zeros, zeros)

    for step in range(5):
        value = torch.tensor([[float(step), 0.0, 0.0]])
        sensor.record(value, identity, value, -value)
    measurement = sensor.measure()
    assert torch.equal(measurement.position_w, torch.tensor([[0.0, 0.0, 0.0]]))
    assert torch.equal(measurement.linear_velocity_b, torch.tensor([[0.0, 0.0, 0.0]]))

    value = torch.tensor([[5.0, 0.0, 0.0]])
    sensor.record(value, identity, value, -value)
    measurement = sensor.measure()
    assert torch.equal(measurement.position_w, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(measurement.linear_velocity_b, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(measurement.angular_velocity_b, torch.tensor([[-1.0, 0.0, 0.0]]))


def test_pose_sensor_returns_exact_state_without_consuming_random_numbers() -> None:
    count = 8
    sensor = DelayedPoseSensor(count, 0, torch.device("cpu"))
    positions = torch.randn((count, 3))
    identity = torch.zeros((count, 4))
    identity[:, 0] = 1.0
    velocities = torch.randn((count, 3))
    sensor.record(positions, identity, velocities, -velocities)

    random_state = torch.random.get_rng_state()
    measurement = sensor.measure()
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert torch.equal(measurement.position_w, positions)
    assert torch.equal(measurement.quaternion_wxyz, identity)
    assert torch.equal(measurement.linear_velocity_b, velocities)
    assert torch.equal(measurement.angular_velocity_b, -velocities)


def test_current_field_interpolation_and_periodic_current() -> None:
    grid_values = [
        [float(ix), float(iy), float(iz)]
        for ix in range(2)
        for iy in range(2)
        for iz in range(2)
    ]
    interpolated = current_fields.calculate_trilinear_current_field(
        positions=torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]),
        bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        grid_shape=[2, 2, 2],
        grid_values=grid_values,
    )
    assert torch.allclose(interpolated, torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]))

    periodic = current_fields.calculate_periodic_water_current(
        torch.tensor([0.0, 1.0]),
        amplitude_w=torch.tensor([0.04, 0.02, 0.01]),
        period_s=torch.tensor([4.0, 4.0, 2.0]),
        phase_rad=torch.tensor([0.0, torch.pi / 2.0, 0.0]),
    )
    assert torch.allclose(
        periodic,
        torch.tensor([[0.0, 0.02, 0.0], [0.04, 0.0, 0.0]]),
        atol=1.0e-6,
    )


def test_tether_is_slack_then_pulls_toward_its_anchor() -> None:
    common = dict(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        attach_offset_b=[0.0, 0.0, 0.0],
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=models.quat_conjugate_wxyz,
        quat_apply_fn=models.quat_apply_wxyz,
    )
    slack_force, _ = tether.calculate_tether_wrench(
        anchor_pos_w=[1.0, 0.0, 0.0],
        slack_length=2.0,
        **common,
    )
    taut_force, _ = tether.calculate_tether_wrench(
        anchor_pos_w=[3.0, 0.0, 0.0],
        slack_length=1.0,
        **common,
    )
    assert torch.equal(slack_force, torch.zeros_like(slack_force))
    assert taut_force[0, 0] > 0.0
    assert torch.equal(taut_force[0, 1:], torch.zeros(2))

"""Minimal numerical regression checks for the experiment's shared physics."""

from __future__ import annotations

import torch

from environment.hydrodynamics import current_fields, models, pool_effects
from robot.dynamics import tether
from robot.propulsion import curves
from robot.propulsion.dynamics import FirstOrderThrusterResponse, ThrusterCommandProcessor
from robot.runtime import T60_RUNTIME


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

    added_mass = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    coriolis = model.calculate_added_mass_coriolis_wrench(relative_velocity, added_mass)
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

    added_mass = torch.eye(6).unsqueeze(0)
    added_mass[0, 1, 5] = 0.4
    relative_acceleration = torch.tensor([[0.2, -0.1, 0.3, 0.02, -0.04, 0.06]])
    assert torch.allclose(
        model.calculate_added_mass_inertia_wrench(relative_acceleration, added_mass),
        -torch.bmm(added_mass, relative_acceleration.unsqueeze(-1)).squeeze(-1),
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
        torch.tensor([[0.0, 0.0, -1.0]]),
        -1.0,
        0.5,
        1.4,
        1.2,
        1.15,
        0.95,
        0.9,
    )
    assert torch.allclose(surface_buoyancy, torch.tensor([[0.95]]))


def test_thruster_curve_preserves_deadband_and_wrench_geometry() -> None:
    commands = torch.tensor([[-0.125] * 8, [0.125] * 8, [1.0] * 8], dtype=torch.float64)
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


def test_horizontal_positive_pwm_selects_complete_installed_flu_branch() -> None:
    pwm = torch.full((1, 8), 1700.0, dtype=torch.float64)
    forces = curves.thruster_body_forces_from_pwm_us(pwm)[0]
    coefficients = torch.tensor(
        T60_RUNTIME.model.thruster_force_curve_coefficients,
        dtype=torch.float64,
    )
    q = 1700.0 - T60_RUNTIME.model.thruster_pwm_center_us - T60_RUNTIME.model.thruster_pwm_deadband_us

    # Positive physical PWM selects each normalized positive branch. Preserve
    # its Fx/Fy/Fz vector as a unit.
    expected_horizontal_flu = coefficients[4:, 2, :] * q**2 + coefficients[4:, 3, :] * q
    assert torch.allclose(forces[4:], expected_horizontal_flu)
    assert torch.all(forces[4:6, 0] < 0.0)  # T5/T6: forward rotation, installed toward F-
    assert torch.all(forces[6:8, 0] > 0.0)  # T7/T8: forward, force toward +F

    # Mirrored installations retain equal F/Z and mirrored L components.
    assert torch.allclose(forces[4, [0, 2]], forces[5, [0, 2]])
    assert torch.allclose(forces[4, 1], -forces[5, 1])
    assert torch.allclose(forces[6, [0, 2]], forces[7, [0, 2]])
    assert torch.allclose(forces[6, 1], -forces[7, 1])

    negative_pwm = torch.full((1, 8), 1300.0, dtype=torch.float64)
    negative_forces = curves.thruster_body_forces_from_pwm_us(negative_pwm)[0]
    expected_negative_horizontal_flu = coefficients[4:, 0, :] * q**2 + coefficients[4:, 1, :] * q
    assert torch.allclose(negative_forces[4:], expected_negative_horizontal_flu)


def test_installed_curve_jacobian_matches_all_three_flu_components() -> None:
    commands = torch.full((1, 8), 0.8, dtype=torch.float64)
    epsilon = 1.0e-5
    numerical = (
        curves.measured_thruster_body_forces(commands + epsilon)
        - curves.measured_thruster_body_forces(commands - epsilon)
    ) / (2.0 * epsilon)
    analytical = curves.measured_thruster_force_jacobian(commands)
    assert torch.allclose(analytical, numerical, rtol=1.0e-6, atol=1.0e-6)


def test_nominal_thruster_response_and_delay_are_explicit_and_applied() -> None:
    physics_dt_s = 1.0 / 200.0
    delay_steps = T60_RUNTIME.thruster_command_delay_steps_for_dt(physics_dt_s)
    assert T60_RUNTIME.thruster_time_constant_s == 0.08
    assert T60_RUNTIME.thruster_command_delay_s == 0.13
    assert delay_steps == 26

    processor = ThrusterCommandProcessor(1, 8, delay_steps, torch.device("cpu"))
    commands = torch.ones((1, 8))
    for _ in range(delay_steps):
        applied = processor.process(commands, delay_steps, 0.0, physics_dt_s)
        assert torch.equal(applied, torch.zeros_like(applied))
    applied = processor.process(commands, delay_steps, 0.0, physics_dt_s)
    assert torch.equal(applied, commands)

    limiter = ThrusterCommandProcessor(1, 8, 0, torch.device("cpu"))
    out_of_range = torch.tensor([[-2.0, 2.0] * 4])
    assert torch.equal(
        limiter.process(out_of_range, 0, 0.0, physics_dt_s),
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

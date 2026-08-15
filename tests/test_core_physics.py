"""Minimal numerical regression checks for the experiment's shared physics."""

from __future__ import annotations

import torch

from environment.hydrodynamics import current_fields, models, pool_effects
from robot.dynamics import tether
from robot.propulsion import curves


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

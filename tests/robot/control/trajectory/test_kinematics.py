"""Minimal checks for trajectory generation and retiming."""

from __future__ import annotations

import torch

from robot.control import trajectory as kinematics


def _parameters(count: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    return {
        "trajectory_type": torch.arange(count, dtype=torch.long) % 8,
        "axis": torch.randint(0, 3, (count,), dtype=torch.long, generator=generator),
        "amp_x": 0.60 + 0.18 * torch.rand(count, generator=generator),
        "amp_y": 0.55 + 0.20 * torch.rand(count, generator=generator),
        "amp_z": 0.08 + 0.12 * torch.rand(count, generator=generator),
        "requested_period_s": 10.0 + 10.0 * torch.rand(count, generator=generator),
        "phase_x": 2.0 * torch.pi * torch.rand(count, generator=generator),
        "phase_y": 2.0 * torch.pi * torch.rand(count, generator=generator),
    }


def _tables(parameters: dict[str, torch.Tensor]):
    return kinematics.build_retimed_tables(
        **parameters,
        radius_min=0.20,
        radius_max=0.75,
        chirp_rate=2.0,
        harmonic_ratio=0.08,
        limits=kinematics.TrajectoryKinematicLimits(),
    )


def _reference(parameters, tables, elapsed):
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(tables, elapsed)
    return kinematics.evaluate_retimed_reference(
        parameters["trajectory_type"],
        parameters["axis"],
        parameters["amp_x"],
        parameters["amp_y"],
        parameters["amp_z"],
        parameters["phase_x"],
        parameters["phase_y"],
        phase,
        phase_rate,
        phase_acceleration,
        radius_min=0.20,
        radius_max=0.75,
        harmonic_ratio=0.08,
    )


def test_smooth_startup_begins_at_rest_and_joins_unit_speed() -> None:
    trajectory_time, speed_scale, speed_scale_rate = kinematics.smooth_startup_time(
        torch.tensor([0.0, 2.0, 4.0, 5.0], dtype=torch.float64),
        4.0,
    )
    assert torch.allclose(
        trajectory_time,
        torch.tensor([0.0, 0.3125, 2.0, 3.0], dtype=torch.float64),
    )
    assert torch.allclose(speed_scale, torch.tensor([0.0, 0.5, 1.0, 1.0], dtype=torch.float64))
    assert torch.equal(speed_scale_rate[[0, 2, 3]], torch.zeros(3, dtype=torch.float64))


def test_retimed_references_are_finite_and_within_motion_limits() -> None:
    parameters = _parameters(32)
    tables = _tables(parameters)
    fractions = torch.arange(129, dtype=torch.float32).reshape(1, -1) / 129.0
    elapsed = fractions * tables.effective_period_s.reshape(-1, 1)
    _, velocity, acceleration, curvature = _reference(parameters, tables, elapsed)
    limits = kinematics.TrajectoryKinematicLimits()

    assert torch.isfinite(curvature).all()
    assert torch.linalg.vector_norm(velocity, dim=-1).max() <= limits.max_speed_mps * 1.01
    assert torch.linalg.vector_norm(acceleration, dim=-1).max() <= limits.max_acceleration_mps2 * 1.01
    assert torch.all(tables.effective_period_s >= parameters["requested_period_s"] - 1.0e-5)


def test_periodic_reference_is_continuous_at_wrap() -> None:
    parameters = _parameters()
    tables = _tables(parameters)
    before = _reference(parameters, tables, tables.effective_period_s - 1.0e-4)
    after = _reference(parameters, tables, torch.full_like(tables.effective_period_s, 1.0e-4))

    for before_value, after_value in zip(before[:3], after[:3], strict=True):
        assert torch.allclose(before_value, after_value, atol=2.0e-3, rtol=2.0e-3)


def test_nominal_trajectory_envelope_fits_target_pool() -> None:
    trajectory_types = torch.tensor(
        [
            kinematics.CIRCLE,
            kinematics.LISSAJOUS,
            kinematics.AXIS_SINE,
            kinematics.AXIS_SINE,
            kinematics.AXIS_SINE,
            kinematics.WAVY_LOOP,
            kinematics.BREATHING_LOOP,
            kinematics.CHIRP,
            kinematics.RACETRACK,
            kinematics.RANDOM_SMOOTH,
            kinematics.LATERAL_SINE,
            kinematics.VERTICAL_SINE,
            kinematics.SPATIAL_HELIX,
        ],
        dtype=torch.long,
    )
    count = trajectory_types.numel()
    phase = torch.linspace(0.0, 2.0 * torch.pi, 4097).repeat(count, 1)
    offsets = kinematics.evaluate_geometry(
        trajectory_types,
        torch.tensor([0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
        torch.full((count,), 0.78),
        torch.full((count,), 0.75),
        torch.full((count,), 0.20),
        torch.zeros(count),
        torch.zeros(count),
        phase,
        radius_min=0.30,
        radius_max=1.20,
        harmonic_ratio=0.08,
    )
    targets = offsets + torch.tensor([2.5, 1.75, 0.375])
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([5.0, 3.5, 0.75])

    assert torch.all(targets > lower)
    assert torch.all(targets < upper)

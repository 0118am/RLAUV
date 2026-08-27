"""Minimal checks for trajectory generation and retiming."""

from __future__ import annotations

import torch

from robot.control import trajectory as kinematics
from robot.control.trajectory import guidance


def _limits() -> kinematics.TrajectoryKinematicLimits:
    return kinematics.TrajectoryKinematicLimits(
        max_speed_mps=0.60,
        max_acceleration_mps2=0.45,
        max_yaw_rate_radps=0.80,
        max_jerk_mps3=0.36,
        retime_samples=256,
    )


def _parameters(count: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    return {
        "trajectory_type": torch.arange(count, dtype=torch.long) % 8,
        "axis": torch.randint(0, 3, (count,), dtype=torch.long, generator=generator),
        "wave_count": torch.ones(count, dtype=torch.long),
        "amp_x": 0.60 + 0.18 * torch.rand(count, generator=generator),
        "amp_y": 0.55 + 0.20 * torch.rand(count, generator=generator),
        "amp_z": 0.08 + 0.12 * torch.rand(count, generator=generator),
        "requested_period_s": 10.0 + 10.0 * torch.rand(count, generator=generator),
        "phase_x": 2.0 * torch.pi * torch.rand(count, generator=generator),
        "phase_y": 2.0 * torch.pi * torch.rand(count, generator=generator),
        "target_speed_mps": torch.full((count,), 0.20),
    }


def _tables(parameters: dict[str, torch.Tensor]):
    return kinematics.build_retimed_tables(
        **parameters,
        radius_min=0.20,
        radius_max=0.75,
        chirp_rate=2.0,
        harmonic_ratio=0.08,
        limits=_limits(),
    )


def _reference(parameters, tables, elapsed):
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(tables, elapsed)
    return kinematics.evaluate_retimed_reference(
        parameters["trajectory_type"],
        parameters["axis"],
        parameters["wave_count"],
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
    limits = _limits()

    assert torch.isfinite(curvature).all()
    assert torch.linalg.vector_norm(velocity, dim=-1).max() <= limits.max_speed_mps * 1.01
    assert torch.linalg.vector_norm(acceleration, dim=-1).max() <= limits.max_acceleration_mps2 * 1.01
    assert torch.all(tables.effective_period_s > 0.0)


def test_periodic_reference_is_continuous_at_wrap() -> None:
    parameters = _parameters()
    tables = _tables(parameters)
    before = _reference(parameters, tables, tables.effective_period_s - 1.0e-4)
    after = _reference(parameters, tables, torch.full_like(tables.effective_period_s, 1.0e-4))

    for before_value, after_value in zip(before[:3], after[:3], strict=True):
        assert torch.allclose(before_value, after_value, atol=2.0e-3, rtol=2.0e-3)


def test_reverse_spatial_helix_traverses_the_same_curve_backward() -> None:
    sample_phase = torch.linspace(0.0, 2.0 * torch.pi, 257)
    phase = torch.stack((-sample_phase, sample_phase))
    trajectory_type = torch.tensor(
        [kinematics.SPATIAL_HELIX, kinematics.REVERSE_SPATIAL_HELIX],
        dtype=torch.long,
    )
    phase_rate = torch.full_like(phase, 0.2)
    position, velocity, acceleration, curvature = (
        kinematics.evaluate_retimed_reference(
            trajectory_type,
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.full((2,), 2.5),
            torch.full((2,), 1.5),
            torch.full((2,), 0.5),
            torch.zeros(2),
            torch.zeros(2),
            phase,
            phase_rate,
            torch.zeros_like(phase),
            radius_min=0.30,
            radius_max=1.20,
            harmonic_ratio=0.08,
        )
    )

    assert torch.allclose(position[0], position[1], atol=1.0e-6)
    assert torch.allclose(velocity[0], -velocity[1], atol=1.0e-6)
    assert torch.allclose(acceleration[0], acceleration[1], atol=1.0e-6)
    assert torch.allclose(curvature[0], curvature[1], atol=1.0e-6)
    signed_heading_rate = (
        velocity[:, :, 0] * acceleration[:, :, 1]
        - velocity[:, :, 1] * acceleration[:, :, 0]
    ) / velocity[:, :, :2].square().sum(dim=-1).clamp_min(1.0e-8)
    assert torch.all(signed_heading_rate[0] > 0.0)
    assert torch.all(signed_heading_rate[1] < 0.0)


def test_speed_controlled_training_commands_respect_discrete_orientation_rate() -> None:
    kinds = (
        kinematics.AXIS_SINE,
        kinematics.LATERAL_WAVE,
        kinematics.VERTICAL_WAVE,
        kinematics.SPATIAL_HELIX,
        kinematics.REVERSE_SPATIAL_HELIX,
    )
    trajectory_type = torch.tensor(
        [kind for kind in kinds for _ in range(4)],
        dtype=torch.long,
    )
    target_speed = torch.tensor([0.1, 0.2, 0.3, 0.4] * len(kinds))
    count = trajectory_type.numel()
    axis_sine = trajectory_type == kinematics.AXIS_SINE
    lateral = trajectory_type == kinematics.LATERAL_WAVE
    vertical = trajectory_type == kinematics.VERTICAL_WAVE
    wave = lateral | vertical
    helix = (trajectory_type == kinematics.SPATIAL_HELIX) | (
        trajectory_type == kinematics.REVERSE_SPATIAL_HELIX
    )
    zeros = torch.zeros(count)
    parameters = {
        "trajectory_type": trajectory_type,
        "axis": torch.zeros(count, dtype=torch.long),
        "wave_count": torch.ones(count, dtype=torch.long),
        "amp_x": torch.where(axis_sine | wave | helix, torch.full((count,), 0.75), zeros),
        "amp_y": torch.where(wave | helix, torch.full((count,), 0.65), zeros),
        "amp_z": torch.where(vertical, torch.full((count,), 0.20), torch.where(helix, torch.full((count,), 0.16), zeros)),
        "requested_period_s": torch.full((count,), 10.0),
        "phase_x": zeros,
        "phase_y": zeros,
        "target_speed_mps": target_speed,
    }
    limits = _limits()
    tables = kinematics.build_retimed_tables(
        **parameters,
        radius_min=0.30,
        radius_max=1.20,
        chirp_rate=1.6,
        harmonic_ratio=0.08,
        limits=limits,
    )

    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    for index in range(count):
        elapsed = torch.arange(0.0, float(tables.effective_period_s[index]), 0.02)
        one = kinematics.RetimedTrajectoryTables(
            phase=tables.phase[index : index + 1],
            elapsed_s=tables.elapsed_s[index : index + 1],
            phase_rate=tables.phase_rate[index : index + 1],
            phase_acceleration=tables.phase_acceleration[index : index + 1],
            requested_period_s=tables.requested_period_s[index : index + 1],
            effective_period_s=tables.effective_period_s[index : index + 1],
            retimed=tables.retimed[index : index + 1],
        )
        phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(
            one, elapsed.reshape(1, -1)
        )
        _, velocity, acceleration, _ = kinematics.evaluate_retimed_reference(
            trajectory_type[index : index + 1],
            parameters["axis"][index : index + 1],
            parameters["wave_count"][index : index + 1],
            parameters["amp_x"][index : index + 1],
            parameters["amp_y"][index : index + 1],
            parameters["amp_z"][index : index + 1],
            zeros[index : index + 1],
            zeros[index : index + 1],
            phase,
            phase_rate,
            phase_acceleration,
            radius_min=0.30,
            radius_max=1.20,
            harmonic_ratio=0.08,
        )
        assert torch.linalg.vector_norm(velocity, dim=-1).max() <= limits.max_speed_mps * 1.01
        assert torch.linalg.vector_norm(acceleration, dim=-1).max() <= limits.max_acceleration_mps2 * 1.01

        fixed = torch.full(
            (elapsed.numel(),),
            bool(axis_sine[index]),
            dtype=torch.bool,
        )
        heading_velocity = guidance.horizontal_heading_velocity(velocity[0], fixed)
        previous = guidance.quaternion_from_level_heading(
            heading_velocity[:1], identity
        )
        max_rate = 0.0
        for sample in heading_velocity[1:]:
            current = guidance.quaternion_from_level_heading(
                sample.reshape(1, 3), previous
            )
            rate = guidance.quaternion_step_angular_velocity_body(previous, current, 0.02)
            max_rate = max(max_rate, float(torch.linalg.vector_norm(rate)))
            previous = current
        assert max_rate <= limits.max_yaw_rate_radps * 1.01


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
            kinematics.LATERAL_WAVE,
            kinematics.VERTICAL_WAVE,
            kinematics.SPATIAL_HELIX,
            kinematics.REVERSE_SPATIAL_HELIX,
        ],
        dtype=torch.long,
    )
    count = trajectory_types.numel()
    phase = torch.linspace(0.0, 2.0 * torch.pi, 4097).repeat(count, 1)
    offsets = kinematics.evaluate_geometry(
        trajectory_types,
        torch.tensor([0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        torch.ones(count, dtype=torch.long),
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


def test_precision_evaluation_helix_is_5x3x1_at_point_one_mps() -> None:
    trajectory_type = torch.tensor([kinematics.SPATIAL_HELIX], dtype=torch.long)
    zeros = torch.zeros(1)
    limits = kinematics.TrajectoryKinematicLimits(
        max_speed_mps=0.38,
        max_acceleration_mps2=0.45,
        max_yaw_rate_radps=0.80,
        max_jerk_mps3=0.36,
        retime_samples=256,
    )
    tables = kinematics.build_retimed_tables(
        trajectory_type,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        torch.tensor([2.5]),
        torch.tensor([1.5]),
        torch.tensor([0.5]),
        torch.ones(1),
        zeros,
        zeros,
        torch.tensor([0.10]),
        radius_min=0.30,
        radius_max=1.20,
        chirp_rate=1.6,
        harmonic_ratio=0.08,
        limits=limits,
    )
    elapsed = torch.linspace(0.0, float(tables.effective_period_s[0]), 4097).reshape(1, -1)
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(
        tables, elapsed
    )
    position, velocity, acceleration, curvature = kinematics.evaluate_retimed_reference(
        trajectory_type,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        torch.tensor([2.5]),
        torch.tensor([1.5]),
        torch.tensor([0.5]),
        zeros,
        zeros,
        phase,
        phase_rate,
        phase_acceleration,
        radius_min=0.30,
        radius_max=1.20,
        harmonic_ratio=0.08,
    )

    extent = position.amax(dim=1) - position.amin(dim=1)
    assert torch.allclose(extent, torch.tensor([[5.0, 3.0, 1.0]]), atol=1.0e-5)
    assert torch.linalg.vector_norm(velocity, dim=-1).max() <= 0.101
    assert torch.linalg.vector_norm(acceleration, dim=-1).max() <= 0.01
    assert curvature.max() <= 0.85


def test_precision_evaluation_lissajous_is_5x3x1_and_capped_at_point_35_mps() -> None:
    trajectory_type = torch.tensor([kinematics.LISSAJOUS], dtype=torch.long)
    zeros = torch.zeros(1)
    limits = kinematics.TrajectoryKinematicLimits(
        max_speed_mps=0.38,
        max_acceleration_mps2=0.45,
        max_yaw_rate_radps=0.80,
        max_jerk_mps3=0.36,
        retime_samples=256,
    )
    tables = kinematics.build_retimed_tables(
        trajectory_type,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        torch.tensor([2.5]),
        torch.tensor([1.5]),
        torch.tensor([0.5]),
        torch.ones(1),
        zeros,
        zeros,
        torch.tensor([0.35]),
        radius_min=0.30,
        radius_max=1.20,
        chirp_rate=1.6,
        harmonic_ratio=0.08,
        limits=limits,
    )
    elapsed = torch.linspace(0.0, float(tables.effective_period_s[0]), 4097).reshape(1, -1)
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(
        tables, elapsed
    )
    position, velocity, acceleration, _ = kinematics.evaluate_retimed_reference(
        trajectory_type,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        torch.tensor([2.5]),
        torch.tensor([1.5]),
        torch.tensor([0.5]),
        zeros,
        zeros,
        phase,
        phase_rate,
        phase_acceleration,
        radius_min=0.30,
        radius_max=1.20,
        harmonic_ratio=0.08,
    )

    extent = position.amax(dim=1) - position.amin(dim=1)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    sample_dt = elapsed[0, 1] - elapsed[0, 0]
    jerk = (acceleration[:, 2:] - acceleration[:, :-2]) / (2.0 * sample_dt)
    assert torch.allclose(extent, torch.tensor([[5.0, 3.0, 1.0]]), atol=1.0e-5)
    assert speed.max() <= 0.350001
    assert speed.max() >= 0.34
    assert torch.linalg.vector_norm(acceleration, dim=-1).max() <= 0.451
    assert torch.linalg.vector_norm(jerk, dim=-1).max() <= 0.36 * 1.01

    wall_time = torch.arange(
        0.0,
        float(tables.effective_period_s[0]) + 4.0,
        0.02,
    )
    trajectory_time, startup_rate, startup_acceleration = kinematics.smooth_startup_time(
        wall_time,
        4.0,
    )
    startup_phase, nominal_rate, nominal_acceleration = kinematics.sample_retimed_phase(
        tables,
        trajectory_time.reshape(1, -1),
    )
    startup_phase_rate = nominal_rate * startup_rate.reshape(1, -1)
    startup_phase_acceleration = (
        nominal_acceleration * startup_rate.square().reshape(1, -1)
        + nominal_rate * startup_acceleration.reshape(1, -1)
    )
    _, startup_velocity, startup_linear_acceleration, _ = (
        kinematics.evaluate_retimed_reference(
            trajectory_type,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            torch.tensor([2.5]),
            torch.tensor([1.5]),
            torch.tensor([0.5]),
            zeros,
            zeros,
            startup_phase,
            startup_phase_rate,
            startup_phase_acceleration,
            radius_min=0.30,
            radius_max=1.20,
            harmonic_ratio=0.08,
        )
    )
    startup_jerk = (
        startup_linear_acceleration[:, 2:]
        - startup_linear_acceleration[:, :-2]
    ) / 0.04
    assert torch.linalg.vector_norm(startup_velocity, dim=-1).max() <= 0.350001
    assert torch.linalg.vector_norm(startup_linear_acceleration, dim=-1).max() <= 0.451
    assert torch.linalg.vector_norm(startup_jerk, dim=-1).max() <= 0.36 * 1.01


def test_high_curvature_traveling_wave_pair_stays_within_limits() -> None:
    base_amplitudes = torch.tensor(
        [
            [ax, ay, az]
            for ax in (2.25, 2.5)
            for ay in (1.35, 1.5)
            for az in (0.4, 0.5)
        ]
    )
    count = base_amplitudes.shape[0]
    trajectory_type = torch.full((count,), kinematics.LATERAL_WAVE, dtype=torch.long)
    zeros = torch.zeros(count)
    limits = kinematics.TrajectoryKinematicLimits(
        max_speed_mps=0.38,
        max_acceleration_mps2=0.45,
        max_yaw_rate_radps=0.80,
        max_jerk_mps3=0.36,
        retime_samples=256,
    )
    scaled = base_amplitudes * torch.tensor([1.0, 0.5, 1.0])
    tables = kinematics.build_retimed_tables(
        trajectory_type,
        torch.zeros(count, dtype=torch.long),
        torch.full((count,), 3, dtype=torch.long),
        scaled[:, 0],
        scaled[:, 1],
        scaled[:, 2],
        torch.ones(count),
        zeros,
        zeros,
        torch.full((count,), 0.02),
        radius_min=0.30,
        radius_max=1.20,
        chirp_rate=1.6,
        harmonic_ratio=0.08,
        limits=limits,
    )
    elapsed = (
        torch.linspace(0.0, 1.0, 4097).reshape(1, -1)
        * tables.effective_period_s.reshape(-1, 1)
    )
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(
        tables, elapsed
    )
    _, velocity, acceleration, curvature = kinematics.evaluate_retimed_reference(
        trajectory_type,
        torch.zeros(count, dtype=torch.long),
        torch.full((count,), 3, dtype=torch.long),
        scaled[:, 0],
        scaled[:, 1],
        scaled[:, 2],
        zeros,
        zeros,
        phase,
        phase_rate,
        phase_acceleration,
        radius_min=0.30,
        radius_max=1.20,
        harmonic_ratio=0.08,
    )

    speed = torch.linalg.vector_norm(velocity, dim=-1)
    sample_dt = tables.effective_period_s / 4096.0
    jerk = (acceleration[:, 2:] - acceleration[:, :-2]) / (
        2.0 * sample_dt.reshape(-1, 1, 1)
    )
    assert speed.max() <= 0.0201
    assert torch.linalg.vector_norm(acceleration, dim=-1).max() <= 0.45
    assert torch.linalg.vector_norm(jerk, dim=-1).max() <= 0.36 * 1.01
    assert (speed * curvature).max() <= 0.80 * 1.01

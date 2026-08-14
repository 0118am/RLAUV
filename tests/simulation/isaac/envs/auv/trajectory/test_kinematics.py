"""Pure reference-generator tests; these do not require Isaac Sim."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[6]
MODULE_PATH = REPO_ROOT / "simulation/isaac/envs/auv/trajectory/kinematics.py"
SPEC = importlib.util.spec_from_file_location("trajectory_kinematics", MODULE_PATH)
kinematics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = kinematics
SPEC.loader.exec_module(kinematics)


def _sample_parameters(count: int = 8):
    torch.manual_seed(73)
    types = torch.arange(count, dtype=torch.long) % 8
    return {
        "trajectory_type": types,
        "axis": torch.randint(0, 3, (count,), dtype=torch.long),
        "amp_x": 0.60 + 0.18 * torch.rand(count),
        "amp_y": 0.55 + 0.20 * torch.rand(count),
        "amp_z": 0.08 + 0.12 * torch.rand(count),
        "requested_period_s": 10.0 + 10.0 * torch.rand(count),
        "phase_x": 2.0 * torch.pi * torch.rand(count),
        "phase_y": 2.0 * torch.pi * torch.rand(count),
    }


def _build_tables(parameters: dict[str, torch.Tensor]):
    return kinematics.build_retimed_tables(
        **parameters,
        radius_min=0.20,
        radius_max=0.75,
        chirp_rate=2.0,
        harmonic_ratio=0.08,
        limits=kinematics.TrajectoryKinematicLimits(),
    )


def test_smooth_startup_time_begins_at_rest_and_joins_unit_speed_continuously():
    elapsed = torch.tensor([0.0, 2.0, 4.0, 5.0], dtype=torch.float64)

    trajectory_time, speed_scale, speed_scale_rate = kinematics.smooth_startup_time(elapsed, 4.0)

    assert torch.allclose(
        trajectory_time,
        torch.tensor([0.0, 0.3125, 2.0, 3.0], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert torch.allclose(
        speed_scale,
        torch.tensor([0.0, 0.5, 1.0, 1.0], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert speed_scale_rate[0].item() == 0.0
    assert torch.allclose(speed_scale_rate[2:], torch.zeros(2, dtype=torch.float64), atol=1.0e-12)


def test_startup_chain_rule_makes_reference_velocity_zero_at_reset():
    parameters = {
        "trajectory_type": torch.tensor([0], dtype=torch.long),
        "axis": torch.tensor([0], dtype=torch.long),
        "amp_x": torch.tensor([0.7]),
        "amp_y": torch.tensor([0.7]),
        "amp_z": torch.tensor([0.1]),
        "requested_period_s": torch.tensor([12.0]),
        "phase_x": torch.tensor([0.3]),
        "phase_y": torch.tensor([0.0]),
    }
    tables = _build_tables(parameters)
    episode_time = torch.tensor([0.0])
    trajectory_time, speed_scale, speed_scale_rate = kinematics.smooth_startup_time(episode_time, 4.0)
    phase, nominal_rate, nominal_acceleration = kinematics.sample_retimed_phase(tables, trajectory_time)
    phase_rate = nominal_rate * speed_scale
    phase_acceleration = nominal_acceleration * speed_scale.square() + nominal_rate * speed_scale_rate

    _, velocity, acceleration, _ = kinematics.evaluate_retimed_reference(
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

    assert torch.equal(velocity, torch.zeros_like(velocity))
    assert torch.equal(acceleration, torch.zeros_like(acceleration))


def test_startup_reference_respects_existing_kinematic_limits():
    parameters = _sample_parameters(64)
    tables = _build_tables(parameters)
    duration_s = 4.0
    episode_time = torch.linspace(0.0, duration_s + 2.0, 301).repeat(64, 1)
    trajectory_time, speed_scale, speed_scale_rate = kinematics.smooth_startup_time(
        episode_time,
        duration_s,
    )
    phase, nominal_rate, nominal_acceleration = kinematics.sample_retimed_phase(tables, trajectory_time)
    phase_rate = nominal_rate * speed_scale
    phase_acceleration = nominal_acceleration * speed_scale.square() + nominal_rate * speed_scale_rate
    _, velocity, acceleration, _ = kinematics.evaluate_retimed_reference(
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
    dt = episode_time[0, 1] - episode_time[0, 0]
    jerk = (acceleration[:, 2:] - acceleration[:, :-2]) / (2.0 * dt)
    limits = kinematics.TrajectoryKinematicLimits()

    assert torch.linalg.vector_norm(velocity, dim=-1).max().item() <= limits.max_speed_mps + 1.0e-4
    assert torch.linalg.vector_norm(acceleration, dim=-1).max().item() <= limits.max_acceleration_mps2 + 1.0e-3
    assert torch.linalg.vector_norm(jerk, dim=-1).max().item() <= limits.max_jerk_mps3 + 1.0e-2


def _speed_controlled_parameters():
    speeds = torch.tensor([0.1, 0.2, 0.3, 0.4] * 3)
    types = torch.tensor(
        [kinematics.LATERAL_SINE] * 4
        + [kinematics.VERTICAL_SINE] * 4
        + [kinematics.SPATIAL_HELIX] * 4,
        dtype=torch.long,
    )
    count = types.numel()
    return types, speeds, {
        "trajectory_type": types,
        "axis": torch.zeros(count, dtype=torch.long),
        "amp_x": torch.full((count,), 0.75),
        "amp_y": torch.full((count,), 0.65),
        "amp_z": torch.where(types == kinematics.VERTICAL_SINE, torch.full((count,), 0.50), torch.full((count,), 0.16)),
        "requested_period_s": torch.full((count,), 10.0),
        "phase_x": torch.zeros(count),
        "phase_y": torch.zeros(count),
        "target_speed_mps": speeds,
    }


def test_speed_controlled_geometry_has_explicit_lateral_vertical_and_spatial_shapes():
    types, _, parameters = _speed_controlled_parameters()
    phase = torch.tensor([[0.0, 0.25 * torch.pi]]).expand(types.numel(), -1)
    positions = kinematics.evaluate_geometry(
        types,
        parameters["axis"],
        parameters["amp_x"],
        parameters["amp_y"],
        parameters["amp_z"],
        parameters["phase_x"],
        parameters["phase_y"],
        phase,
        radius_min=0.20,
        radius_max=0.75,
        harmonic_ratio=0.08,
    )

    assert torch.equal(positions[:4, :, (0, 2)], torch.zeros_like(positions[:4, :, (0, 2)]))
    assert torch.equal(positions[4:8, :, (0, 1)], torch.zeros_like(positions[4:8, :, (0, 1)]))
    assert torch.all(torch.linalg.vector_norm(positions[8:, 0], dim=-1) > 0.0)
    assert torch.all(positions[8:, 1, 2].abs() > 0.10)


def test_all_three_training_shapes_realize_each_requested_speed_level_without_retiming():
    types, speeds, parameters = _speed_controlled_parameters()
    tables = kinematics.build_retimed_tables(
        **parameters,
        radius_min=0.20,
        radius_max=0.75,
        chirp_rate=1.6,
        harmonic_ratio=0.08,
        limits=kinematics.TrajectoryKinematicLimits(),
    )
    elapsed = torch.arange(1024, dtype=torch.float32).reshape(1, -1) / 1024.0
    elapsed = elapsed * tables.effective_period_s.reshape(-1, 1)
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(tables, elapsed)
    _, velocity, acceleration, _ = kinematics.evaluate_retimed_reference(
        types,
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
    speed = torch.linalg.vector_norm(velocity, dim=-1)

    sine = types != kinematics.SPATIAL_HELIX
    assert torch.allclose(speed[sine].amax(dim=1), speeds[sine], atol=2.0e-4, rtol=2.0e-4)
    assert torch.allclose(speed[~sine], speeds[~sine].unsqueeze(-1).expand_as(speed[~sine]), atol=1.0e-4, rtol=2.0e-4)
    assert torch.linalg.vector_norm(acceleration, dim=-1).max().item() <= 0.45 + 1.0e-3
    assert not torch.any(tables.retimed)

    # Verify jerk at the environment's 50 Hz policy rate rather than only at
    # table knots; this catches interpolation artifacts between reset-time
    # samples.
    for env_id in range(types.numel()):
        one = kinematics.RetimedTrajectoryTables(
            phase=tables.phase[env_id : env_id + 1],
            elapsed_s=tables.elapsed_s[env_id : env_id + 1],
            phase_rate=tables.phase_rate[env_id : env_id + 1],
            phase_acceleration=tables.phase_acceleration[env_id : env_id + 1],
            requested_period_s=tables.requested_period_s[env_id : env_id + 1],
            effective_period_s=tables.effective_period_s[env_id : env_id + 1],
            retimed=tables.retimed[env_id : env_id + 1],
        )
        policy_time = torch.arange(0.0, float(one.effective_period_s[0]), 0.02).unsqueeze(0)
        q, q_rate, q_acceleration = kinematics.sample_retimed_phase(one, policy_time)
        _, _, policy_acceleration, _ = kinematics.evaluate_retimed_reference(
            types[env_id : env_id + 1],
            parameters["axis"][env_id : env_id + 1],
            parameters["amp_x"][env_id : env_id + 1],
            parameters["amp_y"][env_id : env_id + 1],
            parameters["amp_z"][env_id : env_id + 1],
            parameters["phase_x"][env_id : env_id + 1],
            parameters["phase_y"][env_id : env_id + 1],
            q,
            q_rate,
            q_acceleration,
            radius_min=0.20,
            radius_max=0.75,
            harmonic_ratio=0.08,
        )
        policy_jerk = torch.linalg.vector_norm(
            (policy_acceleration[:, 2:] - policy_acceleration[:, :-2]) / 0.04,
            dim=-1,
        )
        assert policy_jerk.max().item() <= 0.36 + 1.0e-2


def test_kinematic_limits_hold_for_one_thousand_random_curve_parameters():
    """Check limits on a broad batch, including arbitrary reset phases."""

    parameters = _sample_parameters(1_000)
    limits = kinematics.TrajectoryKinematicLimits()
    tables = _build_tables(parameters)
    assert torch.all(tables.effective_period_s >= parameters["requested_period_s"] - 1.0e-5)

    fractions = torch.arange(257, dtype=torch.float32).reshape(1, -1) / 257.0
    elapsed = fractions * tables.effective_period_s.reshape(-1, 1)
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(tables, elapsed)
    _, velocity, acceleration, _ = kinematics.evaluate_retimed_reference(
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
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    sample_dt = tables.effective_period_s.reshape(-1, 1) / fractions.shape[1]
    jerk = torch.linalg.vector_norm(
        (torch.roll(acceleration, -1, 1) - torch.roll(acceleration, 1, 1))
        / (2.0 * sample_dt).unsqueeze(-1),
        dim=-1,
    )
    _, first, second = kinematics._geometry_derivatives(  # noqa: SLF001 - verifies the heading constraint itself.
        parameters["trajectory_type"],
        parameters["axis"],
        parameters["amp_x"],
        parameters["amp_y"],
        parameters["amp_z"],
        parameters["phase_x"],
        parameters["phase_y"],
        phase,
        radius_min=0.20,
        radius_max=0.75,
        harmonic_ratio=0.08,
    )
    orientation_rate = kinematics._orientation_rate_per_phase(first, second) * phase_rate  # noqa: SLF001
    orientation_rate = torch.where(
        (parameters["trajectory_type"] == 2).reshape(-1, 1), torch.zeros_like(orientation_rate), orientation_rate
    )
    assert speed.max().item() <= limits.max_speed_mps * 1.01
    assert acceleration_norm.max().item() <= limits.max_acceleration_mps2 * 1.01
    assert jerk.max().item() <= limits.max_jerk_mps3 * 1.01
    assert orientation_rate.max().item() <= limits.max_orientation_rate_radps * 1.01


def test_periodic_sampling_is_continuous_at_the_wrap():
    parameters = _sample_parameters()
    tables = _build_tables(parameters)
    before = tables.effective_period_s - 1.0e-4
    after = torch.full_like(before, 1.0e-4)
    values = []
    for elapsed in (before, after):
        phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(tables, elapsed)
        values.append(
            kinematics.evaluate_retimed_reference(
                parameters["trajectory_type"], parameters["axis"], parameters["amp_x"], parameters["amp_y"],
                parameters["amp_z"], parameters["phase_x"], parameters["phase_y"], phase, phase_rate,
                phase_acceleration, radius_min=0.20, radius_max=0.75, harmonic_ratio=0.08,
            )
        )
    for before_value, after_value in zip(values[0][:3], values[1][:3], strict=True):
        assert torch.allclose(before_value, after_value, atol=2.0e-3, rtol=2.0e-3)


def test_random_smooth_moves_and_ood_curves_are_retimed():
    parameters = _sample_parameters()
    tables = _build_tables(parameters)
    fractions = torch.arange(128, dtype=torch.float32).reshape(1, -1) / 128.0
    elapsed = fractions * tables.effective_period_s.reshape(-1, 1)
    phase, phase_rate, phase_acceleration = kinematics.sample_retimed_phase(tables, elapsed)
    _, velocity, _, curvature = kinematics.evaluate_retimed_reference(
        parameters["trajectory_type"], parameters["axis"], parameters["amp_x"], parameters["amp_y"],
        parameters["amp_z"], parameters["phase_x"], parameters["phase_y"], phase, phase_rate,
        phase_acceleration, radius_min=0.20, radius_max=0.75, harmonic_ratio=0.08,
    )
    random_smooth_mask = parameters["trajectory_type"] == 7
    random_smooth_speed = torch.linalg.vector_norm(velocity[random_smooth_mask], dim=-1)
    assert torch.quantile(random_smooth_speed, 0.95).item() >= 0.05
    assert torch.isfinite(curvature).all()
    for trajectory_type in (1, 5):
        mask = parameters["trajectory_type"] == trajectory_type
        assert torch.all(tables.effective_period_s[mask] > parameters["requested_period_s"][mask])


def test_racetrack_geometry_is_closed_with_finite_join_derivatives():
    phase = torch.tensor([[0.0, 2.0 * torch.pi]], dtype=torch.float32)
    params = _sample_parameters(1)
    params["trajectory_type"][:] = 6
    position, first, second = kinematics._geometry_derivatives(  # noqa: SLF001
        params["trajectory_type"], params["axis"], params["amp_x"], params["amp_y"], params["amp_z"],
        params["phase_x"], params["phase_y"], phase, radius_min=0.20, radius_max=0.75, harmonic_ratio=0.08,
    )
    assert torch.allclose(position[:, 0], position[:, 1], atol=2.0e-4)
    assert torch.allclose(first[:, 0], first[:, 1], atol=2.0e-3)
    assert torch.allclose(second[:, 0], second[:, 1], atol=2.0e-2)
    assert torch.isfinite(second).all()

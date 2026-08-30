"""Boundary tests for the Isaac composition layer."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from environment.profile import HydrodynamicsProfile, load_environment_profile_json
from robot.runtime import T60_RUNTIME
from simulation.composition import RuntimeComposition


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PROFILE = (
    REPOSITORY_ROOT
    / "environment/hydrodynamics/coefficients/auv_open_water_openfoam_full_hydrodynamics_v2.json"
)


def test_nominal_hydrodynamic_matrices_follow_port_starboard_structure() -> None:
    environment = load_environment_profile_json(ENVIRONMENT_PROFILE)
    parity = (1, -1, 1, -1, 1, -1)
    for matrix in (
        environment.hydrodynamics.added_mass,
        environment.hydrodynamics.linear_damping,
        environment.hydrodynamics.quadratic_damping,
    ):
        assert len(matrix) == 6
        assert all(len(row) == 6 for row in matrix)
        assert all(
            matrix[row][column] == 0.0
            for row in range(6)
            for column in range(6)
            if parity[row] != parity[column]
        )
        assert any(
            matrix[row][column] != 0.0
            for row in range(6)
            for column in range(6)
            if row != column and parity[row] == parity[column]
        )

    added_mass = np.asarray(environment.hydrodynamics.added_mass)
    np.testing.assert_allclose(added_mass, added_mass.T, atol=0.0, rtol=0.0)
    assert np.linalg.eigvalsh(added_mass)[0] > 0.0


def test_added_mass_must_be_positive_semidefinite() -> None:
    environment = load_environment_profile_json(ENVIRONMENT_PROFILE)
    matrix = np.eye(6)
    matrix[-1, -1] = -0.8e-8
    HydrodynamicsProfile.model_validate(
        environment.hydrodynamics.model_dump(mode="python") | {"added_mass": matrix.tolist()}
    )

    matrix[-1, -1] = -2.0e-8
    with pytest.raises(ValueError, match="positive semidefinite"):
        HydrodynamicsProfile.model_validate(
            environment.hydrodynamics.model_dump(mode="python") | {"added_mass": matrix.tolist()}
        )


def test_composition_combines_environment_and_robot_sources() -> None:
    environment = load_environment_profile_json(ENVIRONMENT_PROFILE)
    robot = replace(T60_RUNTIME, thruster_time_constant_s=0.17)
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 100.0),
        decimation=4,
        domain_randomization=SimpleNamespace(),
    )

    RuntimeComposition(environment=environment, robot=robot).apply(cfg)

    assert cfg.environment_profile_name == environment.name
    assert cfg.linear_damping == environment.hydrodynamics.linear_damping
    assert cfg.mass == robot.model.mass_kg
    assert cfg.dyn_time_constant == pytest.approx(0.17)
    assert cfg.thruster_command_delay_s == pytest.approx(0.05)
    assert cfg.thruster_command_delay_steps == 5
    assert cfg.pose_sensor_delay_s == pytest.approx(0.05)
    assert cfg.pose_sensor_delay_steps == 5
    assert not hasattr(cfg, "pose_sensor_position_error_max_m")
    assert not hasattr(cfg, "pose_sensor_orientation_error_max_rad")
    assert cfg.sim.dt == pytest.approx(1.0 / 100.0)
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1.0 / 25.0)
    assert cfg.quadratic_damping == environment.hydrodynamics.quadratic_damping
    assert cfg.added_mass == environment.hydrodynamics.added_mass
    assert not cfg.water_current_field_enabled
    assert not hasattr(cfg.domain_randomization, "payload_samples")
    assert not hasattr(cfg.domain_randomization, "mass_relative_amplitude_by_stage")
    assert not hasattr(cfg.domain_randomization, "center_of_mass_offset_amplitude_by_stage")
    assert not hasattr(cfg.domain_randomization, "com_to_cob_relative_amplitude_by_stage")
    assert not hasattr(cfg.domain_randomization, "volume_range")
    assert cfg.domain_randomization.enabled_features == ()

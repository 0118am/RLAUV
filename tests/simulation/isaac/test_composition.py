"""Boundary tests for the Isaac composition layer."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment.profiles.environment_profile import load_environment_profile_json
from robot.runtime import T60_RUNTIME
from simulation.isaac.composition import IsaacComposition


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_PROFILE = (
    REPOSITORY_ROOT
    / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
)


def test_composition_combines_environment_and_robot_sources() -> None:
    environment = load_environment_profile_json(ENVIRONMENT_PROFILE)
    robot = replace(T60_RUNTIME, thruster_time_constant_s=0.17)
    cfg = SimpleNamespace(domain_randomization=SimpleNamespace())

    IsaacComposition(environment=environment, robot=robot).apply(cfg)

    assert cfg.environment_profile_name == environment.name
    assert cfg.linear_damping == environment.hydrodynamics.linear_damping
    assert cfg.mass == robot.model.mass_kg
    assert cfg.dyn_time_constant == pytest.approx(0.17)
    assert cfg.domain_randomization.mass_range == [robot.model.mass_kg] * 2
    assert cfg.domain_randomization.enabled_features == []

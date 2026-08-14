"""Boundary tests for the Isaac composition layer."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment.profiles.environment_profile import (
    environment_profile_from_dict,
    load_environment_profile_json,
)
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


@pytest.mark.parametrize("robot_section", ["rigid_body", "thrusters", "battery", "tether"])
def test_environment_profile_rejects_robot_sections(robot_section: str) -> None:
    with pytest.raises(ValueError, match="water and pool physics"):
        environment_profile_from_dict(
            {
                "name": "invalid-cross-domain-profile",
                robot_section: {},
            }
        )


def test_isaac_config_has_no_shadow_physics_defaults() -> None:
    source = (REPOSITORY_ROOT / "simulation/isaac/config.py").read_text(encoding="utf-8")

    for assignment in (
        "    mass =",
        "    volume =",
        "    water_rho =",
        "    linear_damping =",
        "    added_mass_diag =",
        "    dyn_time_constant =",
        "    battery_voltage_nominal =",
        "    tether_enabled =",
    ):
        assert assignment not in source

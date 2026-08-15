"""Boundary tests for the Isaac composition layer."""

from dataclasses import replace
import json
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
OPENFOAM_CONFIG_UPDATES = (
    REPOSITORY_ROOT
    / "environment/openfoam/results_jn2_port_starboard_symmetric_minimal_level6_v1/config_updates.json"
)


def test_nominal_hydrodynamic_matrices_match_openfoam_fit_output() -> None:
    environment = load_environment_profile_json(ENVIRONMENT_PROFILE)
    with OPENFOAM_CONFIG_UPDATES.open(encoding="utf-8") as stream:
        fitted = json.load(stream)

    assert environment.hydrodynamics.added_mass == fitted["added_mass_diag"]
    assert environment.hydrodynamics.linear_damping == fitted["linear_damping"]
    assert environment.hydrodynamics.quadratic_damping == fitted["quadratic_damping"]


def test_composition_combines_environment_and_robot_sources() -> None:
    environment = load_environment_profile_json(ENVIRONMENT_PROFILE)
    robot = replace(T60_RUNTIME, thruster_time_constant_s=0.17)
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 200.0),
        domain_randomization=SimpleNamespace(),
    )

    IsaacComposition(environment=environment, robot=robot).apply(cfg)

    assert cfg.environment_profile_name == environment.name
    assert cfg.linear_damping == environment.hydrodynamics.linear_damping
    assert cfg.mass == robot.model.mass_kg
    assert cfg.dyn_time_constant == pytest.approx(0.17)
    assert cfg.thruster_command_delay_s == pytest.approx(0.13)
    assert cfg.thruster_command_delay_steps == 26
    assert cfg.thruster_command_delay_applied_s == pytest.approx(0.13)
    assert cfg.quadratic_damping == environment.hydrodynamics.quadratic_damping
    assert cfg.added_mass_diag == environment.hydrodynamics.added_mass
    assert not cfg.water_current_field_enabled
    assert not hasattr(cfg, "high_order_residual_enabled")
    assert cfg.domain_randomization.thruster_command_delay_steps_range == [26, 26]
    assert cfg.domain_randomization.mass_range == [robot.model.mass_kg] * 2
    assert cfg.domain_randomization.enabled_features == []

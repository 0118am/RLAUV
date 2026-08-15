"""Pure-Python coverage for training configuration and campaign management."""

from __future__ import annotations

from pathlib import Path

from simulation.isaac.training import (
    materialize_training_profiles,
)
from environment.profiles.domain_randomization import load_domain_randomization_spec_json
from environment.profiles.environment_profile import load_environment_profile_json


def test_notebook_values_materialize_isolated_training_profiles(tmp_path: Path) -> None:
    profiles = materialize_training_profiles(
        "unit profile",
        output_root=tmp_path,
        hydrodynamics={"water_current_w": [0.1, 0.0, 0.0]},
        randomization={"water_current_max_by_stage": [0.0, 0.05, 0.1, 0.15, 0.25]},
    )

    environment = load_environment_profile_json(profiles.environment)
    randomization = load_domain_randomization_spec_json(profiles.randomization)

    assert profiles.environment.parent == tmp_path / "unit-profile"
    assert environment.hydrodynamics.water_current_w == [0.1, 0.0, 0.0]
    assert randomization.base_profile_name == environment.name
    assert randomization.parameters.water_current_max_by_stage[-1] == 0.25
    assert randomization.metadata["configured_by"] == "train.ipynb"

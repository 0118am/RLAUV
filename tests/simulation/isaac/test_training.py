"""Pure-Python coverage for training configuration and campaign management."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from simulation.isaac.training import (
    build_default_campaign,
    campaign_payload,
    materialize_training_profiles,
    render_campaign_status,
    write_campaign_config,
)
from environment.profiles.domain_randomization import load_domain_randomization_spec_json
from environment.profiles.environment_profile import load_environment_profile_json
from simulation.isaac.trajectory.train import _require_training_environment


def test_default_campaign_is_json_serializable_and_owned_by_isaac(tmp_path: Path) -> None:
    campaign = build_default_campaign(
        isaaclab_root=tmp_path / "IsaacLab",
        rlpolicy_root=tmp_path / "rlpolicy",
    )

    payload = campaign_payload(campaign)
    encoded = json.dumps(payload)

    assert "simulation/isaac/trajectory/competence_curriculum.py" in " ".join(
        campaign.supervisor_command()
    )
    assert payload["train"]["reward_profile"] == "policy_6"
    assert payload["train"]["trajectory_curriculum"]["stage_0_types"] == (8, 9, 10)
    assert "t60_policy_6_supervisor_state.json" in encoded


def test_campaign_config_and_empty_status_do_not_require_isaac_sim(tmp_path: Path) -> None:
    campaign = build_default_campaign(
        isaaclab_root=tmp_path / "IsaacLab",
        rlpolicy_root=tmp_path / "rlpolicy",
        run_name="unit_campaign",
    )

    path = write_campaign_config(campaign)
    status = render_campaign_status(campaign, log_tail_lines=0)

    assert json.loads(path.read_text(encoding="utf-8"))["train"]["run_name"] == "unit_campaign"
    assert "Campaign: unit_campaign" in status
    assert "Latest run: none" in status


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


def test_training_worker_requires_env_isaaclab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", "/opt/conda/envs/another_env")

    with pytest.raises(RuntimeError, match="conda activate env_isaaclab"):
        _require_training_environment()


def test_training_worker_accepts_env_isaaclab_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", "/opt/conda/envs/env_isaaclab")

    _require_training_environment()

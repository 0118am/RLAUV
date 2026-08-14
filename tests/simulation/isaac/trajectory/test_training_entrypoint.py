"""Training-entry environment guard tests."""

import sys

import pytest

from simulation.isaac.trajectory.train import _require_training_environment


def test_training_entry_requires_env_isaaclab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", "/opt/conda/envs/another_env")

    with pytest.raises(RuntimeError, match="conda activate env_isaaclab"):
        _require_training_environment()


def test_training_entry_accepts_env_isaaclab_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", "/opt/conda/envs/env_isaaclab")

    _require_training_environment()


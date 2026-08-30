"""Subprocess-environment contracts for notebook-managed campaigns."""

from __future__ import annotations

from pathlib import Path

from simulation.training import campaign


def test_experiment_environment_recovers_kernel_conda_prefix(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "env_isaaclab"
    executable = prefix / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (prefix / "conda-meta").mkdir()
    monkeypatch.setattr(campaign.sys, "executable", str(executable))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", "/stale/virtualenv")

    env = campaign.experiment_environment()

    assert env["CONDA_PREFIX"] == str(prefix)
    assert "VIRTUAL_ENV" not in env
    assert env["LD_LIBRARY_PATH"].split(":", maxsplit=1)[0] == str(prefix / "lib")


def test_experiment_environment_allows_explicit_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "env_isaaclab"
    executable = prefix / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (prefix / "conda-meta").mkdir()
    monkeypatch.setattr(campaign.sys, "executable", str(executable))

    env = campaign.experiment_environment({"CONDA_PREFIX": "/explicit/prefix"})

    assert env["CONDA_PREFIX"] == "/explicit/prefix"

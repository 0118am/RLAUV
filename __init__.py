"""Gym registration entry point for the AUV IsaacLab task package."""

from __future__ import annotations


def _register_environments():
    """Import IsaacLab components, register task IDs, and return public classes."""

    from .simulation.assembly import AUVTrajEnv, AUVTrajEnvCfg, register_environment

    register_environment()
    return AUVTrajEnv, AUVTrajEnvCfg


# Pytest may inspect this file as a bare ``__init__`` module when the checkout
# directory contains a hyphen. In that context there is no package anchor for
# relative imports and task registration is neither possible nor needed.
if __package__:
    AUVTrajEnv, AUVTrajEnvCfg = _register_environments()
    __all__ = ["AUVTrajEnv", "AUVTrajEnvCfg"]

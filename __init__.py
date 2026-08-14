"""Gym registration entry point for the AUV IsaacLab task package."""

from __future__ import annotations


def _register_environments():
    """Import IsaacLab components, register task IDs, and return public classes."""

    import gymnasium as gym

    from .simulation.isaac.config import AUVTrajEnvCfg
    from .simulation.isaac.env import AUVTrajEnv
    from .simulation.isaac.ppo import config as ppo_config

    gym.register(
        id="Isaac-AUV-Traj-Direct-v1",
        entry_point=AUVTrajEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": AUVTrajEnvCfg,
            "rsl_rl_cfg_entry_point": ppo_config.AUVTrajPPORunnerCfg,
        },
    )
    return AUVTrajEnv, AUVTrajEnvCfg


# Pytest may inspect this file as a bare ``__init__`` module when the checkout
# directory contains a hyphen. In that context there is no package anchor for
# relative imports and task registration is neither possible nor needed.
if __package__:
    AUVTrajEnv, AUVTrajEnvCfg = _register_environments()
    __all__ = ["AUVTrajEnv", "AUVTrajEnvCfg"]

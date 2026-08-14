# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration entry point for the AUV IsaacLab task package."""

from __future__ import annotations


def _register_environments():
    """Import IsaacLab components, register task IDs, and return public classes."""

    import gymnasium as gym

    from .simulation.isaac.agents.ppo import config as ppo_config
    from .simulation.isaac.envs.auv.config import AUVTrajEnvCfg
    from .simulation.isaac.envs.auv.env import AUVTrajEnv

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

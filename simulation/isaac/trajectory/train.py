# IsaacLab launcher-compatible portions:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the AUV trajectory policy with the repository-owned RSL-RL runner.

This is the Isaac Sim worker entry point.  It intentionally mirrors the
supported IsaacLab RSL-RL launcher contract while keeping the AUV-specific
runner, algorithm registration, logging location, and lifecycle in this
repository. Human-facing numeric selection lives in the repository-root
``train.ipynb`` and delegates lifecycle work to ``simulation.isaac.training``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MINIMUM_RSL_RL_VERSION = "3.0.1"


def _require_training_environment() -> None:
    """Fail early when training is launched outside ``env_isaaclab``."""

    expected = "env_isaaclab"
    active_conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    interpreter_env = Path(sys.prefix).name
    if expected not in {active_conda_env, interpreter_env}:
        raise RuntimeError(
            "Trajectory training requires the env_isaaclab environment. "
            "Run `conda activate env_isaaclab` before invoking isaaclab.sh."
        )


def _require_rsl_rl_version() -> None:
    """Validate the RSL-RL API version expected by the local runner."""

    from packaging import version

    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) >= version.parse(MINIMUM_RSL_RL_VERSION):
        return
    launcher = r".\isaaclab.bat" if platform.system() == "Windows" else "./isaaclab.sh"
    raise RuntimeError(
        f"rsl-rl-lib {installed_version} is installed; {MINIMUM_RSL_RL_VERSION} or newer is required. "
        f"Install it with `{launcher} -p -m pip install rsl-rl-lib=={MINIMUM_RSL_RL_VERSION}`."
    )


def _build_parser(app_launcher_type: type) -> argparse.ArgumentParser:
    """Build the local worker CLI without importing Isaac Sim at module import."""

    from simulation.isaac.trajectory import cli as cli_args

    parser = argparse.ArgumentParser(description="Train the AUV trajectory policy with RSL-RL.")
    parser.add_argument("--video", action="store_true", help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel simulation environments.")
    parser.add_argument("--task", type=str, default=None, help="Registered IsaacLab task name.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Gym registry key containing the RSL-RL configuration.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment and agent seed; -1 selects one randomly.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Number of policy training iterations.")
    parser.add_argument("--distributed", action="store_true", help="Use multiple GPUs or nodes.")
    parser.add_argument(
        "--export_io_descriptors",
        action="store_true",
        help="Export IO descriptors for manager-based environments.",
    )
    parser.add_argument(
        "--ray-proc-id",
        "-rid",
        type=int,
        default=None,
        help="Process identifier supplied by IsaacLab's Ray integration.",
    )
    cli_args.add_rsl_rl_args(parser)
    app_launcher_type.add_app_launcher_args(parser)
    return parser


def _run_training(args_cli: argparse.Namespace, app_launcher: object) -> None:
    """Resolve Hydra configuration and execute one RSL-RL training process."""

    import torch
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    from isaaclab.envs import (
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    from simulation.isaac.ppo.algorithm import RolloutAdaptivePPO
    from simulation.isaac.trajectory import cli as cli_args
    from simulation.isaac.trajectory.training_worker import (
        configure_torch_runtime,
        execute_training,
    )

    _require_rsl_rl_version()

    # RSL-RL resolves algorithm names from this module when constructing the
    # local OnPolicyRunner subclass.
    on_policy_runner_module.RolloutAdaptivePPO = RolloutAdaptivePPO

    configure_torch_runtime(torch)

    @hydra_task_config(args_cli.task, args_cli.agent)
    def train(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
    ) -> None:
        execute_training(env_cfg, agent_cfg, args_cli, app_launcher, cli_args, __file__)

    train()


def main() -> None:
    """Parse the worker CLI, launch Isaac Sim, train, and always close it."""

    _require_training_environment()

    from isaaclab.app import AppLauncher

    parser = _build_parser(AppLauncher)
    args_cli, hydra_args = parser.parse_known_args()
    if args_cli.video:
        args_cli.enable_cameras = True
    sys.argv = [sys.argv[0], *hydra_args]

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _run_training(args_cli, app_launcher)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

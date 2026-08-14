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
from datetime import datetime
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import sys
import time


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

    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    from simulation.isaac.ppo.algorithm import RolloutAdaptivePPO
    from simulation.isaac.ppo.runner import GpuBatchedOnPolicyRunner
    from simulation.isaac.trajectory import cli as cli_args

    _require_rsl_rl_version()

    # RSL-RL resolves algorithm names from this module when constructing the
    # local OnPolicyRunner subclass.
    on_policy_runner_module.RolloutAdaptivePPO = RolloutAdaptivePPO

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, args_cli.agent)
    def train(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
    ) -> None:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations

        env_cfg.seed = agent_cfg.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training requires a CUDA device.")
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"
            distributed_seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = distributed_seed
            agent_cfg.seed = distributed_seed

        # ``agent.experiment_name`` may be absolute. os.path.join deliberately
        # preserves that contract and otherwise follows the IsaacLab default.
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        run_directory_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name requested from command line: {run_directory_name}")
        if agent_cfg.run_name:
            run_directory_name += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, run_directory_name)

        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.log_dir = log_dir

        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )
        try:
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)

            resume_path = None
            if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
                resume_path = get_checkpoint_path(
                    log_root_path,
                    agent_cfg.load_run,
                    agent_cfg.load_checkpoint,
                )

            if args_cli.video:
                video_kwargs = {
                    "video_folder": os.path.join(log_dir, "videos", "train"),
                    "step_trigger": lambda step: step % args_cli.video_interval == 0,
                    "video_length": args_cli.video_length,
                    "disable_logger": True,
                }
                print("[INFO] Recording videos during training.")
                print_dict(video_kwargs, nesting=4)
                env = gym.wrappers.RecordVideo(env, **video_kwargs)

            start_time = time.time()
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = GpuBatchedOnPolicyRunner(
                    env,
                    agent_cfg.to_dict(),
                    log_dir=log_dir,
                    device=agent_cfg.device,
                )
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(
                    env,
                    agent_cfg.to_dict(),
                    log_dir=log_dir,
                    device=agent_cfg.device,
                )
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

            runner.add_git_repo_to_log(__file__)
            if resume_path is not None:
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                runner.load(resume_path)

            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        finally:
            env.close()

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

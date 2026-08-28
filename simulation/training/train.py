# IsaacLab launcher-compatible portions:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the AUV trajectory policy with the selected RSL-RL PPO extension.

This is the Isaac Sim worker entry point.  It intentionally mirrors the
supported IsaacLab RSL-RL launcher contract while keeping the AUV-specific
configuration, logging location, and lifecycle in this repository.
Human-facing numeric selection lives in the repository-root
``train.ipynb`` and delegates lifecycle work to ``simulation.training``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser(app_launcher_type: type) -> argparse.ArgumentParser:
    """Build the local worker CLI without importing Isaac Sim at module import."""

    parser = argparse.ArgumentParser(description="Train the AUV trajectory policy with RSL-RL.")
    parser.add_argument("--video", action="store_true", help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel simulation environments.")
    parser.add_argument("--task", type=str, default=None, help="Registered IsaacLab task name.")
    parser.add_argument("--seed", type=int, default=None, help="Environment and agent seed; -1 selects one randomly.")
    parser.add_argument(
        "--training_recipe",
        type=Path,
        required=True,
        help="Versioned repository training recipe JSON.",
    )
    parser.add_argument("--distributed", action="store_true", help="Use multiple GPUs or nodes.")
    add_rsl_rl_args(parser)
    app_launcher_type.add_app_launcher_args(parser)
    return parser


def _run_training(args_cli: argparse.Namespace, app_launcher: object) -> None:
    """Resolve Hydra configuration and execute one RSL-RL training process."""

    import torch
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(
        env_cfg: DirectRLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
    ) -> None:
        execute_training(env_cfg, agent_cfg, args_cli, app_launcher, __file__)

    train()


def main() -> None:
    """Parse the worker CLI, launch Isaac Sim, train, and always close it."""

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


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Apply the shared train/eval CLI overrides to an RSL-RL config."""

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10_000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def configure_training(
    env_cfg: Any,
    agent_cfg: Any,
    args_cli: Any,
    app_launcher: Any,
) -> tuple[Any, Any]:
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.seed += app_launcher.local_rank
        env_cfg.seed = agent_cfg.seed
    return env_cfg, agent_cfg


def build_log_paths(agent_cfg: Any) -> tuple[str, str]:
    # An absolute experiment name deliberately overrides the conventional root.
    root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {root}")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {run_name}")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    return root, os.path.join(root, run_name)


def maybe_record_video(env: Any, args_cli: Any, log_dir: str, gym: Any, print_dict: Any) -> Any:
    if not args_cli.video:
        return env
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    return gym.wrappers.RecordVideo(env, **video_kwargs)


def execute_training(
    env_cfg: Any,
    agent_cfg: Any,
    args_cli: Any,
    app_launcher: Any,
    source_file: str,
) -> None:
    import gymnasium as gym
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    from simulation.training.recipe import (
        apply_training_recipe,
        load_training_recipe,
        materialize_run_inputs,
        run_input_paths,
    )
    from simulation.training.ppo.runner import AUVOnPolicyRunner

    recipe = load_training_recipe(args_cli.training_recipe)
    env_cfg, agent_cfg = configure_training(
        env_cfg,
        agent_cfg,
        args_cli,
        app_launcher,
    )
    log_root, log_dir = build_log_paths(agent_cfg)
    resume_path = (
        get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if agent_cfg.resume
        else None
    )
    if resume_path is not None:
        recipe = load_training_recipe(run_input_paths(Path(resume_path).parent).recipe)
    env_cfg, agent_cfg = apply_training_recipe(recipe, env_cfg, agent_cfg)
    run_inputs = materialize_run_inputs(recipe, log_dir)
    env_cfg.environment_profile = str(run_inputs.environment)
    env_cfg.domain_randomization_spec = str(run_inputs.domain_randomization)
    env_cfg.log_dir = log_dir
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    try:
        env = maybe_record_video(env, args_cli, log_dir, gym, print_dict)
        started = time.time()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = AUVOnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(source_file)
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations)
        print(f"Training time: {round(time.time() - started, 2)} seconds")
    finally:
        env.close()


if __name__ == "__main__":
    main()

# IsaacLab launcher-compatible portions:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the AUV trajectory policy with the repository-owned RSL-RL runner.

This is the Isaac Sim worker entry point.  It intentionally mirrors the
supported IsaacLab RSL-RL launcher contract while keeping the AUV-specific
runner, algorithm registration, logging location, and lifecycle in this
repository. Human-facing numeric selection lives in the repository-root
``train.ipynb`` and delegates lifecycle work to ``simulation.training``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    parser.add_argument(
        "--training_recipe",
        type=Path,
        required=True,
        help="Versioned repository training recipe JSON.",
    )
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
    add_rsl_rl_args(parser)
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

    from simulation.training.ppo.algorithm import RolloutAdaptivePPO
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
        execute_training(env_cfg, agent_cfg, args_cli, app_launcher, __file__)

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
    # -- play arguments
    arg_group.add_argument("--play_checkpoint", type=str, default=None, help="Checkpoint file to play from")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    return update_rsl_rl_cfg(rslrl_cfg, args_cli)


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


def configure_torch_runtime(torch: Any) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def configure_training(
    env_cfg: Any,
    agent_cfg: Any,
    args_cli: Any,
    app_launcher: Any,
) -> tuple[Any, Any]:
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
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


def resolve_resume_path(agent_cfg: Any, log_root: str, get_checkpoint_path: Any) -> str | None:
    if not (agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"):
        return None
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


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


def create_runner(
    env: Any,
    agent_cfg: Any,
    log_dir: str,
    gpu_runner_type: Any,
    distillation_runner_type: Any,
) -> Any:
    arguments = (env, agent_cfg.to_dict())
    keywords = {"log_dir": log_dir, "device": agent_cfg.device}
    if agent_cfg.class_name == "OnPolicyRunner":
        return gpu_runner_type(*arguments, **keywords)
    if agent_cfg.class_name == "DistillationRunner":
        return distillation_runner_type(*arguments, **keywords)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def execute_training(
    env_cfg: Any,
    agent_cfg: Any,
    args_cli: Any,
    app_launcher: Any,
    source_file: str,
) -> None:
    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner
    from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    from simulation.training.ppo.runner import GpuBatchedOnPolicyRunner
    from simulation.training.manifest import (
        build_run_manifest,
        load_run_manifest,
        validate_manifest_selection,
        write_run_manifest,
    )
    from simulation.training.recipe import load_training_recipe, materialize_run_inputs

    env_cfg, agent_cfg = configure_training(
        env_cfg,
        agent_cfg,
        args_cli,
        app_launcher,
    )
    log_root, log_dir = build_log_paths(agent_cfg)
    recipe = load_training_recipe(args_cli.training_recipe)
    if env_cfg.mlp_architecture != recipe.mlp_architecture:
        raise ValueError(
            f"Environment architecture {env_cfg.mlp_architecture!r} does not match recipe "
            f"{recipe.mlp_architecture!r}."
        )
    if env_cfg.tracking_reward_profile != recipe.reward_profile:
        raise ValueError(
            f"Environment reward {env_cfg.tracking_reward_profile!r} does not match recipe "
            f"{recipe.reward_profile!r}."
        )
    run_inputs = materialize_run_inputs(recipe, log_dir)
    env_cfg.environment_profile = str(run_inputs.environment)
    env_cfg.domain_randomization_spec = str(run_inputs.domain_randomization)
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
        resume_path = resolve_resume_path(agent_cfg, log_root, get_checkpoint_path)
        if resume_path is not None:
            resume_manifest = load_run_manifest(os.path.dirname(resume_path))
            validate_manifest_selection(
                resume_manifest,
                mlp_architecture=recipe.mlp_architecture,
                reward_profile=recipe.reward_profile,
            )
            if resume_manifest.recipe_name != recipe.name:
                raise ValueError(
                    f"Resume run uses recipe {resume_manifest.recipe_name!r}, expected {recipe.name!r}."
                )
        env = maybe_record_video(env, args_cli, log_dir, gym, print_dict)
        started = time.time()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_runner(
            env,
            agent_cfg,
            log_dir,
            GpuBatchedOnPolicyRunner,
            DistillationRunner,
        )
        write_run_manifest(
            build_run_manifest(
                recipe=recipe,
                task_name=args_cli.task,
                env_cfg=env_cfg,
                agent_cfg=agent_cfg,
                run_dir=log_dir,
            )
        )
        runner.add_git_repo_to_log(source_file)
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {round(time.time() - started, 2)} seconds")
    finally:
        env.close()


if __name__ == "__main__":
    main()

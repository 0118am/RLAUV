"""Runtime helpers for one Isaac Sim trajectory-training worker."""

from __future__ import annotations

from datetime import datetime
import os
import time
from typing import Any


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
    cli_args: Any,
) -> tuple[Any, Any]:
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
    cli_args: Any,
    source_file: str,
) -> None:
    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner
    from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    from simulation.isaac.ppo.runner import GpuBatchedOnPolicyRunner

    env_cfg, agent_cfg = configure_training(
        env_cfg,
        agent_cfg,
        args_cli,
        app_launcher,
        cli_args,
    )
    log_root, log_dir = build_log_paths(agent_cfg)
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

"""Configuration, controller, and artifact setup for trajectory evaluation."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from robot.control import PIDGains, PIDTrajectoryController
from simulation.isaac.ppo.evaluation import load_evaluation_actor
from simulation.isaac.trajectory import TRAJECTORY_TYPE_IDS
from simulation.isaac.trajectory.evaluation_cases import (
    build_evaluation_case_label,
    resolve_random_smooth_ranges,
    sanitize_evaluation_label,
)


@dataclass(frozen=True)
class EvaluationSetup:
    env_cfg: Any
    agent_cfg: Any
    log_root: Path
    checkpoint_path: str | None
    checkpoint_name: str


@dataclass(frozen=True)
class EvaluationPaths:
    directory: Path
    logs_csv: Path
    summary_csv: Path
    domain_samples_csv: Path


def _apply_disturbance_overlay(env_cfg: Any, args: Any) -> None:
    env_cfg.evaluation_physics_overlay = {
        "damping_scale": float(args.eval_damping_scale),
        "thruster_tau_scale": float(args.eval_thruster_tau_scale),
    }
    if args.eval_current is not None:
        env_cfg.evaluation_physics_overlay["water_current_w"] = [float(value) for value in args.eval_current]
    smooth_current = args.eval_smooth_current or args.eval_current_variation_std > 0.0
    if smooth_current:
        env_cfg.evaluation_physics_overlay.update(
            {
                "smooth_current": True,
                "current_variation_std": float(args.eval_current_variation_std),
                "current_tau": float(args.eval_current_tau),
                "current_feature_only": not bool(args.eval_domain_randomization),
            }
        )
    if abs(args.eval_thruster_scale - 1.0) > 1.0e-9:
        env_cfg.evaluation_physics_overlay["thruster_force_scale"] = float(args.eval_thruster_scale)


def _apply_trajectory_configuration(env_cfg: Any, args: Any) -> None:
    env_cfg.trajectory_eval_type = TRAJECTORY_TYPE_IDS[args.trajectory]
    env_cfg.trajectory_eval_duration_s = args.duration
    scalar_overrides = {
        "trajectory_eval_amp_x": args.trajectory_amp_x,
        "trajectory_eval_amp_y": args.trajectory_amp_y,
        "trajectory_eval_amp_z": args.trajectory_amp_z,
        "trajectory_eval_period": args.trajectory_period,
        "trajectory_eval_speed_mps": args.trajectory_speed,
        "trajectory_eval_radius_min": args.trajectory_radius_min,
        "trajectory_eval_radius_max": args.trajectory_radius_max,
    }
    for name, value in scalar_overrides.items():
        if value is not None:
            setattr(env_cfg, name, value)

    if args.trajectory == "random_smooth":
        ranges = resolve_random_smooth_ranges(
            trajectory_amp_x=args.trajectory_amp_x,
            trajectory_amp_y=args.trajectory_amp_y,
            trajectory_amp_z=args.trajectory_amp_z,
            trajectory_period=args.trajectory_period,
            trajectory_amp_x_range=args.trajectory_amp_x_range,
            trajectory_amp_y_range=args.trajectory_amp_y_range,
            trajectory_amp_z_range=args.trajectory_amp_z_range,
            trajectory_period_range=args.trajectory_period_range,
        )
        for name, value in ranges.items():
            setattr(env_cfg, name, value)


def configure_evaluation(
    args: Any,
    architecture: Any,
    parse_agent_cfg: Callable[[str, Any], Any],
) -> EvaluationSetup:
    eval_num_envs = args.num_envs or (args.random_curve_count if args.trajectory == "random_smooth" else 1)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=max(1, eval_num_envs),
        use_fabric=not args.disable_fabric,
    )
    env_cfg.eval_mode = True
    env_cfg.eval_domain_randomization = bool(args.eval_domain_randomization)
    env_cfg.eval_disturbance_stage = int(args.eval_disturbance_stage)
    if args.environment_profile is not None:
        env_cfg.environment_profile = args.environment_profile
    env_cfg.domain_randomization_spec = args.domain_randomization_spec
    env_cfg.mlp_architecture = architecture.name
    env_cfg.trajectory_eval_mode = True
    env_cfg.tracking_reward_profile = args.reward_profile
    env_cfg.trajectory_eval_align_initial_target = args.align_initial_target
    env_cfg.cap_episode_length = False
    env_cfg.use_boundaries = bool(args.keep_boundaries)
    env_cfg.domain_randomization.use_custom_randomization = False
    _apply_trajectory_configuration(env_cfg, args)
    _apply_disturbance_overlay(env_cfg, args)

    agent_cfg = parse_agent_cfg(args.task, args)
    if args.experiment_name is None:
        agent_cfg.experiment_name = architecture.experiment_name
    agent_cfg.policy.actor_hidden_dims = list(architecture.actor_hidden_dims)
    agent_cfg.policy.critic_hidden_dims = list(architecture.critic_hidden_dims)
    env_cfg.seed = agent_cfg.seed
    args.seed = agent_cfg.seed

    log_root = Path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)))
    checkpoint_path = None
    checkpoint_name = "pid"
    if args.controller == "ppo":
        if not agent_cfg.load_run:
            raise ValueError("PPO evaluation requires an explicit policy run directory via --load_run.")
        checkpoint_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name == "model_0.pt" and not args.allow_initial_checkpoint:
            raise ValueError(
                "Refusing to evaluate model_0.pt because it is the initial checkpoint, not a trained tracking policy. "
                "Select a later model_N.pt checkpoint or pass --allow_initial_checkpoint for a baseline plot."
            )
    return EvaluationSetup(env_cfg, agent_cfg, log_root, checkpoint_path, checkpoint_name)


def build_controller(env: Any, observations: Any, setup: EvaluationSetup, args: Any) -> Any:
    if args.controller == "ppo":
        policy_cfg = setup.agent_cfg.policy
        print(f"[INFO]: Loading model checkpoint from: {setup.checkpoint_path}")
        controller = load_evaluation_actor(
            setup.checkpoint_path,
            observation_dim=observations["policy"].shape[-1],
            action_dim=env.num_actions,
            hidden_dims=list(policy_cfg.actor_hidden_dims),
            activation=policy_cfg.activation,
            device=env.unwrapped.device,
        )
        print("[INFO]: Controller: PPO feed-forward MLP")
        return controller

    gains = PIDGains(
        position_kp=args.pid_position_kp,
        position_ki=args.pid_position_ki,
        velocity_kd=args.pid_velocity_kd,
        attitude_kp=args.pid_attitude_kp,
        attitude_ki=args.pid_attitude_ki,
        angular_velocity_kd=args.pid_angular_velocity_kd,
    )
    print("[INFO]: Controller: 6-DOF PID with measured nonlinear vector-force allocation")
    return PIDTrajectoryController(
        num_envs=env.unwrapped.num_envs,
        dt=env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation,
        thruster_positions_b=env.unwrapped.thruster_com_offsets[0],
        thruster_force_curve_coefficients=env.unwrapped._thruster_force_curve_coefficients,
        mass_kg=env.unwrapped.masses,
        gains=gains,
    )


def evaluation_case_label(args: Any, domain_randomization_name: str | None) -> str:
    return build_evaluation_case_label(
        disturbance_name=args.disturbance_name,
        sample_domain_randomization=bool(args.eval_domain_randomization),
        domain_randomization_name=domain_randomization_name,
        seed=args.seed,
        current_w=args.eval_current,
        smooth_current=args.eval_smooth_current,
        current_variation_std=args.eval_current_variation_std,
        damping_scale=args.eval_damping_scale,
        thruster_scale=args.eval_thruster_scale,
        thruster_tau_scale=args.eval_thruster_tau_scale,
    )


def prepare_evaluation_paths(
    setup: EvaluationSetup,
    args: Any,
    disturbance_label: str,
) -> EvaluationPaths:
    run_name = setup.agent_cfg.load_run if args.controller == "ppo" else "pid"
    checkpoint_stem = Path(setup.checkpoint_name).stem
    result_parts = [checkpoint_stem]
    if args.trajectory != "lissajous":
        result_parts.append(args.trajectory)
    if args.evaluation_label:
        result_parts.append(sanitize_evaluation_label(args.evaluation_label))
    elif disturbance_label:
        result_parts.append(disturbance_label)
    directory = setup.log_root / run_name / "evaluation" / ("_".join(result_parts) + "_trajectory_eval")
    directory.mkdir(parents=True, exist_ok=True)
    return EvaluationPaths(
        directory=directory,
        logs_csv=directory / "logs.csv",
        summary_csv=directory / "summary_metrics.csv",
        domain_samples_csv=directory / "domain_samples.csv",
    )

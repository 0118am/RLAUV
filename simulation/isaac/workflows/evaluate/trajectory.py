import argparse
import json
import math
import os
from collections import deque
from pathlib import Path
import sys

# The evaluator runs as a file from IsaacLab, so make the task checkout
# importable before resolving the pure-data MLP profile registry.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac.agents.ppo.architectures import available_mlp_architectures, get_mlp_architecture
from simulation.isaac.agents.ppo.evaluation import load_evaluation_actor
from simulation.isaac.controllers import PIDGains, PIDTrajectoryController
from isaaclab.app import AppLauncher
from simulation.isaac.workflows.common.evaluation_cases import (
    build_evaluation_case_label,
    sanitize_evaluation_label,
    validate_evaluation_parameters,
)

from simulation.isaac.workflows.common import trajectory_cli as cli_args  # isort: skip


TRAJECTORY_TYPE_IDS = {
    "lissajous": 1,
    "helix": 3,
    "spiral": 4,
    "chirp": 5,
    "racetrack": 6,
    "random_smooth": 7,
    "lateral_sine": 8,
    "vertical_sine": 9,
    "spatial_helix": 10,
}


def _finite_statistic(values, reducer) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(reducer(finite))

# This script evaluates the trajectory task without manually editing
# obs["policy"]. The desired command comes from AUVTrajEnv itself, matching
# the training-time observation/reward path.
parser = argparse.ArgumentParser(description="Evaluate an AUV trajectory-tracking policy.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--pool_dynamics_profile",
    type=str,
    default=None,
    help="Deterministic PoolDynamicsProfile JSON used for evaluation.",
)
parser.add_argument(
    "--domain_randomization_spec",
    type=str,
    default=None,
    help="Versioned DomainRandomizationSpec JSON to audit or sample during evaluation.",
)
parser.add_argument(
    "--eval_domain_randomization",
    action="store_true",
    default=False,
    help="Sample the selected DomainRandomizationSpec while retaining evaluation initial conditions.",
)
parser.add_argument(
    "--eval_disturbance_stage",
    type=int,
    default=-1,
    help="Force a DR curriculum stage during evaluation; -1 keeps the recipe's step schedule.",
)
parser.add_argument(
    "--evaluation_label",
    type=str,
    default="",
    help="Stable label for a held-out evaluation set; included in its result directory and CSV.",
)
parser.add_argument(
    "--keep_boundaries",
    action="store_true",
    default=False,
    help="Retain task boundaries and report out-of-bounds terminations instead of disabling them.",
)
parser.add_argument("--duration", type=float, default=32.0, help="Trajectory evaluation duration in seconds.")
parser.add_argument(
    "--mlp_architecture",
    choices=available_mlp_architectures(),
    default="mlp_history_5",
    help="Named feed-forward input/layer profile used by the checkpoint.",
)
parser.add_argument(
    "--reward_profile",
    type=str,
    default="policy_0",
    help="Versioned tracking reward policy from simulation/isaac/agents/rewards/policy_N.py.",
)
parser.add_argument("--controller", choices=("ppo", "pid"), default="ppo", help="Tracking controller to evaluate.")
parser.add_argument("--pid_position_kp", type=float, nargs=3, default=(20.0, 20.0, 25.0))
parser.add_argument("--pid_position_ki", type=float, nargs=3, default=(0.5, 0.5, 0.8))
parser.add_argument("--pid_velocity_kd", type=float, nargs=3, default=(15.0, 15.0, 18.0))
parser.add_argument("--pid_attitude_kp", type=float, nargs=3, default=(8.0, 8.0, 6.0))
parser.add_argument("--pid_attitude_ki", type=float, nargs=3, default=(0.2, 0.2, 0.15))
parser.add_argument("--pid_angular_velocity_kd", type=float, nargs=3, default=(3.0, 3.0, 2.5))
parser.add_argument("--custom_weights", type=str, default=None, help="Path to a checkpoint outside the log directory.")
parser.add_argument(
    "--allow_initial_checkpoint",
    action="store_true",
    default=False,
    help="Allow model_0.pt evaluation as an explicit untrained-policy baseline.",
)
parser.add_argument(
    "--trajectory",
    type=str,
    default="lissajous",
    choices=TRAJECTORY_TYPE_IDS.keys(),
    help="Fixed eval trajectory to run. Only lissajous appears in the original eval; the others are OOD tests.",
)
parser.add_argument("--trajectory_amp_x", type=float, default=None, help="Override trajectory x amplitude.")
parser.add_argument("--trajectory_amp_y", type=float, default=None, help="Override trajectory y amplitude/radius.")
parser.add_argument("--trajectory_amp_z", type=float, default=None, help="Override trajectory z amplitude.")
parser.add_argument("--trajectory_period", type=float, default=None, help="Override trajectory period.")
parser.add_argument(
    "--trajectory_speed",
    type=float,
    choices=(0.1, 0.2, 0.3, 0.4),
    default=None,
    help="Speed level for lateral_sine, vertical_sine, or spatial_helix.",
)
parser.add_argument(
    "--trajectory_amp_x_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="Random-smooth x-amplitude sampling range; required with the other three ranges.",
)
parser.add_argument(
    "--trajectory_amp_y_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="Random-smooth y-amplitude sampling range; required with the other three ranges.",
)
parser.add_argument(
    "--trajectory_amp_z_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="Random-smooth z-amplitude sampling range; required with the other three ranges.",
)
parser.add_argument(
    "--trajectory_period_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="Random-smooth requested-period sampling range; required with the amplitude ranges.",
)
parser.add_argument("--trajectory_radius_min", type=float, default=None, help="Override spiral minimum radius.")
parser.add_argument("--trajectory_radius_max", type=float, default=None, help="Override spiral maximum radius.")
parser.add_argument(
    "--random_curve_count",
    type=int,
    default=5,
    help="Number of random smooth curves to evaluate in parallel when --trajectory=random_smooth.",
)
parser.add_argument(
    "--align_initial_target",
    action="store_true",
    default=False,
    help="Start the vehicle on the first eval target instead of at the trajectory center.",
)
parser.add_argument(
    "--disable_trajectory_vis",
    action="store_true",
    default=False,
    help="Disable live desired/actual trajectory drawing in GUI eval.",
)
parser.add_argument(
    "--trail_stride",
    type=int,
    default=4,
    help="Draw live trajectory trails every N policy steps in GUI eval.",
)
parser.add_argument(
    "--trail_max_points",
    type=int,
    default=2500,
    help="Maximum desired/actual points kept in the live GUI trail.",
)
parser.add_argument(
    "--hold_open",
    action="store_true",
    default=False,
    help="Keep Isaac Sim open after eval so the live trails can be inspected.",
)
parser.add_argument(
    "--eval_current",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "VZ"),
    help="Fixed world-frame water current in m/s for disturbance eval.",
)
parser.add_argument(
    "--eval_smooth_current",
    action="store_true",
    default=False,
    help="Let eval_current drift smoothly around its mean with a low-frequency current model.",
)
parser.add_argument(
    "--eval_current_variation_std",
    type=float,
    default=0.0,
    help="Std of smooth current variation in m/s. Used with --eval_smooth_current.",
)
parser.add_argument(
    "--eval_current_tau",
    type=float,
    default=12.0,
    help="Time constant in seconds for smooth current disturbance eval.",
)
parser.add_argument("--eval_damping_scale", type=float, default=1.0, help="Multiply linear/quadratic damping.")
parser.add_argument("--eval_thruster_scale", type=float, default=1.0, help="Multiply all thruster force outputs.")
parser.add_argument(
    "--eval_thruster_tau_scale",
    type=float,
    default=1.0,
    help="Multiply the first-order thruster response time constant.",
)
parser.add_argument(
    "--disturbance_name",
    type=str,
    default=None,
    help="Optional label used in the output directory name for disturbance eval.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
validate_evaluation_parameters(
    duration_s=args_cli.duration,
    current_w=args_cli.eval_current,
    current_variation_std=args_cli.eval_current_variation_std,
    current_tau=args_cli.eval_current_tau,
    damping_scale=args_cli.eval_damping_scale,
    thruster_scale=args_cli.eval_thruster_scale,
    thruster_tau_scale=args_cli.eval_thruster_tau_scale,
    num_envs=args_cli.num_envs,
    random_curve_count=args_cli.random_curve_count,
)
if args_cli.eval_domain_randomization and not args_cli.domain_randomization_spec:
    parser.error("--eval_domain_randomization requires --domain_randomization_spec")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply, quat_error_magnitude



def _direction_error(vector_a: torch.Tensor, vector_b: torch.Tensor, min_norm: float = 1.0e-3) -> torch.Tensor:
    """Return angular direction error, using NaN when either direction is undefined."""

    norm_a = torch.norm(vector_a, dim=1)
    norm_b = torch.norm(vector_b, dim=1)
    cosine = torch.sum(vector_a * vector_b, dim=1) / torch.clamp(norm_a * norm_b, min=min_norm**2)
    angle = torch.acos(torch.clamp(cosine, min=-1.0, max=1.0))
    valid = (norm_a > min_norm) & (norm_b > min_norm)
    return torch.where(valid, angle, torch.full_like(angle, float("nan")))


def _resolve_positive_range(scalar: float | None, value_range: list[float] | None, name: str) -> list[float] | None:
    if scalar is not None and value_range is not None:
        raise ValueError(f"Specify only one of {name} or {name}_range.")
    if scalar is not None:
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return [float(scalar), float(scalar)]
    if value_range is None:
        return None
    lower, upper = (float(value_range[0]), float(value_range[1]))
    if lower <= 0.0 or upper < lower:
        raise ValueError(f"{name}_range must satisfy 0 < MIN <= MAX.")
    return [lower, upper]


def _resolve_random_smooth_ranges() -> dict[str, list[float]]:
    values = {
        "trajectory_amp_x_range": _resolve_positive_range(
            args_cli.trajectory_amp_x, args_cli.trajectory_amp_x_range, "trajectory_amp_x"
        ),
        "trajectory_amp_y_range": _resolve_positive_range(
            args_cli.trajectory_amp_y, args_cli.trajectory_amp_y_range, "trajectory_amp_y"
        ),
        "trajectory_amp_z_range": _resolve_positive_range(
            args_cli.trajectory_amp_z, args_cli.trajectory_amp_z_range, "trajectory_amp_z"
        ),
        "trajectory_period_range": _resolve_positive_range(
            args_cli.trajectory_period, args_cli.trajectory_period_range, "trajectory_period"
        ),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "random_smooth evaluation requires explicit positive amplitude and period ranges; missing "
            + ", ".join(missing)
            + "."
        )
    return {name: value for name, value in values.items() if value is not None}


def _resolve_checkpoint(log_root_path: str, agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    if args_cli.custom_weights:
        return args_cli.custom_weights
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _disturbance_label(
    domain_randomization_spec_name: str | None = None,
) -> str:
    return build_evaluation_case_label(
        disturbance_name=args_cli.disturbance_name,
        sample_domain_randomization=bool(args_cli.eval_domain_randomization),
        domain_randomization_name=domain_randomization_spec_name,
        seed=args_cli.seed,
        current_w=args_cli.eval_current,
        smooth_current=args_cli.eval_smooth_current,
        current_variation_std=args_cli.eval_current_variation_std,
        damping_scale=args_cli.eval_damping_scale,
        thruster_scale=args_cli.eval_thruster_scale,
        thruster_tau_scale=args_cli.eval_thruster_tau_scale,
    )


def _apply_eval_cfg_disturbance(env_cfg) -> None:
    # The environment applies this after resolving its profile and DR recipe.
    env_cfg.evaluation_physics_overlay = {
        "damping_scale": float(args_cli.eval_damping_scale),
        "thruster_tau_scale": float(args_cli.eval_thruster_tau_scale),
    }
    if args_cli.eval_current is not None:
        env_cfg.evaluation_physics_overlay["water_current_w"] = [float(value) for value in args_cli.eval_current]
    smooth_current = args_cli.eval_smooth_current or args_cli.eval_current_variation_std > 0.0
    if smooth_current:
        env_cfg.evaluation_physics_overlay.update(
            {
                "smooth_current": True,
                "current_variation_std": float(args_cli.eval_current_variation_std),
                "current_tau": float(args_cli.eval_current_tau),
                "current_feature_only": not bool(args_cli.eval_domain_randomization),
            }
        )
    if abs(args_cli.eval_thruster_scale - 1.0) > 1.0e-9:
        env_cfg.evaluation_physics_overlay["thruster_force_scale"] = float(args_cli.eval_thruster_scale)


class TrajectoryEvalVisualizer:
    """Small GUI-only helper for drawing desired and actual eval trails."""

    def __init__(
        self,
        enabled: bool,
        trajectory: str,
        checkpoint_name: str,
        max_points: int,
        stride: int,
    ):
        self.enabled = enabled
        self.trajectory = trajectory
        self.checkpoint_name = checkpoint_name
        self.stride = max(1, stride)
        self.desired_points = deque(maxlen=max(2, max_points))
        self.actual_points = deque(maxlen=max(2, max_points))
        self._draw = None
        self._labels = {}

        if not self.enabled:
            return

        self._init_debug_draw()
        self._init_status_window()

    def _init_debug_draw(self):
        try:
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except Exception:
                from omni.isaac.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._clear_draw()
        except Exception as exc:
            self.enabled = False
            print(f"[WARN]: Live trajectory drawing is unavailable: {exc}")

    def _init_status_window(self):
        try:
            import omni.ui as ui

            self._window = ui.Window("Trajectory Eval", width=360, height=150)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    ui.Label(f"trajectory: {self.trajectory}")
                    ui.Label(f"checkpoint: {self.checkpoint_name}")
                    ui.Label("blue: desired target trail")
                    ui.Label("orange: actual AUV trail")
                    self._labels["time"] = ui.Label("time: 0.00 s")
                    self._labels["error"] = ui.Label("pos err: -- m | vel err: -- m/s")
        except Exception as exc:
            print(f"[WARN]: Trajectory status window is unavailable: {exc}")

    @staticmethod
    def _point(tensor: torch.Tensor) -> tuple[float, float, float]:
        values = tensor.detach().cpu().tolist()
        return float(values[0]), float(values[1]), float(values[2])

    def update(
        self,
        step: int,
        time_s: float,
        desired_pos_w: torch.Tensor,
        actual_pos_w: torch.Tensor,
        position_error: float,
        velocity_error: float,
    ):
        if not self.enabled:
            return

        self.desired_points.append(self._point(desired_pos_w))
        self.actual_points.append(self._point(actual_pos_w))

        if step % self.stride == 0:
            self._draw_trails()
            self._update_labels(time_s, position_error, velocity_error)

    def _draw_trails(self):
        if self._draw is None:
            return

        desired = list(self.desired_points)
        actual = list(self.actual_points)
        start_points = desired[:-1] + actual[:-1]
        end_points = desired[1:] + actual[1:]
        colors = [(0.1, 0.45, 1.0, 1.0)] * max(0, len(desired) - 1)
        colors += [(1.0, 0.45, 0.05, 1.0)] * max(0, len(actual) - 1)
        widths = [3.0] * len(start_points)

        self._clear_draw()
        if start_points:
            self._draw.draw_lines(start_points, end_points, colors, widths)
        if desired and actual:
            self._draw.draw_points(
                [desired[-1], actual[-1]],
                [(1.0, 0.95, 0.05, 1.0), (1.0, 0.95, 0.95, 1.0)],
                [18.0, 12.0],
            )

    def _clear_draw(self):
        if self._draw is None:
            return
        if hasattr(self._draw, "clear_lines"):
            self._draw.clear_lines()
        if hasattr(self._draw, "clear_points"):
            self._draw.clear_points()

    def _update_labels(self, time_s: float, position_error: float, velocity_error: float):
        time_label = self._labels.get("time")
        error_label = self._labels.get("error")
        if time_label is not None:
            time_label.text = f"time: {time_s:.2f} s"
        if error_label is not None:
            error_label.text = f"pos err: {position_error:.3f} m | vel err: {velocity_error:.3f} m/s"

    def set_status(self, message: str):
        label = self._labels.get("time")
        if label is not None:
            label.text = message


def main():
    if args_cli.eval_domain_randomization and not args_cli.domain_randomization_spec:
        raise ValueError("--eval_domain_randomization requires --domain_randomization_spec.")
    eval_num_envs = args_cli.num_envs or 1
    if args_cli.trajectory == "random_smooth":
        eval_num_envs = args_cli.num_envs or args_cli.random_curve_count
    eval_num_envs = max(1, eval_num_envs)

    architecture = get_mlp_architecture(args_cli.mlp_architecture)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=eval_num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Use the env target generator for fixed trajectories or random smooth
    # generalization tests.  This avoids manually editing obs["policy"].
    env_cfg.eval_mode = True
    env_cfg.eval_domain_randomization = bool(args_cli.eval_domain_randomization)
    env_cfg.eval_disturbance_stage = int(args_cli.eval_disturbance_stage)
    env_cfg.pool_dynamics_profile = args_cli.pool_dynamics_profile
    env_cfg.domain_randomization_spec = args_cli.domain_randomization_spec
    env_cfg.mlp_architecture = architecture.name
    env_cfg.trajectory_eval_mode = True
    env_cfg.tracking_reward_profile = args_cli.reward_profile
    env_cfg.trajectory_eval_type = TRAJECTORY_TYPE_IDS[args_cli.trajectory]
    env_cfg.trajectory_eval_duration_s = args_cli.duration
    random_smooth_ranges = _resolve_random_smooth_ranges() if args_cli.trajectory == "random_smooth" else None
    if args_cli.trajectory_amp_x is not None:
        env_cfg.trajectory_eval_amp_x = args_cli.trajectory_amp_x
    if args_cli.trajectory_amp_y is not None:
        env_cfg.trajectory_eval_amp_y = args_cli.trajectory_amp_y
    if args_cli.trajectory_amp_z is not None:
        env_cfg.trajectory_eval_amp_z = args_cli.trajectory_amp_z
    if args_cli.trajectory_period is not None:
        env_cfg.trajectory_eval_period = args_cli.trajectory_period
    if args_cli.trajectory_speed is not None:
        env_cfg.trajectory_eval_speed_mps = args_cli.trajectory_speed
    if args_cli.trajectory_radius_min is not None:
        env_cfg.trajectory_eval_radius_min = args_cli.trajectory_radius_min
    if args_cli.trajectory_radius_max is not None:
        env_cfg.trajectory_eval_radius_max = args_cli.trajectory_radius_max
    if args_cli.trajectory == "random_smooth":
        assert random_smooth_ranges is not None
        env_cfg.trajectory_amp_x_range = random_smooth_ranges["trajectory_amp_x_range"]
        env_cfg.trajectory_amp_y_range = random_smooth_ranges["trajectory_amp_y_range"]
        env_cfg.trajectory_amp_z_range = random_smooth_ranges["trajectory_amp_z_range"]
        env_cfg.trajectory_period_range = random_smooth_ranges["trajectory_period_range"]
    env_cfg.trajectory_eval_align_initial_target = args_cli.align_initial_target
    env_cfg.cap_episode_length = False
    env_cfg.use_boundaries = bool(args_cli.keep_boundaries)
    env_cfg.domain_randomization.use_custom_randomization = False
    _apply_eval_cfg_disturbance(env_cfg)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.experiment_name is None:
        agent_cfg.experiment_name = architecture.experiment_name
    agent_cfg.policy.actor_hidden_dims = list(architecture.actor_hidden_dims)
    agent_cfg.policy.critic_hidden_dims = list(architecture.critic_hidden_dims)
    env_cfg.seed = agent_cfg.seed
    args_cli.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None
    checkpoint_name = "pid"
    if args_cli.controller == "ppo":
        resume_path = _resolve_checkpoint(log_root_path, agent_cfg)
        checkpoint_name = os.path.basename(resume_path)
        if checkpoint_name == "model_0.pt" and not args_cli.allow_initial_checkpoint:
            raise ValueError(
                "Refusing to evaluate model_0.pt because it is the initial checkpoint, not a trained tracking policy. "
                "Select a later model_N.pt checkpoint or pass --allow_initial_checkpoint for a baseline plot."
            )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    print(f"[INFO]: Tracking reward profile: {args_cli.reward_profile}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    obs = env.get_observations()
    if args_cli.controller == "ppo":
        policy_cfg = agent_cfg.policy
        policy = load_evaluation_actor(
            resume_path,
            observation_dim=obs["policy"].shape[-1],
            action_dim=env.num_actions,
            hidden_dims=list(policy_cfg.actor_hidden_dims),
            activation=policy_cfg.activation,
            device=env.unwrapped.device,
        )
        print("[INFO]: Controller: PPO feed-forward MLP")
    else:
        gains = PIDGains(
            position_kp=args_cli.pid_position_kp,
            position_ki=args_cli.pid_position_ki,
            velocity_kd=args_cli.pid_velocity_kd,
            attitude_kp=args_cli.pid_attitude_kp,
            attitude_ki=args_cli.pid_attitude_ki,
            angular_velocity_kd=args_cli.pid_angular_velocity_kd,
        )
        policy = PIDTrajectoryController(
            num_envs=env.unwrapped.num_envs,
            dt=env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation,
            thruster_positions_b=env.unwrapped.thruster_com_offsets[0],
            thruster_force_curve_coefficients=env.unwrapped._thruster_force_curve_coefficients,
            position_scale_m=env.unwrapped.cfg.observation_position_scale_m,
            linear_velocity_scale_mps=env.unwrapped.cfg.observation_linear_velocity_scale_mps,
            angular_velocity_scale_radps=env.unwrapped.cfg.observation_angular_velocity_scale_radps,
            linear_acceleration_scale_mps2=env.unwrapped.cfg.observation_linear_acceleration_scale_mps2,
            mass_kg=env.unwrapped.masses,
            gains=gains,
        )
        print("[INFO]: Controller: 6-DOF PID with measured nonlinear vector-force allocation")

    # Results mirror the existing play/eval directory layout but use a distinct
    # suffix so repeated trajectory evaluations do not overwrite prior logs.
    run_name = (agent_cfg.load_run or "custom_weights") if args_cli.controller == "ppo" else "pid"
    checkpoint_stem = checkpoint_name[:-3] if checkpoint_name.endswith(".pt") else checkpoint_name
    disturbance_label = _disturbance_label(
        getattr(env.unwrapped.cfg, "domain_randomization_spec_name", None),
    )
    result_parts = [checkpoint_stem]
    if args_cli.trajectory != "lissajous":
        result_parts.append(args_cli.trajectory)
    if args_cli.evaluation_label:
        result_parts.append(sanitize_evaluation_label(args_cli.evaluation_label))
    elif disturbance_label:
        result_parts.append(disturbance_label)
    result_dir_name = "_".join(result_parts) + "_trajectory_eval"

    save_path = os.path.join(
        "results",
        "rsl_rl",
        agent_cfg.experiment_name,
        run_name,
        result_dir_name,
    )
    os.makedirs(save_path, exist_ok=True)
    logs_csv_path = os.path.join(save_path, "logs.csv")
    summary_csv_path = os.path.join(save_path, "summary_metrics.csv")
    domain_samples_csv_path = os.path.join(save_path, "domain_samples.csv")
    print(f"[INFO]: Saving trajectory eval results into: {save_path}")

    masses = env.unwrapped.masses.reshape(-1)
    volumes = env.unwrapped.volumes.reshape(-1)
    center_of_mass_offsets = env.unwrapped.center_of_mass_offsets
    com_to_cob_offsets = env.unwrapped.com_to_cob_offsets
    principal_inertias = env.unwrapped.inertia_principal_moments
    payload_sample_indices = env.unwrapped.payload_sample_indices
    linear_damping = env.unwrapped.linear_damping.reshape(env.unwrapped.num_envs, -1)
    quadratic_damping = env.unwrapped.quadratic_damping.reshape(env.unwrapped.num_envs, -1)
    added_mass = env.unwrapped.added_mass_diag.reshape(env.unwrapped.num_envs, -1)
    thruster_force_scale = env.unwrapped.thruster_force_scale
    thruster_time_constant = env.unwrapped.thruster_time_constant.reshape(-1)
    thruster_delay_steps = env.unwrapped.thruster_delay_steps.reshape(-1)
    thruster_max_command_rate = env.unwrapped.thruster_max_command_rate.reshape(-1)
    thruster_command_resolution = env.unwrapped.thruster_command_resolution
    thruster_command_dropout_probability = env.unwrapped.thruster_command_dropout_probability
    battery_voltage = env.unwrapped.battery_voltage.reshape(-1)
    observation_noise_std = env.unwrapped.observation_noise_std.reshape(env.unwrapped.num_envs, -1)
    observation_delay_steps = env.unwrapped.observation_delay_steps.reshape(-1)
    observation_update_period_steps = env.unwrapped.observation_update_period_steps.reshape(-1)
    observation_dropout_probability = env.unwrapped.observation_dropout_probability
    observation_lowpass_alpha = env.unwrapped.observation_lowpass_alpha
    observation_bias_drift_std = env.unwrapped.observation_bias_drift_std
    physics_dt = float(env.unwrapped.cfg.sim.dt)
    policy_dt = physics_dt * int(env.unwrapped.cfg.decimation)

    domain_rows = []
    for env_id in range(env.unwrapped.num_envs):
        domain_rows.append(
            {
                "env_id": env_id,
                "seed": int(env.unwrapped.cfg.seed),
                "pool_dynamics_profile_name": env.unwrapped.cfg.pool_dynamics_profile_name,
                "domain_randomization_spec_name": env.unwrapped.cfg.domain_randomization_spec_name or "",
                "sampled_mass_kg": masses[env_id].cpu().item(),
                "sampled_volume_m3": volumes[env_id].cpu().item(),
                "payload_sample_index": payload_sample_indices[env_id].cpu().item(),
                "sampled_center_of_mass_x_m": center_of_mass_offsets[env_id, 0].cpu().item(),
                "sampled_center_of_mass_y_m": center_of_mass_offsets[env_id, 1].cpu().item(),
                "sampled_center_of_mass_z_m": center_of_mass_offsets[env_id, 2].cpu().item(),
                "sampled_com_to_cob_x_m": com_to_cob_offsets[env_id, 0].cpu().item(),
                "sampled_com_to_cob_y_m": com_to_cob_offsets[env_id, 1].cpu().item(),
                "sampled_com_to_cob_z_m": com_to_cob_offsets[env_id, 2].cpu().item(),
                "sampled_principal_inertia_x_kg_m2": principal_inertias[env_id, 0].cpu().item(),
                "sampled_principal_inertia_y_kg_m2": principal_inertias[env_id, 1].cpu().item(),
                "sampled_principal_inertia_z_kg_m2": principal_inertias[env_id, 2].cpu().item(),
                "sampled_linear_damping_l2": torch.linalg.vector_norm(linear_damping[env_id]).cpu().item(),
                "sampled_quadratic_damping_l2": torch.linalg.vector_norm(
                    quadratic_damping[env_id]
                ).cpu().item(),
                "sampled_added_mass_l2": torch.linalg.vector_norm(added_mass[env_id]).cpu().item(),
                "sampled_thruster_scale_mean": thruster_force_scale[env_id].mean().cpu().item(),
                "sampled_thruster_scale_min": thruster_force_scale[env_id].min().cpu().item(),
                "sampled_thruster_scale_max": thruster_force_scale[env_id].max().cpu().item(),
                "sampled_thruster_time_constant_s": thruster_time_constant[env_id].cpu().item(),
                "sampled_thruster_delay_steps": thruster_delay_steps[env_id].cpu().item(),
                "sampled_thruster_delay_s": thruster_delay_steps[env_id].cpu().item() * physics_dt,
                "sampled_thruster_max_command_rate_per_s": thruster_max_command_rate[env_id].cpu().item(),
                "sampled_thruster_command_resolution_mean": thruster_command_resolution[
                    env_id
                ].mean().cpu().item(),
                "sampled_thruster_command_dropout_probability_mean": thruster_command_dropout_probability[
                    env_id
                ].mean().cpu().item(),
                "sampled_battery_voltage_v": battery_voltage[env_id].cpu().item(),
                "sampled_observation_noise_std_mean": observation_noise_std[
                    env_id
                ].mean().cpu().item(),
                "sampled_observation_delay_steps": observation_delay_steps[env_id].cpu().item(),
                "sampled_observation_delay_s": observation_delay_steps[env_id].cpu().item() * policy_dt,
                "sampled_observation_update_period_steps": observation_update_period_steps[env_id].cpu().item(),
                "sampled_observation_update_period_s": observation_update_period_steps[
                    env_id
                ].cpu().item() * policy_dt,
                "sampled_observation_dropout_probability_mean": observation_dropout_probability[
                    env_id
                ].mean().cpu().item(),
                "sampled_observation_lowpass_alpha_mean": observation_lowpass_alpha[env_id].mean().cpu().item(),
                "sampled_observation_bias_drift_std_mean": observation_bias_drift_std[
                    env_id
                ].mean().cpu().item(),
            }
        )
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(domain_samples_csv_path, index=False)
    step_dt = policy_dt
    num_steps = int(math.ceil(args_cli.duration / step_dt))
    log_rows = []
    previous_eval_actions = None
    previous_applied_actions = None
    termination_events = 0
    active_envs = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    any_failure = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    first_failure_time_s = torch.full(
        (env.unwrapped.num_envs,), float("nan"), dtype=torch.float32, device=env.unwrapped.device
    )
    visualizer = TrajectoryEvalVisualizer(
        enabled=not args_cli.disable_trajectory_vis and not getattr(args_cli, "headless", False),
        trajectory=args_cli.trajectory,
        checkpoint_name=checkpoint_name,
        max_points=args_cli.trail_max_points,
        stride=args_cli.trail_stride,
    )

    for step in range(num_steps):
        if not bool(torch.any(active_envs)):
            break
        t = step * step_dt
        with torch.inference_mode():
            active_ids = torch.nonzero(active_envs, as_tuple=False).reshape(-1).tolist()
            # Pull target state from the env after it synchronizes to the
            # current episode time.  This keeps logs aligned with policy input.
            target_pos_w, target_lin_vel_w, target_quat_w = env.unwrapped.get_tracking_targets()
            tracking_kinematics = env.unwrapped.get_tracking_kinematics()
            target_lin_acc_w = tracking_kinematics["target_acceleration_w"]
            target_lin_jerk_w = tracking_kinematics["target_jerk_w"]
            target_curvature_m_inv = tracking_kinematics["target_curvature_m_inv"]
            target_orientation_rate_radps = tracking_kinematics["target_orientation_rate_radps"]
            requested_period_s = tracking_kinematics["requested_period_s"]
            requested_speed_mps = tracking_kinematics["requested_speed_mps"]
            effective_period_s = tracking_kinematics["effective_period_s"]
            retimed = tracking_kinematics["retimed"]
            target_speeds = torch.linalg.vector_norm(target_lin_vel_w, dim=1)
            target_accelerations = torch.linalg.vector_norm(target_lin_acc_w, dim=1)
            target_jerks = torch.linalg.vector_norm(target_lin_jerk_w, dim=1)
            root_pos_w = env.unwrapped._robot.data.root_pos_w
            root_quat_w = env.unwrapped._robot.data.root_quat_w
            root_lin_vel_w = quat_apply(root_quat_w, env.unwrapped._robot.data.root_lin_vel_b)
            root_lin_vel_b = env.unwrapped._robot.data.root_lin_vel_b
            root_ang_vel_b = env.unwrapped._robot.data.root_ang_vel_b
            root_pos_local = root_pos_w - env.unwrapped.scene.env_origins

            position_errors = torch.norm(target_pos_w - root_pos_w, dim=1)
            velocity_errors = torch.norm(target_lin_vel_w - root_lin_vel_w, dim=1)
            attitude_errors = quat_error_magnitude(target_quat_w, root_quat_w)
            body_x_b = torch.zeros_like(root_lin_vel_b)
            body_x_b[:, 0] = 1.0
            nose_direction_w = quat_apply(root_quat_w, body_x_b)
            command_heading_errors = _direction_error(nose_direction_w, target_lin_vel_w)
            motion_sideslip_errors = _direction_error(nose_direction_w, root_lin_vel_w)
            visualized_env_id = int(active_ids[0])
            visualizer.update(
                step,
                t,
                target_pos_w[visualized_env_id],
                root_pos_w[visualized_env_id],
                position_errors[visualized_env_id].cpu().item(),
                velocity_errors[visualized_env_id].cpu().item(),
            )

            raw_policy_actions = policy(obs)
            # Match the environment boundary explicitly. ``action_*`` below
            # means the command passed to the actuator chain, while the raw
            # actor output remains available for diagnosing saturation.
            actions = torch.clamp(raw_policy_actions, -1.0, 1.0)
            # Inactive rows have already completed their one evaluation episode.
            # They still have to be stepped as part of the vector environment,
            # but they cannot affect logged metrics or receive new commands.
            actions = torch.where(active_envs.unsqueeze(-1), actions, torch.zeros_like(actions))
            action_norms = torch.norm(actions, dim=1)
            action_rms = torch.sqrt(torch.mean(actions**2, dim=1))
            action_delta = torch.zeros_like(actions) if previous_eval_actions is None else actions - previous_eval_actions
            action_rate_rms = torch.sqrt(torch.mean(action_delta**2, dim=1))
            previous_eval_actions = actions.clone()
            raw_action_clip_mask = raw_policy_actions.abs() > 1.0
            water_current_w = env.unwrapped.water_current_w
            row_indices: dict[int, int] = {}
            for env_id in active_ids:
                row_indices[env_id] = len(log_rows)
                log_rows.append(
                    {
                        "trajectory": args_cli.trajectory,
                        "reward_profile": args_cli.reward_profile,
                        "disturbance": disturbance_label or "nominal",
                        "env_id": env_id,
                        "episode_id": 0,
                        "episode_step": step,
                        "time": t,
                        "time_s": t,
                        "water_current_x": water_current_w[env_id, 0].cpu().item(),
                        "water_current_y": water_current_w[env_id, 1].cpu().item(),
                        "water_current_z": water_current_w[env_id, 2].cpu().item(),
                        "desired_x": target_pos_w[env_id, 0].cpu().item(),
                        "desired_y": target_pos_w[env_id, 1].cpu().item(),
                        "desired_z": target_pos_w[env_id, 2].cpu().item(),
                        "true_x": root_pos_w[env_id, 0].cpu().item(),
                        "true_y": root_pos_w[env_id, 1].cpu().item(),
                        "true_z": root_pos_w[env_id, 2].cpu().item(),
                        "desired_vx": target_lin_vel_w[env_id, 0].cpu().item(),
                        "desired_vy": target_lin_vel_w[env_id, 1].cpu().item(),
                        "desired_vz": target_lin_vel_w[env_id, 2].cpu().item(),
                        "target_speed_mps": target_speeds[env_id].cpu().item(),
                        "requested_speed_mps": requested_speed_mps[env_id].cpu().item(),
                        "target_acceleration_mps2": target_accelerations[env_id].cpu().item(),
                        "target_jerk_mps3": target_jerks[env_id].cpu().item(),
                        "target_curvature_m_inv": target_curvature_m_inv[env_id].cpu().item(),
                        "target_orientation_rate_radps": target_orientation_rate_radps[env_id].cpu().item(),
                        "requested_period_s": requested_period_s[env_id].cpu().item(),
                        "effective_period_s": effective_period_s[env_id].cpu().item(),
                        "trajectory_retimed": float(retimed[env_id].cpu().item()),
                        "true_vx": root_lin_vel_w[env_id, 0].cpu().item(),
                        "true_vy": root_lin_vel_w[env_id, 1].cpu().item(),
                        "true_vz": root_lin_vel_w[env_id, 2].cpu().item(),
                        "position_w_x_m": root_pos_local[env_id, 0].cpu().item(),
                        "position_w_y_m": root_pos_local[env_id, 1].cpu().item(),
                        "position_w_z_m": root_pos_local[env_id, 2].cpu().item(),
                        "quat_w": root_quat_w[env_id, 0].cpu().item(),
                        "quat_x": root_quat_w[env_id, 1].cpu().item(),
                        "quat_y": root_quat_w[env_id, 2].cpu().item(),
                        "quat_z": root_quat_w[env_id, 3].cpu().item(),
                        "linear_velocity_w_x_mps": root_lin_vel_w[env_id, 0].cpu().item(),
                        "linear_velocity_w_y_mps": root_lin_vel_w[env_id, 1].cpu().item(),
                        "linear_velocity_w_z_mps": root_lin_vel_w[env_id, 2].cpu().item(),
                        "angular_velocity_b_x_radps": root_ang_vel_b[env_id, 0].cpu().item(),
                        "angular_velocity_b_y_radps": root_ang_vel_b[env_id, 1].cpu().item(),
                        "angular_velocity_b_z_radps": root_ang_vel_b[env_id, 2].cpu().item(),
                        **{
                            f"action_{action_index}": actions[env_id, action_index].cpu().item()
                            for action_index in range(actions.shape[1])
                        },
                        **{
                            f"raw_policy_action_{action_index}": raw_policy_actions[env_id, action_index].cpu().item()
                            for action_index in range(raw_policy_actions.shape[1])
                        },
                        **{
                            f"raw_policy_action_clipped_{action_index}": float(
                                raw_action_clip_mask[env_id, action_index].cpu().item()
                            )
                            for action_index in range(raw_policy_actions.shape[1])
                        },
                        "position_error": position_errors[env_id].cpu().item(),
                        "velocity_error": velocity_errors[env_id].cpu().item(),
                        "attitude_error": attitude_errors[env_id].cpu().item(),
                        "command_heading_error_rad": command_heading_errors[env_id].cpu().item(),
                        "motion_sideslip_error_rad": motion_sideslip_errors[env_id].cpu().item(),
                        "action_norm": action_norms[env_id].cpu().item(),
                        "action_rms": action_rms[env_id].cpu().item(),
                        "action_rate_rms": action_rate_rms[env_id].cpu().item(),
                        "raw_policy_action_norm": torch.norm(raw_policy_actions[env_id]).cpu().item(),
                        "raw_policy_action_rms": torch.sqrt(
                            torch.mean(raw_policy_actions[env_id] ** 2)
                        ).cpu().item(),
                        "raw_policy_action_clip_fraction": raw_action_clip_mask[env_id]
                        .to(dtype=torch.float32)
                        .mean()
                        .cpu()
                        .item(),
                    }
                )

            obs, rewards, _, _ = env.step(actions)
            # DirectRLEnv resets automatically. Each vector row contributes
            # exactly one episode; after its first safety termination it is
            # excluded from all later logs and aggregate metrics.
            terminated = env.unwrapped.reset_terminated.clone() & active_envs
            termination_events += int(torch.count_nonzero(terminated).cpu().item())
            any_failure |= terminated
            first_failure_time_s = torch.where(
                terminated & torch.isnan(first_failure_time_s),
                torch.full_like(first_failure_time_s, t + step_dt),
                first_failure_time_s,
            )
            applied_actions = env.unwrapped.thruster_command_processor.rate_limited_state
            realized_thruster_force_n = env.unwrapped.realized_thruster_force_n
            realized_force_wrench = env.unwrapped._thrust[:, 0, :]
            realized_torque_wrench = env.unwrapped._moment[:, 0, :]
            applied_delta = torch.zeros_like(applied_actions)
            if previous_applied_actions is not None:
                applied_delta = applied_actions - previous_applied_actions
            previous_applied_actions = applied_actions.clone()
            requested_to_applied = actions - applied_actions
            for env_id in active_ids:
                row = log_rows[row_indices[env_id]]
                row["reward"] = rewards[env_id].cpu().item()
                if bool(terminated[env_id]):
                    # Auto-reset has already cleared these tensors, so zeros
                    # would falsely describe the terminal transition. Preserve
                    # reward/done but mark post-step actuator data unavailable.
                    row.update(
                        {
                            **{f"applied_action_{index}": float("nan") for index in range(applied_actions.shape[1])},
                            **{
                                f"requested_to_applied_action_delta_{index}": float("nan")
                                for index in range(applied_actions.shape[1])
                            },
                            **{
                                f"realized_thruster_force_{index}_n": float("nan")
                                for index in range(realized_thruster_force_n.shape[1])
                            },
                            "requested_to_applied_action_rms": float("nan"),
                            "applied_action_rate_rms": float("nan"),
                            "realized_thruster_force_abs_mean_n": float("nan"),
                            "realized_thruster_force_abs_max_n": float("nan"),
                            "realized_wrench_force_x_n": float("nan"),
                            "realized_wrench_force_y_n": float("nan"),
                            "realized_wrench_force_z_n": float("nan"),
                            "realized_wrench_torque_x_nm": float("nan"),
                            "realized_wrench_torque_y_nm": float("nan"),
                            "realized_wrench_torque_z_nm": float("nan"),
                            "safety_terminated": 1.0,
                        }
                    )
                    continue
                row.update(
                    {
                        **{
                            f"applied_action_{action_index}": applied_actions[env_id, action_index].cpu().item()
                            for action_index in range(applied_actions.shape[1])
                        },
                        **{
                            f"requested_to_applied_action_delta_{action_index}": requested_to_applied[
                                env_id, action_index
                            ]
                            .cpu()
                            .item()
                            for action_index in range(applied_actions.shape[1])
                        },
                        **{
                            f"realized_thruster_force_{action_index}_n": realized_thruster_force_n[
                                env_id, action_index
                            ]
                            .cpu()
                            .item()
                            for action_index in range(realized_thruster_force_n.shape[1])
                        },
                        "requested_to_applied_action_rms": torch.sqrt(
                            torch.mean(requested_to_applied[env_id] ** 2)
                        )
                        .cpu()
                        .item(),
                        "applied_action_rate_rms": torch.sqrt(torch.mean(applied_delta[env_id] ** 2)).cpu().item(),
                        "realized_thruster_force_abs_mean_n": realized_thruster_force_n[env_id]
                        .abs()
                        .mean()
                        .cpu()
                        .item(),
                        "realized_thruster_force_abs_max_n": realized_thruster_force_n[env_id]
                        .abs()
                        .max()
                        .cpu()
                        .item(),
                        "realized_wrench_force_x_n": realized_force_wrench[env_id, 0].cpu().item(),
                        "realized_wrench_force_y_n": realized_force_wrench[env_id, 1].cpu().item(),
                        "realized_wrench_force_z_n": realized_force_wrench[env_id, 2].cpu().item(),
                        "realized_wrench_torque_x_nm": realized_torque_wrench[env_id, 0].cpu().item(),
                        "realized_wrench_torque_y_nm": realized_torque_wrench[env_id, 1].cpu().item(),
                        "realized_wrench_torque_z_nm": realized_torque_wrench[env_id, 2].cpu().item(),
                        "safety_terminated": float(terminated[env_id].cpu().item()),
                    }
                )
            active_envs &= ~terminated

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(logs_csv_path, index=False)

    # Keep scalar summary metrics in a separate CSV for quick experiment
    # comparisons without loading the full trajectory log.
    position_errors = log_df["position_error"].to_numpy()
    velocity_errors = log_df["velocity_error"].to_numpy()
    reference_path_lengths: list[float] = []
    reference_speed_p95: list[float] = []
    for _, curve_rows in log_df.groupby("env_id", sort=True):
        ordered = curve_rows.sort_values("time_s")
        desired_positions = ordered[["desired_x", "desired_y", "desired_z"]].to_numpy(dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(desired_positions, axis=0), axis=1)
        reference_path_lengths.append(float(np.sum(segment_lengths)))
        reference_speed_p95.append(float(np.quantile(ordered["target_speed_mps"].to_numpy(), 0.95)))
    max_target_speed = float(log_df["target_speed_mps"].max())
    max_target_acceleration = float(log_df["target_acceleration_mps2"].max())
    max_target_jerk = float(log_df["target_jerk_mps3"].max())
    max_target_orientation_rate = float(log_df["target_orientation_rate_radps"].max())
    reference_limits = {
        "max_speed_mps": float(env.unwrapped.cfg.trajectory_max_speed_mps),
        "max_acceleration_mps2": float(env.unwrapped.cfg.trajectory_max_acceleration_mps2),
        "max_orientation_rate_radps": float(env.unwrapped.cfg.trajectory_max_orientation_rate_radps),
        "max_jerk_mps3": float(env.unwrapped.cfg.trajectory_max_jerk_mps3),
        "retime_samples": int(env.unwrapped.cfg.trajectory_retime_samples),
    }
    reference_generator_version = str(env.unwrapped.cfg.trajectory_generator_version)
    within_kinematic_envelope = (
        max_target_speed <= reference_limits["max_speed_mps"] * 1.01
        and max_target_acceleration <= reference_limits["max_acceleration_mps2"] * 1.01
        and max_target_jerk <= reference_limits["max_jerk_mps3"] * 1.01
        and max_target_orientation_rate <= reference_limits["max_orientation_rate_radps"] * 1.01
    )
    reference_valid = (
        bool(reference_path_lengths)
        and min(reference_path_lengths) >= 0.10
        and min(reference_speed_p95) >= 0.05
        and within_kinematic_envelope
    )
    per_curve_requested_periods = [
        float(value)
        for value in log_df.groupby("env_id", sort=True)["requested_period_s"].first().tolist()
    ]
    per_curve_effective_periods = [
        float(value)
        for value in log_df.groupby("env_id", sort=True)["effective_period_s"].first().tolist()
    ]
    summary = {
        "controller": args_cli.controller,
        "trajectory": args_cli.trajectory,
        "reward_profile": args_cli.reward_profile,
        "reference_generator_version": reference_generator_version,
        "seed": int(env.unwrapped.cfg.seed),
        "pool_dynamics_profile_name": env.unwrapped.cfg.pool_dynamics_profile_name,
        "domain_randomization_spec_name": (
            getattr(env.unwrapped.cfg, "domain_randomization_spec_name", None) or ""
        ),
        "disturbance": disturbance_label or "nominal",
        "num_curves": int(log_df["env_id"].nunique()),
        "thruster_time_constant_mean_s": float(domain_df["sampled_thruster_time_constant_s"].mean()),
        "thruster_command_delay_mean_s": float(domain_df["sampled_thruster_delay_s"].mean()),
        "thruster_command_delay_max_s": float(domain_df["sampled_thruster_delay_s"].max()),
        "thruster_command_rate_limit_mean_per_s": float(
            domain_df["sampled_thruster_max_command_rate_per_s"].mean()
        ),
        "thruster_command_resolution_mean": float(
            domain_df["sampled_thruster_command_resolution_mean"].mean()
        ),
        "thruster_command_dropout_probability_mean": float(
            domain_df["sampled_thruster_command_dropout_probability_mean"].mean()
        ),
        "observation_delay_mean_s": float(domain_df["sampled_observation_delay_s"].mean()),
        "observation_update_period_mean_s": float(
            domain_df["sampled_observation_update_period_s"].mean()
        ),
        "observation_dropout_probability_mean": float(
            domain_df["sampled_observation_dropout_probability_mean"].mean()
        ),
        "observation_lowpass_alpha_mean": float(
            domain_df["sampled_observation_lowpass_alpha_mean"].mean()
        ),
        "observation_noise_std_mean": float(domain_df["sampled_observation_noise_std_mean"].mean()),
        "observation_bias_drift_std_mean": float(
            domain_df["sampled_observation_bias_drift_std_mean"].mean()
        ),
        "reference_valid": int(reference_valid),
        "reference_within_kinematic_envelope": int(within_kinematic_envelope),
        "reference_path_length_mean_m": float(np.mean(reference_path_lengths)),
        "reference_path_length_m_by_env_json": json.dumps(reference_path_lengths),
        "min_reference_path_length_m": float(min(reference_path_lengths)),
        "target_speed_p95_mps_by_env_json": json.dumps(reference_speed_p95),
        "min_curve_target_speed_p95_mps": float(min(reference_speed_p95)),
        "target_speed_mean_mps": float(log_df["target_speed_mps"].mean()),
        "target_speed_p95_mps": float(log_df["target_speed_mps"].quantile(0.95)),
        "target_speed_max_mps": max_target_speed,
        "target_acceleration_mean_mps2": float(log_df["target_acceleration_mps2"].mean()),
        "target_acceleration_p95_mps2": float(log_df["target_acceleration_mps2"].quantile(0.95)),
        "target_acceleration_max_mps2": max_target_acceleration,
        "target_jerk_mean_mps3": float(log_df["target_jerk_mps3"].mean()),
        "target_jerk_p95_mps3": float(log_df["target_jerk_mps3"].quantile(0.95)),
        "target_jerk_max_mps3": max_target_jerk,
        "target_curvature_mean_m_inv": float(log_df["target_curvature_m_inv"].mean()),
        "target_curvature_p95_m_inv": float(log_df["target_curvature_m_inv"].quantile(0.95)),
        "target_curvature_max_m_inv": float(log_df["target_curvature_m_inv"].max()),
        "target_orientation_rate_mean_radps": float(log_df["target_orientation_rate_radps"].mean()),
        "target_orientation_rate_p95_radps": float(log_df["target_orientation_rate_radps"].quantile(0.95)),
        "target_orientation_rate_max_radps": max_target_orientation_rate,
        "requested_period_mean_s": float(log_df["requested_period_s"].mean()),
        "requested_period_s_by_env_json": json.dumps(per_curve_requested_periods),
        "effective_period_mean_s": float(log_df["effective_period_s"].mean()),
        "effective_period_max_s": float(log_df["effective_period_s"].max()),
        "effective_period_s_by_env_json": json.dumps(per_curve_effective_periods),
        "retimed_curve_fraction": float(log_df.groupby("env_id")["trajectory_retimed"].first().mean()),
        "position_rmse": float(np.sqrt(np.mean(position_errors**2))),
        "position_error_p95": float(np.quantile(position_errors, 0.95)),
        "position_mae": float(np.mean(position_errors)),
        "max_position_error": float(np.max(position_errors)),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_errors**2))),
        "mean_command_heading_error_deg": (
            None
            if (heading_mean := _finite_statistic(log_df["command_heading_error_rad"], np.mean)) is None
            else float(np.degrees(heading_mean))
        ),
        "mean_motion_sideslip_error_deg": (
            None
            if (sideslip_mean := _finite_statistic(log_df["motion_sideslip_error_rad"], np.mean)) is None
            else float(np.degrees(sideslip_mean))
        ),
        "mean_action_rms": float(log_df["action_rms"].mean()),
        "mean_action_rate_rms": float(log_df["action_rate_rms"].mean()),
        "raw_policy_action_clip_fraction": float(log_df["raw_policy_action_clip_fraction"].mean()),
        "mean_requested_to_applied_action_rms": _finite_statistic(
            log_df["requested_to_applied_action_rms"], np.mean
        ),
        "mean_applied_action_rate_rms": _finite_statistic(log_df["applied_action_rate_rms"], np.mean),
        "mean_realized_thruster_force_abs_n": _finite_statistic(
            log_df["realized_thruster_force_abs_mean_n"], np.mean
        ),
        "max_realized_thruster_force_abs_n": _finite_statistic(
            log_df["realized_thruster_force_abs_max_n"], np.max
        ),
        "mean_realized_wrench_force_norm_n": _finite_statistic(
            np.linalg.norm(
                log_df[
                    [
                        "realized_wrench_force_x_n",
                        "realized_wrench_force_y_n",
                        "realized_wrench_force_z_n",
                    ]
                ].to_numpy(),
                axis=1,
            ),
            np.mean,
        ),
        "mean_reward_per_step": float(log_df["reward"].mean()),
        "mean_water_current_norm": float(
            np.mean(
                np.linalg.norm(
                    log_df[["water_current_x", "water_current_y", "water_current_z"]].to_numpy(),
                    axis=1,
                )
            )
        ),
        "max_water_current_norm": float(
            np.max(
                np.linalg.norm(
                    log_df[["water_current_x", "water_current_y", "water_current_z"]].to_numpy(),
                    axis=1,
                )
            )
        ),
        "eval_damping_scale": float(args_cli.eval_damping_scale),
        "eval_thruster_scale": float(args_cli.eval_thruster_scale),
        "eval_thruster_tau_scale": float(args_cli.eval_thruster_tau_scale),
        "evaluation_label": sanitize_evaluation_label(args_cli.evaluation_label),
        "eval_disturbance_stage": int(args_cli.eval_disturbance_stage),
        "failure_events": int(termination_events),
        "any_failure_rate": float(any_failure.to(dtype=torch.float32).mean().cpu().item()),
        "terminations_per_episode": float(termination_events / max(1, int(log_df["env_id"].nunique()))),
        "mean_time_to_first_failure_s": float(torch.nanmean(first_failure_time_s).cpu().item())
        if bool(torch.any(any_failure))
        else None,
        "survival_rate": float((~any_failure).to(dtype=torch.float32).mean().cpu().item()),
    }
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    print(f"[INFO]: Summary metrics: {summary}")

    if args_cli.hold_open and not getattr(args_cli, "headless", False):
        visualizer.set_status("eval complete; close Isaac Sim to exit")
        print("[INFO]: Evaluation complete. Close Isaac Sim to exit.")
        while simulation_app.is_running():
            simulation_app.update()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

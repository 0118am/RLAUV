"""Command-line entry point for policy and PID evaluation."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

# The evaluator runs as a file from IsaacLab, so make the task checkout
# importable before resolving the pure-data MLP profile registry.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_DIRECTORY
]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.training.ppo.networks import get_mlp_architecture
from simulation.training.recipe import (
    DEFAULT_TRAINING_RECIPE,
    TrainingRecipe,
    apply_trajectory_kinematic_limits,
    load_training_recipe,
    run_input_paths,
)
from simulation.training.evaluation.campaign import evaluation_paths
from simulation.training.evaluation.policy import load_actor
from robot.control.trajectory import TRAJECTORY_AXIS_BY_NAME, TRAJECTORY_TYPE_IDS
from robot.control import PIDGains, PIDTrajectoryController
from isaaclab.app import AppLauncher
from simulation.training.evaluation.config import (
    DEFAULT_CURRENT_TAU_S,
    DEFAULT_DYNAMICS_SCALE,
    DEFAULT_EVALUATION_DURATION_S,
    DEFAULT_RANDOM_CURVE_COUNT,
    build_evaluation_case_label,
    resolve_random_smooth_ranges,
    sanitize_evaluation_label,
)

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
    "--environment_profile",
    type=str,
    default=None,
    help="Deterministic environment/hydrodynamics profile JSON used for evaluation.",
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
    "--duration",
    type=float,
    default=DEFAULT_EVALUATION_DURATION_S,
    help="Trajectory evaluation duration in seconds.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Path to the PPO model_*.pt checkpoint.",
)
parser.add_argument("--controller", choices=("ppo", "pid"), default="ppo", help="Tracking controller to evaluate.")
_DEFAULT_PID_GAINS = PIDGains()
parser.add_argument("--pid_position_kp", type=float, nargs=3, default=_DEFAULT_PID_GAINS.position_kp)
parser.add_argument("--pid_position_ki", type=float, nargs=3, default=_DEFAULT_PID_GAINS.position_ki)
parser.add_argument("--pid_velocity_kd", type=float, nargs=3, default=_DEFAULT_PID_GAINS.velocity_kd)
parser.add_argument("--pid_attitude_kp", type=float, nargs=3, default=_DEFAULT_PID_GAINS.attitude_kp)
parser.add_argument("--pid_attitude_ki", type=float, nargs=3, default=_DEFAULT_PID_GAINS.attitude_ki)
parser.add_argument(
    "--pid_angular_velocity_kd",
    type=float,
    nargs=3,
    default=_DEFAULT_PID_GAINS.angular_velocity_kd,
)
parser.add_argument(
    "--trajectory",
    type=str,
    default="lissajous",
    choices=TRAJECTORY_TYPE_IDS.keys(),
    help="Fixed evaluation trajectory to run.",
)
parser.add_argument(
    "--trajectory_wave_count",
    type=int,
    default=1,
    help="Number of transverse sine waves during each forward half-cycle.",
)
parser.add_argument("--trajectory_amp_x", type=float, default=None, help="Override trajectory x amplitude.")
parser.add_argument("--trajectory_amp_y", type=float, default=None, help="Override trajectory y amplitude/radius.")
parser.add_argument("--trajectory_amp_z", type=float, default=None, help="Override trajectory z amplitude.")
parser.add_argument("--trajectory_period", type=float, default=None, help="Override trajectory period.")
parser.add_argument(
    "--trajectory_speed",
    type=float,
    default=None,
    help=(
        "Peak speed for surge/sway/heave sine, or path speed for lissajous, "
        "lateral_wave, vertical_wave, spatial_helix, and reverse_spatial_helix."
    ),
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
parser.add_argument(
    "--trajectory_radius_min",
    type=float,
    default=None,
    help="Override breathing-loop minimum radius.",
)
parser.add_argument(
    "--trajectory_radius_max",
    type=float,
    default=None,
    help="Override breathing-loop maximum radius.",
)
parser.add_argument(
    "--random_curve_count",
    type=int,
    default=DEFAULT_RANDOM_CURVE_COUNT,
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
    default=DEFAULT_CURRENT_TAU_S,
    help="Time constant in seconds for smooth current disturbance eval.",
)
parser.add_argument(
    "--eval_damping_scale",
    type=float,
    default=DEFAULT_DYNAMICS_SCALE,
    help="Multiply linear/quadratic damping.",
)
parser.add_argument(
    "--eval_thruster_scale",
    type=float,
    default=DEFAULT_DYNAMICS_SCALE,
    help="Multiply all thruster force outputs.",
)
parser.add_argument(
    "--eval_thruster_tau_scale",
    type=float,
    default=DEFAULT_DYNAMICS_SCALE,
    help="Multiply the first-order thruster response time constant.",
)
parser.add_argument(
    "--disturbance_name",
    type=str,
    default=None,
    help="Optional label used in the output directory name for disturbance eval.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.controller == "ppo":
    if args_cli.checkpoint is None:
        parser.error("PPO evaluation requires --checkpoint")
    args_cli.checkpoint = args_cli.checkpoint.expanduser().resolve()
    inputs = run_input_paths(args_cli.checkpoint.parent)
    evaluation_recipe = load_training_recipe(inputs.recipe)
    args_cli.mlp_architecture = evaluation_recipe.mlp_architecture
    args_cli.reward_profile = evaluation_recipe.reward_profile
    if args_cli.environment_profile is None:
        args_cli.environment_profile = str(inputs.environment)
    if args_cli.domain_randomization_spec is None:
        args_cli.domain_randomization_spec = str(inputs.domain_randomization)
else:
    args_cli.mlp_architecture = "mlp_33d"
    evaluation_recipe = load_training_recipe(DEFAULT_TRAINING_RECIPE)
    args_cli.reward_profile = evaluation_recipe.reward_profile
    if args_cli.environment_profile is None:
        args_cli.environment_profile = str(
            evaluation_recipe.resolve_path(evaluation_recipe.environment_base)
        )
    if args_cli.domain_randomization_spec is None:
        args_cli.domain_randomization_spec = str(
            evaluation_recipe.resolve_path(evaluation_recipe.domain_randomization_base)
        )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from collections import deque
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

from simulation.training.evaluation.metrics import (
    build_evaluation_summary,
    write_evaluation_summary,
)
from simulation.training.evaluation.runtime import collect_domain_samples, run_evaluation


class TrajectoryEvalVisualizer:
    """Draw one desired/actual trajectory pair without affecting evaluation data."""

    def __init__(
        self,
        enabled: bool,
        trajectory: str,
        checkpoint_name: str,
        max_points: int,
        stride: int,
    ) -> None:
        self.enabled = enabled
        self.trajectory = trajectory
        self.checkpoint_name = checkpoint_name
        self.stride = stride
        self.desired_points = deque(maxlen=max_points)
        self.actual_points = deque(maxlen=max_points)
        self._draw = None
        self._labels = {}

        if self.enabled:
            self._init_debug_draw()
            self._init_status_window()

    def _init_debug_draw(self) -> None:
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._clear_draw()

    def _init_status_window(self) -> None:
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
    ) -> None:
        if not self.enabled:
            return

        self.desired_points.append(self._point(desired_pos_w))
        self.actual_points.append(self._point(actual_pos_w))
        if step % self.stride == 0:
            self._draw_trails()
            self._update_labels(time_s, position_error, velocity_error)

    def _draw_trails(self) -> None:
        if self._draw is None:
            return
        desired = list(self.desired_points)
        actual = list(self.actual_points)
        start_points = desired[:-1] + actual[:-1]
        end_points = desired[1:] + actual[1:]
        colors = [(0.1, 0.45, 1.0, 1.0)] * max(0, len(desired) - 1)
        colors += [(1.0, 0.45, 0.05, 1.0)] * max(0, len(actual) - 1)
        self._clear_draw()
        if start_points:
            self._draw.draw_lines(start_points, end_points, colors, [3.0] * len(start_points))
        if desired and actual:
            self._draw.draw_points(
                [desired[-1], actual[-1]],
                [(1.0, 0.95, 0.05, 1.0), (1.0, 0.95, 0.95, 1.0)],
                [18.0, 12.0],
            )

    def _clear_draw(self) -> None:
        if self._draw is None:
            return
        self._draw.clear_lines()
        self._draw.clear_points()

    def _update_labels(self, time_s: float, position_error: float, velocity_error: float) -> None:
        time_label = self._labels.get("time")
        error_label = self._labels.get("error")
        if time_label is not None:
            time_label.text = f"time: {time_s:.2f} s"
        if error_label is not None:
            error_label.text = f"pos err: {position_error:.3f} m | vel err: {velocity_error:.3f} m/s"

    def set_status(self, message: str) -> None:
        label = self._labels.get("time")
        if label is not None:
            label.text = message


@dataclass(frozen=True)
class EvaluationSetup:
    env_cfg: Any
    log_root: Path
    run_name: str
    checkpoint_path: str | None
    checkpoint_name: str
    architecture: Any


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
    env_cfg.trajectory_eval_axis = TRAJECTORY_AXIS_BY_NAME.get(args.trajectory, 0)
    env_cfg.trajectory_eval_wave_count = args.trajectory_wave_count
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
    recipe: TrainingRecipe,
) -> EvaluationSetup:
    eval_num_envs = (
        args.num_envs
        if args.num_envs is not None
        else args.random_curve_count if args.trajectory == "random_smooth" else 1
    )
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=eval_num_envs,
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
    env_cfg.use_boundaries = True
    apply_trajectory_kinematic_limits(
        env_cfg,
        recipe.kinematic_limits,
    )
    env_cfg.trajectory_startup_duration_s = float(
        recipe.trajectory_startup_duration_s
    )
    _apply_trajectory_configuration(env_cfg, args)
    _apply_disturbance_overlay(env_cfg, args)

    if args.seed is None:
        args.seed = int(env_cfg.seed)
    env_cfg.seed = args.seed

    checkpoint_path = None
    checkpoint_name = "pid"
    run_name = "pid"
    if args.controller == "ppo":
        checkpoint_path = str(args.checkpoint)
        run_dir = args.checkpoint.parent
        log_root = run_dir.parent
        run_name = run_dir.name
        checkpoint_name = args.checkpoint.name
    else:
        log_root = Path(os.path.abspath(os.path.join("logs", "rsl_rl", architecture.experiment_name)))
    return EvaluationSetup(env_cfg, log_root, run_name, checkpoint_path, checkpoint_name, architecture)


def build_controller(env: Any, setup: EvaluationSetup, args: Any) -> Any:
    if args.controller == "ppo":
        print(f"[INFO]: Loading model checkpoint from: {setup.checkpoint_path}")
        controller = load_actor(
            setup.checkpoint_path,
            setup.architecture,
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
    robot_runtime = env.unwrapped.robot_runtime
    return PIDTrajectoryController(
        num_envs=env.unwrapped.num_envs,
        dt=env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation,
        thruster_positions_b=robot_runtime.thruster_com_offsets[0],
        thruster_force_curve_coefficients=robot_runtime.thruster_force_curve_coefficients,
        mass_kg=robot_runtime.masses,
        gains=gains,
    )


def main() -> None:
    architecture = get_mlp_architecture(args_cli.mlp_architecture)
    setup = configure_evaluation(args_cli, architecture, evaluation_recipe)
    print(f"[INFO]: Tracking reward profile: {args_cli.reward_profile}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=setup.env_cfg))
    observations = env.get_observations()
    policy = build_controller(env, setup, args_cli)
    disturbance_label = build_evaluation_case_label(
        disturbance_name=args_cli.disturbance_name,
        sample_domain_randomization=bool(args_cli.eval_domain_randomization),
        domain_randomization_name=env.unwrapped.cfg.domain_randomization_spec_name,
        seed=args_cli.seed,
        current_w=args_cli.eval_current,
        smooth_current=args_cli.eval_smooth_current,
        current_variation_std=args_cli.eval_current_variation_std,
        damping_scale=args_cli.eval_damping_scale,
        thruster_scale=args_cli.eval_thruster_scale,
        thruster_tau_scale=args_cli.eval_thruster_tau_scale,
    )
    case_label = (
        sanitize_evaluation_label(args_cli.evaluation_label)
        if args_cli.evaluation_label
        else disturbance_label
    )
    paths = evaluation_paths(
        setup.log_root / setup.run_name / "evaluation",
        setup.checkpoint_name,
        args_cli.trajectory,
        case_label,
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]: Saving trajectory eval results into: {paths.directory}")

    domain_samples = collect_domain_samples(env)
    domain_samples.to_csv(paths.domain_samples_csv, index=False)
    visualizer = TrajectoryEvalVisualizer(
        enabled=not args_cli.disable_trajectory_vis and not args_cli.headless,
        trajectory=args_cli.trajectory,
        checkpoint_name=setup.checkpoint_name,
        max_points=args_cli.trail_max_points,
        stride=args_cli.trail_stride,
    )
    result = run_evaluation(
        env,
        policy,
        observations,
        duration_s=args_cli.duration,
        trajectory=args_cli.trajectory,
        reward_profile=args_cli.reward_profile,
        disturbance_label=disturbance_label,
        visualizer=visualizer,
    )
    result.log.to_csv(paths.logs_csv, index=False)
    summary = build_evaluation_summary(
        result.log,
        domain_samples,
        env,
        args_cli,
        disturbance_label,
        termination_events=result.termination_events,
        any_failure=result.any_failure,
        first_failure_time_s=result.first_failure_time_s,
    )
    write_evaluation_summary(summary, str(paths.summary_csv))
    print(f"[INFO]: Summary metrics: {summary}")

    if args_cli.hold_open and not args_cli.headless:
        visualizer.set_status("eval complete; close Isaac Sim to exit")
        print("[INFO]: Evaluation complete. Close Isaac Sim to exit.")
        while simulation_app.is_running():
            simulation_app.update()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

import argparse
from pathlib import Path
import sys

# The evaluator runs as a file from IsaacLab, so make the task checkout
# importable before resolving the pure-data MLP profile registry.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac.ppo.architectures import available_mlp_architectures, get_mlp_architecture
from robot.control.trajectory import TRAJECTORY_TYPE_IDS
from isaaclab.app import AppLauncher
from simulation.isaac.trajectory.evaluation_cases import (
    DEFAULT_CURRENT_TAU_S,
    DEFAULT_DYNAMICS_SCALE,
    DEFAULT_EVALUATION_DURATION_S,
    DEFAULT_RANDOM_CURVE_COUNT,
    validate_evaluation_parameters,
)

from simulation.isaac.trajectory import cli as cli_args  # isort: skip

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
    "--keep_boundaries",
    action="store_true",
    default=False,
    help="Retain task boundaries and report out-of-bounds terminations instead of disabling them.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=DEFAULT_EVALUATION_DURATION_S,
    help="Trajectory evaluation duration in seconds.",
)
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
    help="Versioned tracking reward policy from simulation/isaac/rewards/policy_N.py.",
)
parser.add_argument("--controller", choices=("ppo", "pid"), default="ppo", help="Tracking controller to evaluate.")
parser.add_argument("--pid_position_kp", type=float, nargs=3, default=(20.0, 20.0, 25.0))
parser.add_argument("--pid_position_ki", type=float, nargs=3, default=(0.5, 0.5, 0.8))
parser.add_argument("--pid_velocity_kd", type=float, nargs=3, default=(15.0, 15.0, 18.0))
parser.add_argument("--pid_attitude_kp", type=float, nargs=3, default=(8.0, 8.0, 6.0))
parser.add_argument("--pid_attitude_ki", type=float, nargs=3, default=(0.2, 0.2, 0.15))
parser.add_argument("--pid_angular_velocity_kd", type=float, nargs=3, default=(3.0, 3.0, 2.5))
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
    help="Fixed evaluation trajectory to run.",
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

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from simulation.isaac.trajectory.evaluation_metrics import (
    build_evaluation_summary,
    write_evaluation_summary,
)
from simulation.isaac.trajectory.evaluation_runtime import collect_domain_samples, run_evaluation
from simulation.isaac.trajectory.evaluation_setup import (
    build_controller,
    configure_evaluation,
    evaluation_case_label,
    prepare_evaluation_paths,
)
from simulation.isaac.trajectory.evaluation_visualization import TrajectoryEvalVisualizer


def main() -> None:
    architecture = get_mlp_architecture(args_cli.mlp_architecture)
    setup = configure_evaluation(args_cli, architecture, cli_args.parse_rsl_rl_cfg)
    print(f"[INFO]: Tracking reward profile: {args_cli.reward_profile}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=setup.env_cfg))
    observations = env.get_observations()
    policy = build_controller(env, observations, setup, args_cli)
    disturbance_label = evaluation_case_label(
        args_cli,
        getattr(env.unwrapped.cfg, "domain_randomization_spec_name", None),
    )
    paths = prepare_evaluation_paths(setup, args_cli, disturbance_label)
    print(f"[INFO]: Saving trajectory eval results into: {paths.directory}")

    domain_samples = collect_domain_samples(env)
    domain_samples.to_csv(paths.domain_samples_csv, index=False)
    visualizer = TrajectoryEvalVisualizer(
        enabled=not args_cli.disable_trajectory_vis and not getattr(args_cli, "headless", False),
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

    if args_cli.hold_open and not getattr(args_cli, "headless", False):
        visualizer.set_status("eval complete; close Isaac Sim to exit")
        print("[INFO]: Evaluation complete. Close Isaac Sim to exit.")
        while simulation_app.is_running():
            simulation_app.update()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

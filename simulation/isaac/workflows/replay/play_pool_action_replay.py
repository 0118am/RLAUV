"""Replay measured thruster commands open-loop in Isaac and write a standard state log."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Replay measured AUV actions in Isaac with no policy feedback.")
parser.add_argument("input_log", type=Path, help="Standard measured replay CSV containing action_0...action_7.")
parser.add_argument("--output", type=Path, required=True, help="Standard simulated replay CSV path.")
parser.add_argument("--profile", type=Path, help="Measured PoolDynamicsProfile JSON applied before environment creation.")
parser.add_argument("--measured-env-id", type=int, help="Select env_id if the input contains multiple streams.")
parser.add_argument("--task", default="Isaac-AUV-Traj-Direct-v1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration", type=float, help="Optional replay duration limit in seconds.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports below require the launched Isaac application."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply

from simulation.isaac.workflows.replay.validate_pool_replay import load_replay_csv


OUTPUT_COLUMNS = (
    "time_s",
    "position_w_x_m",
    "position_w_y_m",
    "position_w_z_m",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "linear_velocity_w_x_mps",
    "linear_velocity_w_y_mps",
    "linear_velocity_w_z_mps",
    "angular_velocity_b_x_radps",
    "angular_velocity_b_y_radps",
    "angular_velocity_b_z_radps",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
    "action_7",
)


def _action_at_time(time_s: torch.Tensor, actions: torch.Tensor, query_time_s: float) -> torch.Tensor:
    query = torch.tensor(query_time_s, dtype=time_s.dtype, device=time_s.device)
    median_dt = torch.median(time_s[1:] - time_s[:-1])
    tolerance = torch.clamp(median_dt * 1.0e-6, min=torch.finfo(time_s.dtype).eps * 16.0)
    index = torch.searchsorted(time_s, query + tolerance, right=True) - 1
    index = torch.clamp(index, min=0, max=time_s.numel() - 1)
    return actions[index]


def _write_state_row(writer: csv.writer, env, time_s: float, action: torch.Tensor) -> None:
    root = env.unwrapped
    position_local = root._robot.data.root_pos_w[0] - root.scene.env_origins[0]
    quaternion = root._robot.data.root_quat_w[0]
    linear_velocity_w = root._robot.data.root_lin_vel_w[0]
    angular_velocity_b = root._robot.data.root_ang_vel_b[0]
    writer.writerow(
        [
            float(time_s),
            *[float(value) for value in position_local.detach().cpu().tolist()],
            *[float(value) for value in quaternion.detach().cpu().tolist()],
            *[float(value) for value in linear_velocity_w.detach().cpu().tolist()],
            *[float(value) for value in angular_velocity_b.detach().cpu().tolist()],
            *[float(value) for value in action.detach().cpu().tolist()],
        ]
    )


def main() -> None:
    measured = load_replay_csv(args_cli.input_log, args_cli.measured_env_id)
    if measured.actions is None:
        raise ValueError("input_log must contain contiguous action_0...action_7 columns.")
    if measured.actions.shape[1] != 8:
        raise ValueError(f"AUV action replay requires 8 action columns, got {measured.actions.shape[1]}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = int(args_cli.seed)
    if args_cli.profile is not None:
        # Let the environment resolve/apply the profile so the same order and
        # profile is used consistently by replay, train, and evaluation.
        env_cfg.pool_dynamics_profile = str(args_cli.profile)
    env_cfg.eval_mode = True
    env_cfg.cap_episode_length = False
    env_cfg.episode_length_before_reset = None
    env_cfg.use_boundaries = False
    env_cfg.domain_randomization.use_custom_randomization = False

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    env.reset()
    root = env.unwrapped
    device = root.device
    initial_position_w = measured.position_w[0].to(dtype=torch.float32, device=device) + root.scene.env_origins[0]
    initial_quaternion = measured.quaternion_wxyz[0].to(dtype=torch.float32, device=device)
    initial_linear_velocity_w = measured.linear_velocity_w[0].to(dtype=torch.float32, device=device)
    initial_angular_velocity_b = measured.angular_velocity_b[0].to(dtype=torch.float32, device=device)
    initial_angular_velocity_w = quat_apply(initial_quaternion.reshape(1, 4), initial_angular_velocity_b.reshape(1, 3))[0]
    root._robot.write_root_pose_to_sim(
        torch.cat((initial_position_w, initial_quaternion)).reshape(1, 7),
        root._robot._ALL_INDICES,
    )
    root._robot.write_root_velocity_to_sim(
        torch.cat((initial_linear_velocity_w, initial_angular_velocity_w)).reshape(1, 6),
        root._robot._ALL_INDICES,
    )

    replay_start = float(measured.time_s[0].item())
    replay_duration = float((measured.time_s[-1] - measured.time_s[0]).item())
    if args_cli.duration is not None:
        if args_cli.duration <= 0.0:
            raise ValueError("duration must be positive.")
        replay_duration = min(replay_duration, float(args_cli.duration))
    policy_dt = float(root.cfg.sim.dt * root.cfg.decimation)
    step_count = int(torch.floor(torch.tensor(replay_duration / policy_dt)).item()) + 1
    action_times = measured.time_s.to(dtype=torch.float32, device=device)
    actions = measured.actions.to(dtype=torch.float32, device=device)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(OUTPUT_COLUMNS)
        with torch.inference_mode():
            for step in range(step_count):
                elapsed = step * policy_dt
                source_time = replay_start + elapsed
                action = _action_at_time(action_times, actions, source_time)
                _write_state_row(writer, env, source_time, action)
                env.step(action.reshape(1, 8))

    env.close()
    print(f"[INFO]: Wrote open-loop replay log: {args_cli.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

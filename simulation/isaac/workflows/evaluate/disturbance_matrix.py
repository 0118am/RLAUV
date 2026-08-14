#!/usr/bin/env python3
"""Build or run a disturbance evaluation matrix for trajectory policies."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CASES = {
    "nominal": [],
    "current_x_0p1": ["--eval_current", "0.1", "0.0", "0.0", "--disturbance_name", "current_x_0p1"],
    "current_y_0p1": ["--eval_current", "0.0", "0.1", "0.0", "--disturbance_name", "current_y_0p1"],
    "current_diag_0p2": [
        "--eval_current",
        "0.1414",
        "0.1414",
        "0.0",
        "--disturbance_name",
        "current_diag_0p2",
    ],
    "current_x_0p2": ["--eval_current", "0.2", "0.0", "0.0", "--disturbance_name", "current_x_0p2"],
    "smooth_current_x_0p2": [
        "--eval_current",
        "0.2",
        "0.0",
        "0.0",
        "--eval_smooth_current",
        "--eval_current_variation_std",
        "0.01",
        "--eval_current_tau",
        "12.0",
        "--disturbance_name",
        "smooth_current_x_0p2",
    ],
    "combined_hard": [
        "--eval_current",
        "0.2",
        "0.0",
        "0.02",
        "--eval_smooth_current",
        "--eval_current_variation_std",
        "0.012",
        "--eval_current_tau",
        "10.0",
        "--eval_damping_scale",
        "1.3",
        "--eval_thruster_scale",
        "0.85",
        "--eval_thruster_tau_scale",
        "1.5",
        "--disturbance_name",
        "combined_hard",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AUV trajectory disturbance eval cases.")
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/jining_yang/IsaacLab"))
    parser.add_argument("--task", default="Isaac-AUV-Traj-Direct-v1")
    parser.add_argument("--load-run", default="2026-06-19_21-25-13")
    parser.add_argument("--checkpoint", default="model_399.pt")
    parser.add_argument("--duration", default="32.0")
    parser.add_argument("--trajectories", nargs="+", default=["lissajous", "helix", "spiral"])
    parser.add_argument("--cases", nargs="+", default=["nominal", "current_x_0p1", "current_x_0p2", "combined_hard"])
    parser.add_argument("--num-envs", default="1")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of only printing them.")
    parser.add_argument("--no-headless", action="store_true", help="Show Isaac Sim GUI.")
    parser.add_argument("--no-align", action="store_true", help="Do not start at the initial target.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, trajectory: str, case: str) -> list[str]:
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}. Available: {sorted(CASES)}")
    script = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/"
        "isaac-auv-env/simulation/isaac/workflows/evaluate/trajectory.py"
    )
    command = [
        "./isaaclab.sh",
        "-p",
        script,
        "--task",
        args.task,
        "--load_run",
        args.load_run,
        "--checkpoint",
        args.checkpoint,
        "--trajectory",
        trajectory,
        "--duration",
        args.duration,
        "--num_envs",
        args.num_envs,
        "--disable_trajectory_vis",
    ]
    if not args.no_headless:
        command.append("--headless")
    if not args.no_align:
        command.append("--align_initial_target")
    command.extend(CASES[case])
    return command


def main() -> None:
    args = parse_args()
    for case in args.cases:
        for trajectory in args.trajectories:
            command = build_command(args, trajectory, case)
            print(" ".join(command))
            if args.run:
                subprocess.run(command, cwd=args.isaaclab_root, check=True)


if __name__ == "__main__":
    main()

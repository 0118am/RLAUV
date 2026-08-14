#!/usr/bin/env python3
"""Export an AUV trajectory RSL-RL actor checkpoint to ONNX."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

# Export is also invoked as a standalone file, outside the notebook's
# ``sys.path`` setup.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac.ppo.architectures import available_mlp_architectures, get_mlp_architecture
from simulation.isaac.ppo.evaluation import TrajectoryEvaluationActor, load_evaluation_actor


class FeedForwardActorOnnx(torch.nn.Module):
    """Pure actor wrapper for a named feed-forward MLP policy."""

    def __init__(self, policy: TrajectoryEvaluationActor):
        super().__init__()
        self.actor = policy.actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trajectory policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_*.pt checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory. Defaults to <checkpoint run>/exports.",
    )
    parser.add_argument("--prefix", default="auv_traj_policy")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--activation", default="elu")
    parser.add_argument(
        "--mlp_architecture",
        choices=available_mlp_architectures(),
        default="mlp_history_5",
        help="Named MLP input/layer profile stored with the training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architecture_profile = get_mlp_architecture(args.mlp_architecture)
    observation_dim = architecture_profile.observation_dim
    hidden_dims = list(architecture_profile.actor_hidden_dims)
    policy = load_evaluation_actor(
        args.checkpoint,
        observation_dim=observation_dim,
        action_dim=args.action_dim,
        hidden_dims=hidden_dims,
        activation=args.activation,
        device="cpu",
    )

    output_dir = args.output_dir or args.checkpoint.resolve().parent / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    output_path = output_dir / f"{args.prefix}_{architecture_profile.name}_{args.date}_{stem}.onnx"
    dummy_obs = torch.zeros(1, observation_dim)
    exporter = FeedForwardActorOnnx(policy)
    torch.onnx.export(
        exporter,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(output_path)


if __name__ == "__main__":
    main()

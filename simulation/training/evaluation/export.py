#!/usr/bin/env python3
"""Export a trained AUV trajectory actor checkpoint to ONNX."""

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

from simulation.training.ppo.networks import available_mlp_architectures, get_mlp_architecture
from simulation.training.ppo.networks import TrajectoryEvaluationActor, load_evaluation_actor
from simulation.training.manifest import load_run_manifest, validate_manifest_selection


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
    parser.add_argument("--action-dim", type=int, default=None, help="Optional manifest consistency assertion.")
    parser.add_argument("--activation", default=None, help="Optional manifest consistency assertion.")
    parser.add_argument(
        "--mlp_architecture",
        choices=available_mlp_architectures(),
        default=None,
        help="Optional assertion for the architecture stored with the run.",
    )
    parser.add_argument(
        "--reward_profile",
        default=None,
        help="Optional assertion for the reward profile stored with the run.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Defaults to <checkpoint run>/params/run_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.run_manifest or args.checkpoint.resolve().parent / "params" / "run_manifest.json"
    manifest = load_run_manifest(manifest_path)
    validate_manifest_selection(
        manifest,
        mlp_architecture=args.mlp_architecture,
        reward_profile=args.reward_profile,
    )
    if args.action_dim is not None and args.action_dim != manifest.action_dim:
        raise ValueError(
            f"Requested action dimension {args.action_dim} does not match manifest {manifest.action_dim}."
        )
    if args.activation is not None and args.activation != manifest.activation:
        raise ValueError(
            f"Requested activation {args.activation!r} does not match manifest {manifest.activation!r}."
        )
    architecture_profile = get_mlp_architecture(manifest.mlp_architecture)
    observation_dim = architecture_profile.observation_dim
    hidden_dims = list(architecture_profile.actor_hidden_dims)
    policy = load_evaluation_actor(
        args.checkpoint,
        observation_dim=observation_dim,
        action_dim=manifest.action_dim,
        hidden_dims=hidden_dims,
        activation=manifest.activation,
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

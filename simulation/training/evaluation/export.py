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

from simulation.training.evaluation.policy import load_actor
from simulation.training.recipe import load_training_recipe, run_input_paths


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    recipe = load_training_recipe(run_input_paths(checkpoint.parent).recipe)
    architecture = recipe.architecture
    policy = load_actor(
        checkpoint,
        architecture,
        device="cpu",
    )
    deployable_policy = torch.nn.Sequential(
        policy,
        torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
    )

    output_dir = args.output_dir or checkpoint.parent / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    output_path = output_dir / f"{args.prefix}_{architecture.name}_{args.date}_{stem}.onnx"
    torch.onnx.export(
        deployable_policy,
        (torch.zeros(1, architecture.observation_dim),),
        output_path,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamo=True,
        external_data=False,
    )
    print(output_path)


if __name__ == "__main__":
    main()

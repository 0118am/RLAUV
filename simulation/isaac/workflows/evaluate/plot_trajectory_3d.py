#!/usr/bin/env python3
"""Plot 3D desired-vs-actual trajectory eval logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 3D trajectory tracking plots from eval logs.csv files.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Result run directory containing eval subfolders.")
    parser.add_argument("--checkpoint", default="model_399", help="Checkpoint stem to plot, for example model_399.")
    parser.add_argument("--env-id", type=int, default=0, help="Environment id to plot from each log.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectory_3d"),
        help="Directory for generated plot artifacts; override for a publication export.",
    )
    return parser.parse_args()


def find_logs(run_dir: Path, checkpoint: str) -> list[Path]:
    logs = sorted(run_dir.glob(f"{checkpoint}*_trajectory_eval/logs.csv"))
    if not logs:
        raise FileNotFoundError(f"No logs.csv files found for {checkpoint} under {run_dir}")
    return logs


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def load_log(path: Path, env_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["env_id"] == env_id].copy()
    if df.empty:
        raise ValueError(f"No rows for env_id={env_id} in {path}")
    return df


def metrics_for(df: pd.DataFrame) -> dict[str, float | str]:
    desired = df[["desired_x", "desired_y", "desired_z"]].to_numpy()
    actual = df[["true_x", "true_y", "true_z"]].to_numpy()
    error = actual - desired
    pos_error = np.linalg.norm(error, axis=1)
    vel_error = df["velocity_error"].to_numpy()
    bias = error.mean(axis=0)
    return {
        "trajectory": str(df["trajectory"].iloc[0]),
        "position_rmse": float(np.sqrt(np.mean(pos_error**2))),
        "position_mae": float(np.mean(pos_error)),
        "max_position_error": float(np.max(pos_error)),
        "velocity_rmse": float(np.sqrt(np.mean(vel_error**2))),
        "bias_x_actual_minus_desired": float(bias[0]),
        "bias_y_actual_minus_desired": float(bias[1]),
        "bias_z_actual_minus_desired": float(bias[2]),
        "bias_norm": float(np.linalg.norm(bias)),
    }


def plot_gallery(log_paths: list[Path], checkpoint: str, env_id: int, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_log(path, env_id) for path in log_paths]
    metrics = [metrics_for(df) for df in frames]
    order = {name: i for i, name in enumerate(["helix", "lissajous", "spiral", "chirp", "racetrack", "random_smooth"])}
    pairs = sorted(zip(frames, metrics), key=lambda item: order.get(item[1]["trajectory"], 99))

    fig = plt.figure(figsize=(6.2 * len(pairs), 5.8), constrained_layout=True)
    for idx, (df, row) in enumerate(pairs, start=1):
        ax = fig.add_subplot(1, len(pairs), idx, projection="3d")
        desired = df[["desired_x", "desired_y", "desired_z"]].to_numpy()
        actual = df[["true_x", "true_y", "true_z"]].to_numpy()
        ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color="#1f77b4", linewidth=2.4, label="desired")
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#ff7f0e", linewidth=1.8, label="actual")
        ax.scatter(desired[0, 0], desired[0, 1], desired[0, 2], color="#1f77b4", s=36, marker="o")
        ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2], color="#ff7f0e", s=36, marker="o")
        ax.scatter(desired[-1, 0], desired[-1, 1], desired[-1, 2], color="#1f77b4", s=44, marker="^")
        ax.scatter(actual[-1, 0], actual[-1, 1], actual[-1, 2], color="#ff7f0e", s=44, marker="^")
        xyz = np.vstack([desired, actual])
        set_axes_equal(ax, xyz)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=24, azim=-58)
        title = (
            f"{row['trajectory']}: 3D tracking\n"
            f"RMSE {row['position_rmse']:.3f} m | max {row['max_position_error']:.3f} m\n"
            f"bias {row['bias_norm']:.3f} m"
        )
        ax.set_title(title)
        ax.legend(loc="upper left")

    image_path = output_dir / f"{checkpoint}_3d_trajectory_gallery.png"
    metrics_path = output_dir / f"{checkpoint}_3d_metrics.csv"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    pd.DataFrame(metrics).sort_values("trajectory").to_csv(metrics_path, index=False)
    return image_path, metrics_path


def main() -> None:
    args = parse_args()
    logs = find_logs(args.run_dir, args.checkpoint)
    image_path, metrics_path = plot_gallery(logs, args.checkpoint, args.env_id, args.output_dir)
    print(f"wrote {image_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()

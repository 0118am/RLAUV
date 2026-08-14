import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)(?:_[a-z_]+)?_trajectory_eval", str(path))
    if match is None:
        raise ValueError(f"Could not parse checkpoint iteration from: {path}")
    return int(match.group(1))


def trajectory_name(path: Path) -> str:
    match = re.search(r"model_\d+(?:_([a-z_]+))?_trajectory_eval", str(path))
    if match is None:
        raise ValueError(f"Could not parse trajectory name from: {path}")
    return match.group(1) or "lissajous"


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory eval RMSE across checkpoints.")
    parser.add_argument("--results_root", type=Path, required=True, help="Run results directory to scan.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--csv", type=Path, default=None, help="Output summary CSV path.")
    args = parser.parse_args()

    summary_paths = sorted(
        args.results_root.glob("model_*_trajectory_eval/summary_metrics.csv"),
        key=checkpoint_iteration,
    )
    if not summary_paths:
        raise SystemExit(f"No trajectory summary_metrics.csv files found under: {args.results_root}")

    rows = []
    for summary_path in summary_paths:
        row = pd.read_csv(summary_path).iloc[0].to_dict()
        row["checkpoint"] = checkpoint_iteration(summary_path.parent)
        row["trajectory"] = row.get("trajectory", trajectory_name(summary_path.parent))
        row["summary_path"] = str(summary_path)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("checkpoint")
    csv_path = args.csv or args.results_root / "checkpoint_rmse_summary.csv"
    out_path = args.out or args.results_root / "checkpoint_rmse_curve.png"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for trajectory, group in df.groupby("trajectory"):
        group = group.sort_values("checkpoint")
        ax.plot(group["checkpoint"], group["position_rmse"], marker="o", label=f"{trajectory} position RMSE [m]")
    best_ix = df["position_rmse"].idxmin()
    best = df.loc[best_ix]
    ax.scatter(best["checkpoint"], best["position_rmse"], color="tab:red", zorder=5)
    ax.annotate(
        f"best pos RMSE\n{best['trajectory']} model_{int(best['checkpoint'])}: {best['position_rmse']:.3f} m",
        xy=(best["checkpoint"], best["position_rmse"]),
        xytext=(12, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
    )
    ax.set_title("AUV trajectory eval across checkpoints")
    ax.set_xlabel("checkpoint iteration")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)

    print(f"Loaded {len(df)} checkpoint summaries.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {out_path}")
    print(df[["trajectory", "checkpoint", "position_rmse", "position_mae", "max_position_error", "velocity_rmse"]])


if __name__ == "__main__":
    main()

"""Launch Isaac action replay and reject startup exits or missing output artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.workflows.replay.validate_pool_replay import load_replay_csv  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and verify an Isaac open-loop pool action replay.")
    parser.add_argument("input_log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional checked-run JSON report.")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--measured-env-id", type=int)
    parser.add_argument("--task", default="Isaac-AUV-Traj-Direct-v1")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/jining_yang/IsaacLab"))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--gpu-preflight-timeout-s", type=float, default=10.0)
    parser.add_argument("--disable-fabric", action="store_true")
    return parser


def build_isaac_replay_command(args: argparse.Namespace) -> list[str]:
    script = (
        "source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/"
        "simulation/isaac/workflows/replay/play_pool_action_replay.py"
    )
    command = [
        str(args.isaaclab_root / "isaaclab.sh"),
        "-p",
        script,
        str(args.input_log.resolve()),
        "--output",
        str(args.output.resolve()),
        "--task",
        args.task,
        "--device",
        args.device,
        "--headless",
    ]
    if args.profile is not None:
        command.extend(("--profile", str(args.profile.resolve())))
    if args.measured_env_id is not None:
        command.extend(("--measured-env-id", str(args.measured_env_id)))
    if args.duration is not None:
        command.extend(("--duration", str(args.duration)))
    if args.disable_fabric:
        command.append("--disable_fabric")
    return command


def check_gpu_preflight(timeout_s: float = 10.0) -> str:
    if float(timeout_s) <= 0.0:
        raise ValueError("gpu_preflight_timeout_s must be positive.")
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=float(timeout_s), check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is unavailable; CUDA replay cannot be verified.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi did not respond within {timeout_s:g} s.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"NVIDIA GPU preflight failed: {detail or f'exit {result.returncode}'}")
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("NVIDIA GPU preflight returned no device.")
    return output


def validate_isaac_replay_output(
    path: Path,
    minimum_samples: int = 2,
    *,
    expected_start_time_s: float | None = None,
    expected_duration_s: float | None = None,
) -> dict[str, Any]:
    if int(minimum_samples) != minimum_samples or int(minimum_samples) < 2:
        raise ValueError("minimum_samples must be an integer >= 2.")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Isaac replay did not create a non-empty output log: {path}")
    trajectory = load_replay_csv(path)
    sample_count = int(trajectory.time_s.numel())
    if sample_count < int(minimum_samples):
        raise RuntimeError(f"Isaac replay output has {sample_count} samples; expected at least {minimum_samples}.")
    if trajectory.actions is None or trajectory.actions.shape[1] != 8:
        raise RuntimeError("Isaac replay output must contain all 8 action channels.")
    start_time_s = float(trajectory.time_s[0].item())
    duration_s = float((trajectory.time_s[-1] - trajectory.time_s[0]).item())
    median_dt_s = float(torch.median(trajectory.time_s[1:] - trajectory.time_s[:-1]).item())
    time_tolerance_s = max(1.0e-6, 1.01 * median_dt_s)
    if expected_start_time_s is not None and abs(start_time_s - float(expected_start_time_s)) > time_tolerance_s:
        raise RuntimeError(
            f"Isaac replay starts at {start_time_s:g} s; expected {float(expected_start_time_s):g} s."
        )
    if expected_duration_s is not None:
        expected_duration = float(expected_duration_s)
        if expected_duration <= 0.0:
            raise ValueError("expected_duration_s must be positive when provided.")
        if duration_s + time_tolerance_s < expected_duration:
            raise RuntimeError(
                f"Isaac replay covers {duration_s:g} s; expected at least {expected_duration:g} s."
            )
    return {
        "output_log": str(path),
        "sample_count": sample_count,
        "start_time_s": start_time_s,
        "duration_s": duration_s,
        "expected_duration_s": expected_duration_s,
        "duration_coverage": None if expected_duration_s is None else min(1.0, duration_s / float(expected_duration_s)),
        "action_count": int(trajectory.actions.shape[1]),
        "finite_and_strictly_increasing": True,
    }


def run_checked_replay(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.timeout_s) <= 0.0:
        raise ValueError("timeout_s must be positive.")
    if not args.input_log.is_file():
        raise FileNotFoundError(f"Input replay log does not exist: {args.input_log}")
    measured = load_replay_csv(args.input_log, args.measured_env_id)
    expected_duration_s = float((measured.time_s[-1] - measured.time_s[0]).item())
    if args.duration is not None:
        if float(args.duration) <= 0.0:
            raise ValueError("duration must be positive.")
        expected_duration_s = min(expected_duration_s, float(args.duration))
    if not (args.isaaclab_root / "isaaclab.sh").is_file():
        raise FileNotFoundError(f"IsaacLab launcher does not exist under: {args.isaaclab_root}")
    gpu_info = None
    if str(args.device).startswith("cuda"):
        gpu_info = check_gpu_preflight(args.gpu_preflight_timeout_s)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(args.output.name + ".partial")
    if partial_output.exists():
        partial_output.unlink()
    command_args = argparse.Namespace(**vars(args))
    command_args.output = partial_output
    command = build_isaac_replay_command(command_args)
    environment = os.environ.copy()
    environment.setdefault("CONDA_PREFIX", sys.prefix)
    environment["PATH"] = f"{sys.prefix}/bin:{environment.get('PATH', '')}"
    if environment.get("TERM") in (None, "", "dumb"):
        environment["TERM"] = "xterm"
    environment.setdefault("XDG_CACHE_HOME", "/tmp/isaac-cache")
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        result = subprocess.run(
            command,
            cwd=args.isaaclab_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=float(args.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Isaac replay exceeded timeout {args.timeout_s:g} s.") from exc
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout[-4000:], result.stderr[-4000:]) if part).strip()
        raise RuntimeError(f"Isaac replay failed with exit code {result.returncode}:\n{detail}")
    artifact = validate_isaac_replay_output(
        partial_output,
        expected_start_time_s=float(measured.time_s[0].item()),
        expected_duration_s=expected_duration_s,
    )
    partial_output.replace(args.output)
    artifact["output_log"] = str(args.output)
    return {
        "passed": True,
        "command": command,
        "device": args.device,
        "gpu_preflight": gpu_info,
        "process_exit_code": result.returncode,
        "artifact": artifact,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_checked_replay(args)
        exit_code = 0
    except Exception as exc:
        report = {"passed": False, "error": str(exc)}
        exit_code = 2
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

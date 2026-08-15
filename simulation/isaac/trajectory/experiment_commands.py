"""Compatibility facade for training and evaluation command helpers."""

from .evaluation_commands import (
    build_eval_command,
    eval_dir,
    eval_dir_name,
    eval_request_case_label,
    logs_path,
    run_eval_matrix,
    summary_path,
    validate_trajectories,
)
from .training_commands import (
    benchmark_gpu_throughput,
    build_gpu_benchmark_commands,
    build_train_command,
    train_policy,
)

__all__ = [
    "benchmark_gpu_throughput",
    "build_eval_command",
    "build_gpu_benchmark_commands",
    "build_train_command",
    "eval_dir",
    "eval_dir_name",
    "eval_request_case_label",
    "logs_path",
    "run_eval_matrix",
    "summary_path",
    "train_policy",
    "validate_trajectories",
]

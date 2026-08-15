"""Training command construction and launch helpers."""

from __future__ import annotations

from collections.abc import Sequence
import json

from simulation.isaac.trajectory.experiment_models import (
    ExperimentSpec, TrainRequest, TrajectoryCurriculumRequest,
)
from simulation.isaac.trajectory.experiment_process import run_command
from simulation.isaac.trajectory.experiment_runs import latest_run_dir


def _trajectory_curriculum_overrides(curriculum: TrajectoryCurriculumRequest) -> list[str]:
    values = {
        "trajectory_curriculum": curriculum.enabled,
        "trajectory_amp_x_range": curriculum.amplitude_x_range,
        "trajectory_amp_y_range": curriculum.amplitude_y_range,
        "trajectory_amp_z_range": curriculum.amplitude_z_range,
        "trajectory_period_range": curriculum.period_range,
        "trajectory_speed_levels_mps": curriculum.speed_levels_mps,
        "trajectory_curriculum_stage_steps": curriculum.stage_steps,
        "trajectory_curriculum_stage_0_types": curriculum.stage_0_types,
        "trajectory_curriculum_stage_1_types": curriculum.stage_1_types,
        "trajectory_curriculum_stage_2_types": curriculum.stage_2_types,
        "trajectory_curriculum_stage_3_types": curriculum.stage_3_types,
        "trajectory_curriculum_amp_scales": curriculum.amplitude_scales,
        "trajectory_curriculum_z_amp_scales": curriculum.vertical_amplitude_scales,
        "trajectory_curriculum_period_min": curriculum.period_min_by_stage,
        "trajectory_curriculum_period_max": curriculum.period_max_by_stage,
    }
    return [
        f"env.{name}={json.dumps(value, separators=(',', ':'))}"
        for name, value in values.items()
    ]


def _mlp_architecture_overrides(spec: ExperimentSpec) -> list[str]:
    """Forward one named MLP recipe to both IsaacLab and RSL-RL.

    The environment owns the causal history buffer, while the runner owns the
    layer widths.  Passing them together makes a checkpoint self-describing in
    its saved Hydra configuration and prevents train/eval input-shape drift.
    """

    architecture = spec.architecture
    values = {
        "env.mlp_architecture": architecture.name,
        "agent.experiment_name": spec.rsl_experiment_name,
        "agent.policy.actor_hidden_dims": list(architecture.actor_hidden_dims),
        "agent.policy.critic_hidden_dims": list(architecture.critic_hidden_dims),
    }
    return [
        f"{name}={json.dumps(value, separators=(',', ':'))}"
        for name, value in values.items()
    ]


def build_train_command(spec: ExperimentSpec, request: TrainRequest) -> list[str]:
    command = [
        "./isaaclab.sh",
        "-p",
        spec.train_script,
        "--task",
        spec.task_name,
        "--num_envs",
        str(request.num_envs),
        "--seed",
        str(request.seed),
    ]
    command.extend(_mlp_architecture_overrides(spec))
    if request.max_iterations is not None:
        command.extend(["--max_iterations", str(request.max_iterations)])
    if request.run_name:
        command.extend(["--run_name", request.run_name])
    if request.headless:
        command.append("--headless")
    command.extend(request.extra_args)
    if request.rollout_steps_per_env is not None:
        command.append(f"agent.num_steps_per_env={request.rollout_steps_per_env}")
    command.append(f"env.tracking_reward_profile={request.reward_profile}")
    if request.environment_profile is not None:
        command.append(f"env.environment_profile={request.environment_profile}")
    if request.domain_randomization_spec is not None:
        command.append(f"env.domain_randomization_spec={request.domain_randomization_spec}")
    if request.domain_randomization_features is not None:
        command.append("env.domain_randomization_feature_override_enabled=true")
        command.append(
            "env.domain_randomization.enabled_features="
            + json.dumps(list(request.domain_randomization_features), separators=(",", ":"))
        )
    if request.trajectory_curriculum is not None:
        command.extend(_trajectory_curriculum_overrides(request.trajectory_curriculum))
    if request.resume_load_run:
        if not request.resume_checkpoint:
            raise ValueError("resume_load_run requires resume_checkpoint.")
        # These are argparse flags of IsaacLab's RSL-RL launcher, rather than
        # Hydra fields.  In particular, the launcher's ``--resume`` default
        # otherwise overwrites ``agent.resume=true`` after Hydra resolves it.
        command.extend(
            (
                "--resume",
                "--load_run",
                request.resume_load_run,
                "--checkpoint",
                request.resume_checkpoint,
            )
        )
    return command


def train_policy(spec: ExperimentSpec, request: TrainRequest, *, execute: bool = False) -> tuple[int | None, str | None]:
    result = run_command(build_train_command(spec, request), cwd=spec.isaaclab_root, execute=execute, label="TRAIN")
    selected = latest_run_dir(spec, request.reward_profile).name if execute else None
    if selected:
        print(f"Selected completed training run: {selected}")
    return result, selected


def build_gpu_benchmark_commands(
    spec: ExperimentSpec,
    request: TrainRequest,
    env_candidates: Sequence[int],
) -> list[list[str]]:
    return [
        build_train_command(
            spec,
            TrainRequest(
                reward_profile=request.reward_profile,
                seed=request.seed,
                num_envs=num_envs,
                run_name=f"gpu_bench_{request.reward_profile}_{num_envs}",
                headless=request.headless,
                extra_args=request.extra_args,
                max_iterations=1,
                rollout_steps_per_env=request.rollout_steps_per_env,
                environment_profile=request.environment_profile,
                domain_randomization_spec=request.domain_randomization_spec,
                trajectory_curriculum=request.trajectory_curriculum,
            ),
        )
        for num_envs in env_candidates
    ]


def benchmark_gpu_throughput(
    spec: ExperimentSpec,
    request: TrainRequest,
    env_candidates: Sequence[int],
    *,
    execute: bool = False,
) -> None:
    for command in build_gpu_benchmark_commands(spec, request, env_candidates):
        run_command(command, cwd=spec.isaaclab_root, execute=execute, label="GPU BENCHMARK")

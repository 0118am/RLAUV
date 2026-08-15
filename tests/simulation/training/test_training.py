"""Focused contracts for recipes, manifests, and the explicit runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from environment.profiles.composition import resolve_runtime_composition
from environment.randomization import reset_current, reset_hydrodynamics
from environment.runtime import BodyKinematics, EnvironmentRuntimeState
from robot.randomization import reset_actuators, reset_battery, reset_rigid_body
from robot.runtime_state import RobotRuntimeState
from simulation.training import DEFAULT_TRAINING_RECIPE, load_training_recipe, materialize_run_inputs
from simulation.training.campaign import build_train_command
from simulation.training.evaluation.campaign import build_eval_command
from simulation.training.manifest import (
    build_run_manifest,
    load_run_manifest,
    validate_manifest_selection,
    write_run_manifest,
)
from simulation.training.recipe import EvalRequest, ExperimentSpec, TrainRequest


def _materialized_manifest(run_dir: Path):
    recipe = load_training_recipe()
    materialize_run_inputs(recipe, run_dir)
    architecture = recipe.architecture
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.01),
        decimation=2,
        scene=SimpleNamespace(num_envs=4),
    )
    agent_cfg = SimpleNamespace(
        seed=17,
        num_steps_per_env=recipe.rollout_steps_per_env,
        max_iterations=recipe.max_iterations,
        policy=SimpleNamespace(
            actor_hidden_dims=list(architecture.actor_hidden_dims),
            critic_hidden_dims=list(architecture.critic_hidden_dims),
            activation="elu",
        ),
    )
    return recipe, write_run_manifest(
        build_run_manifest(
            recipe=recipe,
            task_name="Isaac-AUV-Traj-Direct-v1",
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            run_dir=run_dir,
        )
    )


def test_recipe_materializes_run_local_inputs_and_rejects_unknown_fields(tmp_path: Path) -> None:
    recipe = load_training_recipe()
    paths = materialize_run_inputs(recipe, tmp_path / "run")

    assert paths.recipe.parent == tmp_path / "run" / "params" / "inputs"
    assert paths.environment.is_file() and paths.domain_randomization.is_file()
    assert "thruster_reaction_torque_coeff_scale_range" not in paths.domain_randomization.read_text()

    recipe_data = json.loads(DEFAULT_TRAINING_RECIPE.read_text(encoding="utf-8"))
    recipe_data["randomization_overrides"][
        "thruster_reaction_torque_coeff_scale_range"
    ] = [0.9, 1.1]
    invalid = tmp_path / "invalid_recipe.json"
    invalid.write_text(json.dumps(recipe_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown randomization_overrides field"):
        load_training_recipe(invalid)


def test_manifest_drives_evaluation_and_rejects_explicit_mismatches(tmp_path: Path) -> None:
    recipe = load_training_recipe()
    spec = ExperimentSpec(
        isaaclab_root=tmp_path,
        rlpolicy_root=tmp_path / "policies",
        mlp_architecture=recipe.mlp_architecture,
    )
    run_dir = spec.logs_root / "run-a"
    _, manifest_path = _materialized_manifest(run_dir)
    manifest = load_run_manifest(run_dir)

    command = build_eval_command(
        spec,
        EvalRequest(trajectories=("lissajous",)),
        "run-a",
        "model_50.pt",
        "lissajous",
    )
    assert command[command.index("--run_manifest") + 1] == str(manifest_path)
    assert command[command.index("--environment_profile") + 1] == str(
        manifest.input_path("environment")
    )
    assert "hash" not in manifest_path.read_text(encoding="utf-8").lower()
    with pytest.raises(ValueError, match="does not match run manifest"):
        validate_manifest_selection(manifest, mlp_architecture="mlp_30d")
    with pytest.raises(ValueError, match="does not match run manifest"):
        validate_manifest_selection(manifest, reward_profile="policy_0")


def test_explicit_runtime_nominal_dr_and_wrench_contract() -> None:
    recipe = load_training_recipe()
    environment_profile, randomization_spec = recipe.resolve_profiles()
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.01),
        domain_randomization=SimpleNamespace(),
        evaluation_current_override=False,
        evaluation_current_variation_std=0.0,
        evaluation_current_tau=12.0,
        evaluation_thruster_force_scale_override=False,
        evaluation_thruster_force_scale=1.0,
    )
    composition = resolve_runtime_composition(environment_profile, randomization_spec)
    composition.apply(cfg)
    num_envs = 2
    environment = EnvironmentRuntimeState(
        cfg,
        num_envs=num_envs,
        device="cpu",
        gravity_w=torch.tensor([0.0, 0.0, -9.81]),
        pool_center_local=(2.5, 2.5, 1.0),
    )
    robot = RobotRuntimeState(
        cfg,
        model=composition.robot.model,
        num_envs=num_envs,
        action_dim=8,
        device="cpu",
    )
    env_ids = torch.arange(num_envs)

    reset_rigid_body(robot, cfg, env_ids, 0, enabled=False)
    reset_current(environment, cfg, env_ids, 0, enabled=False)
    reset_hydrodynamics(environment, cfg, env_ids, 0, enabled=False)
    reset_actuators(robot, cfg, env_ids, 0, enabled=False)
    reset_battery(robot, cfg, env_ids, 0, enabled=False)
    stage = len(cfg.domain_randomization.water_current_max_by_stage) - 1
    payload_scale = reset_rigid_body(robot, cfg, env_ids, stage, enabled=True)
    reset_current(environment, cfg, env_ids, stage, enabled=True)
    reset_hydrodynamics(environment, cfg, env_ids, stage, enabled=True)
    reset_actuators(robot, cfg, env_ids, stage, enabled=True)
    reset_battery(robot, cfg, env_ids, stage, enabled=True)
    if payload_scale is not None:
        environment.apply_payload_hydrodynamic_scale(
            env_ids,
            linear_damping=payload_scale.linear_damping,
            quadratic_damping=payload_scale.quadratic_damping,
            added_mass=payload_scale.added_mass,
        )

    positions = torch.tensor([[2.5, 2.5, 1.0]]).repeat(num_envs, 1)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
    zeros = torch.zeros((num_envs, 3))
    kinematics = BodyKinematics(
        root_position_w=positions,
        root_position_local_w=positions,
        root_quat_w=quaternions,
        root_linear_velocity_w=zeros,
        root_linear_velocity_b=zeros,
        root_angular_velocity_b=zeros,
        scene_origins_w=zeros,
    )
    robot.advance_battery(torch.zeros((num_envs, 1)))
    raw_forces = robot.advance_thruster_forces(
        torch.full((num_envs, 8), 0.1), physics_time_s=0.01, physics_dt=0.01
    )
    effective = environment.calculate_effective_state(
        kinematics, sim_time_s=0.01, additional_scale=1.0
    )
    thrust = robot.compose_thruster_wrench(
        raw_forces,
        relative_velocity_b=effective.relative_velocity_b,
        environment_thruster_scale=effective.thruster_scale,
    )
    relative_acceleration = environment.update_relative_acceleration(
        effective.relative_velocity_b, physics_dt=0.01
    )
    fluid = environment.compose_fluid_wrench(
        kinematics,
        effective,
        volumes=robot.volumes,
        com_to_cob_offsets=robot.com_to_cob_offsets,
        relative_acceleration_b=relative_acceleration,
    )
    tether = robot.compose_tether_wrench(
        kinematics,
        water_current_w=effective.water_current_w,
        gravity_w=environment.gravity_w,
        physics_dt=0.01,
        additional_scale=1.0,
    )
    total = (thrust[0] + fluid[0] + tether[0], thrust[1] + fluid[1] + tether[1])
    assert all(value.shape == (num_envs, 3) for value in total)
    assert all(torch.isfinite(value).all() for value in total)

    assembly = (Path(__file__).resolve().parents[3] / "simulation" / "assembly.py").read_text()
    assert assembly.count("permanent_wrench_composer.set_forces_and_torques(") == 1


def test_training_command_uses_recipe_and_managed_worker_path(tmp_path: Path) -> None:
    recipe = load_training_recipe()
    spec = ExperimentSpec(
        isaaclab_root=tmp_path,
        rlpolicy_root=tmp_path / "policies",
        mlp_architecture=recipe.mlp_architecture,
    )
    request = TrainRequest(
        reward_profile=recipe.reward_profile,
        training_recipe=DEFAULT_TRAINING_RECIPE,
        num_envs=1,
    )
    command = build_train_command(spec, request)
    assert spec.train_script in command
    assert "--training_recipe" in command


def test_only_assembly_uses_direct_physx_write_interfaces() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    assembly = repo_root / "simulation" / "assembly.py"
    markers = (
        "root_physx_view",
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
        "permanent_wrench_composer.set_forces_and_torques",
    )
    assert all(marker in assembly.read_text(encoding="utf-8") for marker in markers)
    for source_root in (
        repo_root / "environment",
        repo_root / "robot",
        repo_root / "simulation" / "training",
    ):
        for path in source_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8", errors="ignore")
            assert not any(marker in source for marker in markers), path

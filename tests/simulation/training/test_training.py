"""Focused contracts for recipes and the explicit runtimes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math

import pytest
import torch

from environment.randomization import reset_current, reset_hydrodynamics
from environment.runtime import BodyKinematics, EnvironmentRuntimeState
from robot.randomization import reset_actuators
from robot.control.trajectory.observation_contract import (
    BASE_OBSERVATION_DIM,
    OBSERVATION_CONTRACT_VERSION,
    TRAJECTORY_OBSERVATION,
)
from robot.control.trajectory import AXIS_SINE, LATERAL_WAVE, VERTICAL_WAVE
from robot.runtime_state import RobotRuntimeState
from simulation.composition import resolve_runtime_composition
from simulation.domain_randomization import DomainRandomizationProfile, disturbance_stage_count
from simulation.training import (
    apply_training_recipe,
    load_training_recipe,
    materialize_run_inputs,
)
from simulation.training.ppo.networks import get_mlp_architecture
from simulation.training.recipe import expand_trajectory_stage_commands
from simulation.training.rewards import PRECISION_V9


def test_recipe_materializes_run_local_inputs(tmp_path: Path) -> None:
    recipe = load_training_recipe()
    paths = materialize_run_inputs(recipe, tmp_path / "run")

    assert paths.recipe.parent == tmp_path / "run" / "params" / "inputs"
    assert paths.environment.is_file() and paths.domain_randomization.is_file()


def test_speed_curvature_stages_apply_without_parallel_legacy_fields() -> None:
    recipe = load_training_recipe()
    env_cfg = SimpleNamespace()
    agent_cfg = SimpleNamespace(
        policy=SimpleNamespace(actor_hidden_dims=[], critic_hidden_dims=[])
    )

    apply_training_recipe(recipe, env_cfg, agent_cfg)

    assert env_cfg.trajectory_curriculum_stage_start_steps == [
        0,
        6_400,
        19_200,
        38_400,
    ]
    assert len(env_cfg.trajectory_curriculum_stage_types[-1]) == 35
    assert set(env_cfg.trajectory_curriculum_stage_types[0]) == {AXIS_SINE}
    assert set(env_cfg.trajectory_curriculum_stage_types[-1]) == {
        AXIS_SINE,
        LATERAL_WAVE,
        VERTICAL_WAVE,
    }
    assert set(env_cfg.trajectory_curriculum_stage_axes[0]) == {0, 1, 2}
    assert set(env_cfg.trajectory_curriculum_stage_wave_counts[-1]) == {1, 2, 3}
    assert max(env_cfg.trajectory_curriculum_stage_speeds_mps[-1]) == 0.5
    axis_speed_levels_by_axis = {
        axis: {
            speed
            for trajectory_type, command_axis, speed in zip(
                env_cfg.trajectory_curriculum_stage_types[-1],
                env_cfg.trajectory_curriculum_stage_axes[-1],
                env_cfg.trajectory_curriculum_stage_speeds_mps[-1],
                strict=True,
            )
            if trajectory_type == AXIS_SINE and command_axis == axis
        }
        for axis in (0, 1, 2)
    }
    assert axis_speed_levels_by_axis == {
        0: {0.1, 0.2, 0.3, 0.4, 0.5},
        1: {0.1, 0.2, 0.25},
        2: {0.1, 0.2, 0.3},
    }
    assert not hasattr(env_cfg, "trajectory_speed_levels_mps")
    assert not hasattr(env_cfg, "trajectory_curriculum_amp_scales")
    assert not hasattr(env_cfg, "trajectory_curriculum_stage_geometry_scales")
    assert agent_cfg.policy.activation == recipe.architecture.activation


def test_precision_recipe_covers_full_curve_and_actuator_transient() -> None:
    recipe = load_training_recipe()
    environment, randomization = recipe.resolve_profiles()
    curriculum = recipe.trajectory_curriculum
    architecture = get_mlp_architecture(recipe.mlp_architecture)

    assert curriculum.amplitude_x_range[1] == 2.5
    assert curriculum.amplitude_y_range[1] == 1.5
    assert curriculum.amplitude_z_range[1] == 0.5
    assert tuple(stage.start_step for stage in curriculum.stages) == (
        0,
        6_400,
        19_200,
        38_400,
    )
    assert tuple(
        sum(
            len(expand_trajectory_stage_commands(stage))
            for stage in curriculum.stages[: index + 1]
        )
        for index in range(len(curriculum.stages))
    ) == (3, 13, 27, 35)
    assert architecture.history_steps * 0.04 == 0.32
    assert recipe.schema_version == 6
    assert recipe.action_distribution == "tanh_gaussian_v1"
    assert OBSERVATION_CONTRACT_VERSION == "t60_trajectory_obs_v8"
    assert BASE_OBSERVATION_DIM == 33
    assert TRAJECTORY_OBSERVATION.field("projected_gravity_b").width == 3
    assert TRAJECTORY_OBSERVATION.field("motor_command").width == 8
    assert TRAJECTORY_OBSERVATION.field("target_linear_velocity_b").physical_scale == 0.5
    assert architecture.observation_dim == 201
    assert architecture.critic_privileged_dim == 61
    assert architecture.critic_observation_dim == 262
    training_types = {
        command[0]
        for stage in curriculum.stages
        for command in expand_trajectory_stage_commands(stage)
    }
    assert training_types == {AXIS_SINE, LATERAL_WAVE, VERTICAL_WAVE}
    assert recipe.reward_profile == PRECISION_V9.name
    assert recipe.rollout_steps_per_env == 128
    assert recipe.use_boundaries
    assert recipe.trajectory_startup_duration_s == 4.0
    assert recipe.initial_state.position_radius_m == 0.2
    assert set(randomization.parameters.enabled_features) == {
        "current",
        "hydrodynamics",
        "actuators",
    }
    assert randomization.parameters.thruster_time_constant_range == (0.064, 0.096)
    assert randomization.schema_version == 9
    assert randomization.parameters.disturbance_curriculum_stage_steps == (
        12_800,
        25_600,
        44_800,
    )
    assert randomization.parameters.linear_damping_log_std_by_stage == (
        0.0,
        0.04,
        0.07,
        0.1,
    )
    assert randomization.parameters.quadratic_damping_log_std_by_stage == (
        0.0,
        0.05,
        0.1,
        0.15,
    )
    assert randomization.parameters.fluid_added_mass_log_std_by_stage == (
        0.0,
        0.025,
        0.05,
        0.1,
    )
    assert randomization.parameters.common_thruster_scale_reduction_by_stage == (
        0.0,
        0.05,
        0.1,
        0.15,
    )
    assert not hasattr(randomization.parameters, "damping_scale_by_stage")
    assert not hasattr(randomization.parameters, "mass_range")
    assert not hasattr(randomization.parameters, "com_to_cob_offset_radius")
    assert not hasattr(randomization.parameters, "mass_relative_amplitude_by_stage")
    assert not hasattr(randomization.parameters, "center_of_mass_offset_amplitude_by_stage")
    assert not hasattr(randomization.parameters, "com_to_cob_relative_amplitude_by_stage")
    assert not hasattr(randomization.parameters, "volume_range")
    assert not hasattr(randomization.parameters, "payload_samples")

    bounds = environment.pool_boundary.bounds
    center = tuple(
        0.5 * (bounds[index] + bounds[index + 1]) for index in (0, 2, 4)
    )
    body_half_extents = (0.5615 / 2.0, 0.401999756 / 2.0, 0.190621773 / 2.0)
    conservative_clearance = math.sqrt(sum(value * value for value in body_half_extents)) + 0.20
    for axis, amplitude in enumerate((2.5, 1.5, 0.5)):
        lower, upper = bounds[2 * axis : 2 * axis + 2]
        assert center[axis] - amplitude - body_half_extents[axis] >= lower
        assert center[axis] + amplitude + body_half_extents[axis] <= upper
        assert center[axis] - amplitude - lower >= conservative_clearance
        assert upper - center[axis] - amplitude >= conservative_clearance


def test_removed_shared_damping_scale_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="damping_scale_by_stage"):
        DomainRandomizationProfile(damping_scale_by_stage=(0.1,))


def test_hydrodynamics_dr_has_independent_mean_one_lognormal_scales() -> None:
    torch.manual_seed(20_260_825)
    recipe = load_training_recipe()
    _, randomization = recipe.resolve_profiles()
    parameters = randomization.parameters
    sample_count = 100_000
    nominal_linear = torch.arange(1.0, 7.0)
    nominal_quadratic = torch.arange(7.0, 13.0)
    nominal_fluid_added_mass = torch.arange(13.0, 19.0)
    state = SimpleNamespace(
        device=torch.device("cpu"),
        nominal_linear_damping=nominal_linear,
        nominal_quadratic_damping=nominal_quadratic,
        nominal_fluid_added_mass=nominal_fluid_added_mass,
        linear_damping=nominal_linear.repeat(sample_count, 1),
        quadratic_damping=nominal_quadratic.repeat(sample_count, 1),
        fluid_added_mass=nominal_fluid_added_mass.repeat(sample_count, 1),
        linear_damping_randomization_scale=torch.ones(sample_count, 1),
        quadratic_damping_randomization_scale=torch.ones(sample_count, 1),
        fluid_added_mass_randomization_scale=torch.ones(sample_count, 6),
        damping_speed_linear_randomization_scale=torch.ones(sample_count, 6),
        damping_speed_quadratic_randomization_scale=torch.ones(sample_count, 6),
    )
    cfg = SimpleNamespace(domain_randomization=parameters)
    env_ids = torch.arange(sample_count)
    stage = disturbance_stage_count(parameters) - 1

    reset_hydrodynamics(state, cfg, env_ids, stage, enabled=True)

    linear_scale = state.linear_damping_randomization_scale[:, 0]
    quadratic_scale = state.quadratic_damping_randomization_scale[:, 0]
    fluid_added_mass_scale = state.fluid_added_mass_randomization_scale
    torch.testing.assert_close(
        state.linear_damping,
        nominal_linear.reshape(1, 6) * linear_scale.reshape(-1, 1),
    )
    torch.testing.assert_close(
        state.quadratic_damping,
        nominal_quadratic.reshape(1, 6) * quadratic_scale.reshape(-1, 1),
    )
    torch.testing.assert_close(
        state.fluid_added_mass,
        nominal_fluid_added_mass.reshape(1, 6) * fluid_added_mass_scale,
    )
    assert torch.all(linear_scale > 0.0)
    assert torch.all(quadratic_scale > 0.0)
    assert torch.all(fluid_added_mass_scale > 0.0)

    linear_log_std = parameters.linear_damping_log_std_by_stage[stage]
    quadratic_log_std = parameters.quadratic_damping_log_std_by_stage[stage]
    fluid_added_mass_log_std = parameters.fluid_added_mass_log_std_by_stage[stage]
    expected_linear_std = math.sqrt(math.expm1(linear_log_std**2))
    expected_quadratic_std = math.sqrt(math.expm1(quadratic_log_std**2))
    expected_fluid_added_mass_std = math.sqrt(
        math.expm1(fluid_added_mass_log_std**2)
    )
    assert float(linear_scale.mean()) == pytest.approx(1.0, abs=0.0015)
    assert float(quadratic_scale.mean()) == pytest.approx(1.0, abs=0.0015)
    assert float(linear_scale.std(unbiased=False)) == pytest.approx(
        expected_linear_std, abs=0.002
    )
    assert float(quadratic_scale.std(unbiased=False)) == pytest.approx(
        expected_quadratic_std, abs=0.002
    )
    torch.testing.assert_close(
        fluid_added_mass_scale.mean(dim=0),
        torch.ones(6),
        atol=1.5e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fluid_added_mass_scale.std(dim=0, unbiased=False),
        torch.full((6,), expected_fluid_added_mass_std),
        atol=2.0e-3,
        rtol=0.0,
    )
    correlation = torch.corrcoef(torch.stack((linear_scale, quadratic_scale)))[0, 1]
    assert abs(float(correlation)) < 0.01


def test_hydrodynamics_dr_preserves_full_matrix_structure() -> None:
    torch.manual_seed(42)
    sample_count = 32
    nominal_linear = torch.diag(torch.arange(1.0, 7.0))
    nominal_linear[0, 4] = nominal_linear[4, 0] = 0.25
    nominal_quadratic = torch.diag(torch.arange(7.0, 13.0))
    nominal_quadratic[1, 5] = nominal_quadratic[5, 1] = -0.4
    nominal_fluid_added_mass = torch.diag(torch.arange(13.0, 19.0))
    nominal_fluid_added_mass[2, 4] = nominal_fluid_added_mass[4, 2] = 0.5
    state = SimpleNamespace(
        device=torch.device("cpu"),
        nominal_linear_damping=nominal_linear,
        nominal_quadratic_damping=nominal_quadratic,
        nominal_fluid_added_mass=nominal_fluid_added_mass,
        linear_damping=nominal_linear.repeat(sample_count, 1, 1),
        quadratic_damping=nominal_quadratic.repeat(sample_count, 1, 1),
        fluid_added_mass=nominal_fluid_added_mass.repeat(sample_count, 1, 1),
        linear_damping_randomization_scale=torch.ones(sample_count, 1),
        quadratic_damping_randomization_scale=torch.ones(sample_count, 1),
        fluid_added_mass_randomization_scale=torch.ones(sample_count, 6),
        damping_speed_linear_randomization_scale=torch.ones(sample_count, 6),
        damping_speed_quadratic_randomization_scale=torch.ones(sample_count, 6),
    )
    cfg = SimpleNamespace(
        domain_randomization=SimpleNamespace(
            linear_damping_log_std_by_stage=(0.1,),
            quadratic_damping_log_std_by_stage=(0.15,),
            fluid_added_mass_log_std_by_stage=(0.1,),
            damping_speed_linear_scale_range=None,
            damping_speed_quadratic_scale_range=None,
        )
    )
    env_ids = torch.arange(sample_count)

    reset_hydrodynamics(state, cfg, env_ids, 0, enabled=True)

    torch.testing.assert_close(
        state.linear_damping,
        nominal_linear.unsqueeze(0)
        * state.linear_damping_randomization_scale.reshape(-1, 1, 1),
    )
    torch.testing.assert_close(
        state.quadratic_damping,
        nominal_quadratic.unsqueeze(0)
        * state.quadratic_damping_randomization_scale.reshape(-1, 1, 1),
    )
    assert torch.count_nonzero(state.linear_damping[:, nominal_linear == 0.0]) == 0
    assert torch.count_nonzero(state.quadratic_damping[:, nominal_quadratic == 0.0]) == 0
    assert not torch.equal(
        state.linear_damping_randomization_scale,
        state.quadratic_damping_randomization_scale,
    )
    root_scale = torch.sqrt(state.fluid_added_mass_randomization_scale)
    expected_fluid_added_mass = (
        nominal_fluid_added_mass.unsqueeze(0)
        * root_scale.unsqueeze(1)
        * root_scale.unsqueeze(2)
    )
    torch.testing.assert_close(state.fluid_added_mass, expected_fluid_added_mass)
    torch.testing.assert_close(
        state.fluid_added_mass, state.fluid_added_mass.transpose(1, 2)
    )
    assert torch.all(torch.linalg.eigvalsh(state.fluid_added_mass) > 0.0)


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
    robot.reset_dynamic_buffers(env_ids, physics_time_s=0.0)

    reset_current(environment, cfg, env_ids, 0, enabled=False)
    reset_hydrodynamics(environment, cfg, env_ids, 0, enabled=False)
    reset_actuators(robot, cfg, env_ids, 0, enabled=False)
    torch.testing.assert_close(robot.common_thruster_force_scale, torch.ones((num_envs, 1)))
    stage = disturbance_stage_count(cfg.domain_randomization) - 1
    enabled_features = set(cfg.domain_randomization.enabled_features)
    reset_current(
        environment, cfg, env_ids, stage, enabled="current" in enabled_features
    )
    reset_hydrodynamics(
        environment, cfg, env_ids, stage, enabled="hydrodynamics" in enabled_features
    )
    reset_actuators(
        robot, cfg, env_ids, stage, enabled="actuators" in enabled_features
    )
    assert robot.common_thruster_force_scale.shape == (num_envs, 1)
    assert torch.all(robot.common_thruster_force_scale >= 0.85)
    assert torch.all(robot.common_thruster_force_scale <= 1.0)
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
    probe_forces = torch.arange(1, num_envs * 8 * 3 + 1, dtype=torch.float32).reshape(
        num_envs, 8, 3
    )
    robot.thruster_force_scale[:] = 1.0
    robot.common_thruster_force_scale[:] = torch.tensor([[0.8], [0.9]])
    robot.compose_thruster_wrench(
        probe_forces,
        relative_velocity_b=torch.zeros((num_envs, 6)),
        environment_thruster_scale=torch.ones((num_envs, 8)),
    )
    torch.testing.assert_close(
        robot.realized_thruster_forces_b,
        probe_forces * robot.common_thruster_force_scale.unsqueeze(-1),
    )
    raw_forces = robot.advance_thruster_forces(
        torch.full((num_envs, 8), 0.1), physics_time_s=0.01
    )
    effective = environment.calculate_effective_state(
        kinematics, sim_time_s=0.01, additional_scale=1.0
    )
    torch.testing.assert_close(
        effective.fluid_added_mass, environment.fluid_added_mass
    )
    thrust = robot.compose_thruster_wrench(
        raw_forces,
        relative_velocity_b=effective.relative_velocity_b,
        environment_thruster_scale=effective.thruster_scale,
    )
    velocity_b = torch.cat(
        (kinematics.root_linear_velocity_b, kinematics.root_angular_velocity_b), dim=-1
    )
    environment.update_current_acceleration(
        velocity_b, effective.relative_velocity_b, physics_dt=0.01
    )
    fluid = environment.compose_fluid_wrench(
        kinematics,
        effective,
        volumes=robot.volumes,
        com_to_cob_offsets=robot.com_to_cob_offsets,
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

"""Training command coverage for versioned randomization recipes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from environment.profiles.domain_randomization import (
    DomainRandomizationSpec,
    apply_domain_randomization_spec,
    load_domain_randomization_spec_json,
    validate_domain_randomization_base_profile,
)
from environment.profiles.pool_profile import (
    NOMINAL_POOL_DYNAMICS_PROFILE,
    DomainRandomizationProfile,
    PoolDynamicsProfile,
    apply_pool_dynamics_profile,
    load_pool_dynamics_profile_json,
    write_pool_dynamics_profile_json,
)
from environment.hydrodynamics.models import mean_one_lognormal_scale, scale_hydrodynamic_coefficients
from environment.calibration.build_pool_profile_from_calibration import (
    collect_domain_randomization_sources,
    main as build_profile_main,
    write_profile_and_randomization_spec_atomically,
)
from simulation.isaac.workflows.common.trajectory_experiment import (
    CompetenceGateCriteria,
    EvalRequest,
    ExperimentSpec,
    TrainRequest,
    TrajectoryCurriculumRequest,
    assess_competence_gate,
    build_eval_command,
    build_train_command,
    collect_summary_df,
    curriculum_current_sweep_requests,
    curriculum_eval_requests,
    curriculum_segment_request,
    eval_request_case_label,
    is_completed_run,
    summary_path,
)


def test_train_command_forwards_domain_randomization_spec() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), mlp_architecture="mlp_history_5")
    request = TrainRequest(
        reward_profile="policy_0",
        seed=17,
        pool_dynamics_profile="/profiles/measured_pool.json",
        domain_randomization_spec="/profiles/auv_pool_openfoam_hydrodynamics_dr_v1.json",
    )

    command = build_train_command(spec, request)

    assert "env.pool_dynamics_profile=/profiles/measured_pool.json" in command
    assert "env.domain_randomization_spec=/profiles/auv_pool_openfoam_hydrodynamics_dr_v1.json" in command
    assert 'agent.experiment_name="auv_traj_mlp_history_5"' in command
    assert 'env.mlp_architecture="mlp_history_5"' in command
    assert not any(part.startswith("env.mlp_history_") for part in command)
    assert "agent.policy.actor_hidden_dims=[512,384,256,128]" in command
    assert "--agent" not in command
    seed_index = command.index("--seed")
    assert command[seed_index + 1] == "17"

    baseline_command = build_train_command(
        ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), mlp_architecture="mlp_30d"), request
    )
    assert 'env.mlp_architecture="mlp_30d"' in baseline_command
    assert command != baseline_command


def test_train_command_forwards_explicit_domain_randomization_feature_subset() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), mlp_architecture="mlp_history_5")
    request = TrainRequest(
        reward_profile="policy_0",
        domain_randomization_spec="/profiles/auv_pool_openfoam_hydrodynamics_dr_v1.json",
        domain_randomization_features=("actuators", "battery"),
    )

    command = build_train_command(spec, request)

    assert "env.domain_randomization_feature_override_enabled=true" in command
    assert 'env.domain_randomization.enabled_features=["actuators","battery"]' in command


def test_train_command_forwards_notebook_trajectory_curriculum() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), mlp_architecture="mlp_history_5")
    curriculum = TrajectoryCurriculumRequest(
        enabled=True,
        amplitude_x_range=(0.60, 0.78),
        amplitude_y_range=(0.55, 0.75),
        amplitude_z_range=(0.08, 0.20),
        period_range=(10.0, 26.0),
        stage_steps=(9_750, 22_500, 40_500),
        stage_0_types=(0, 2, 7),
        stage_1_types=(7, 7, 0, 2, 3, 6),
        stage_2_types=(7, 7, 7, 0, 2, 3, 6),
        stage_3_types=(7, 7, 7, 7, 7, 0, 2, 3, 6),
        amplitude_scales=(0.55, 0.75, 0.90, 1.0),
        vertical_amplitude_scales=(0.25, 0.5, 0.75, 1.0),
        period_min_by_stage=(20.0, 10.0, 10.0, 10.0),
        period_max_by_stage=(20.0, 10.0, 10.0, 10.0),
    )
    request = TrainRequest(
        reward_profile="policy_0",
        max_iterations=500,
        rollout_steps_per_env=256,
        trajectory_curriculum=curriculum,
    )

    command = build_train_command(spec, request)

    assert "--max_iterations" in command
    assert command[command.index("--max_iterations") + 1] == "500"
    assert "--agent" not in command
    assert "agent.num_steps_per_env=256" in command
    assert "env.trajectory_curriculum=true" in command
    assert "env.trajectory_curriculum_stage_steps=[9750,22500,40500]" in command
    assert "env.trajectory_curriculum_stage_0_types=[0,2,7]" in command
    assert "env.trajectory_curriculum_amp_scales=[0.55,0.75,0.9,1.0]" in command


def test_train_command_uses_isaaclab_cli_flags_for_resume() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), mlp_architecture="mlp_history_5")
    request = TrainRequest(
        reward_profile="policy_5",
        resume_load_run="2026-07-17_19-21-56_trajectory_policy_5",
        resume_checkpoint="model_24.pt",
    )

    command = build_train_command(spec, request)

    assert command[command.index("--resume")] == "--resume"
    assert command[command.index("--load_run") + 1] == request.resume_load_run
    assert command[command.index("--checkpoint") + 1] == request.resume_checkpoint
    assert not any(part.startswith("agent.resume=") for part in command)


def test_training_notebook_is_the_single_human_recipe_selection_entry() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    train_notebook = (repository_root / "simulation/isaac/notebooks/train.ipynb").read_text(
        encoding="utf-8"
    )
    eval_notebook = (repository_root / "simulation/isaac/notebooks/evaluate.ipynb").read_text(
        encoding="utf-8"
    )
    train_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(train_notebook)["cells"]
        if cell.get("cell_type") == "code"
    )
    eval_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(eval_notebook)["cells"]
        if cell.get("cell_type") == "code"
    )

    assert train_notebook.count("DOMAIN_RANDOMIZATION_SPEC =") == 1
    assert train_notebook.count("TRAJECTORY_CURRICULUM =") == 1

    env_config = (repository_root / "simulation/isaac/envs/auv/config.py").read_text(encoding="utf-8")
    env_runtime = (repository_root / "simulation/isaac/envs/auv/env.py").read_text(encoding="utf-8")
    assert "auv_pool_openfoam_hydrodynamics_v1.json" in env_config
    assert "implicit hydrodynamic fallbacks are disabled" in env_runtime
    assert "NOMINAL_POOL_DYNAMICS_PROFILE" not in env_runtime
    assert 'domain_randomization_spec = ""' in env_config
    assert "trajectory_curriculum = False" in env_config
    assert "trajectory_curriculum_stage_steps = []" in env_config
    assert "trajectory_amp_x_range = [0.0, 0.0]" in env_config
    assert "trajectory_train_types = [0]" in env_config
    assert "observation_base_dim = 30" in env_config
    assert "critic_privileged_fields = []" in env_config
    assert "attitude_error_quat" in env_config
    assert 'MLP_ARCHITECTURE = "mlp_history_5"' in train_source
    assert 'MLP_ARCHITECTURE = "mlp_history_5"' in eval_source
    assert "importlib.reload(experiment_tools)" in train_notebook
    assert "importlib.reload(experiment_tools)" in eval_notebook
    assert "auv_pool_openfoam_hydrodynamics_v1.json" in eval_notebook
    assert "auv_pool_openfoam_hydrodynamics_dr_v1.json" in eval_notebook
    assert "SAMPLE_DOMAIN_RANDOMIZATION = True" in eval_source
    assert 'FINAL_RUN = ""' in eval_source
    assert 'FINAL_CHECKPOINT = "latest"' in eval_source
    assert "LOAD_RUN = FINAL_RUN" in eval_source
    assert "EVAL_CHECKPOINTS = FINAL_CHECKPOINT" in eval_source
    assert "checkpoint=EVAL_CHECKPOINTS" in eval_source
    assert "SHOW_INLINE_PLOTS = True" in eval_source
    assert "PPO_MAX_ITERATIONS = 500" in train_notebook
    assert "USE_COMPETENCE_GATE = True" in train_source
    assert "GATE_SEGMENT_ITERATIONS = 25" in train_source
    assert "simulation/isaac/workflows/train/competence_curriculum.py" in train_source
    assert "_stop_previous_campaign" in train_source
    assert "Stopped previous" in train_source
    assert "curriculum_nominal" in (repository_root / "simulation/isaac/workflows/common/trajectory_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "curriculum_robust" in (repository_root / "simulation/isaac/workflows/common/trajectory_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "stage_steps=(9_750, 22_500, 40_500)" in train_notebook
    assert "stage_0_types=(8, 9, 10)" in train_notebook
    assert "0.1/0.2/0.3/0.4 m/s" in train_notebook
    assert "speed_levels_mps=(0.1, 0.2, 0.3, 0.4)" in train_notebook
    assert "135-D Actor" in train_notebook

    agent_config = (repository_root / "simulation/isaac/agents/ppo/config.py").read_text(encoding="utf-8")
    assert 'experiment_name = "auv_traj_mlp"' in agent_config
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in agent_config

    tools = (repository_root / "simulation/isaac/workflows/common/trajectory_experiment.py").read_text(encoding="utf-8")
    assert "mlp_architecture" in tools
    assert "evaluation_console.log" in tools

    eval_script = (repository_root / "simulation/isaac/workflows/evaluate/trajectory.py").read_text(
        encoding="utf-8"
    )
    assert "raw_policy_actions = policy(obs)" in eval_script
    assert "actions = torch.clamp(raw_policy_actions, -1.0, 1.0)" in eval_script
    assert 'f"raw_policy_action_{action_index}"' in eval_script


def test_only_trajectory_task_remains_registered() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    registration = (repository_root / "__init__.py").read_text(encoding="utf-8")
    environment_source = (repository_root / "simulation/isaac/envs/auv/env.py").read_text(encoding="utf-8")
    config_source = (repository_root / "simulation/isaac/envs/auv/config.py").read_text(encoding="utf-8")

    assert registration.count("gym.register(") == 1
    assert registration.count('id="Isaac-AUV-Traj-Direct-v1"') == 1
    assert environment_source.count("class AUVTrajEnv(") == 1
    assert config_source.count("class AUVTrajEnvCfg(") == 1


def test_architecture_run_namespaces_and_completed_run_selection(tmp_path: Path) -> None:
    spec = ExperimentSpec(isaaclab_root=tmp_path)
    assert spec.experiment_name == "auv_traj_mlp_history_5"
    assert spec.architecture.observation_dim == 135
    assert spec.architecture.critic_privileged_dim == 77
    assert spec.architecture.critic_observation_dim == 212
    assert ExperimentSpec(isaaclab_root=tmp_path, mlp_architecture="mlp_30d").experiment_name == "auv_traj_mlp"
    assert ExperimentSpec(isaaclab_root=tmp_path, mlp_architecture="mlp_30d").architecture.critic_observation_dim == 107
    run_dir = spec.logs_root / "run-a"
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True)
    (run_dir / "model_1.pt").touch()
    (params_dir / "env.yaml").write_text(
        "tracking_reward_profile: policy_1\n",
        encoding="utf-8",
    )

    assert is_completed_run(run_dir, "policy_1")
    assert not is_completed_run(run_dir, "policy_0")


def test_training_notebook_selects_openfoam_pool_profile_and_its_curriculum() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    notebook = json.loads(
        (repository_root / "simulation/isaac/notebooks/train.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    profile_path = repository_root / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
    recipe_path = repository_root / "simulation/isaac/configs/domain_randomization/auv_pool_openfoam_hydrodynamics_dr_v1.json"

    assert str(profile_path.relative_to(repository_root)) in source
    assert str(recipe_path.relative_to(repository_root)) in source

    profile = load_pool_dynamics_profile_json(profile_path)
    recipe = load_domain_randomization_spec_json(recipe_path)
    validate_domain_randomization_base_profile(recipe, profile)

    assert profile.hydrodynamics.water_current_periodic_enabled is True
    assert profile.hydrodynamics.high_order_residual_enabled is False
    assert profile.free_surface.sloshing_enabled is True
    assert profile.thrusters.max_command_rate == 0.0
    added_mass = torch.tensor(profile.hydrodynamics.added_mass, dtype=torch.float64)
    linear_damping = torch.tensor(profile.hydrodynamics.linear_damping, dtype=torch.float64)
    quadratic_damping = torch.tensor(profile.hydrodynamics.quadratic_damping, dtype=torch.float64)
    expected_added_mass = torch.tensor([
        [14.029239, 0.0, -0.140840, 0.0, -0.171501, 0.0],
        [0.0, 20.233100, 0.0, 0.142224, 0.0, 0.012447],
        [0.086838, 0.0, 35.182178, 0.0, 0.177760, 0.0],
        [0.0, 0.105818, 0.0, 0.138202, 0.0, -0.005946],
        [-0.184144, 0.0, 0.254834, 0.0, 0.416299, 0.0],
        [0.0, 0.005376, 0.0, -0.005700, 0.0, 0.161243],
    ], dtype=torch.float64)
    expected_linear_damping = torch.tensor([
        [19.156401, 0.0, 5.997051, 0.0, -1.298618, 0.0],
        [0.0, 20.179327, 0.0, 0.510583, 0.0, -1.223627],
        [-8.268416, 0.0, 121.922336, 0.0, 5.878941, 0.0],
        [0.0, 0.812006, 0.0, 0.215864, 0.0, -0.040070],
        [0.154307, 0.0, 1.072468, 0.0, 0.417244, 0.0],
        [0.0, 1.172288, 0.0, -0.023618, 0.0, 0.274687],
    ], dtype=torch.float64)
    expected_quadratic_damping = torch.tensor([
        [4.551593, 0.0, -8.183631, 0.0, 0.628409, 0.0],
        [0.0, 2.762384, 0.0, -0.229215, 0.0, 0.242430],
        [4.954815, 0.0, -31.263433, 0.0, 0.691957, 0.0],
        [0.0, -0.356197, 0.0, 0.020291, 0.0, 0.048761],
        [-1.028514, 0.0, -3.197941, 0.0, 0.232852, 0.0],
        [0.0, 0.331193, 0.0, -0.016809, 0.0, 0.118359],
    ], dtype=torch.float64)
    assert torch.allclose(added_mass, expected_added_mass, rtol=0.0, atol=5.0e-7)
    assert torch.allclose(linear_damping, expected_linear_damping, rtol=0.0, atol=5.0e-7)
    assert torch.allclose(quadratic_damping, expected_quadratic_damping, rtol=0.0, atol=5.0e-7)
    assert not torch.allclose(added_mass, added_mass.T)
    assert "Raw fitted asymmetry" in profile.description
    assert recipe.parameters.additional_hydrodynamics_scale_by_stage == [0.0, 0.0, 0.35, 0.7, 1.0]
    assert recipe.parameters.added_mass_log_std_by_stage == [0.0, 0.0, 0.05, 0.08, 0.12]
    assert recipe.parameters.thruster_max_command_rate_range == [0.0, 0.0]


def test_added_mass_dr_multiplies_the_selected_openfoam_profile_baseline() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    profile = load_pool_dynamics_profile_json(
        repository_root / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
    )
    recipe = load_domain_randomization_spec_json(
        repository_root / "simulation/isaac/configs/domain_randomization/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )
    cfg = SimpleNamespace(domain_randomization=SimpleNamespace())

    apply_pool_dynamics_profile(cfg, profile, include_legacy_domain_randomization=False)
    assert cfg.thruster_max_command_rate == 0.0
    profile_baseline = torch.tensor(profile.hydrodynamics.added_mass, dtype=torch.float64).reshape(1, 6, 6)
    assert torch.equal(torch.tensor(cfg.added_mass_diag, dtype=torch.float64).reshape(1, 6, 6), profile_baseline)

    apply_domain_randomization_spec(cfg, recipe, base_profile=profile)
    assert cfg.domain_randomization.thruster_max_command_rate_range == [0.0, 0.0]
    assert torch.equal(torch.tensor(cfg.added_mass_diag, dtype=torch.float64).reshape(1, 6, 6), profile_baseline)

    normal_latent = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]], dtype=torch.float64)
    scale = mean_one_lognormal_scale(normal_latent, recipe.parameters.added_mass_log_std_by_stage[-1])
    sampled = scale_hydrodynamic_coefficients(profile_baseline, scale)

    expected = profile_baseline * torch.sqrt(scale.unsqueeze(1) * scale.unsqueeze(2))
    assert torch.allclose(sampled, expected)
    assert not torch.allclose(sampled, sampled.transpose(-1, -2))
    assert torch.allclose(torch.diagonal(sampled, dim1=-2, dim2=-1), torch.diagonal(profile_baseline, dim1=-2, dim2=-1) * scale)
    assert torch.all(torch.diagonal(sampled, dim1=-2, dim2=-1) > 0.0)


def test_openfoam_actuator_and_observation_recipe_reaches_runtime_cfg() -> None:

    repository_root = Path(__file__).resolve().parents[4]
    profile = load_pool_dynamics_profile_json(
        repository_root / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
    )
    recipe = load_domain_randomization_spec_json(
        repository_root / "simulation/isaac/configs/domain_randomization/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )
    cfg = SimpleNamespace(domain_randomization=SimpleNamespace())

    apply_pool_dynamics_profile(cfg, profile, include_legacy_domain_randomization=False)
    assert cfg.dyn_time_constant == pytest.approx(profile.thrusters.dyn_time_constant)

    # Training and an EvalRequest with sample_domain_randomization=True both
    # use this exact application path; eval_mode only controls whether reset
    # sampling is enabled at runtime.
    apply_domain_randomization_spec(cfg, recipe, base_profile=profile)
    assert cfg.domain_randomization.use_custom_randomization is True
    assert cfg.domain_randomization.observation_delay_steps_range == [0, 0]
    assert cfg.domain_randomization.observation_dropout_probability_range == [0.0, 0.0]
    # No command-chain latency is invented before timing measurements exist.
    assert cfg.domain_randomization.thruster_command_delay_steps_range == [0, 0]


def test_eval_command_forwards_reproducible_randomized_case() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"))
    recipe_path = (
        Path(__file__).resolve().parents[4]
        / "simulation/isaac/configs/domain_randomization/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )
    request = EvalRequest(
        reward_profile="policy_0",
        seed=23,
        pool_dynamics_profile="/profiles/measured_pool.json",
        domain_randomization_spec=recipe_path,
        sample_domain_randomization=True,
    )

    command = build_eval_command(spec, request, "run-a", "model_100.pt", "helix")

    assert command[command.index("--seed") + 1] == "23"
    assert command[command.index("--pool_dynamics_profile") + 1] == "/profiles/measured_pool.json"
    assert command[command.index("--domain_randomization_spec") + 1] == str(recipe_path)
    assert "--eval_domain_randomization" in command
    assert command[command.index("--mlp_architecture") + 1] == "mlp_history_5"
    case_label = eval_request_case_label(request)
    assert case_label == "dr_auv-pool-openfoam-hydrodynamics-dr-v1_seed23"


def test_eval_randomization_requires_a_recipe() -> None:
    with pytest.raises(ValueError, match="requires domain_randomization_spec"):
        eval_request_case_label(
            EvalRequest(reward_profile="policy_0", sample_domain_randomization=True)
        )


def test_competence_gate_uses_two_labeled_held_out_sets_and_two_pass_promotion(tmp_path: Path) -> None:
    curriculum = TrajectoryCurriculumRequest(
        enabled=True,
        amplitude_x_range=(0.60, 0.78),
        amplitude_y_range=(0.55, 0.75),
        amplitude_z_range=(0.08, 0.20),
        period_range=(10.0, 26.0),
        stage_steps=(9_750, 22_500, 40_500),
        stage_0_types=(0, 2, 7),
        stage_1_types=(7, 7, 0, 2, 3, 6),
        stage_2_types=(7, 7, 7, 0, 2, 3, 6),
        stage_3_types=(7, 7, 7, 7, 7, 0, 2, 3, 6),
        amplitude_scales=(0.55, 0.75, 0.90, 1.0),
        vertical_amplitude_scales=(0.25, 0.5, 0.75, 1.0),
        period_min_by_stage=(20.0, 10.0, 10.0, 10.0),
        period_max_by_stage=(20.0, 10.0, 10.0, 10.0),
    )
    request = TrainRequest(
        reward_profile="policy_1",
        domain_randomization_spec="/profiles/dr.json",
        trajectory_curriculum=curriculum,
    )
    nominal_request, robust_request = curriculum_eval_requests(request, stage=0)
    assert nominal_request.evaluation_label == "curve_v2_curriculum_nominal"
    assert nominal_request.align_initial_target
    assert not nominal_request.sample_domain_randomization
    assert robust_request.evaluation_label == "curve_v2_curriculum_robust"
    assert robust_request.sample_domain_randomization
    assert robust_request.eval_disturbance_stage == 4
    assert nominal_request.trajectories == ("random_smooth",)

    nominal_path = tmp_path / "nominal.csv"
    robust_path = tmp_path / "robust.csv"
    gate_columns = (
        "position_error_p95,velocity_rmse,any_failure_rate,reference_valid,"
        "reference_within_kinematic_envelope,min_reference_path_length_m,"
        "min_curve_target_speed_p95_mps,target_speed_max_mps,target_acceleration_max_mps2,"
        "target_orientation_rate_max_radps,target_jerk_max_mps3\n"
    )
    nominal_path.write_text(
        gate_columns + "0.5,0.3,0.0,1,1,0.5,0.2,0.59,0.44,0.79,0.35\n", encoding="utf-8"
    )
    robust_path.write_text(
        gate_columns + "0.7,0.4,0.0,1,1,0.5,0.2,0.59,0.44,0.79,0.35\n", encoding="utf-8"
    )
    decision = assess_competence_gate(
        checkpoint="model_24.pt",
        stage=0,
        nominal_summary=nominal_path,
        robust_summary=robust_path,
        previous_consecutive_passes=1,
        criteria=CompetenceGateCriteria(),
    )
    assert decision.passed
    assert decision.promoted
    assert decision.next_stage == 1
    assert decision.consecutive_passes == 0

    segment = curriculum_segment_request(request, stage=1, segment_iterations=25)
    assert segment.curriculum_gate_stage == 1
    assert segment.max_iterations == 25
    assert segment.disturbance_curriculum_global_step_offset == 0


def test_resumed_curriculum_segment_preserves_disturbance_progress() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), mlp_architecture="mlp_history_5")
    request = TrainRequest(
        reward_profile="policy_5",
        rollout_steps_per_env=256,
    )

    segment = curriculum_segment_request(
        request,
        stage=0,
        segment_iterations=25,
        completed_iterations=25,
    )
    assert segment.disturbance_curriculum_global_step_offset == 6_400

    command = build_train_command(spec, segment)
    assert "env.disturbance_curriculum_global_step_offset=6400" in command


def test_current_only_sweep_uses_matched_held_out_curves() -> None:
    curriculum = TrajectoryCurriculumRequest(
        enabled=True,
        amplitude_x_range=(0.4, 1.6),
        amplitude_y_range=(0.25, 1.0),
        amplitude_z_range=(0.08, 0.35),
        period_range=(16.0, 28.0),
        stage_steps=(7_500, 20_000, 37_500),
        stage_0_types=(0, 0, 1, 2),
        stage_1_types=(0, 1, 2, 3, 6),
        stage_2_types=(0, 1, 2, 3, 4, 5, 6),
        stage_3_types=(0, 1, 2, 3, 4, 5, 6),
        amplitude_scales=(0.35, 0.55, 0.8, 1.0),
        vertical_amplitude_scales=(0.25, 0.5, 0.75, 1.0),
        period_min_by_stage=(26.0, 22.0, 18.0, 16.0),
        period_max_by_stage=(30.0, 30.0, 28.0, 28.0),
    )
    request = TrainRequest(reward_profile="policy_5", trajectory_curriculum=curriculum)
    sweep, robust = curriculum_current_sweep_requests(
        request,
        stage=0,
        magnitudes_mps=(0.0, 0.10),
        direction_w=(0.0, 2.0, 0.0),
    )

    assert [item.seed for item in sweep] == [10_173, 10_173]
    assert [item.eval_current for item in sweep] == [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0)]
    assert not any(item.sample_domain_randomization for item in sweep)
    assert robust.eval_disturbance_stage == 4

    command = build_eval_command(
        ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab")),
        sweep[1],
        "run-a",
        "model_24.pt",
        "random_smooth",
    )
    assert command[command.index("--eval_current") + 1 : command.index("--eval_current") + 4] == ["0.0", "0.1", "0.0"]
    assert "--trajectory_amp_x_range" in command
    assert "--align_initial_target" in command


def test_random_smooth_eval_requires_non_static_ranges_and_forwards_them() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"))
    with pytest.raises(ValueError, match="requires explicit positive"):
        build_eval_command(
            spec,
            EvalRequest(reward_profile="policy_0"),
            "run-a",
            "model_24.pt",
            "random_smooth",
        )
    with pytest.raises(ValueError, match="must satisfy"):
        build_eval_command(
            spec,
            EvalRequest(
                reward_profile="policy_0",
                trajectory_amp_x_range=(0.0, 0.78),
                trajectory_amp_y_range=(0.55, 0.75),
                trajectory_amp_z_range=(0.08, 0.20),
                trajectory_period_range=(10.0, 20.0),
            ),
            "run-a",
            "model_24.pt",
            "random_smooth",
        )
    command = build_eval_command(
        spec,
        EvalRequest(
            reward_profile="policy_0",
            trajectory_amp_x_range=(0.60, 0.78),
            trajectory_amp_y_range=(0.55, 0.75),
            trajectory_amp_z_range=(0.08, 0.20),
            trajectory_period_range=(10.0, 20.0),
        ),
        "run-a",
        "model_24.pt",
        "random_smooth",
    )
    assert command[command.index("--trajectory_amp_x_range") + 1 : command.index("--trajectory_amp_x_range") + 3] == [
        "0.6",
        "0.78",
    ]
    assert "--trajectory_amp_x" not in command


def test_competence_gate_rejects_static_or_legacy_reference_summaries(tmp_path: Path) -> None:
    columns = (
        "position_error_p95,velocity_rmse,any_failure_rate,reference_valid,"
        "reference_within_kinematic_envelope,min_reference_path_length_m,"
        "min_curve_target_speed_p95_mps,target_speed_max_mps,target_acceleration_max_mps2,"
        "target_orientation_rate_max_radps,target_jerk_max_mps3\n"
    )
    static = tmp_path / "static.csv"
    static.write_text(columns + "0.1,0.1,0.0,0,1,0.0,0.0,0.0,0.0,0.0,0.0\n", encoding="utf-8")
    decision = assess_competence_gate(
        checkpoint="model_24.pt",
        stage=0,
        nominal_summary=static,
        robust_summary=static,
    )
    assert not decision.passed
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("position_error_p95,velocity_rmse,any_failure_rate\n0.1,0.1,0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference_valid"):
        assess_competence_gate(
            checkpoint="model_24.pt",
            stage=0,
            nominal_summary=legacy,
            robust_summary=legacy,
        )


def test_summary_collection_keeps_nominal_and_randomized_cases_separate(tmp_path: Path) -> None:
    spec = ExperimentSpec(isaaclab_root=tmp_path)
    run_name = "run-a"
    nominal_path = summary_path(spec, run_name, "model_100.pt", "helix")
    randomized_path = summary_path(
        spec,
        run_name,
        "model_100.pt",
        "helix",
        "dr_recipe_seed23",
    )
    nominal_path.parent.mkdir(parents=True)
    randomized_path.parent.mkdir(parents=True)
    nominal_path.write_text(
        "trajectory,disturbance,position_rmse,velocity_rmse\nhelix,nominal,0.2,0.3\n",
        encoding="utf-8",
    )
    randomized_path.write_text(
        "trajectory,disturbance,position_rmse,velocity_rmse\n"
        "helix,dr_recipe_seed23,0.4,0.5\n",
        encoding="utf-8",
    )

    nominal = collect_summary_df(spec, run_name, case_label="")
    randomized = collect_summary_df(spec, run_name, case_label="dr_recipe_seed23")

    assert nominal["position_rmse"].tolist() == [0.2]
    assert randomized["position_rmse"].tolist() == [0.4]


def test_calibration_builder_can_export_bound_randomization_recipe(tmp_path: Path) -> None:
    updates_path = tmp_path / "calibration_updates.json"
    profile_path = tmp_path / "measured_profile.json"
    randomization_path = tmp_path / "measured_profile_dr.json"
    updates_path.write_text(
        json.dumps(
            {
                "domain_randomization_updates": {
                    "use_custom_randomization": True,
                    "mass_range": [10.0, 10.2],
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = build_profile_main(
        [
            "--updates",
            str(updates_path),
            "--name",
            "measured-campaign",
            "--output",
            str(profile_path),
            "--domain-randomization-output",
            str(randomization_path),
        ]
    )
    exported = load_domain_randomization_spec_json(randomization_path)
    deterministic_profile = load_pool_dynamics_profile_json(profile_path)

    assert exit_code == 0
    assert deterministic_profile.domain_randomization is None
    assert exported.base_profile_name == "measured-campaign"
    assert exported.parameters.mass_range == [10.0, 10.2]
    assert "mass_range" in exported.parameter_sources


def test_calibration_source_records_filename_without_absolute_directory(tmp_path: Path) -> None:
    first = tmp_path / "machine-a" / "updates.json"
    second = tmp_path / "machine-b" / "updates.json"
    payload = json.dumps({"domain_randomization_updates": {"mass_range": [10.0, 10.2]}})
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    first_sources = collect_domain_randomization_sources([first], [])
    second_sources = collect_domain_randomization_sources([second], [])

    assert first_sources == second_sources
    assert first_sources["mass_range"] == "Calibration update file: updates.json"


def test_atomic_profile_recipe_export_preserves_existing_pair_on_validation_failure(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.json"
    recipe_path = tmp_path / "recipe.json"
    profile_path.write_text("old profile\n", encoding="utf-8")
    recipe_path.write_text("old recipe\n", encoding="utf-8")
    invalid_spec = DomainRandomizationSpec(
        name="partial",
        parameters=DomainRandomizationProfile(mass_range=[10.0, 10.1]),
    )

    with pytest.raises(ValueError, match="parameters must be complete"):
        write_profile_and_randomization_spec_atomically(
            NOMINAL_POOL_DYNAMICS_PROFILE,
            profile_path,
            invalid_spec,
            recipe_path,
        )

    assert profile_path.read_text(encoding="utf-8") == "old profile\n"
    assert recipe_path.read_text(encoding="utf-8") == "old recipe\n"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.bak")) == []


def test_builder_rejects_inherited_uncertainty_without_provenance(tmp_path: Path) -> None:
    base_path = tmp_path / "legacy_profile.json"
    profile_path = tmp_path / "deterministic_profile.json"
    recipe_path = tmp_path / "recipe.json"
    write_pool_dynamics_profile_json(
        PoolDynamicsProfile(
            name="legacy-profile",
            domain_randomization=DomainRandomizationProfile(
                use_custom_randomization=True,
                mass_range=[10.0, 10.2],
            ),
        ),
        base_path,
    )

    exit_code = build_profile_main(
        [
            "--base-profile",
            str(base_path),
            "--output",
            str(profile_path),
            "--domain-randomization-output",
            str(recipe_path),
        ]
    )

    assert exit_code == 1
    assert not profile_path.exists()
    assert not recipe_path.exists()

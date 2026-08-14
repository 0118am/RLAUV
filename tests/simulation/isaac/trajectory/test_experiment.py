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
from environment.profiles.environment_profile import load_environment_profile_json
from environment.hydrodynamics.models import mean_one_lognormal_scale, scale_hydrodynamic_coefficients
from robot.runtime import T60_RUNTIME
from simulation.isaac.composition import resolve_isaac_composition
from simulation.isaac.training import build_default_campaign, campaign_payload
from simulation.isaac.trajectory.experiment import (
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
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"), mlp_architecture="mlp_history_5")
    request = TrainRequest(
        reward_profile="policy_0",
        seed=17,
        environment_profile="/profiles/measured_pool.json",
        domain_randomization_spec="/profiles/auv_pool_openfoam_hydrodynamics_dr_v1.json",
    )

    command = build_train_command(spec, request)

    assert "env.environment_profile=/profiles/measured_pool.json" in command
    assert "env.domain_randomization_spec=/profiles/auv_pool_openfoam_hydrodynamics_dr_v1.json" in command
    assert 'agent.experiment_name="/repo/rlpolicy/auv_traj_mlp_history_5"' in command
    assert 'env.mlp_architecture="mlp_history_5"' in command
    assert not any(part.startswith("env.mlp_history_") for part in command)
    assert "agent.policy.actor_hidden_dims=[512,384,256,128]" in command
    assert "--agent" not in command
    seed_index = command.index("--seed")
    assert command[seed_index + 1] == "17"

    baseline_command = build_train_command(
        ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"), mlp_architecture="mlp_30d"), request
    )
    assert 'env.mlp_architecture="mlp_30d"' in baseline_command
    assert command != baseline_command


def test_train_command_forwards_explicit_domain_randomization_feature_subset() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"), mlp_architecture="mlp_history_5")
    request = TrainRequest(
        reward_profile="policy_0",
        domain_randomization_spec="/profiles/auv_pool_openfoam_hydrodynamics_dr_v1.json",
        domain_randomization_features=("actuators", "battery"),
    )

    command = build_train_command(spec, request)

    assert "env.domain_randomization_feature_override_enabled=true" in command
    assert 'env.domain_randomization.enabled_features=["actuators","battery"]' in command


def test_train_command_forwards_python_trajectory_curriculum() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"), mlp_architecture="mlp_history_5")
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
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"), mlp_architecture="mlp_history_5")
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


def test_train_notebook_is_the_single_human_recipe_selection_entry() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    training_source = (repository_root / "simulation/isaac/training.py").read_text(encoding="utf-8")
    train_notebook = (repository_root / "train.ipynb").read_text(encoding="utf-8")
    train_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(train_notebook)["cells"]
        if cell.get("cell_type") == "code"
    )
    worker_source = (repository_root / "simulation/isaac/trajectory/train.py").read_text(encoding="utf-8")
    eval_notebook = (repository_root / "eval.ipynb").read_text(
        encoding="utf-8"
    )
    eval_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(eval_notebook)["cells"]
        if cell.get("cell_type") == "code"
    )

    campaign = build_default_campaign(isaaclab_root="/opt/IsaacLab")
    assert campaign.experiment.mlp_architecture == "mlp_history_5"
    assert campaign.experiment.architecture.observation_dim == 135
    assert campaign.train.reward_profile == "policy_6"
    assert campaign.total_iterations == 500
    assert campaign.segment_iterations == 25
    assert campaign.use_competence_gate

    env_config = (repository_root / "simulation/isaac/config.py").read_text(encoding="utf-8")
    env_runtime = (repository_root / "simulation/isaac/env.py").read_text(encoding="utf-8")
    assert "auv_pool_openfoam_hydrodynamics_v1.json" in env_config
    assert "implicit physics fallbacks are disabled" in env_runtime
    assert "NOMINAL_POOL_DYNAMICS_PROFILE" not in env_runtime
    assert 'domain_randomization_spec = ""' in env_config
    assert "trajectory_curriculum = False" in env_config
    assert "trajectory_curriculum_stage_steps = []" in env_config
    assert "trajectory_amp_x_range = [0.0, 0.0]" in env_config
    assert "trajectory_train_types = [0]" in env_config
    assert "observation_base_dim = 30" in env_config
    assert "critic_privileged_fields = []" in env_config
    assert "attitude_error_quat" in env_config
    assert 'MLP_ARCHITECTURE = "mlp_history_5"' in eval_source
    assert "importlib.reload(experiment_tools)" in eval_notebook
    assert "auv_pool_openfoam_hydrodynamics_v1.json" in eval_notebook
    assert "auv_pool_openfoam_hydrodynamics_dr_v1.json" in eval_notebook
    assert "SAMPLE_DOMAIN_RANDOMIZATION = True" in eval_source
    assert "POLICY_RUN_DIR = None" in eval_source
    assert 'FINAL_CHECKPOINT = "latest"' in eval_source
    assert "selected_dir = Path(POLICY_RUN_DIR).expanduser().resolve()" in eval_source
    assert "EVAL_CHECKPOINTS = FINAL_CHECKPOINT" in eval_source
    assert "checkpoint=EVAL_CHECKPOINTS" in eval_source
    assert "SHOW_INLINE_PLOTS = True" in eval_source
    assert 'RLPOLICY_ROOT = REPO_ROOT / "simulation/isaac/rlpolicy"' in eval_source
    assert "COMPETENCE_SUPERVISOR_SCRIPT" in training_source
    assert "simulation/isaac/trajectory/competence_curriculum.py" in training_source
    assert "materialize_training_profiles" in train_source
    assert "water_current_max_by_stage" in train_source
    assert "added_mass_log_std_by_stage" in train_source
    assert "stop_campaign" in train_source
    assert "os.killpg" in training_source
    assert "runpy" not in worker_source
    assert "official_script" not in worker_source
    assert "curriculum_nominal" in (repository_root / "simulation/isaac/trajectory/experiment.py").read_text(
        encoding="utf-8"
    )
    assert "curriculum_robust" in (repository_root / "simulation/isaac/trajectory/experiment.py").read_text(
        encoding="utf-8"
    )
    curriculum = campaign.train.trajectory_curriculum
    assert curriculum is not None
    assert curriculum.stage_steps == (9_750, 22_500, 40_500)
    assert curriculum.stage_0_types == (8, 9, 10)
    assert curriculum.speed_levels_mps == (0.1, 0.2, 0.3, 0.4)

    agent_config = (repository_root / "simulation/isaac/ppo/config.py").read_text(encoding="utf-8")
    assert 'experiment_name = "auv_traj_mlp"' in agent_config
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in agent_config

    tools = (repository_root / "simulation/isaac/trajectory/experiment.py").read_text(encoding="utf-8")
    assert "mlp_architecture" in tools
    assert "evaluation_console.log" in tools

    eval_script = (repository_root / "simulation/isaac/trajectory/evaluate.py").read_text(
        encoding="utf-8"
    )
    assert "raw_policy_actions = policy(obs)" in eval_script
    assert "actions = torch.clamp(raw_policy_actions, -1.0, 1.0)" in eval_script
    assert 'f"raw_policy_action_{action_index}"' in eval_script


def test_only_trajectory_task_remains_registered() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    registration = (repository_root / "__init__.py").read_text(encoding="utf-8")
    environment_source = (repository_root / "simulation/isaac/env.py").read_text(encoding="utf-8")
    config_source = (repository_root / "simulation/isaac/config.py").read_text(encoding="utf-8")

    assert registration.count("gym.register(") == 1
    assert registration.count('id="Isaac-AUV-Traj-Direct-v1"') == 1
    assert environment_source.count("class AUVTrajEnv(") == 1
    assert config_source.count("class AUVTrajEnvCfg(") == 1


def test_architecture_run_namespaces_and_completed_run_selection(tmp_path: Path) -> None:
    spec = ExperimentSpec(isaaclab_root=tmp_path, rlpolicy_root=tmp_path / "rlpolicy")
    assert spec.experiment_name == "auv_traj_mlp_history_5"
    assert spec.architecture.observation_dim == 135
    assert spec.architecture.critic_privileged_dim == 77
    assert spec.architecture.critic_observation_dim == 212
    assert ExperimentSpec(isaaclab_root=tmp_path, rlpolicy_root=tmp_path / "rlpolicy", mlp_architecture="mlp_30d").experiment_name == "auv_traj_mlp"
    assert ExperimentSpec(isaaclab_root=tmp_path, rlpolicy_root=tmp_path / "rlpolicy", mlp_architecture="mlp_30d").architecture.critic_observation_dim == 107
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


def test_python_training_recipe_selects_openfoam_environment_profile_and_its_curriculum() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    profile_path = repository_root / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
    recipe_path = repository_root / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    campaign = build_default_campaign(isaaclab_root="/opt/IsaacLab")

    assert campaign.train.environment_profile == profile_path
    assert campaign.train.domain_randomization_spec == recipe_path
    payload = campaign_payload(campaign)
    assert payload["train"]["environment_profile"] == str(profile_path)
    assert payload["train"]["domain_randomization_spec"] == str(recipe_path)

    profile = load_environment_profile_json(profile_path)
    recipe = load_domain_randomization_spec_json(recipe_path)
    validate_domain_randomization_base_profile(recipe, profile)

    assert profile.hydrodynamics.water_current_periodic_enabled is True
    assert profile.hydrodynamics.high_order_residual_enabled is False
    assert profile.free_surface.sloshing_enabled is True
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert not ({"rigid_body", "thrusters", "battery", "tether", "observation"} & profile_payload.keys())
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
    profile = load_environment_profile_json(
        repository_root / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
    )
    recipe = load_domain_randomization_spec_json(
        repository_root / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )
    cfg = SimpleNamespace(domain_randomization=SimpleNamespace())

    resolve_isaac_composition(profile).apply(cfg)
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


def test_openfoam_actuator_recipe_reaches_runtime_cfg() -> None:

    repository_root = Path(__file__).resolve().parents[4]
    profile = load_environment_profile_json(
        repository_root / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
    )
    recipe = load_domain_randomization_spec_json(
        repository_root / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )
    cfg = SimpleNamespace(domain_randomization=SimpleNamespace())

    resolve_isaac_composition(profile).apply(cfg)
    assert cfg.dyn_time_constant == pytest.approx(T60_RUNTIME.thruster_time_constant_s)

    # Training and an EvalRequest with sample_domain_randomization=True both
    # use this exact application path; eval_mode only controls whether reset
    # sampling is enabled at runtime.
    apply_domain_randomization_spec(cfg, recipe, base_profile=profile)
    assert cfg.domain_randomization.use_custom_randomization is True
    assert "observations" not in cfg.domain_randomization.enabled_features
    assert not hasattr(cfg.domain_randomization, "observation_delay_steps_range")
    # No command-chain latency is invented before timing measurements exist.
    assert cfg.domain_randomization.thruster_command_delay_steps_range == [0, 0]


def test_eval_command_forwards_reproducible_randomized_case() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"))
    recipe_path = (
        Path(__file__).resolve().parents[4]
        / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )
    request = EvalRequest(
        reward_profile="policy_0",
        seed=23,
        environment_profile="/profiles/measured_pool.json",
        domain_randomization_spec=recipe_path,
        sample_domain_randomization=True,
    )

    command = build_eval_command(spec, request, "run-a", "model_100.pt", "helix")

    assert command[command.index("--seed") + 1] == "23"
    assert command[command.index("--environment_profile") + 1] == "/profiles/measured_pool.json"
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
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"), mlp_architecture="mlp_history_5")
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
        ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy")),
        sweep[1],
        "run-a",
        "model_24.pt",
        "random_smooth",
    )
    assert command[command.index("--eval_current") + 1 : command.index("--eval_current") + 4] == ["0.0", "0.1", "0.0"]
    assert "--trajectory_amp_x_range" in command
    assert "--align_initial_target" in command


def test_random_smooth_eval_requires_non_static_ranges_and_forwards_them() -> None:
    spec = ExperimentSpec(isaaclab_root=Path("/opt/IsaacLab"), rlpolicy_root=Path("/repo/rlpolicy"))
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
    spec = ExperimentSpec(isaaclab_root=tmp_path, rlpolicy_root=tmp_path / "rlpolicy")
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



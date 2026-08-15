"""Configuration contract for the AUV trajectory-tracking environment.

This module owns policy-facing spaces and tunable simulation parameters. It
does not perform simulation work; :mod:`env` composes these values at runtime.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from environment.profiles.features import DOMAIN_RANDOMIZATION_FEATURES
from robot.control.trajectory import LISSAJOUS
from robot.control.trajectory.observation_contract import ACTION_DIM, BASE_OBSERVATION_DIM
from simulation.isaac.trajectory.evaluation_cases import DEFAULT_EVALUATION_DURATION_S
from .robot_asset import AUV_CFG
from .visualization import AUVTrajEnvWindow


DEFAULT_ENVIRONMENT_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
)


@configclass
class AUVTrajEnvCfg(DirectRLEnvCfg):
    """Trajectory policy, asset, actuator, and pool-effect configuration."""

    ui_window_class_type = AUVTrajEnvWindow

    # Start focused on the first underwater environment, but keep the viewport
    # camera static afterwards so mouse orbit/pan/zoom remains under user
    # control. The vehicle starts at z=-8 m relative to its environment.
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.2, 2.2, -6.8),
        lookat=(0.0, 0.0, -8.0),
        origin_type="env",
        env_index=0,
    )

    # One policy action spans four 200 Hz physics steps, yielding a 50 Hz
    # policy/control rate. Rendering once per policy action avoids unnecessary
    # intermediate frames during headless training.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physx=PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
        ),
    )
    # Training entry points override this with the runner seed.  A deterministic
    # default keeps direct Gym/evaluation construction reproducible without
    # mutating the global RNG inside the environment implementation.
    seed = 0

    # robot
    robot_cfg: RigidObjectCfg = AUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    # Marker updates are useful for interactive diagnosis but add GPU/Fabric
    # work even in headless jobs. Evaluation can override this when needed.
    debug_vis = False

    # Deployable current-sample layout:
    # [position_error_b(3), target_linear_velocity_b(3), linear_velocity_error_b(3),
    #  attitude_error_quat(4), angular_velocity_b(3), target_angular_velocity_b(3),
    #  target_linear_acceleration_b(3), applied_action(8)]. Vehicle terms come from
    # simulator state, target terms from the reference generator,
    # and applied_action from the controller after its delay/rate limiter. No
    # simulator-only hydrodynamic or DR truth enters the Actor.
    #
    # ``mlp_architecture`` is selected by simulation.isaac.training. The current
    # 30-D sample is always present; when history is requested the environment
    # appends only earlier, deployable samples of the named fields.  The mixin
    # derives the final Gym spaces before DirectRLEnv is constructed.
    # Use IsaacLab's integer space specification across Hydra. Serializing a
    # Gym Box through Hydra loses its dtype and IsaacLab 2.3 reconstructs its
    # bounds as float64, producing false precision-loss warnings. Runtime code
    # below materializes the exact bounded float32 spaces after Hydra parsing.
    observation_space = BASE_OBSERVATION_DIM
    action_space = ACTION_DIM
    state_space = BASE_OBSERVATION_DIM
    # env
    decimation = 4
    cap_episode_length = True
    # Requested training periods are 20 s in stage 0 and 10 s thereafter.
    # curve_v2 may lengthen an effective period to meet kinematic limits, so
    # summaries record actual period and path length instead of assuming an
    # integer number of laps in this horizon.
    episode_length_s = 40.0
    episode_length_before_reset = None
    # Only ``mlp_architecture`` is user-selected. The two derived fields are
    # copied from its profile at environment construction and persisted in the
    # run config for auditability; do not override them independently.
    mlp_architecture = "mlp_30d"
    mlp_history_steps = 0
    mlp_history_fields = []
    critic_privileged_fields = []
    # Optional MLP-profile Critic selection. It never changes the Actor input.
    critic_privileged_fields_override = []
    use_boundaries = True
    max_auv_x = 7
    max_auv_y = 7
    max_auv_z = 7
    # Isaac/PhysX world coordinates are z-up.  The water surface is at z=-1 m
    # and a positive physical depth therefore has a negative world z value.
    starting_depth = -8
    eval_mode = False
    # Every runtime resolves this profile. There is no hidden hydrodynamic
    # coefficient fallback in the environment config.
    environment_profile = str(DEFAULT_ENVIRONMENT_PROFILE_PATH)
    environment_profile_name = None
    # Evaluation-only modifiers are applied by the environment after the
    # deterministic profile and DR recipe. Keep this serializable so result
    # logs can identify exactly which overlay took effect.
    evaluation_physics_overlay = {}
    # Canonicalized evaluation-only state populated after the deterministic
    # profile and DR recipe have been resolved.  Reset helpers consume these
    # values so an automatic DirectRLEnv reset cannot silently remove a
    # command-line evaluation override.
    evaluation_current_override = False
    evaluation_current_variation_std = 0.0
    evaluation_current_tau = 12.0
    evaluation_thruster_force_scale_override = False
    evaluation_thruster_force_scale = 1.0
    # Internal recipe hand-off populated by the Python training manager.
    # Keep training-distribution choices out of this runtime config.
    domain_randomization_spec = ""
    domain_randomization_spec_name = None
    # A training request may explicitly replace the recipe's enabled feature
    # groups. The separate boolean preserves a recipe-defined subset when no
    # per-run override was requested (including when the requested subset is
    # deliberately empty).
    domain_randomization_feature_override_enabled = False
    # Evaluation keeps deterministic initial conditions unless explicitly
    # asked to sample the selected training recipe.
    eval_domain_randomization = False
    # Evaluation may explicitly probe one DR level; ``-1`` keeps the normal
    # step-based curriculum.
    eval_disturbance_stage = -1
    domain_randomization_log_interval_steps = 250

    # By default every reset starts exactly at the trajectory's t=0 pose and
    # velocity. For nonzero speed, the target quaternion points body +X along
    # that velocity. Disable this only for experiments that deliberately need
    # randomized initial tracking error.
    trajectory_match_initial_state = True
    # Every reset initializes both trajectory and vehicle velocity at zero.
    # The reference then reaches its retimed speed smoothly over this interval;
    # the vehicle follows it through ordinary learned/controller forces.
    trajectory_startup_duration_s = 4.0
    # Fallback random-reset distribution used only when the option above is
    # disabled.
    trajectory_initial_position_radius = 0.35
    init_guidance_rate = 0.5
    init_vel_max = 0.2
    trajectory_eval_mode = False
    # Versioned definitions live in rewards/policy_N.py. Use ``custom`` to retain
    # direct scalar/Hydra coefficient overrides instead of applying a named policy.
    tracking_reward_profile = "policy_0"
    # AUV uses an x-forward body frame. For the trajectory task, command yaw
    # and pitch so the nose follows the full 3D target-velocity direction.
    trajectory_align_heading_with_velocity = True
    trajectory_heading_min_speed = 1.0e-3
    # Speed-controlled training commands. The Python training recipe selects the
    # three shape IDs; every selected environment independently samples one of
    # these four levels at reset.  For sine commands this is peak speed, while
    # for the spatial helix it is the speed along the curve.
    trajectory_speed_levels_mps = [0.1, 0.2, 0.3, 0.4]
    trajectory_lateral_sine_amplitude_m = 0.65
    trajectory_vertical_sine_amplitude_m = 0.50
    trajectory_spatial_helix_radius_x_m = 0.75
    trajectory_spatial_helix_radius_y_m = 0.65
    trajectory_spatial_helix_amplitude_z_m = 0.16
    # Neutral direct-construction fallback. The complete training curriculum is
    # selected in simulation.isaac.training and forwarded by Hydra.
    trajectory_amp_x_range = [0.0, 0.0]
    trajectory_amp_y_range = [0.0, 0.0]
    trajectory_amp_z_range = [0.0, 0.0]
    trajectory_period_range = [24.0, 24.0]
    trajectory_curriculum = False
    # Curriculum stages advance from the environment's global policy-step
    # counter during one direct training run.
    trajectory_curriculum_stage_steps = []
    trajectory_curriculum_stage_0_types = [0]
    trajectory_curriculum_stage_1_types = [0]
    trajectory_curriculum_stage_2_types = [0]
    trajectory_curriculum_stage_3_types = [0]
    trajectory_curriculum_amp_scales = [1.0]
    trajectory_curriculum_z_amp_scales = [1.0]
    trajectory_curriculum_period_min = [16.0]
    trajectory_curriculum_period_max = [28.0]
    # Deterministic eval commands.  trajectory_eval_type maps to:
    # 1=lissajous, 3=wavy loop, 4=spiral, 5=chirp, 6=racetrack,
    # 7=random_smooth, 8=lateral_sine, 9=vertical_sine,
    # 10=spatial_helix.
    # Training-stage type sets are supplied by simulation.isaac.training; eval
    # commands remain independent so held-out shapes can be selected here.
    trajectory_eval_type = LISSAJOUS
    trajectory_eval_amp_x = 0.75
    trajectory_eval_amp_y = 0.65
    trajectory_eval_amp_z = 0.16
    trajectory_eval_period = 12.0
    trajectory_eval_duration_s = DEFAULT_EVALUATION_DURATION_S
    trajectory_eval_radius_min = 0.3
    trajectory_eval_radius_max = 1.2
    trajectory_eval_chirp_rate = 1.6
    trajectory_eval_speed_mps = 0.2
    # The random-smooth generator perturbs a base ellipse only with this
    # low-amplitude second harmonic.  Keeping it <= 0.10 preserves a positive
    # tangential speed and rules out cusp-like sharp turns.
    trajectory_random_smooth_harmonic_ratio = 0.08
    # ``curve_v2`` re-times every command locally so the reference itself is
    # feasible for the simulated vehicle. These are deliberately provisional
    # simulator limits derived from the final random-smooth curriculum envelope;
    # replace them with pool-identified bounds before deployment.
    trajectory_generator_version = "curve_v2"
    trajectory_max_speed_mps = 0.60
    trajectory_max_acceleration_mps2 = 0.45
    trajectory_max_orientation_rate_radps = 0.80
    trajectory_max_jerk_mps3 = 0.36
    trajectory_retime_samples = 256
    trajectory_eval_align_initial_target = True
    trajectory_train_types = [0]

    # trajectory rewards
    rew_scale_terminated = 0.0
    rew_scale_pos = 3.2
    rew_scale_ang = 0.25
    rew_scale_ang_vel = 0.02
    rew_scale_track_vel = 0.6
    rew_scale_forward = 0.0
    rew_scale_motion_alignment = 0.8
    # Keep the fallback values consistent with policy_1. The rate term is a
    # per-policy-step command difference and suppresses high-frequency chatter.
    rew_scale_actions = 0.01
    rew_scale_action_rate = 0.05
    # ``requested`` preserves policies 0--4. policy_5 switches this to the
    # actuator command after delay and rate limiting.
    rew_action_source = "requested"
    rew_pos_sigma = 0.7
    rew_ang_sigma = 0.75
    rew_track_vel_sigma = 0.35
    rew_ang_vel_sigma = 0.5
    rew_forward_min_speed = 1.0e-3
    # The calibrated high-order residual is applied as a separate external
    # body wrench through the PhysX wrench composer.  It can be modulated for
    # explicit dynamics-change experiments without mutating nominal factors.
    physx_high_order_wrench_enabled = True
    physx_high_order_wrench_base_scale = 1.0
    physx_high_order_wrench_modulation_amplitude = 0.0
    physx_high_order_wrench_modulation_frequency_hz = 0.0
    physx_high_order_wrench_modulation_phase_rad = 0.0

    # Neutral runtime defaults for trajectory tracking. Training distributions are
    # selected through a versioned recipe in simulation.isaac.training.
    class domain_randomization:
        # Recipes and per-run overrides select feature groups explicitly.
        enabled_features = list(DOMAIN_RANDOMIZATION_FEATURES)

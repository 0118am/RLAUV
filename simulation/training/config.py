"""Configuration contract for the AUV trajectory-tracking environment.

This module owns policy-facing spaces and tunable training-task parameters. It
does not perform simulation work; :mod:`simulation.assembly` composes
these values at runtime.
"""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from robot.control.trajectory import LISSAJOUS, TRAJECTORY_GENERATOR_VERSION
from robot.control.trajectory.observation_contract import ACTION_DIM, BASE_OBSERVATION_DIM
from simulation.training.rewards import PRECISION_V9 as DEFAULT_REWARD_POLICY
from simulation.training.evaluation.config import DEFAULT_EVALUATION_DURATION_S
from simulation.assets import T60_ASSET_CFG
from .visualization import AUVTrajEnvWindow


DEFAULT_ENVIRONMENT_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "environment/hydrodynamics/coefficients/auv_open_water_openfoam_full_hydrodynamics_v2.json"
)


@configclass
class AUVTrajEnvCfg(DirectRLEnvCfg):
    """Trajectory policy, asset, actuator, and pool-effect configuration."""

    ui_window_class_type = AUVTrajEnvWindow

    # Start focused on the first pool, but keep the viewport camera static
    # afterwards so mouse orbit/pan/zoom remains under user control. Pool-local
    # FLU coordinates start at the lower corner and the center is positive.
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.7, 3.95, 0.95),
        # Resolved from the selected pool profile before DirectRLEnv starts.
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
    )

    # One policy action spans four 100 Hz physics steps. Control and fused-state
    # observation updates therefore both run at 25 Hz. Rendering once per
    # policy action avoids unnecessary intermediate frames during headless training.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
    robot_cfg: RigidObjectCfg = T60_ASSET_CFG

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=7.0,
        replicate_physics=True,
    )
    # Marker updates are useful for interactive diagnosis but add GPU/Fabric
    # work even in headless jobs. Evaluation can override this when needed.
    debug_vis = False

    # Deployable current-sample layout:
    # [position_error_b(3), target_linear_velocity_b(3), linear_velocity_error_b(3),
    #  attitude_error_quat(4), angular_velocity_b(3), target_angular_velocity_b(3),
    #  target_linear_acceleration_b(3), previous_motor_command(8)]. Vehicle terms come from
    # the exact 50 ms delayed fused state, target terms from the reference generator,
    # and previous_motor_command from the bounded Actor output emitted during
    # the preceding policy interval; it is not the command currently emerging
    # from the fixed communication-delay pipeline. No
    # simulator-only hydrodynamic or DR truth enters the Actor.
    #
    # ``mlp_architecture`` is selected by simulation.training. The current
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
    # curve_v5 may lengthen an effective period to meet kinematic limits, so
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
    # The 5 x 3 x 1 m command envelope has explicit vehicle-size margins in
    # the environment profile. Leaving that envelope is a true safety failure.
    use_boundaries = True
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

    # Every reset initializes both trajectory and vehicle velocity at zero.
    # The reference then reaches its retimed speed smoothly over this interval;
    # the vehicle follows it through ordinary learned-policy forces.
    trajectory_startup_duration_s = 4.0
    # Training always starts around, rather than exactly on, the t=0 target.
    # The four independent error limits prevent a zero-error reset shortcut.
    trajectory_initial_position_radius = 0.20
    trajectory_initial_attitude_error_max_rad = math.pi / 18.0
    trajectory_initial_linear_velocity_error_max_mps = 0.10
    trajectory_initial_angular_velocity_error_max_radps = 0.20
    trajectory_eval_mode = False
    # The versioned precision reward is the only supported reward contract.
    tracking_reward_profile = DEFAULT_REWARD_POLICY.name
    # Keep every target level. Smooth paths align yaw with horizontal velocity;
    # reciprocating line tasks retain a fixed yaw because their tangent reverses
    # discontinuously at rest. Vertical translation is commanded through heave.
    trajectory_align_yaw_with_horizontal_velocity = True
    trajectory_heading_min_horizontal_speed = 1.0e-3
    # Speed-controlled training commands are sampled from explicit cumulative
    # command levels. Axis-sine speed/amplitude pairs train acceleration and
    # reversal; traveling-sine speed/wave-count/amplitude pairs train curvature.
    # The training recipe or an explicit random-smooth evaluation supplies
    # these ranges; direct construction has no hidden command distribution.
    trajectory_amp_x_range = None
    trajectory_amp_y_range = None
    trajectory_amp_z_range = None
    trajectory_period_range = None
    trajectory_curriculum = False
    # Curriculum stages advance from the environment's global policy-step
    # counter during one direct training run.
    trajectory_curriculum_stage_start_steps = []
    trajectory_curriculum_stage_types = []
    trajectory_curriculum_stage_axes = []
    trajectory_curriculum_stage_wave_counts = []
    trajectory_curriculum_stage_speeds_mps = []
    trajectory_curriculum_stage_amplitude_scales = []
    # Deterministic eval commands.  trajectory_eval_type maps to:
    # 1=lissajous, 3=wavy_loop, 4=breathing_loop, 5=chirp, 6=racetrack,
    # 7=random_smooth, 8=lateral_wave, 9=vertical_wave,
    # 10=spatial_helix, 11=reverse_spatial_helix.
    # Training-stage type sets are supplied by simulation.training; eval
    # commands remain independent so held-out shapes can be selected here.
    trajectory_eval_type = LISSAJOUS
    trajectory_eval_axis = 0
    trajectory_eval_wave_count = 1
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
    # ``curve_v5`` re-times every command locally so the reference itself is
    # feasible for the simulated vehicle. These are deliberately provisional
    # simulator limits derived from the final random-smooth curriculum envelope;
    # replace them with pool-identified bounds before deployment.
    trajectory_generator_version = TRAJECTORY_GENERATOR_VERSION
    trajectory_max_speed_mps = None
    trajectory_max_acceleration_mps2 = None
    trajectory_max_yaw_rate_radps = None
    trajectory_max_jerk_mps3 = None
    trajectory_retime_samples = None
    trajectory_eval_align_initial_target = True

    # trajectory rewards
    # Hydra requires concrete fields before the named reward is applied. Source
    # them from the reward object so config.py never carries a shadow set.
    rew_scale_terminated = DEFAULT_REWARD_POLICY.termination_penalty
    rew_scale_pos = DEFAULT_REWARD_POLICY.position_weight
    rew_scale_attitude_recovery = DEFAULT_REWARD_POLICY.attitude_recovery_weight
    rew_scale_attitude_precision = DEFAULT_REWARD_POLICY.attitude_precision_weight
    rew_scale_track_vel = DEFAULT_REWARD_POLICY.velocity_weight
    rew_scale_angular_velocity_broad = (
        DEFAULT_REWARD_POLICY.angular_velocity_broad_weight
    )
    rew_scale_angular_velocity_precision = (
        DEFAULT_REWARD_POLICY.angular_velocity_precision_weight
    )
    rew_scale_actions = DEFAULT_REWARD_POLICY.action_weight
    rew_scale_action_rate = DEFAULT_REWARD_POLICY.action_rate_weight
    rew_action_rate_scale_per_s = DEFAULT_REWARD_POLICY.action_rate_scale_per_s
    rew_pos_sigma = DEFAULT_REWARD_POLICY.position_sigma
    rew_attitude_recovery_transition = (
        DEFAULT_REWARD_POLICY.attitude_recovery_transition
    )
    rew_attitude_recovery_zero = DEFAULT_REWARD_POLICY.attitude_recovery_zero
    rew_attitude_precision_sigma = DEFAULT_REWARD_POLICY.attitude_precision_sigma
    rew_track_vel_sigma = DEFAULT_REWARD_POLICY.velocity_sigma
    rew_angular_velocity_broad_sigma = (
        DEFAULT_REWARD_POLICY.angular_velocity_broad_sigma
    )
    rew_angular_velocity_precision_sigma = (
        DEFAULT_REWARD_POLICY.angular_velocity_precision_sigma
    )
    # Neutral runtime defaults for trajectory tracking. Training distributions are
    # selected through a versioned recipe in simulation.training.
    class domain_randomization:
        # Recipes and per-run overrides select feature groups explicitly.
        enabled_features = []

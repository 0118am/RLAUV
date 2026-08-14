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
from robot.dynamics.parameters import AUV

from .robot_asset import AUV_CFG
from .visualization import AUVTrajEnvWindow


DEFAULT_POOL_DYNAMICS_PROFILE_PATH = (
    Path(__file__).resolve().parents[4]
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
    # ``mlp_architecture`` is selected by trajectory_train.ipynb.  The current
    # 30-D sample is always present; when history is requested the environment
    # appends only earlier, deployable samples of the named fields.  The mixin
    # derives the final Gym spaces before DirectRLEnv is constructed.
    # Use IsaacLab's integer space specification across Hydra. Serializing a
    # Gym Box through Hydra loses its dtype and IsaacLab 2.3 reconstructs its
    # bounds as float64, producing false precision-loss warnings. Runtime code
    # below materializes the exact bounded float32 spaces after Hydra parsing.
    observation_space = 30
    action_space = 8
    state_space = 30
    # env
    decimation = 4
    cap_episode_length = True
    # Requested training periods are 20 s in stage 0 and 10 s thereafter.
    # curve_v2 may lengthen an effective period to meet kinematic limits, so
    # summaries record actual period and path length instead of assuming an
    # integer number of laps in this horizon.
    episode_length_s = 40.0
    episode_length_before_reset = None
    observation_base_dim = 30
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
    pool_dynamics_profile = str(DEFAULT_POOL_DYNAMICS_PROFILE_PATH)
    pool_dynamics_profile_name = None
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
    # Internal recipe hand-off populated by simulation/isaac/notebooks/train.ipynb.
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
    # ``-1`` preserves the normal step-based DR curriculum.  Checkpoint-based
    # competence evaluation sets this explicitly so a fresh evaluator process
    # can probe a chosen uncertainty level instead of always starting at zero.
    eval_disturbance_stage = -1
    # A checkpoint-gated campaign runs each segment in a fresh environment
    # process.  The supervisor supplies the number of policy steps completed
    # by earlier segments so the disturbance curriculum does not restart at
    # stage zero on every resume.  Ordinary one-shot training keeps ``0``.
    disturbance_curriculum_global_step_offset = 0
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
    # Speed-controlled training commands.  The training notebook selects the
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
    # selected in simulation/isaac/notebooks/train.ipynb and forwarded by Hydra.
    trajectory_amp_x_range = [0.0, 0.0]
    trajectory_amp_y_range = [0.0, 0.0]
    trajectory_amp_z_range = [0.0, 0.0]
    trajectory_period_range = [24.0, 24.0]
    trajectory_curriculum = False
    # A competence-gated campaign runs short, resumed training segments.  It
    # selects this stage from held-out evaluation results rather than from the
    # fresh process's local ``common_step_counter``.  ``-1`` keeps the original
    # step-based behavior for ordinary one-shot training.
    curriculum_gate_stage = -1
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
    # 1=lissajous, 3=legacy helix, 4=spiral, 5=chirp, 6=racetrack,
    # 7=random_smooth, 8=lateral_sine, 9=vertical_sine,
    # 10=spatial_helix.
    # Training-stage type sets are supplied by trajectory_train.ipynb; eval
    # commands remain independent so held-out shapes can be selected here.
    trajectory_eval_type = 1
    trajectory_eval_amp_x = 0.75
    trajectory_eval_amp_y = 0.65
    trajectory_eval_amp_z = 0.16
    trajectory_eval_period = 12.0
    trajectory_eval_duration_s = 32.0
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

    # Fixed physical scales keep training and deployed observations identical.
    observation_position_scale_m = 2.0
    observation_linear_velocity_scale_mps = 1.0
    observation_angular_velocity_scale_radps = 1.0
    observation_linear_acceleration_scale_mps2 = 0.5

    # trajectory rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
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
    observation_noise_std = 0.0
    observation_bias_range = 0.0
    observation_delay_steps = 0
    observation_update_period_steps = 1
    observation_dropout_probability = 0.0
    observation_lowpass_alpha = 1.0
    observation_bias_drift_std = 0.0
    pool_boundary_effects_enabled = False
    pool_bounds = [-7.0, 7.0, -7.0, 7.0, -15.0, -1.0]
    pool_boundary_effect_distance = 0.75
    pool_boundary_damping_scale = 1.5
    pool_boundary_added_mass_scale = 1.2
    pool_boundary_thrust_scale = 0.85
    free_surface_effects_enabled = False
    free_surface_z = -1.0
    free_surface_effect_distance = 0.5
    free_surface_heave_damping_scale = 1.4
    free_surface_roll_pitch_damping_scale = 1.2
    free_surface_added_mass_scale = 1.15
    free_surface_buoyancy_scale = 0.95
    free_surface_thrust_scale = 0.90
    free_surface_sloshing_enabled = False
    free_surface_sloshing_pool_bounds = [-7.0, 7.0, -7.0, 7.0]
    free_surface_sloshing_water_depth = 14.0
    free_surface_sloshing_mode_numbers = [[1, 0]]
    free_surface_sloshing_amplitudes_m = [0.0]
    free_surface_sloshing_phases_rad = [0.0]
    free_surface_sloshing_depth_axis_sign = -1.0
    tether_enabled = False
    tether_anchor_pos_w = [0.0, 0.0, 8.0]
    tether_attach_offset_b = [-0.2, 0.0, 0.0]
    tether_slack_length = 2.0
    tether_stiffness = 20.0
    tether_damping = 5.0
    tether_drag_coeff = 0.0
    tether_winch_enabled = False
    tether_winch_target_length = 2.0
    tether_winch_reel_speed = 0.0
    tether_winch_min_length = 0.0
    tether_winch_max_length = 20.0
    tether_num_segments = 1
    tether_segment_diameter = 0.004
    tether_segment_density = 1100.0
    tether_segment_buoyancy_density = AUV.water_density_kg_m3
    thruster_inflow_loss_enabled = False
    thruster_inflow_loss_coefficient = 0.25
    thruster_inflow_reference_speed = 1.0
    thruster_inflow_min_scale = 0.5
    thruster_wake_interaction_enabled = False
    thruster_wake_loss_coefficient = 0.10
    thruster_wake_length = 0.6
    thruster_wake_radius = 0.08
    thruster_wake_expansion_rate = 0.15
    thruster_wake_min_scale = 0.7
    thruster_reaction_torque_coeff = 0.0
    thruster_spin_directions = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]

    # dynamics
    center_of_mass_offset = list(AUV.center_of_mass_offset_m)
    inertia_diag = [list(row) for row in AUV.inertia_tensor_body_kg_m2]
    com_to_cob_offset = list(AUV.center_of_buoyancy_from_com_m)
    water_rho = AUV.water_density_kg_m3 # kg/m^3
    # The supplied calibration is steady-state only.  Nominally apply it
    # directly instead of inventing an unmeasured motor lag or rate limit.
    dyn_time_constant = 0.0
    thruster_command_delay_steps = 0
    thruster_max_command_rate = 0.0
    thruster_command_resolution = 0.0
    thruster_command_dropout_probability = 0.0
    battery_voltage_nominal = 16.0
    battery_voltage = 16.0
    battery_min_voltage = 12.0
    battery_voltage_drop_per_s = 0.0
    battery_voltage_thrust_exponent = 2.0
    mass = AUV.mass_kg # kg, AUV validation vehicle
    volume = AUV.displaced_volume_m3 # m^3, independently identified displacement

    # Fossen-style hydrodynamic parameters.  Damping is applied to relative
    # velocity nu_r = nu - nu_current, not absolute vehicle velocity.
    water_current_w = [0.0, 0.0, 0.0]
    # Deterministic pump/return-cycle approximation loaded from the selected
    # pool profile: A * sin(2*pi*t/T + phase), independently on world x/y/z.
    water_current_periodic_enabled = False
    water_current_periodic_amplitude_w = [0.0, 0.0, 0.0]
    water_current_periodic_period_s = [20.0, 20.0, 20.0]
    water_current_periodic_phase_rad = [0.0, 0.0, 0.0]
    water_current_field_enabled = False
    water_current_field_bounds = [-7.0, 7.0, -7.0, 7.0, -15.0, -1.0]
    water_current_field_shape = [1, 1, 1]
    water_current_field_values = []
    # Neutral placeholders are overwritten unconditionally from
    # ``pool_dynamics_profile`` before any runtime tensors are allocated.
    linear_damping = [0.0] * 6
    quadratic_damping = [0.0] * 6
    speed_dependent_damping_enabled = False
    damping_speed_points = [0.0, 1.0]
    linear_damping_speed_scales = []
    quadratic_damping_speed_scales = []
    added_mass_diag = [0.0] * 6
    added_mass_inertia_scale = 1.0
    added_mass_accel_filter_alpha = 0.35
    # Disabled until residual-wrench identification supplies physically
    # constrained factor matrices and explicitly enables the extension.
    high_order_residual_enabled = False
    high_order_residual_added_mass_factor = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    high_order_residual_linear_damping_factor = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    high_order_residual_quadratic_damping_factor = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    high_order_residual_cubic_damping_factor = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # The calibrated high-order residual is applied as a separate external
    # body wrench through the PhysX wrench composer.  It can be modulated for
    # explicit dynamics-change experiments without mutating nominal factors.
    physx_high_order_wrench_enabled = True
    physx_high_order_wrench_base_scale = 1.0
    physx_high_order_wrench_modulation_amplitude = 0.0
    physx_high_order_wrench_modulation_frequency_hz = 0.0
    physx_high_order_wrench_modulation_phase_rad = 0.0

    # Neutral runtime defaults for trajectory tracking. Training distributions are
    # selected through a versioned recipe in simulation/isaac/notebooks/train.ipynb.
    class domain_randomization:
        use_custom_randomization = False
        # All legacy recipes predate feature selection, so their absence is
        # equivalent to this complete list. Individual training runs can set
        # a subset via ``TrainRequest.domain_randomization_features``.
        enabled_features = list(DOMAIN_RANDOMIZATION_FEATURES)
        # Maximum radius for the isotropic Gaussian COM-to-COB offset.
        com_to_cob_offset_radius = 0.0
        volume_range = [AUV.displaced_volume_m3, AUV.displaced_volume_m3]
        mass_range = [AUV.mass_kg, AUV.mass_kg]
        payload_samples = []
        thruster_command_delay_steps_range = [0, 0]
        # Preserve the nominal command bandwidth when no versioned recipe is
        # applied. The active OpenFOAM-pool recipe fixes the same value.
        thruster_max_command_rate_range = [0.0, 0.0]
        thruster_command_resolution_range = [0.0, 0.0]
        thruster_command_dropout_probability_range = [0.0, 0.0]
        thruster_wake_loss_coefficient_scale_range = [1.0, 1.0]
        thruster_reaction_torque_coeff_scale_range = [1.0, 1.0]
        damping_speed_linear_scale_range = [1.0, 1.0]
        damping_speed_quadratic_scale_range = [1.0, 1.0]
        battery_voltage_range = [16.0, 16.0]
        battery_voltage_drop_per_s_range = [0.0, 0.0]
        observation_noise_std_range = [0.0, 0.0]
        observation_bias_range = [0.0, 0.0]
        observation_delay_steps_range = [0, 0]
        observation_update_period_steps_range = [1, 1]
        observation_dropout_probability_range = [0.0, 0.0]
        observation_lowpass_alpha_range = [1.0, 1.0]
        observation_bias_drift_std_range = [0.0, 0.0]
        disturbance_curriculum = False
        disturbance_curriculum_stage_steps = []
        water_current_max_by_stage = [0.0]
        water_current_vertical_max_by_stage = [0.0]
        water_current_smooth = False
        water_current_tau_range = [12.0, 12.0]
        water_current_variation_std_by_stage = [0.0]
        damping_scale_by_stage = [0.0]
        added_mass_log_std_by_stage = [0.0]
        thruster_scale_by_stage = [0.0]
        thruster_tau_scale_by_stage = [0.0]
        additional_hydrodynamics_scale_by_stage = [0.0]

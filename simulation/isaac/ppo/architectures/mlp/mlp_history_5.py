"""Feed-forward actor with five 50-Hz samples of actuator-relevant history."""

from ..base import MlpArchitecture, TRAJECTORY_CRITIC_PRIVILEGED_FIELDS


ARCHITECTURE = MlpArchitecture(
    name="mlp_history_5",
    # Five previous samples provide a 0.10 s causal window for the 0.05 s
    # thruster response and the 4.0 normalized-command/s rate limiter.
    history_steps=5,
    # Cache the vehicle/actuator response that carries hydrodynamic memory.
    # Do not repeat reference velocity/acceleration: the onboard trajectory
    # generator supplies their current values exactly at every policy step.
    history_fields=(
        "position_error_b",
        "linear_velocity_error_b",
        "attitude_error_quat",
        "angular_velocity_b",
        "applied_action",
    ),
    # All profiles use the same complete simulator-state baseline. It is
    # consumed only by V(o, z_priv) during PPO training, never by the Actor.
    critic_privileged_fields=TRAJECTORY_CRITIC_PRIVILEGED_FIELDS,
    # 30 + 5 * (3 + 3 + 4 + 3 + 8) = 135 Actor inputs; the Critic appends
    # the 77-D privileged state declared in base.py.
    actor_hidden_dims=(512, 384, 256, 128),
    critic_hidden_dims=(512, 384, 256, 128),
    experiment_name="auv_traj_mlp_history_5",
)

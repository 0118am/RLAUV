"""Baseline feed-forward actor with the current deployable observation only."""

from ..base import MlpArchitecture, TRAJECTORY_CRITIC_PRIVILEGED_FIELDS


ARCHITECTURE = MlpArchitecture(
    name="mlp_30d",
    history_steps=0,
    history_fields=(),
    critic_privileged_fields=TRAJECTORY_CRITIC_PRIVILEGED_FIELDS,
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    # Retain the established namespace so legacy 30-D MLP checkpoints remain
    # discoverable, while new history profiles receive their own namespace.
    experiment_name="auv_traj_mlp",
)

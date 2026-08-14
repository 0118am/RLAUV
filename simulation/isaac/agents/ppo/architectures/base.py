"""Pure-data contracts for deployable feed-forward trajectory actors."""

from __future__ import annotations

from dataclasses import dataclass


BASE_OBSERVATION_DIM = 30


# These are simulator-only terms used by the PPO value baseline.  They are
# deliberately absent from Actor history fields: a vehicle cannot measure the
# sampled hydrodynamic/actuator truth or the exact pool-effect state on board.
TRAJECTORY_CRITIC_PRIVILEGED_FIELDS = (
    "true_root_state",
    "water_current_b",
    "filtered_relative_acceleration_b",
    "effective_linear_damping_ratio",
    "effective_quadratic_damping_ratio",
    "effective_added_mass_ratio",
    "effective_buoyant_volume_ratio",
    "mass_ratio",
    "principal_inertia_ratio",
    "center_of_mass_offset_b",
    "com_to_cob_offset_b",
    "realized_thruster_force",
    "thruster_force_scale",
    "thruster_parameters",
    "battery_state",
    "tether_slack_ratio",
)

CRITIC_PRIVILEGED_FIELD_DIMENSIONS = {
    # local position (3), true world attitude (4), true body linear/angular
    # velocity (3 + 3).
    "true_root_state": 13,
    "water_current_b": 3,
    "filtered_relative_acceleration_b": 6,
    "effective_linear_damping_ratio": 6,
    "effective_quadratic_damping_ratio": 6,
    "effective_added_mass_ratio": 6,
    "effective_buoyant_volume_ratio": 1,
    "mass_ratio": 1,
    "principal_inertia_ratio": 3,
    "center_of_mass_offset_b": 3,
    "com_to_cob_offset_b": 3,
    # The final force after battery, pool/free-surface, inflow, and wake
    # effects is more informative than exposing an implementation-specific
    # sequence of intermediate thrust scales.
    "realized_thruster_force": 8,
    "thruster_force_scale": 8,
    # tau, delay, command-rate, resolution, dropout, wake loss, reaction
    # torque coefficient.
    "thruster_parameters": 7,
    "battery_state": 2,
    "tether_slack_ratio": 1,
    "high_order_residual_wrench_b": 6,
    "physx_hydrodynamics_scale": 1,
}


@dataclass(frozen=True)
class MlpArchitecture:
    """One complete MLP input and layer-width recipe.

    ``history_steps`` counts only past policy samples. The current 30-D
    observation is always first; selected prior observations follow newest to
    oldest. Keeping this contract data-only makes it usable by train, eval,
    ONNX export, and the real-robot input adapter.
    """

    name: str
    history_steps: int
    history_fields: tuple[str, ...]
    critic_privileged_fields: tuple[str, ...]
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    experiment_name: str

    @property
    def history_feature_dim(self) -> int:
        field_dimensions = {
            "position_error_b": 3,
            "linear_velocity_error_b": 3,
            "attitude_error_quat": 4,
            "angular_velocity_b": 3,
            "applied_action": 8,
        }
        return sum(field_dimensions[name] for name in self.history_fields)

    @property
    def observation_dim(self) -> int:
        return BASE_OBSERVATION_DIM + self.history_steps * self.history_feature_dim

    @property
    def critic_privileged_dim(self) -> int:
        return sum(CRITIC_PRIVILEGED_FIELD_DIMENSIONS[name] for name in self.critic_privileged_fields)

    @property
    def critic_observation_dim(self) -> int:
        return self.observation_dim + self.critic_privileged_dim

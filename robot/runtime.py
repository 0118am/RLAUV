"""Robot-owned runtime configuration used by simulator adapters.

Geometry, rigid-body properties, and measured force curves remain in the
specialized ``robot`` modules.  This profile owns the remaining vehicle-side
actuator and fused-state sensor settings so simulators do not invent shadow
defaults for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot.dynamics.parameters import AUV, AUVModel


@dataclass(frozen=True)
class RobotRuntimeProfile:
    """Vehicle-side runtime settings shared by physics backends."""

    model: AUVModel = AUV

    # User-confirmed T60 actuator and fused-state sensor measurements.
    thruster_command_delay_s: float = 0.05
    thruster_command_delay_source: str = (
        "User-confirmed fixed command communication delay: 50 ms."
    )
    thruster_time_constant_s: float = 0.04
    thruster_time_constant_source: str = (
        "Log-identified T60 first-order thrust time constant after removing the fixed "
        "50 ms command delay: 40 ms."
    )
    pose_sensor_delay_s: float = 0.05
    pose_sensor_source: str = (
        "User-confirmed reliable fused-state sensor with a deterministic 50 ms delay "
        "and no injected measurement noise."
    )
    thruster_inflow_loss_enabled: bool = False
    thruster_inflow_loss_coefficient: float = 0.25
    thruster_inflow_reference_speed: float = 1.0
    thruster_inflow_min_scale: float = 0.5
    thruster_wake_interaction_enabled: bool = False
    thruster_wake_loss_coefficient: float = 0.10
    thruster_wake_length: float = 0.6
    thruster_wake_radius: float = 0.08
    thruster_wake_expansion_rate: float = 0.15
    thruster_wake_min_scale: float = 0.7

    def validate(self) -> None:
        nonnegative = {
            "thruster_command_delay_s": self.thruster_command_delay_s,
            "thruster_time_constant_s": self.thruster_time_constant_s,
            "pose_sensor_delay_s": self.pose_sensor_delay_s,
            "thruster_inflow_loss_coefficient": self.thruster_inflow_loss_coefficient,
            "thruster_wake_loss_coefficient": self.thruster_wake_loss_coefficient,
            "thruster_wake_expansion_rate": self.thruster_wake_expansion_rate,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name, value in {
            "thruster_inflow_reference_speed": self.thruster_inflow_reference_speed,
            "thruster_wake_length": self.thruster_wake_length,
            "thruster_wake_radius": self.thruster_wake_radius,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name, value in {
            "thruster_inflow_min_scale": self.thruster_inflow_min_scale,
            "thruster_wake_min_scale": self.thruster_wake_min_scale,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

    def pose_sensor_delay_steps_for_dt(self, physics_dt_s: float) -> int:
        """Convert the fused-state delay to the 100 Hz truth-history grid."""

        if float(physics_dt_s) <= 0.0:
            raise ValueError("physics_dt_s must be positive.")
        return int(round(self.pose_sensor_delay_s / float(physics_dt_s)))

    def thruster_command_delay_steps_for_dt(self, physics_dt_s: float) -> int:
        """Convert the fixed communication delay to physics steps."""

        if float(physics_dt_s) <= 0.0:
            raise ValueError("physics_dt_s must be positive.")
        return int(round(self.thruster_command_delay_s / float(physics_dt_s)))

    def to_runtime_cfg_updates(self, physics_dt_s: float = 1.0 / 100.0) -> dict[str, Any]:
        """Return adapter fields without creating a second physical source."""

        self.validate()
        sensor_delay_steps = self.pose_sensor_delay_steps_for_dt(physics_dt_s)
        command_delay_steps = self.thruster_command_delay_steps_for_dt(physics_dt_s)
        return {
            "mass": self.model.mass_kg,
            "volume": self.model.displaced_volume_m3,
            "inertia_diag": [list(row) for row in self.model.inertia_tensor_body_kg_m2],
            "center_of_mass_offset": list(self.model.center_of_mass_offset_m),
            "com_to_cob_offset": list(self.model.center_of_buoyancy_from_com_m),
            "body_bounds_size_m": list(self.model.visual_bounds_size_m),
            "thruster_installation_source_step_sha256": (
                self.model.thruster_installation_source_step_sha256
            ),
            "thruster_installation_frame": self.model.thruster_installation_frame,
            "thruster_reaction_torque_model": "absent_no_bench_torque_measurement",
            "thruster_command_delay_s": self.thruster_command_delay_s,
            "thruster_command_delay_steps": command_delay_steps,
            "thruster_command_delay_source": self.thruster_command_delay_source,
            "dyn_time_constant": self.thruster_time_constant_s,
            "thruster_time_constant_source": self.thruster_time_constant_source,
            "pose_sensor_delay_s": self.pose_sensor_delay_s,
            "pose_sensor_delay_steps": sensor_delay_steps,
            "pose_sensor_source": self.pose_sensor_source,
            "thruster_inflow_loss_enabled": self.thruster_inflow_loss_enabled,
            "thruster_inflow_loss_coefficient": self.thruster_inflow_loss_coefficient,
            "thruster_inflow_reference_speed": self.thruster_inflow_reference_speed,
            "thruster_inflow_min_scale": self.thruster_inflow_min_scale,
            "thruster_wake_interaction_enabled": self.thruster_wake_interaction_enabled,
            "thruster_wake_loss_coefficient": self.thruster_wake_loss_coefficient,
            "thruster_wake_length": self.thruster_wake_length,
            "thruster_wake_radius": self.thruster_wake_radius,
            "thruster_wake_expansion_rate": self.thruster_wake_expansion_rate,
            "thruster_wake_min_scale": self.thruster_wake_min_scale,
        }


T60_RUNTIME = RobotRuntimeProfile()

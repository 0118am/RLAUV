"""Robot-owned runtime configuration used by simulator adapters.

Geometry, rigid-body properties, and measured force curves remain in the
specialized ``robot`` modules.  This profile owns the remaining vehicle-side
actuator, fused-state sensor, and tether settings so simulators do not invent
shadow defaults for them.
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
    thruster_time_constant_s: float = 0.08
    thruster_time_constant_source: str = (
        "User-confirmed T60 thrust build-up time constant: 80 ms."
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
    tether_enabled: bool = False
    tether_anchor_pos_w: tuple[float, float, float] = (0.0, 0.0, 8.0)
    tether_attach_offset_b: tuple[float, float, float] = (-0.2, 0.0, 0.0)
    tether_slack_length: float = 2.0
    tether_stiffness: float = 20.0
    tether_damping: float = 5.0
    tether_drag_coeff: float = 0.0
    tether_winch_enabled: bool = False
    tether_winch_target_length: float = 2.0
    tether_winch_reel_speed: float = 0.0
    tether_winch_min_length: float = 0.0
    tether_winch_max_length: float = 20.0
    tether_num_segments: int = 1
    tether_segment_diameter: float = 0.004
    tether_segment_density: float = 1100.0

    def validate(self) -> None:
        nonnegative = {
            "thruster_time_constant_s": self.thruster_time_constant_s,
            "pose_sensor_delay_s": self.pose_sensor_delay_s,
            "thruster_inflow_loss_coefficient": self.thruster_inflow_loss_coefficient,
            "thruster_wake_loss_coefficient": self.thruster_wake_loss_coefficient,
            "thruster_wake_expansion_rate": self.thruster_wake_expansion_rate,
            "tether_slack_length": self.tether_slack_length,
            "tether_stiffness": self.tether_stiffness,
            "tether_damping": self.tether_damping,
            "tether_drag_coeff": self.tether_drag_coeff,
            "tether_winch_target_length": self.tether_winch_target_length,
            "tether_winch_reel_speed": self.tether_winch_reel_speed,
            "tether_winch_min_length": self.tether_winch_min_length,
            "tether_winch_max_length": self.tether_winch_max_length,
            "tether_segment_density": self.tether_segment_density,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name, value in {
            "thruster_inflow_reference_speed": self.thruster_inflow_reference_speed,
            "thruster_wake_length": self.thruster_wake_length,
            "thruster_wake_radius": self.thruster_wake_radius,
            "tether_segment_diameter": self.tether_segment_diameter,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name, value in {
            "thruster_inflow_min_scale": self.thruster_inflow_min_scale,
            "thruster_wake_min_scale": self.thruster_wake_min_scale,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if int(self.tether_num_segments) != self.tether_num_segments or self.tether_num_segments < 1:
            raise ValueError("tether_num_segments must be a positive integer.")
        if self.tether_winch_max_length < self.tether_winch_min_length:
            raise ValueError("tether_winch_max_length must be >= tether_winch_min_length.")

    def pose_sensor_delay_steps_for_dt(self, physics_dt_s: float) -> int:
        """Convert the fused-state delay to the 100 Hz truth-history grid."""

        if float(physics_dt_s) <= 0.0:
            raise ValueError("physics_dt_s must be positive.")
        return int(round(self.pose_sensor_delay_s / float(physics_dt_s)))

    def to_runtime_cfg_updates(self, physics_dt_s: float = 1.0 / 100.0) -> dict[str, Any]:
        """Return adapter fields without creating a second physical source."""

        self.validate()
        sensor_delay_steps = self.pose_sensor_delay_steps_for_dt(physics_dt_s)
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
            "tether_enabled": self.tether_enabled,
            "tether_anchor_pos_w": list(self.tether_anchor_pos_w),
            "tether_attach_offset_b": list(self.tether_attach_offset_b),
            "tether_slack_length": self.tether_slack_length,
            "tether_stiffness": self.tether_stiffness,
            "tether_damping": self.tether_damping,
            "tether_drag_coeff": self.tether_drag_coeff,
            "tether_winch_enabled": self.tether_winch_enabled,
            "tether_winch_target_length": self.tether_winch_target_length,
            "tether_winch_reel_speed": self.tether_winch_reel_speed,
            "tether_winch_min_length": self.tether_winch_min_length,
            "tether_winch_max_length": self.tether_winch_max_length,
            "tether_num_segments": int(self.tether_num_segments),
            "tether_segment_diameter": self.tether_segment_diameter,
            "tether_segment_density": self.tether_segment_density,
        }


T60_RUNTIME = RobotRuntimeProfile()

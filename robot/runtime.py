"""Robot-owned runtime configuration used by simulator adapters.

Geometry, rigid-body properties, and measured force curves remain in the
specialized ``robot`` modules.  This profile owns the remaining vehicle-side
actuator, battery, and tether settings so simulators do not invent shadow
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

    # Provisional external T200/Basic-ESC timing references. Replace these
    # nominal values with the T60's own synchronized PWM-to-force bench data
    # when available; source strings are exported into every run config.
    thruster_time_constant_s: float = 0.08
    thruster_command_delay_s: float = 0.13
    thruster_standstill_startup_delay_reference_s: float = 0.25
    thruster_time_constant_source: str = (
        "Rossol et al., OCEANS 2022, Blue Robotics T200 thrust reversal: "
        "first-order comparison T=0.08 s; https://www.dfki.de/fileadmin/user_upload/import/"
        "12795_oceans2022_sos_paper_final.pdf"
    )
    thruster_command_delay_source: str = (
        "NUS Bumblebee RoboSub 2022, oscilloscope measurement of Blue Robotics Basic ESC: "
        "approximately 130 ms; https://bumblebee.sg/pdf/Bumblebee_Robosub_Paper_2022.pdf"
    )
    thruster_standstill_startup_delay_source: str = (
        "Rossol et al., OCEANS 2022, Blue Robotics T200 standstill startup: approximately "
        "250 ms; recorded for audit but not added on top of the fixed ESC delay; "
        "https://www.dfki.de/fileadmin/user_upload/import/12795_oceans2022_sos_paper_final.pdf"
    )
    thruster_max_command_rate: float = 0.0
    thruster_command_resolution: float = 0.0
    thruster_command_dropout_probability: float = 0.0
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
    battery_nominal_voltage: float = 16.0
    battery_initial_voltage: float = 16.0
    battery_min_voltage: float = 12.0
    battery_voltage_drop_per_s: float = 0.0
    battery_thrust_exponent: float = 2.0

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
            "thruster_command_delay_s": self.thruster_command_delay_s,
            "thruster_standstill_startup_delay_reference_s": (
                self.thruster_standstill_startup_delay_reference_s
            ),
            "thruster_max_command_rate": self.thruster_max_command_rate,
            "thruster_command_resolution": self.thruster_command_resolution,
            "thruster_command_dropout_probability": self.thruster_command_dropout_probability,
            "thruster_inflow_loss_coefficient": self.thruster_inflow_loss_coefficient,
            "thruster_wake_loss_coefficient": self.thruster_wake_loss_coefficient,
            "thruster_wake_expansion_rate": self.thruster_wake_expansion_rate,
            "battery_min_voltage": self.battery_min_voltage,
            "battery_voltage_drop_per_s": self.battery_voltage_drop_per_s,
            "battery_thrust_exponent": self.battery_thrust_exponent,
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
            "battery_nominal_voltage": self.battery_nominal_voltage,
            "battery_initial_voltage": self.battery_initial_voltage,
            "tether_segment_diameter": self.tether_segment_diameter,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name, value in {
            "thruster_command_dropout_probability": self.thruster_command_dropout_probability,
            "thruster_inflow_min_scale": self.thruster_inflow_min_scale,
            "thruster_wake_min_scale": self.thruster_wake_min_scale,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if int(self.tether_num_segments) != self.tether_num_segments or self.tether_num_segments < 1:
            raise ValueError("tether_num_segments must be a positive integer.")
        if self.tether_winch_max_length < self.tether_winch_min_length:
            raise ValueError("tether_winch_max_length must be >= tether_winch_min_length.")

    def thruster_command_delay_steps_for_dt(self, physics_dt_s: float) -> int:
        """Quantize the physical command delay to the simulator time grid."""

        if float(physics_dt_s) <= 0.0:
            raise ValueError("physics_dt_s must be positive.")
        return int(round(self.thruster_command_delay_s / float(physics_dt_s)))

    def to_isaac_cfg_updates(self, physics_dt_s: float = 1.0 / 200.0) -> dict[str, Any]:
        """Return adapter fields without creating a second physical source."""

        self.validate()
        command_delay_steps = self.thruster_command_delay_steps_for_dt(physics_dt_s)
        return {
            "mass": self.model.mass_kg,
            "volume": self.model.displaced_volume_m3,
            "inertia_diag": [list(row) for row in self.model.inertia_tensor_body_kg_m2],
            "center_of_mass_offset": list(self.model.center_of_mass_offset_m),
            "com_to_cob_offset": list(self.model.center_of_buoyancy_from_com_m),
            "water_rho": self.model.water_density_kg_m3,
            "dyn_time_constant": self.thruster_time_constant_s,
            "thruster_command_delay_s": self.thruster_command_delay_s,
            "thruster_command_delay_steps": command_delay_steps,
            "thruster_command_delay_applied_s": command_delay_steps * float(physics_dt_s),
            "thruster_standstill_startup_delay_reference_s": (
                self.thruster_standstill_startup_delay_reference_s
            ),
            "thruster_time_constant_source": self.thruster_time_constant_source,
            "thruster_command_delay_source": self.thruster_command_delay_source,
            "thruster_standstill_startup_delay_source": (
                self.thruster_standstill_startup_delay_source
            ),
            "thruster_max_command_rate": self.thruster_max_command_rate,
            "thruster_command_resolution": self.thruster_command_resolution,
            "thruster_command_dropout_probability": self.thruster_command_dropout_probability,
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
            "battery_voltage_nominal": self.battery_nominal_voltage,
            "battery_voltage": self.battery_initial_voltage,
            "battery_min_voltage": self.battery_min_voltage,
            "battery_voltage_drop_per_s": self.battery_voltage_drop_per_s,
            "battery_voltage_thrust_exponent": self.battery_thrust_exponent,
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
            "tether_segment_buoyancy_density": self.model.water_density_kg_m3,
        }


T60_RUNTIME = RobotRuntimeProfile()

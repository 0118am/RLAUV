"""Compose external environment and robot sources into an Isaac runtime cfg."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environment.profiles.domain_randomization import (
    DomainRandomizationSpec,
    apply_domain_randomization_spec,
    resolve_domain_randomization_spec,
)
from environment.profiles.environment_profile import (
    EnvironmentProfile,
    resolve_environment_profile,
)
from robot.runtime import RobotRuntimeProfile, T60_RUNTIME


def _neutral_randomization_updates(robot: RobotRuntimeProfile) -> dict[str, Any]:
    """Materialize the complete no-randomization cross-domain contract."""

    model = robot.model
    return {
        "use_custom_randomization": False,
        "enabled_features": [],
        "com_to_cob_offset_radius": 0.0,
        "volume_range": [model.displaced_volume_m3, model.displaced_volume_m3],
        "mass_range": [model.mass_kg, model.mass_kg],
        "payload_samples": [],
        "thruster_command_delay_steps_range": [robot.thruster_command_delay_steps] * 2,
        "thruster_max_command_rate_range": [robot.thruster_max_command_rate] * 2,
        "thruster_command_resolution_range": [robot.thruster_command_resolution] * 2,
        "thruster_command_dropout_probability_range": [
            robot.thruster_command_dropout_probability
        ] * 2,
        "thruster_wake_loss_coefficient_scale_range": [1.0, 1.0],
        "thruster_reaction_torque_coeff_scale_range": [1.0, 1.0],
        "damping_speed_linear_scale_range": [1.0, 1.0],
        "damping_speed_quadratic_scale_range": [1.0, 1.0],
        "battery_voltage_range": [robot.battery_initial_voltage] * 2,
        "battery_voltage_drop_per_s_range": [robot.battery_voltage_drop_per_s] * 2,
        "disturbance_curriculum": False,
        "disturbance_curriculum_stage_steps": [],
        "water_current_smooth": False,
        "water_current_tau_range": [12.0, 12.0],
        "water_current_max_by_stage": [0.0],
        "water_current_vertical_max_by_stage": [0.0],
        "water_current_variation_std_by_stage": [0.0],
        "damping_scale_by_stage": [0.0],
        "added_mass_log_std_by_stage": [0.0],
        "thruster_scale_by_stage": [0.0],
        "thruster_tau_scale_by_stage": [0.0],
        "additional_hydrodynamics_scale_by_stage": [0.0],
    }


@dataclass(frozen=True)
class IsaacComposition:
    """Resolved external sources for one Isaac environment construction."""

    environment: EnvironmentProfile
    robot: RobotRuntimeProfile = T60_RUNTIME
    randomization: DomainRandomizationSpec | None = None

    def apply(self, cfg: Any) -> Any:
        self.robot.validate()
        for key, value in self.environment.to_cfg_updates().items():
            setattr(cfg, key, copy.deepcopy(value))
        for key, value in self.robot.to_isaac_cfg_updates().items():
            setattr(cfg, key, copy.deepcopy(value))

        if self.randomization is None:
            for key, value in _neutral_randomization_updates(self.robot).items():
                setattr(cfg.domain_randomization, key, copy.deepcopy(value))
            cfg.domain_randomization_spec_name = None
        else:
            apply_domain_randomization_spec(
                cfg,
                self.randomization,
                base_profile=self.environment,
            )
        cfg.environment_profile_name = self.environment.name
        return cfg


def resolve_isaac_composition(
    environment_profile: EnvironmentProfile | str | Path,
    domain_randomization_spec: DomainRandomizationSpec | str | Path | None = None,
    *,
    robot: RobotRuntimeProfile = T60_RUNTIME,
) -> IsaacComposition:
    """Resolve, validate, and return one explicit Isaac composition."""

    environment = resolve_environment_profile(environment_profile)
    randomization = (
        None
        if domain_randomization_spec in (None, "")
        else resolve_domain_randomization_spec(domain_randomization_spec)
    )
    return IsaacComposition(environment=environment, robot=robot, randomization=randomization)

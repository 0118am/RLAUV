"""Versioned domain-randomization recipes independent of measured profiles.

Measured :class:`PoolDynamicsProfile` objects describe one deterministic
vehicle/pool configuration.  This module describes how training samples around
that configuration.  Keeping the two artifacts separate allows several
training distributions to reference the same measured profile without
changing the physical source of truth.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping

from .features import DOMAIN_RANDOMIZATION_FEATURES
from .pool_profile import (
    DomainRandomizationProfile,
    PoolDynamicsProfile,
)


DOMAIN_RANDOMIZATION_SCHEMA_VERSION = 1


def domain_randomization_parameter_names() -> frozenset[str]:
    return frozenset(item.name for item in fields(DomainRandomizationProfile))


def domain_randomization_parameters_requiring_sources(
    parameters: DomainRandomizationProfile,
) -> frozenset[str]:
    """Return active stochastic parameters that require provenance."""

    required: set[str] = set()
    for name, value in parameters.to_cfg_updates().items():
        if isinstance(value, bool) or value is None:
            continue
        if name == "payload_samples":
            if len(value) > 0:
                required.add(name)
        elif name == "com_to_cob_offset_radius":
            if float(value) > 0.0:
                required.add(name)
        elif name == "disturbance_curriculum_stage_steps":
            if len(value) > 0:
                required.add(name)
        elif name.endswith("_by_stage"):
            if any(float(item) != 0.0 for item in value):
                required.add(name)
        elif name.endswith("_range"):
            if len(value) == 2 and float(value[0]) != float(value[1]):
                required.add(name)
    return frozenset(required)


@dataclass(frozen=True)
class DomainRandomizationSpec:
    """Serializable training recipe for reset/step-time randomization."""

    name: str
    parameters: DomainRandomizationProfile
    schema_version: int = DOMAIN_RANDOMIZATION_SCHEMA_VERSION
    description: str = ""
    base_profile_name: str | None = None
    parameter_sources: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise ValueError("DomainRandomizationSpec.schema_version must be an integer.")
        if self.schema_version != DOMAIN_RANDOMIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported domain-randomization schema version {self.schema_version}; "
                f"expected {DOMAIN_RANDOMIZATION_SCHEMA_VERSION}."
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DomainRandomizationSpec.name must be non-empty.")
        if not isinstance(self.description, str):
            raise ValueError("DomainRandomizationSpec.description must be a string.")
        if not isinstance(self.parameters, DomainRandomizationProfile):
            raise ValueError("DomainRandomizationSpec.parameters must be a DomainRandomizationProfile.")
        if not isinstance(self.parameter_sources, Mapping):
            raise ValueError("DomainRandomizationSpec.parameter_sources must be a mapping.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("DomainRandomizationSpec.metadata must be a mapping.")
        if self.base_profile_name is not None and (
            not isinstance(self.base_profile_name, str) or not self.base_profile_name.strip()
        ):
            raise ValueError("base_profile_name must be a non-empty string when provided.")
        self.parameters.validate()
        parameter_names = domain_randomization_parameter_names()
        # ``enabled_features`` was introduced after the original versioned
        # recipe schema. Its omission intentionally means "all features".
        optional_legacy_parameters = {"enabled_features"}
        missing_parameters = sorted(
            parameter_names - optional_legacy_parameters - set(self.parameters.to_cfg_updates())
        )
        if missing_parameters:
            raise ValueError(
                "DomainRandomizationSpec.parameters must be complete; missing: "
                + ", ".join(missing_parameters)
            )
        if any(not isinstance(key, str) for key in self.parameter_sources):
            raise ValueError("parameter_sources keys must be strings.")
        unknown_sources = sorted(set(self.parameter_sources) - parameter_names)
        if unknown_sources:
            raise ValueError(
                "parameter_sources contains fields absent from parameters: " + ", ".join(unknown_sources)
            )
        for key, source in self.parameter_sources.items():
            if not isinstance(key, str) or not isinstance(source, str) or not source.strip():
                raise ValueError("parameter_sources must map parameter names to non-empty source strings.")
        missing_sources = sorted(
            domain_randomization_parameters_requiring_sources(self.parameters)
            - set(self.parameter_sources)
        )
        if missing_sources:
            raise ValueError(
                "Active domain-randomization parameters require parameter_sources: "
                + ", ".join(missing_sources)
            )
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON-compatible values.") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "base_profile_name": self.base_profile_name,
            "parameters": copy.deepcopy(self.parameters.to_cfg_updates()),
            "parameter_sources": copy.deepcopy(dict(self.parameter_sources)),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

def domain_randomization_spec_from_dict(data: Mapping[str, Any]) -> DomainRandomizationSpec:
    """Build a validated spec and reject misspelled fields."""

    allowed = {
        "schema_version",
        "name",
        "description",
        "base_profile_name",
        "parameters",
        "parameter_sources",
        "metadata",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("Unknown DomainRandomizationSpec field(s): " + ", ".join(unknown))

    parameters_data = data.get("parameters")
    if not isinstance(parameters_data, Mapping):
        raise TypeError("DomainRandomizationSpec.parameters must be a mapping.")
    valid_parameter_names = domain_randomization_parameter_names()
    unknown_parameters = sorted(set(parameters_data) - valid_parameter_names)
    if unknown_parameters:
        raise ValueError("Unknown domain-randomization parameter(s): " + ", ".join(unknown_parameters))

    parameter_sources = data.get("parameter_sources", {})
    metadata = data.get("metadata", {})
    if not isinstance(parameter_sources, Mapping):
        raise TypeError("DomainRandomizationSpec.parameter_sources must be a mapping.")
    if not isinstance(metadata, Mapping):
        raise TypeError("DomainRandomizationSpec.metadata must be a mapping.")

    spec = DomainRandomizationSpec(
        schema_version=data.get("schema_version", DOMAIN_RANDOMIZATION_SCHEMA_VERSION),
        name=data.get("name", ""),
        description=data.get("description", ""),
        base_profile_name=data.get("base_profile_name"),
        parameters=DomainRandomizationProfile(**copy.deepcopy(dict(parameters_data))),
        parameter_sources=copy.deepcopy(dict(parameter_sources)),
        metadata=copy.deepcopy(dict(metadata)),
    )
    spec.validate()
    return spec


def load_domain_randomization_spec_json(path: str | Path) -> DomainRandomizationSpec:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant {value!r} is not allowed.")
            ),
        )
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a JSON object.")
    return domain_randomization_spec_from_dict(data)


def write_domain_randomization_spec_json(spec: DomainRandomizationSpec, path: str | Path) -> None:
    spec.validate()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(spec.to_dict(), stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def resolve_domain_randomization_spec(
    value: DomainRandomizationSpec | str | Path,
) -> DomainRandomizationSpec:
    if isinstance(value, DomainRandomizationSpec):
        value.validate()
        return value
    return load_domain_randomization_spec_json(value)


def complete_domain_randomization_profile(
    parameters: DomainRandomizationProfile,
    base_profile: PoolDynamicsProfile,
) -> DomainRandomizationProfile:
    """Fill a calibration overlay with explicit neutral values.

    Standalone recipes must not inherit hidden training ranges from an
    environment config. Absolute mass/volume/battery values are centered on
    the bound deterministic profile; all other missing entries are neutral.
    """

    base_profile.validate()
    parameters.validate()
    provided = parameters.to_cfg_updates()
    stage_count = max(
        (
            len(value)
            for name in (
                "water_current_max_by_stage",
                "water_current_vertical_max_by_stage",
                "water_current_variation_std_by_stage",
                "damping_scale_by_stage",
                "added_mass_log_std_by_stage",
                "thruster_scale_by_stage",
                "thruster_tau_scale_by_stage",
                "additional_hydrodynamics_scale_by_stage",
            )
            if (value := provided.get(name)) is not None
        ),
        default=1,
    )
    thrusters = base_profile.thrusters
    battery = base_profile.battery
    neutral: dict[str, Any] = {
        "use_custom_randomization": False,
        "enabled_features": list(DOMAIN_RANDOMIZATION_FEATURES),
        "com_to_cob_offset_radius": 0.0,
        "volume_range": [base_profile.rigid_body.volume, base_profile.rigid_body.volume],
        "mass_range": [base_profile.rigid_body.mass, base_profile.rigid_body.mass],
        "payload_samples": [],
        "thruster_command_delay_steps_range": [thrusters.command_delay_steps, thrusters.command_delay_steps],
        "thruster_max_command_rate_range": [thrusters.max_command_rate, thrusters.max_command_rate],
        "thruster_command_resolution_range": [thrusters.command_resolution, thrusters.command_resolution],
        "thruster_command_dropout_probability_range": [
            thrusters.command_dropout_probability,
            thrusters.command_dropout_probability,
        ],
        "thruster_wake_loss_coefficient_scale_range": [1.0, 1.0],
        "thruster_reaction_torque_coeff_scale_range": [1.0, 1.0],
        "damping_speed_linear_scale_range": [1.0, 1.0],
        "damping_speed_quadratic_scale_range": [1.0, 1.0],
        "battery_voltage_range": [battery.initial_voltage, battery.initial_voltage],
        "battery_voltage_drop_per_s_range": [battery.voltage_drop_per_s, battery.voltage_drop_per_s],
        "observation_noise_std_range": [0.0, 0.0],
        "observation_bias_range": [0.0, 0.0],
        "observation_delay_steps_range": [0, 0],
        "observation_update_period_steps_range": [1, 1],
        "observation_dropout_probability_range": [0.0, 0.0],
        "observation_lowpass_alpha_range": [1.0, 1.0],
        "observation_bias_drift_std_range": [0.0, 0.0],
        "disturbance_curriculum": False,
        "disturbance_curriculum_stage_steps": [],
        "water_current_smooth": False,
        "water_current_tau_range": [12.0, 12.0],
        "water_current_max_by_stage": [0.0] * stage_count,
        "water_current_vertical_max_by_stage": [0.0] * stage_count,
        "water_current_variation_std_by_stage": [0.0] * stage_count,
        "damping_scale_by_stage": [0.0] * stage_count,
        "added_mass_log_std_by_stage": [0.0] * stage_count,
        "thruster_scale_by_stage": [0.0] * stage_count,
        "thruster_tau_scale_by_stage": [0.0] * stage_count,
        "additional_hydrodynamics_scale_by_stage": [0.0] * stage_count,
    }
    neutral.update(copy.deepcopy(provided))
    completed = DomainRandomizationProfile(**neutral)
    completed.validate()
    return completed


def domain_randomization_spec_from_pool_profile(
    profile: PoolDynamicsProfile,
    *,
    name: str | None = None,
    description: str = "",
    parameter_sources: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DomainRandomizationSpec:
    """Extract legacy calibration uncertainty into a separate bound recipe."""

    profile.validate()
    if profile.domain_randomization is None:
        raise ValueError(f"Pool profile {profile.name!r} has no domain_randomization section to export.")
    completed_parameters = complete_domain_randomization_profile(
        profile.domain_randomization,
        profile,
    )
    spec = DomainRandomizationSpec(
        name=name or f"{profile.name}-dr-v1",
        description=description,
        base_profile_name=profile.name,
        parameters=completed_parameters,
        parameter_sources=copy.deepcopy(dict(parameter_sources or {})),
        metadata=copy.deepcopy(dict(metadata or {})),
    )
    spec.validate()
    return spec


def validate_domain_randomization_base_profile(
    spec: DomainRandomizationSpec,
    base_profile: PoolDynamicsProfile,
) -> None:
    """Reject a recipe accidentally paired with a different measured profile."""

    base_profile.validate()
    if spec.base_profile_name is not None and spec.base_profile_name != base_profile.name:
        raise ValueError(
            f"Domain-randomization spec expects base profile {spec.base_profile_name!r}, "
            f"got {base_profile.name!r}."
        )


def apply_domain_randomization_spec(
    cfg: Any,
    spec: DomainRandomizationSpec,
    *,
    base_profile: PoolDynamicsProfile | None = None,
) -> Any:
    """Apply a domain-randomization recipe to an AUV config."""

    spec.validate()
    if spec.base_profile_name is not None and base_profile is None:
        raise ValueError("A bound DomainRandomizationSpec requires a resolved base profile.")
    if base_profile is not None:
        validate_domain_randomization_base_profile(spec, base_profile)
    if not hasattr(cfg, "domain_randomization"):
        raise AttributeError("cfg must define domain_randomization to apply a randomization spec.")
    for key, value in spec.parameters.to_cfg_updates().items():
        setattr(cfg.domain_randomization, key, copy.deepcopy(value))
    cfg.domain_randomization_spec_name = spec.name
    return cfg

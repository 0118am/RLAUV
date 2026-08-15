"""Versioned domain-randomization recipes independent of deterministic sources."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .environment_profile import EnvironmentProfile
from ._validation import (
    validate_integer_sequence,
    validate_nonnegative,
    validate_nonnegative_sequence,
    validate_payload_samples,
    validate_range,
)


DOMAIN_RANDOMIZATION_SCHEMA_VERSION = 1


NumberSequence = Sequence[float]


@dataclass(frozen=True)
class DomainRandomizationProfile:
    """Optional reset-time randomization ranges for calibrated uncertainty."""

    use_custom_randomization: bool | None = None
    enabled_features: Sequence[str] | None = None
    com_to_cob_offset_radius: float | None = None
    volume_range: NumberSequence | None = None
    mass_range: NumberSequence | None = None
    payload_samples: Sequence[Mapping[str, Any]] | None = None
    thruster_command_delay_steps_range: NumberSequence | None = None
    thruster_max_command_rate_range: NumberSequence | None = None
    thruster_command_resolution_range: NumberSequence | None = None
    thruster_command_dropout_probability_range: NumberSequence | None = None
    thruster_wake_loss_coefficient_scale_range: NumberSequence | None = None
    thruster_reaction_torque_coeff_scale_range: NumberSequence | None = None
    damping_speed_linear_scale_range: NumberSequence | None = None
    damping_speed_quadratic_scale_range: NumberSequence | None = None
    battery_voltage_range: NumberSequence | None = None
    battery_voltage_drop_per_s_range: NumberSequence | None = None
    disturbance_curriculum: bool | None = None
    disturbance_curriculum_stage_steps: Sequence[int] | None = None
    water_current_smooth: bool | None = None
    water_current_tau_range: NumberSequence | None = None
    water_current_max_by_stage: NumberSequence | None = None
    water_current_vertical_max_by_stage: NumberSequence | None = None
    water_current_variation_std_by_stage: NumberSequence | None = None
    damping_scale_by_stage: NumberSequence | None = None
    added_mass_log_std_by_stage: NumberSequence | None = None
    thruster_scale_by_stage: NumberSequence | None = None
    thruster_tau_scale_by_stage: NumberSequence | None = None
    additional_hydrodynamics_scale_by_stage: NumberSequence | None = None

    def _validate_optional_ranges(self) -> None:
        for name in (
            "volume_range",
            "mass_range",
            "thruster_max_command_rate_range",
            "thruster_command_resolution_range",
            "thruster_command_dropout_probability_range",
            "thruster_wake_loss_coefficient_scale_range",
            "thruster_reaction_torque_coeff_scale_range",
            "damping_speed_linear_scale_range",
            "damping_speed_quadratic_scale_range",
            "battery_voltage_range",
            "battery_voltage_drop_per_s_range",
            "water_current_tau_range",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            validate_range(value, f"domain_randomization.{name}")
            if name in {"mass_range", "volume_range", "battery_voltage_range", "water_current_tau_range"}:
                if float(value[0]) <= 0.0:
                    raise ValueError(f"domain_randomization.{name} must be positive.")
            elif float(value[0]) < 0.0:
                raise ValueError(f"domain_randomization.{name} must be non-negative.")
            if name == "thruster_command_dropout_probability_range" and (
                float(value[0]) < 0.0 or float(value[1]) > 1.0
            ):
                raise ValueError(
                    "domain_randomization.thruster_command_dropout_probability_range must be in [0, 1]."
                )
        if self.thruster_command_delay_steps_range is not None:
            validate_range(
                self.thruster_command_delay_steps_range,
                "domain_randomization.thruster_command_delay_steps_range",
                integer=True,
            )
            if int(self.thruster_command_delay_steps_range[0]) < 0:
                raise ValueError(
                    "domain_randomization.thruster_command_delay_steps_range must be non-negative."
                )

    def _validate_curriculum_arrays(self) -> None:
        stage_names = (
            "water_current_max_by_stage",
            "water_current_vertical_max_by_stage",
            "water_current_variation_std_by_stage",
            "damping_scale_by_stage",
            "added_mass_log_std_by_stage",
            "thruster_scale_by_stage",
            "thruster_tau_scale_by_stage",
            "additional_hydrodynamics_scale_by_stage",
        )
        for name in stage_names:
            value = getattr(self, name)
            if value is None:
                continue
            validate_nonnegative_sequence(value, f"domain_randomization.{name}")
            if name in {
                "damping_scale_by_stage",
                "thruster_scale_by_stage",
                "thruster_tau_scale_by_stage",
                "additional_hydrodynamics_scale_by_stage",
            } and any(float(item) > 1.0 for item in value):
                raise ValueError(
                    f"domain_randomization.{name} must not exceed 1.0 because it is used as a ± amplitude."
                )
        stage_lengths = [len(value) for name in stage_names if (value := getattr(self, name)) is not None]
        if stage_lengths and any(length != stage_lengths[0] for length in stage_lengths):
            raise ValueError("disturbance by-stage arrays must have matching lengths.")
        if self.disturbance_curriculum and stage_lengths and self.disturbance_curriculum_stage_steps is None:
            raise ValueError(
                "domain_randomization.disturbance_curriculum_stage_steps is required when curriculum is enabled."
            )
        if self.disturbance_curriculum_stage_steps is not None:
            validate_integer_sequence(
                self.disturbance_curriculum_stage_steps,
                "domain_randomization.disturbance_curriculum_stage_steps",
                nonnegative=True,
            )
            if (
                self.disturbance_curriculum
                and stage_lengths
                and len(self.disturbance_curriculum_stage_steps) != stage_lengths[0] - 1
            ):
                raise ValueError(
                    "domain_randomization.disturbance_curriculum_stage_steps must have one fewer entry "
                    "than disturbance by-stage arrays."
                )

    def validate(self) -> None:
        if self.enabled_features is not None:
            from .features import normalize_domain_randomization_features

            normalize_domain_randomization_features(self.enabled_features)
        for name in ("use_custom_randomization", "disturbance_curriculum", "water_current_smooth"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"domain_randomization.{name} must be boolean.")
        if self.payload_samples is not None:
            validate_payload_samples(self.payload_samples, "domain_randomization.payload_samples")
        if self.com_to_cob_offset_radius is not None:
            validate_nonnegative(
                self.com_to_cob_offset_radius,
                "domain_randomization.com_to_cob_offset_radius",
            )
        self._validate_optional_ranges()
        self._validate_curriculum_arrays()

    def to_cfg_updates(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


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
        missing_parameters = sorted(parameter_names - set(self.parameters.to_cfg_updates()))
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


def validate_domain_randomization_base_profile(
    spec: DomainRandomizationSpec,
    base_profile: EnvironmentProfile,
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
    base_profile: EnvironmentProfile | None = None,
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

"""Versioned domain-randomization recipes independent of deterministic sources."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from common.schema import (
    FiniteJsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    StrictBoolean,
    StrictFrozenModel,
)


DOMAIN_RANDOMIZATION_SCHEMA_VERSION = 9

DOMAIN_RANDOMIZATION_FEATURES = (
    "current",
    "hydrodynamics",
    "actuators",
)
ALL_DOMAIN_RANDOMIZATION_FEATURES = frozenset(DOMAIN_RANDOMIZATION_FEATURES)
DomainRandomizationFeature: TypeAlias = Literal[*DOMAIN_RANDOMIZATION_FEATURES]


def normalize_domain_randomization_features(features: Iterable[str]) -> tuple[str, ...]:
    """Return an explicit, canonical feature selection."""

    if features is None or isinstance(features, str):
        raise ValueError("domain_randomization.enabled_features must be a sequence.")
    names = tuple(str(name) for name in features)
    unknown = sorted(set(names) - ALL_DOMAIN_RANDOMIZATION_FEATURES)
    if unknown:
        raise ValueError("Unknown domain-randomization feature(s): " + ", ".join(unknown))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate domain-randomization feature(s): " + ", ".join(duplicates))
    return names


PositiveRange: TypeAlias = tuple[PositiveFloat, PositiveFloat]
NonNegativeRange: TypeAlias = tuple[NonNegativeFloat, NonNegativeFloat]
StageValues: TypeAlias = Annotated[tuple[NonNegativeFloat, ...], Field(min_length=1)]

DISTURBANCE_STAGE_FIELDS = (
    "water_current_max_by_stage",
    "water_current_vertical_max_by_stage",
    "water_current_variation_std_by_stage",
    "linear_damping_log_std_by_stage",
    "quadratic_damping_log_std_by_stage",
    "fluid_added_mass_log_std_by_stage",
    "thruster_scale_by_stage",
    "common_thruster_scale_reduction_by_stage",
    "additional_hydrodynamics_scale_by_stage",
)


class DomainRandomizationProfile(StrictFrozenModel):
    """Optional reset-time randomization ranges for calibrated uncertainty."""

    use_custom_randomization: StrictBoolean | None = None
    enabled_features: tuple[DomainRandomizationFeature, ...] | None = None
    thruster_time_constant_range: NonNegativeRange | None = None
    thruster_wake_loss_coefficient_scale_range: NonNegativeRange | None = None
    damping_speed_linear_scale_range: NonNegativeRange | None = None
    damping_speed_quadratic_scale_range: NonNegativeRange | None = None
    disturbance_curriculum: StrictBoolean | None = None
    disturbance_curriculum_stage_steps: tuple[NonNegativeInt, ...] | None = None
    water_current_smooth: StrictBoolean | None = None
    water_current_tau_range: PositiveRange | None = None
    water_current_max_by_stage: StageValues | None = None
    water_current_vertical_max_by_stage: StageValues | None = None
    water_current_variation_std_by_stage: StageValues | None = None
    linear_damping_log_std_by_stage: StageValues | None = None
    quadratic_damping_log_std_by_stage: StageValues | None = None
    fluid_added_mass_log_std_by_stage: StageValues | None = None
    thruster_scale_by_stage: StageValues | None = None
    common_thruster_scale_reduction_by_stage: StageValues | None = None
    additional_hydrodynamics_scale_by_stage: StageValues | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> "DomainRandomizationProfile":
        if self.enabled_features is not None:
            normalize_domain_randomization_features(self.enabled_features)

        range_names = (
            "thruster_time_constant_range",
            "thruster_wake_loss_coefficient_scale_range",
            "damping_speed_linear_scale_range",
            "damping_speed_quadratic_scale_range",
            "water_current_tau_range",
        )
        for name in range_names:
            value = getattr(self, name)
            if value is not None and value[1] < value[0]:
                raise ValueError(f"domain_randomization.{name} upper bound must be >= lower bound.")

        amplitude_names = {
            "thruster_scale_by_stage",
            "additional_hydrodynamics_scale_by_stage",
        }
        for name in amplitude_names:
            values = getattr(self, name)
            if values is not None and any(value > 1.0 for value in values):
                raise ValueError(
                    f"domain_randomization.{name} must not exceed 1.0 because it is used as a +/- amplitude."
                )

        common_reductions = self.common_thruster_scale_reduction_by_stage
        if common_reductions is not None and any(value > 1.0 for value in common_reductions):
            raise ValueError(
                "domain_randomization.common_thruster_scale_reduction_by_stage must not exceed "
                "1.0 because it is a weakening fraction."
            )

        stage_lengths = [
            len(values)
            for name in DISTURBANCE_STAGE_FIELDS
            if (values := getattr(self, name)) is not None
        ]
        if stage_lengths and any(length != stage_lengths[0] for length in stage_lengths):
            raise ValueError("disturbance by-stage arrays must have matching lengths.")

        steps = self.disturbance_curriculum_stage_steps
        if steps is not None and any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(
                "domain_randomization.disturbance_curriculum_stage_steps must be strictly increasing."
            )
        if self.disturbance_curriculum and stage_lengths and steps is None:
            raise ValueError(
                "domain_randomization.disturbance_curriculum_stage_steps is required when curriculum is enabled."
            )
        if (
            self.disturbance_curriculum
            and stage_lengths
            and steps is not None
            and len(steps) != stage_lengths[0] - 1
        ):
            raise ValueError(
                "domain_randomization.disturbance_curriculum_stage_steps must have one fewer entry "
                "than disturbance by-stage arrays."
            )
        return self

DOMAIN_RANDOMIZATION_PARAMETER_NAMES = frozenset(DomainRandomizationProfile.model_fields)


def disturbance_stage_count(profile: Any) -> int:
    """Return the number of stages explicitly present in one DR profile."""

    return max(
        (
            len(value)
            for name in DISTURBANCE_STAGE_FIELDS
            if (value := getattr(profile, name)) is not None
        ),
        default=1,
    )


class DomainRandomizationSpec(StrictFrozenModel):
    """Serializable training recipe for reset/step-time randomization."""

    name: Annotated[str, Field(min_length=1)]
    parameters: DomainRandomizationProfile
    schema_version: Literal[DOMAIN_RANDOMIZATION_SCHEMA_VERSION] = DOMAIN_RANDOMIZATION_SCHEMA_VERSION
    description: str = ""
    metadata: dict[str, FiniteJsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_name(self) -> "DomainRandomizationSpec":
        if not self.name.strip():
            raise ValueError("DomainRandomizationSpec.name must be non-empty.")
        return self


def load_domain_randomization_spec_json(path: str | Path) -> DomainRandomizationSpec:
    return DomainRandomizationSpec.model_validate_json(Path(path).read_bytes())


def write_domain_randomization_spec_json(spec: DomainRandomizationSpec, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")


def resolve_domain_randomization_spec(
    value: DomainRandomizationSpec | str | Path,
) -> DomainRandomizationSpec:
    return value if isinstance(value, DomainRandomizationSpec) else load_domain_randomization_spec_json(value)

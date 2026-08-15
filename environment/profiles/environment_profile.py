"""Strict simulator-independent profile for water and pool physics only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from ._validation import (
    HydroCoefficients,
    count_current_vectors,
    is_sequence,
    validate_6_vector_or_matrix,
    validate_damping_speed_scale_curve,
    validate_nonnegative,
    validate_positive,
    validate_vector,
)


NumberSequence = Sequence[float]
ZERO_HYDRODYNAMIC_COEFFICIENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class HydrodynamicsProfile:
    """Hydrodynamic coefficients and deterministic water-current models."""

    linear_damping: HydroCoefficients = ZERO_HYDRODYNAMIC_COEFFICIENTS
    quadratic_damping: HydroCoefficients = ZERO_HYDRODYNAMIC_COEFFICIENTS
    speed_dependent_damping_enabled: bool = False
    damping_speed_points: NumberSequence = (0.0, 1.0)
    linear_damping_speed_scales: Sequence[Any] = field(default_factory=tuple)
    quadratic_damping_speed_scales: Sequence[Any] = field(default_factory=tuple)
    added_mass: HydroCoefficients = ZERO_HYDRODYNAMIC_COEFFICIENTS
    added_mass_inertia_scale: float = 1.0
    added_mass_accel_filter_alpha: float = 0.35
    water_current_w: NumberSequence = (0.0, 0.0, 0.0)
    water_current_periodic_enabled: bool = False
    water_current_periodic_amplitude_w: NumberSequence = (0.0, 0.0, 0.0)
    water_current_periodic_period_s: NumberSequence = (20.0, 20.0, 20.0)
    water_current_periodic_phase_rad: NumberSequence = (0.0, 0.0, 0.0)
    water_current_field_enabled: bool = False
    water_current_field_bounds: NumberSequence = (-7.0, 7.0, -7.0, 7.0, -15.0, -1.0)
    water_current_field_shape: Sequence[int] = (1, 1, 1)
    water_current_field_values: Sequence[Any] = field(default_factory=tuple)

    def validate(self) -> None:
        validate_6_vector_or_matrix(self.linear_damping, "hydrodynamics.linear_damping")
        validate_6_vector_or_matrix(self.quadratic_damping, "hydrodynamics.quadratic_damping")
        validate_damping_speed_scale_curve(
            self.damping_speed_points,
            self.linear_damping_speed_scales,
            "hydrodynamics.linear_damping_speed_scales",
        )
        validate_damping_speed_scale_curve(
            self.damping_speed_points,
            self.quadratic_damping_speed_scales,
            "hydrodynamics.quadratic_damping_speed_scales",
        )
        if (
            self.speed_dependent_damping_enabled
            and len(self.linear_damping_speed_scales) == 0
            and len(self.quadratic_damping_speed_scales) == 0
        ):
            raise ValueError(
                "hydrodynamics requires at least one damping speed scale curve when "
                "speed_dependent_damping_enabled=True."
            )
        validate_6_vector_or_matrix(self.added_mass, "hydrodynamics.added_mass")
        validate_nonnegative(self.added_mass_inertia_scale, "hydrodynamics.added_mass_inertia_scale")
        if not 0.0 <= float(self.added_mass_accel_filter_alpha) <= 1.0:
            raise ValueError("hydrodynamics.added_mass_accel_filter_alpha must be in [0, 1].")
        validate_vector(self.water_current_w, 3, "hydrodynamics.water_current_w")
        validate_vector(
            self.water_current_periodic_amplitude_w,
            3,
            "hydrodynamics.water_current_periodic_amplitude_w",
        )
        validate_vector(
            self.water_current_periodic_period_s,
            3,
            "hydrodynamics.water_current_periodic_period_s",
        )
        for index, period in enumerate(self.water_current_periodic_period_s):
            validate_positive(period, f"hydrodynamics.water_current_periodic_period_s[{index}]")
        validate_vector(
            self.water_current_periodic_phase_rad,
            3,
            "hydrodynamics.water_current_periodic_phase_rad",
        )
        validate_vector(self.water_current_field_bounds, 6, "hydrodynamics.water_current_field_bounds")
        if not (
            self.water_current_field_bounds[0] < self.water_current_field_bounds[1]
            and self.water_current_field_bounds[2] < self.water_current_field_bounds[3]
            and self.water_current_field_bounds[4] < self.water_current_field_bounds[5]
        ):
            raise ValueError("hydrodynamics.water_current_field_bounds must be min < max on each axis.")
        validate_vector(self.water_current_field_shape, 3, "hydrodynamics.water_current_field_shape")
        shape = tuple(int(item) for item in self.water_current_field_shape)
        if any(item <= 0 or item != raw for item, raw in zip(shape, self.water_current_field_shape)):
            raise ValueError("hydrodynamics.water_current_field_shape must contain positive integers.")
        current_count = count_current_vectors(
            self.water_current_field_values,
            "hydrodynamics.water_current_field_values",
        )
        if self.water_current_field_enabled or current_count > 0:
            expected_count = shape[0] * shape[1] * shape[2]
            if current_count != expected_count:
                raise ValueError(
                    "hydrodynamics.water_current_field_values must contain "
                    f"{expected_count} vectors for grid shape {shape}, got {current_count}."
                )

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "linear_damping": self.linear_damping,
            "quadratic_damping": self.quadratic_damping,
            "speed_dependent_damping_enabled": self.speed_dependent_damping_enabled,
            "damping_speed_points": self.damping_speed_points,
            "linear_damping_speed_scales": self.linear_damping_speed_scales,
            "quadratic_damping_speed_scales": self.quadratic_damping_speed_scales,
            "added_mass_diag": self.added_mass,
            "added_mass_inertia_scale": self.added_mass_inertia_scale,
            "added_mass_accel_filter_alpha": self.added_mass_accel_filter_alpha,
            "water_current_w": self.water_current_w,
            "water_current_periodic_enabled": self.water_current_periodic_enabled,
            "water_current_periodic_amplitude_w": self.water_current_periodic_amplitude_w,
            "water_current_periodic_period_s": self.water_current_periodic_period_s,
            "water_current_periodic_phase_rad": self.water_current_periodic_phase_rad,
            "water_current_field_enabled": self.water_current_field_enabled,
            "water_current_field_bounds": self.water_current_field_bounds,
            "water_current_field_shape": self.water_current_field_shape,
            "water_current_field_values": self.water_current_field_values,
        }


@dataclass(frozen=True)
class PoolBoundaryProfile:
    enabled: bool = False
    bounds: NumberSequence = (-7.0, 7.0, -7.0, 7.0, -15.0, -1.0)
    effect_distance: float = 0.75
    damping_scale_at_boundary: float = 1.5
    added_mass_scale_at_boundary: float = 1.2
    thrust_scale_at_boundary: float = 0.85

    def validate(self) -> None:
        validate_vector(self.bounds, 6, "pool_boundary.bounds")
        if not (
            self.bounds[0] < self.bounds[1]
            and self.bounds[2] < self.bounds[3]
            and self.bounds[4] < self.bounds[5]
        ):
            raise ValueError("pool_boundary.bounds must be ordered as min < max on each axis.")
        validate_positive(self.effect_distance, "pool_boundary.effect_distance")
        validate_positive(self.damping_scale_at_boundary, "pool_boundary.damping_scale_at_boundary")
        validate_positive(self.added_mass_scale_at_boundary, "pool_boundary.added_mass_scale_at_boundary")
        validate_nonnegative(self.thrust_scale_at_boundary, "pool_boundary.thrust_scale_at_boundary")

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "pool_boundary_effects_enabled": self.enabled,
            "pool_bounds": self.bounds,
            "pool_boundary_effect_distance": self.effect_distance,
            "pool_boundary_damping_scale": self.damping_scale_at_boundary,
            "pool_boundary_added_mass_scale": self.added_mass_scale_at_boundary,
            "pool_boundary_thrust_scale": self.thrust_scale_at_boundary,
        }


@dataclass(frozen=True)
class FreeSurfaceProfile:
    enabled: bool = False
    surface_z: float = -1.0
    effect_distance: float = 0.5
    heave_damping_scale: float = 1.4
    roll_pitch_damping_scale: float = 1.2
    added_mass_scale: float = 1.15
    buoyancy_scale: float = 0.95
    thrust_scale: float = 0.90
    sloshing_enabled: bool = False
    sloshing_pool_bounds: NumberSequence = (-7.0, 7.0, -7.0, 7.0)
    sloshing_water_depth: float = 14.0
    sloshing_mode_numbers: Sequence[Any] = ((1, 0),)
    sloshing_amplitudes_m: NumberSequence = (0.0,)
    sloshing_phases_rad: NumberSequence = (0.0,)
    sloshing_depth_axis_sign: float = -1.0

    def validate(self) -> None:
        float(self.surface_z)
        validate_positive(self.effect_distance, "free_surface.effect_distance")
        validate_positive(self.heave_damping_scale, "free_surface.heave_damping_scale")
        validate_positive(self.roll_pitch_damping_scale, "free_surface.roll_pitch_damping_scale")
        validate_positive(self.added_mass_scale, "free_surface.added_mass_scale")
        validate_nonnegative(self.buoyancy_scale, "free_surface.buoyancy_scale")
        validate_nonnegative(self.thrust_scale, "free_surface.thrust_scale")
        validate_vector(self.sloshing_pool_bounds, 4, "free_surface.sloshing_pool_bounds")
        if not (
            float(self.sloshing_pool_bounds[0]) < float(self.sloshing_pool_bounds[1])
            and float(self.sloshing_pool_bounds[2]) < float(self.sloshing_pool_bounds[3])
        ):
            raise ValueError("free_surface.sloshing_pool_bounds must be ordered min < max.")
        validate_positive(self.sloshing_water_depth, "free_surface.sloshing_water_depth")
        if not is_sequence(self.sloshing_mode_numbers) or len(self.sloshing_mode_numbers) == 0:
            raise ValueError("free_surface.sloshing_mode_numbers must contain at least one (m, n) pair.")
        for index, mode in enumerate(self.sloshing_mode_numbers):
            validate_vector(mode, 2, f"free_surface.sloshing_mode_numbers[{index}]")
            if any(int(value) != value or int(value) < 0 for value in mode) or sum(
                int(value) for value in mode
            ) == 0:
                raise ValueError(
                    "free_surface sloshing modes require non-negative integers with m + n > 0."
                )
        validate_vector(
            self.sloshing_amplitudes_m,
            len(self.sloshing_mode_numbers),
            "free_surface.sloshing_amplitudes_m",
        )
        validate_vector(
            self.sloshing_phases_rad,
            len(self.sloshing_mode_numbers),
            "free_surface.sloshing_phases_rad",
        )
        if any(float(value) < 0.0 for value in self.sloshing_amplitudes_m):
            raise ValueError("free_surface.sloshing_amplitudes_m must be non-negative.")
        if sum(float(value) for value in self.sloshing_amplitudes_m) >= float(self.sloshing_water_depth):
            raise ValueError("Total sloshing amplitude must be smaller than water depth.")
        if float(self.sloshing_depth_axis_sign) not in (-1.0, 1.0):
            raise ValueError("free_surface.sloshing_depth_axis_sign must be -1 or 1.")

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "free_surface_effects_enabled": self.enabled,
            "free_surface_z": self.surface_z,
            "free_surface_effect_distance": self.effect_distance,
            "free_surface_heave_damping_scale": self.heave_damping_scale,
            "free_surface_roll_pitch_damping_scale": self.roll_pitch_damping_scale,
            "free_surface_added_mass_scale": self.added_mass_scale,
            "free_surface_buoyancy_scale": self.buoyancy_scale,
            "free_surface_thrust_scale": self.thrust_scale,
            "free_surface_sloshing_enabled": self.sloshing_enabled,
            "free_surface_sloshing_pool_bounds": self.sloshing_pool_bounds,
            "free_surface_sloshing_water_depth": self.sloshing_water_depth,
            "free_surface_sloshing_mode_numbers": self.sloshing_mode_numbers,
            "free_surface_sloshing_amplitudes_m": self.sloshing_amplitudes_m,
            "free_surface_sloshing_phases_rad": self.sloshing_phases_rad,
            "free_surface_sloshing_depth_axis_sign": self.sloshing_depth_axis_sign,
        }


@dataclass(frozen=True)
class EnvironmentProfile:
    """Water, hydrodynamics, and pool effects consumed by simulator adapters."""

    name: str = "nominal-pool-environment"
    description: str = "Neutral simulator-independent pool environment."
    hydrodynamics: HydrodynamicsProfile = field(default_factory=HydrodynamicsProfile)
    pool_boundary: PoolBoundaryProfile = field(default_factory=PoolBoundaryProfile)
    free_surface: FreeSurfaceProfile = field(default_factory=FreeSurfaceProfile)

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("environment profile name must be a non-empty string.")
        self.hydrodynamics.validate()
        self.pool_boundary.validate()
        self.free_surface.validate()

    def to_cfg_updates(self) -> dict[str, Any]:
        """Flatten environment-owned fields for a simulator adapter."""

        self.validate()
        updates: dict[str, Any] = {}
        for section in (self.hydrodynamics, self.pool_boundary, self.free_surface):
            updates.update(section.to_cfg_updates())
        return updates


def _section_from_mapping(cls: type, data: Any, section_name: str) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError(f"{section_name} must be a mapping.")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section_name} field(s): {', '.join(unknown)}.")
    return cls(**{item.name: data[item.name] for item in fields(cls) if item.name in data})


def environment_profile_from_dict(data: Mapping[str, Any]) -> EnvironmentProfile:
    """Build a strict environment profile and reject robot/task sections."""

    if not isinstance(data, Mapping):
        raise TypeError("Environment profile data must be a mapping.")
    section_types = {
        "hydrodynamics": HydrodynamicsProfile,
        "pool_boundary": PoolBoundaryProfile,
        "free_surface": FreeSurfaceProfile,
    }
    allowed = {"name", "description", *section_types}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            "Environment profiles may contain only water and pool physics; "
            f"unknown field(s): {', '.join(unknown)}."
        )
    kwargs: dict[str, Any] = {
        key: data[key]
        for key in ("name", "description")
        if key in data
    }
    for section_name, section_type in section_types.items():
        if section_name in data:
            kwargs[section_name] = _section_from_mapping(
                section_type,
                data[section_name],
                section_name,
            )
    profile = EnvironmentProfile(**kwargs)
    profile.validate()
    return profile


def load_environment_profile_json(path: str | Path) -> EnvironmentProfile:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant {value!r} is not allowed.")
            ),
        )
    return environment_profile_from_dict(data)


def resolve_environment_profile(
    value: EnvironmentProfile | str | Path,
) -> EnvironmentProfile:
    if isinstance(value, EnvironmentProfile):
        value.validate()
        return value
    return load_environment_profile_json(value)


def write_environment_profile_json(
    profile: EnvironmentProfile,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    profile.validate()
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(asdict(profile), stream, allow_nan=False, indent=indent, sort_keys=True)
        stream.write("\n")

"""Strict simulator-independent profile for water and pool physics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from common.schema import (
    FiniteNumber,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    StrictBoolean,
    StrictFrozenModel,
    Vector3,
    Vector4,
    Vector6,
)


Matrix6 = Annotated[tuple[Vector6, ...], Field(min_length=6, max_length=6)]
HydroCoefficients = Vector6 | Matrix6
DampingScaleCurve = tuple[FiniteNumber, ...] | tuple[Vector6, ...]
CurrentFieldValues = (
    tuple[Vector3, ...]
    | tuple[tuple[tuple[Vector3, ...], ...], ...]
)


class HydrodynamicsProfile(StrictFrozenModel):
    """Hydrodynamic coefficients and deterministic water-current models."""

    fluid_density_kg_m3: PositiveFloat
    linear_damping: HydroCoefficients
    quadratic_damping: HydroCoefficients
    speed_dependent_damping_enabled: StrictBoolean
    damping_speed_points: Annotated[tuple[FiniteNumber, ...], Field(min_length=2)]
    linear_damping_speed_scales: DampingScaleCurve
    quadratic_damping_speed_scales: DampingScaleCurve
    added_mass: HydroCoefficients
    water_current_w: Vector3
    water_current_periodic_enabled: StrictBoolean
    water_current_periodic_amplitude_w: Vector3
    water_current_periodic_period_s: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    water_current_periodic_phase_rad: Vector3
    water_current_field_enabled: StrictBoolean
    water_current_field_bounds: Vector6
    water_current_field_shape: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    water_current_field_values: CurrentFieldValues

    @model_validator(mode="after")
    def validate_physics(self) -> "HydrodynamicsProfile":
        if any(right <= left for left, right in zip(self.damping_speed_points, self.damping_speed_points[1:])):
            raise ValueError("hydrodynamics.damping_speed_points must be strictly increasing.")
        for name, scales in (
            ("linear_damping_speed_scales", self.linear_damping_speed_scales),
            ("quadratic_damping_speed_scales", self.quadratic_damping_speed_scales),
        ):
            if scales and len(scales) != len(self.damping_speed_points):
                raise ValueError(
                    f"hydrodynamics.{name} must have one sample per damping_speed_points entry."
                )
            flat = np.asarray(scales, dtype=float)
            if flat.size and np.any(flat < 0.0):
                raise ValueError(f"hydrodynamics.{name} must be non-negative.")
        if (
            self.speed_dependent_damping_enabled
            and not self.linear_damping_speed_scales
            and not self.quadratic_damping_speed_scales
        ):
            raise ValueError(
                "hydrodynamics requires at least one damping speed scale curve when "
                "speed_dependent_damping_enabled=True."
            )

        added_mass = np.asarray(self.added_mass, dtype=float)
        if added_mass.shape == (6, 6):
            if not np.allclose(added_mass, added_mass.T, atol=1.0e-8, rtol=0.0):
                raise ValueError("hydrodynamics.added_mass must satisfy reciprocity (M_A=M_A^T).")
            if float(np.linalg.eigvalsh(added_mass)[0]) < -1.0e-8:
                raise ValueError("hydrodynamics.added_mass must be positive semidefinite.")

        bounds = self.water_current_field_bounds
        if not (bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]):
            raise ValueError("hydrodynamics.water_current_field_bounds must be min < max on each axis.")
        if any(value <= 0 for value in self.water_current_field_shape):
            raise ValueError("hydrodynamics.water_current_field_shape must contain positive integers.")
        current_values = np.asarray(self.water_current_field_values, dtype=float)
        if current_values.size and not np.all(np.isfinite(current_values)):
            raise ValueError("hydrodynamics.water_current_field_values must contain finite values.")
        if self.water_current_field_enabled or current_values.size:
            shape = self.water_current_field_shape
            expected_count = math.prod(shape)
            if current_values.shape not in {(expected_count, 3), (*shape, 3)}:
                raise ValueError(
                    "hydrodynamics.water_current_field_values must have shape "
                    f"({expected_count}, 3) or {(*shape, 3)}, got {current_values.shape}."
                )
        return self

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "water_rho": self.fluid_density_kg_m3,
            "linear_damping": self.linear_damping,
            "quadratic_damping": self.quadratic_damping,
            "speed_dependent_damping_enabled": self.speed_dependent_damping_enabled,
            "damping_speed_points": self.damping_speed_points,
            "linear_damping_speed_scales": self.linear_damping_speed_scales,
            "quadratic_damping_speed_scales": self.quadratic_damping_speed_scales,
            "added_mass": self.added_mass,
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


class PoolBoundaryProfile(StrictFrozenModel):
    enabled: StrictBoolean
    bounds: Vector6
    effect_distance: PositiveFloat
    damping_scale_at_boundary: PositiveFloat
    added_mass_scale_at_boundary: PositiveFloat
    thrust_scale_at_boundary: NonNegativeFloat

    @model_validator(mode="after")
    def validate_bounds(self) -> "PoolBoundaryProfile":
        if not (
            self.bounds[0] < self.bounds[1]
            and self.bounds[2] < self.bounds[3]
            and self.bounds[4] < self.bounds[5]
        ):
            raise ValueError("pool_boundary.bounds must be ordered as min < max on each axis.")
        return self

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "pool_boundary_effects_enabled": self.enabled,
            "pool_bounds": self.bounds,
            "pool_boundary_effect_distance": self.effect_distance,
            "pool_boundary_damping_scale": self.damping_scale_at_boundary,
            "pool_boundary_added_mass_scale": self.added_mass_scale_at_boundary,
            "pool_boundary_thrust_scale": self.thrust_scale_at_boundary,
        }


class FreeSurfaceProfile(StrictFrozenModel):
    enabled: StrictBoolean
    surface_z: FiniteNumber
    effect_distance: PositiveFloat
    heave_damping_scale: PositiveFloat
    roll_pitch_damping_scale: PositiveFloat
    added_mass_scale: PositiveFloat
    buoyancy_scale: NonNegativeFloat
    thrust_scale: NonNegativeFloat
    sloshing_enabled: StrictBoolean
    sloshing_pool_bounds: Vector4
    sloshing_water_depth: PositiveFloat
    sloshing_mode_numbers: Annotated[
        tuple[tuple[NonNegativeInt, NonNegativeInt], ...],
        Field(min_length=1),
    ]
    sloshing_amplitudes_m: tuple[NonNegativeFloat, ...]
    sloshing_phases_rad: tuple[FiniteNumber, ...]
    sloshing_depth_axis_sign: Literal[-1.0, 1.0]

    @model_validator(mode="after")
    def validate_sloshing(self) -> "FreeSurfaceProfile":
        bounds = self.sloshing_pool_bounds
        if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
            raise ValueError("free_surface.sloshing_pool_bounds must be ordered min < max.")
        if any(m + n == 0 for m, n in self.sloshing_mode_numbers):
            raise ValueError("free_surface sloshing modes require m + n > 0.")
        mode_count = len(self.sloshing_mode_numbers)
        if len(self.sloshing_amplitudes_m) != mode_count or len(self.sloshing_phases_rad) != mode_count:
            raise ValueError("free_surface sloshing amplitudes and phases must match the mode count.")
        if sum(self.sloshing_amplitudes_m) >= self.sloshing_water_depth:
            raise ValueError("Total sloshing amplitude must be smaller than water depth.")
        return self

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


class EnvironmentProfile(StrictFrozenModel):
    """Water, hydrodynamics, and pool effects consumed by simulator adapters."""

    name: Annotated[str, Field(min_length=1)]
    description: str
    hydrodynamics: HydrodynamicsProfile
    pool_boundary: PoolBoundaryProfile
    free_surface: FreeSurfaceProfile

    @model_validator(mode="after")
    def validate_name(self) -> "EnvironmentProfile":
        if not self.name.strip():
            raise ValueError("environment profile name must be non-empty.")
        return self

    def to_cfg_updates(self) -> dict[str, Any]:
        """Flatten environment-owned fields for a simulator adapter."""

        updates: dict[str, Any] = {}
        for section in (self.hydrodynamics, self.pool_boundary, self.free_surface):
            updates.update(section.to_cfg_updates())
        return updates


def load_environment_profile_json(path: str | Path) -> EnvironmentProfile:
    return EnvironmentProfile.model_validate_json(Path(path).read_bytes())


def resolve_environment_profile(
    value: EnvironmentProfile | str | Path,
) -> EnvironmentProfile:
    return value if isinstance(value, EnvironmentProfile) else load_environment_profile_json(value)


def write_environment_profile_json(
    profile: EnvironmentProfile,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(profile.model_dump_json(indent=indent) + "\n", encoding="utf-8")

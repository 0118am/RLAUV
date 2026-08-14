"""Structured calibration profiles for high-fidelity pool simulations.

The profile objects in this module are intentionally independent from
IsaacLab.  They collect measured or literature-derived parameters in one
place, validate the shapes that the AUV environment expects, and can apply
those values to any config object with matching attributes.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Sequence

from robot.dynamics.parameters import AUV


NumberSequence = Sequence[float]
HydroCoefficients = Sequence[float] | Sequence[Sequence[float]]
AuditSeverity = str
ZERO_HYDRODYNAMIC_COEFFICIENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


def _as_plain_value(value: Any) -> Any:
    """Return lists/scalars that are friendly to IsaacLab config classes."""

    if isinstance(value, dict):
        return {key: _as_plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_as_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_as_plain_value(item) for item in value]
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _validate_length(value: Sequence[Any], length: int, name: str) -> None:
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {len(value)}.")


def _validate_nonnegative(value: float, name: str) -> None:
    if _finite_float(value, name) < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive(value: float, name: str) -> None:
    if _finite_float(value, name) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_range(value: Sequence[float], name: str, *, integer: bool = False) -> None:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a two-value sequence.")
    _validate_length(value, 2, name)
    lower = _finite_float(value[0], f"{name}[0]")
    upper = _finite_float(value[1], f"{name}[1]")
    if upper < lower:
        raise ValueError(f"{name} upper bound must be >= lower bound.")
    if integer and (int(value[0]) != value[0] or int(value[1]) != value[1]):
        raise ValueError(f"{name} must contain integer values.")


def _validate_nonnegative_sequence(value: Sequence[float], name: str) -> None:
    if not _is_sequence(value) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty sequence.")
    for index, item in enumerate(value):
        if _is_sequence(item):
            raise ValueError(f"{name}[{index}] must be a scalar.")
        _validate_nonnegative(float(item), f"{name}[{index}]")


def _validate_integer_sequence(value: Sequence[int], name: str, *, nonnegative: bool = False) -> None:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a sequence.")
    previous = None
    for index, item in enumerate(value):
        _finite_float(item, f"{name}[{index}]")
        if int(item) != item:
            raise ValueError(f"{name}[{index}] must be an integer.")
        if nonnegative and int(item) < 0:
            raise ValueError(f"{name}[{index}] must be non-negative.")
        if previous is not None and int(item) <= previous:
            raise ValueError(f"{name} must be strictly increasing.")
        previous = int(item)


def _validate_vector(value: Sequence[Any], length: int, name: str) -> None:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a sequence.")
    _validate_length(value, length, name)
    for index, item in enumerate(value):
        if _is_sequence(item):
            raise ValueError(f"{name}[{index}] must be a scalar.")
        _finite_float(item, f"{name}[{index}]")


def _validate_scalar_or_vector(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} group keys must be strings.")
            _validate_scalar_or_vector(item, f"{name}.{key}")
        return
    if _is_sequence(value):
        for index, item in enumerate(value):
            if _is_sequence(item):
                raise ValueError(f"{name}[{index}] must be a scalar.")
            _finite_float(item, f"{name}[{index}]")
    else:
        _finite_float(value, name)


def _validate_scalar_or_vector_bounds(
    value: Any,
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} group keys must be strings.")
            _validate_scalar_or_vector_bounds(item, f"{name}.{key}", lower=lower, upper=upper)
        return
    if _is_sequence(value):
        for index, item in enumerate(value):
            if _is_sequence(item):
                raise ValueError(f"{name}[{index}] must be a scalar.")
            _validate_scalar_bounds(float(item), f"{name}[{index}]", lower=lower, upper=upper)
    else:
        _validate_scalar_bounds(float(value), name, lower=lower, upper=upper)


def _validate_scalar_bounds(value: float, name: str, *, lower: float | None, upper: float | None) -> None:
    value = _finite_float(value, name)
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be >= {lower}.")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}.")


def _count_current_vectors(value: Any, name: str) -> int:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a nested sequence of 3D current vectors.")
    if len(value) == 0:
        return 0
    if all(not _is_sequence(item) for item in value):
        _validate_vector(value, 3, name)
        return 1
    return sum(_count_current_vectors(item, f"{name}[]") for item in value)


def _validate_6_vector_or_matrix(value: HydroCoefficients, name: str) -> None:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a 6-vector or 6x6 matrix.")
    _validate_length(value, 6, name)

    first = value[0]
    if _is_sequence(first):
        for row_index, row in enumerate(value):
            if not _is_sequence(row):
                raise ValueError(f"{name}[{row_index}] must be a 6-value row.")
            _validate_vector(row, 6, f"{name}[{row_index}]")
    else:
        _validate_vector(value, 6, name)


def _validate_inertia_tensor(value: Any, name: str) -> None:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a 3-vector, 3x3 matrix, or flat 9-value matrix.")
    if len(value) == 3 and all(not _is_sequence(item) for item in value):
        _validate_vector(value, 3, name)
        if any(float(item) <= 0.0 for item in value):
            raise ValueError(f"{name} diagonal entries must be positive.")
        moments = [float(item) for item in value]
        if any(moment > sum(moments) - moment + 1.0e-9 for moment in moments):
            raise ValueError(f"{name} must satisfy the rigid-body inertia triangle inequalities.")
        return
    if len(value) == 9 and all(not _is_sequence(item) for item in value):
        rows = [value[0:3], value[3:6], value[6:9]]
    elif len(value) == 3 and all(_is_sequence(item) for item in value):
        rows = value
        for row_index, row in enumerate(rows):
            _validate_vector(row, 3, f"{name}[{row_index}]")
    else:
        raise ValueError(f"{name} must be a 3-vector, 3x3 matrix, or flat 9-value matrix.")

    for index in range(3):
        if float(rows[index][index]) <= 0.0:
            raise ValueError(f"{name} diagonal entries must be positive.")
    for row in range(3):
        for col in range(row + 1, 3):
            if abs(float(rows[row][col]) - float(rows[col][row])) > 1.0e-6:
                raise ValueError(f"{name} must be symmetric.")
    a, b, c = (float(item) for item in rows[0])
    _, d, e = (float(item) for item in rows[1])
    _, _, f = (float(item) for item in rows[2])
    leading_minor_2 = a * d - b * b
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    if leading_minor_2 <= 0.0 or determinant <= 0.0:
        raise ValueError(f"{name} must be positive definite.")
    diagonal = (a, d, f)
    if any(moment > sum(diagonal) - moment + 1.0e-9 for moment in diagonal):
        raise ValueError(f"{name} must satisfy the rigid-body inertia triangle inequalities.")


@dataclass(frozen=True)
class RigidBodyProfile:
    mass: float = AUV.mass_kg
    volume: float = AUV.displaced_volume_m3
    inertia_diag: NumberSequence = field(
        default_factory=lambda: [list(row) for row in AUV.inertia_tensor_body_kg_m2]
    )
    center_of_mass_offset: NumberSequence = field(default_factory=lambda: AUV.center_of_mass_offset_m)
    com_to_cob_offset: NumberSequence = field(default_factory=lambda: AUV.center_of_buoyancy_from_com_m)
    water_rho: float = AUV.water_density_kg_m3
    water_beta: float = 0.001306

    def validate(self) -> None:
        _validate_positive(self.mass, "rigid_body.mass")
        _validate_positive(self.volume, "rigid_body.volume")
        _validate_inertia_tensor(self.inertia_diag, "rigid_body.inertia_diag")
        _validate_vector(self.center_of_mass_offset, 3, "rigid_body.center_of_mass_offset")
        _validate_vector(self.com_to_cob_offset, 3, "rigid_body.com_to_cob_offset")
        _validate_positive(self.water_rho, "rigid_body.water_rho")
        _validate_positive(self.water_beta, "rigid_body.water_beta")

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "mass": self.mass,
            "volume": self.volume,
            "inertia_diag": self.inertia_diag,
            "center_of_mass_offset": self.center_of_mass_offset,
            "com_to_cob_offset": self.com_to_cob_offset,
            "water_rho": self.water_rho,
            "water_beta": self.water_beta,
        }


@dataclass(frozen=True)
class HydrodynamicsProfile:
    linear_damping: HydroCoefficients = ZERO_HYDRODYNAMIC_COEFFICIENTS
    quadratic_damping: HydroCoefficients = ZERO_HYDRODYNAMIC_COEFFICIENTS
    speed_dependent_damping_enabled: bool = False
    damping_speed_points: NumberSequence = (0.0, 1.0)
    linear_damping_speed_scales: Sequence[Any] = field(default_factory=tuple)
    quadratic_damping_speed_scales: Sequence[Any] = field(default_factory=tuple)
    added_mass: HydroCoefficients = ZERO_HYDRODYNAMIC_COEFFICIENTS
    added_mass_inertia_scale: float = 1.0
    added_mass_accel_filter_alpha: float = 0.35
    # Residual augmentation is opt-in. Populate these from independent
    # residual-wrench data; each factor L is evaluated as L @ L.T to preserve
    # passivity/PSD added mass.
    high_order_residual_enabled: bool = False
    high_order_residual_added_mass_factor: HydroCoefficients = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    high_order_residual_linear_damping_factor: HydroCoefficients = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    high_order_residual_quadratic_damping_factor: HydroCoefficients = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    high_order_residual_cubic_damping_factor: HydroCoefficients = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    water_current_w: NumberSequence = (0.0, 0.0, 0.0)
    # Optional deterministic pool-cycle model.  The runtime evaluates each
    # world-axis component as A * sin(2*pi*t/T + phase).
    water_current_periodic_enabled: bool = False
    water_current_periodic_amplitude_w: NumberSequence = (0.0, 0.0, 0.0)
    water_current_periodic_period_s: NumberSequence = (20.0, 20.0, 20.0)
    water_current_periodic_phase_rad: NumberSequence = (0.0, 0.0, 0.0)
    water_current_field_enabled: bool = False
    water_current_field_bounds: NumberSequence = (-7.0, 7.0, -7.0, 7.0, -15.0, -1.0)
    water_current_field_shape: Sequence[int] = (1, 1, 1)
    water_current_field_values: Sequence[Any] = field(default_factory=tuple)

    def validate(self) -> None:
        _validate_6_vector_or_matrix(self.linear_damping, "hydrodynamics.linear_damping")
        _validate_6_vector_or_matrix(self.quadratic_damping, "hydrodynamics.quadratic_damping")
        _validate_damping_speed_scale_curve(
            self.damping_speed_points,
            self.linear_damping_speed_scales,
            "hydrodynamics.linear_damping_speed_scales",
        )
        _validate_damping_speed_scale_curve(
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
        _validate_6_vector_or_matrix(self.added_mass, "hydrodynamics.added_mass")
        _validate_nonnegative(self.added_mass_inertia_scale, "hydrodynamics.added_mass_inertia_scale")
        if not 0.0 <= float(self.added_mass_accel_filter_alpha) <= 1.0:
            raise ValueError("hydrodynamics.added_mass_accel_filter_alpha must be in [0, 1].")
        _validate_6_vector_or_matrix(
            self.high_order_residual_added_mass_factor,
            "hydrodynamics.high_order_residual_added_mass_factor",
        )
        _validate_6_vector_or_matrix(
            self.high_order_residual_linear_damping_factor,
            "hydrodynamics.high_order_residual_linear_damping_factor",
        )
        _validate_6_vector_or_matrix(
            self.high_order_residual_quadratic_damping_factor,
            "hydrodynamics.high_order_residual_quadratic_damping_factor",
        )
        _validate_6_vector_or_matrix(
            self.high_order_residual_cubic_damping_factor,
            "hydrodynamics.high_order_residual_cubic_damping_factor",
        )
        _validate_vector(self.water_current_w, 3, "hydrodynamics.water_current_w")
        _validate_vector(
            self.water_current_periodic_amplitude_w,
            3,
            "hydrodynamics.water_current_periodic_amplitude_w",
        )
        _validate_vector(
            self.water_current_periodic_period_s,
            3,
            "hydrodynamics.water_current_periodic_period_s",
        )
        for index, period in enumerate(self.water_current_periodic_period_s):
            _validate_positive(period, f"hydrodynamics.water_current_periodic_period_s[{index}]")
        _validate_vector(
            self.water_current_periodic_phase_rad,
            3,
            "hydrodynamics.water_current_periodic_phase_rad",
        )
        _validate_vector(self.water_current_field_bounds, 6, "hydrodynamics.water_current_field_bounds")
        if not (
            self.water_current_field_bounds[0] < self.water_current_field_bounds[1]
            and self.water_current_field_bounds[2] < self.water_current_field_bounds[3]
            and self.water_current_field_bounds[4] < self.water_current_field_bounds[5]
        ):
            raise ValueError("hydrodynamics.water_current_field_bounds must be min < max on each axis.")
        _validate_vector(self.water_current_field_shape, 3, "hydrodynamics.water_current_field_shape")
        shape = tuple(int(item) for item in self.water_current_field_shape)
        if any(item <= 0 or item != raw for item, raw in zip(shape, self.water_current_field_shape)):
            raise ValueError("hydrodynamics.water_current_field_shape must contain positive integers.")
        current_count = _count_current_vectors(
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
            "high_order_residual_enabled": self.high_order_residual_enabled,
            "high_order_residual_added_mass_factor": self.high_order_residual_added_mass_factor,
            "high_order_residual_linear_damping_factor": self.high_order_residual_linear_damping_factor,
            "high_order_residual_quadratic_damping_factor": self.high_order_residual_quadratic_damping_factor,
            "high_order_residual_cubic_damping_factor": self.high_order_residual_cubic_damping_factor,
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
class ThrusterProfile:
    dyn_time_constant: float = 0.0
    command_delay_steps: int = 0
    max_command_rate: float = 0.0
    command_resolution: float = 0.0
    command_dropout_probability: float = 0.0
    inflow_loss_enabled: bool = False
    inflow_loss_coefficient: float = 0.25
    inflow_reference_speed: float = 1.0
    inflow_min_scale: float = 0.5
    wake_interaction_enabled: bool = False
    wake_loss_coefficient: float = 0.10
    wake_length: float = 0.6
    wake_radius: float = 0.08
    wake_expansion_rate: float = 0.15
    wake_min_scale: float = 0.7
    reaction_torque_coeff: float = 0.0
    spin_directions: NumberSequence = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0)

    def validate(self) -> None:
        _validate_nonnegative(self.dyn_time_constant, "thrusters.dyn_time_constant")
        if int(self.command_delay_steps) != self.command_delay_steps or self.command_delay_steps < 0:
            raise ValueError("thrusters.command_delay_steps must be a non-negative integer.")
        _validate_nonnegative(self.max_command_rate, "thrusters.max_command_rate")
        _validate_nonnegative(self.command_resolution, "thrusters.command_resolution")
        if not 0.0 <= float(self.command_dropout_probability) <= 1.0:
            raise ValueError("thrusters.command_dropout_probability must be in [0, 1].")
        _validate_nonnegative(self.inflow_loss_coefficient, "thrusters.inflow_loss_coefficient")
        _validate_positive(self.inflow_reference_speed, "thrusters.inflow_reference_speed")
        if not 0.0 <= float(self.inflow_min_scale) <= 1.0:
            raise ValueError("thrusters.inflow_min_scale must be in [0, 1].")
        _validate_nonnegative(self.wake_loss_coefficient, "thrusters.wake_loss_coefficient")
        _validate_positive(self.wake_length, "thrusters.wake_length")
        _validate_positive(self.wake_radius, "thrusters.wake_radius")
        _validate_nonnegative(self.wake_expansion_rate, "thrusters.wake_expansion_rate")
        if not 0.0 <= float(self.wake_min_scale) <= 1.0:
            raise ValueError("thrusters.wake_min_scale must be in [0, 1].")
        _validate_nonnegative(self.reaction_torque_coeff, "thrusters.reaction_torque_coeff")
        _validate_vector(self.spin_directions, 8, "thrusters.spin_directions")

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "dyn_time_constant": self.dyn_time_constant,
            "thruster_command_delay_steps": int(self.command_delay_steps),
            "thruster_max_command_rate": self.max_command_rate,
            "thruster_command_resolution": self.command_resolution,
            "thruster_command_dropout_probability": self.command_dropout_probability,
            "thruster_inflow_loss_enabled": self.inflow_loss_enabled,
            "thruster_inflow_loss_coefficient": self.inflow_loss_coefficient,
            "thruster_inflow_reference_speed": self.inflow_reference_speed,
            "thruster_inflow_min_scale": self.inflow_min_scale,
            "thruster_wake_interaction_enabled": self.wake_interaction_enabled,
            "thruster_wake_loss_coefficient": self.wake_loss_coefficient,
            "thruster_wake_length": self.wake_length,
            "thruster_wake_radius": self.wake_radius,
            "thruster_wake_expansion_rate": self.wake_expansion_rate,
            "thruster_wake_min_scale": self.wake_min_scale,
            "thruster_reaction_torque_coeff": self.reaction_torque_coeff,
            "thruster_spin_directions": self.spin_directions,
        }


@dataclass(frozen=True)
class BatteryProfile:
    nominal_voltage: float = 16.0
    initial_voltage: float = 16.0
    min_voltage: float = 12.0
    voltage_drop_per_s: float = 0.0
    thrust_exponent: float = 2.0

    def validate(self) -> None:
        _validate_positive(self.nominal_voltage, "battery.nominal_voltage")
        _validate_positive(self.initial_voltage, "battery.initial_voltage")
        _validate_nonnegative(self.min_voltage, "battery.min_voltage")
        _validate_nonnegative(self.voltage_drop_per_s, "battery.voltage_drop_per_s")
        _validate_nonnegative(self.thrust_exponent, "battery.thrust_exponent")

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "battery_voltage_nominal": self.nominal_voltage,
            "battery_voltage": self.initial_voltage,
            "battery_min_voltage": self.min_voltage,
            "battery_voltage_drop_per_s": self.voltage_drop_per_s,
            "battery_voltage_thrust_exponent": self.thrust_exponent,
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
        _validate_vector(self.bounds, 6, "pool_boundary.bounds")
        if not (self.bounds[0] < self.bounds[1] and self.bounds[2] < self.bounds[3] and self.bounds[4] < self.bounds[5]):
            raise ValueError("pool_boundary.bounds must be ordered as min < max on each axis.")
        _validate_positive(self.effect_distance, "pool_boundary.effect_distance")
        _validate_positive(self.damping_scale_at_boundary, "pool_boundary.damping_scale_at_boundary")
        _validate_positive(self.added_mass_scale_at_boundary, "pool_boundary.added_mass_scale_at_boundary")
        _validate_nonnegative(self.thrust_scale_at_boundary, "pool_boundary.thrust_scale_at_boundary")

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
        _validate_positive(self.effect_distance, "free_surface.effect_distance")
        _validate_positive(self.heave_damping_scale, "free_surface.heave_damping_scale")
        _validate_positive(self.roll_pitch_damping_scale, "free_surface.roll_pitch_damping_scale")
        _validate_positive(self.added_mass_scale, "free_surface.added_mass_scale")
        _validate_nonnegative(self.buoyancy_scale, "free_surface.buoyancy_scale")
        _validate_nonnegative(self.thrust_scale, "free_surface.thrust_scale")
        _validate_vector(self.sloshing_pool_bounds, 4, "free_surface.sloshing_pool_bounds")
        if not (
            float(self.sloshing_pool_bounds[0]) < float(self.sloshing_pool_bounds[1])
            and float(self.sloshing_pool_bounds[2]) < float(self.sloshing_pool_bounds[3])
        ):
            raise ValueError("free_surface.sloshing_pool_bounds must be ordered min < max.")
        _validate_positive(self.sloshing_water_depth, "free_surface.sloshing_water_depth")
        if not _is_sequence(self.sloshing_mode_numbers) or len(self.sloshing_mode_numbers) == 0:
            raise ValueError("free_surface.sloshing_mode_numbers must contain at least one (m, n) pair.")
        for index, mode in enumerate(self.sloshing_mode_numbers):
            _validate_vector(mode, 2, f"free_surface.sloshing_mode_numbers[{index}]")
            if any(int(value) != value or int(value) < 0 for value in mode) or sum(int(value) for value in mode) == 0:
                raise ValueError("free_surface sloshing modes require non-negative integers with m + n > 0.")
        _validate_vector(
            self.sloshing_amplitudes_m,
            len(self.sloshing_mode_numbers),
            "free_surface.sloshing_amplitudes_m",
        )
        _validate_vector(
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
class TetherProfile:
    enabled: bool = False
    anchor_pos_w: NumberSequence = (0.0, 0.0, 8.0)
    attach_offset_b: NumberSequence = (-0.2, 0.0, 0.0)
    slack_length: float = 2.0
    stiffness: float = 20.0
    damping: float = 5.0
    drag_coeff: float = 0.0
    winch_enabled: bool = False
    winch_target_length: float = 2.0
    winch_reel_speed: float = 0.0
    winch_min_length: float = 0.0
    winch_max_length: float = 20.0
    num_segments: int = 1
    segment_diameter: float = 0.004
    segment_density: float = 1100.0
    segment_buoyancy_density: float = AUV.water_density_kg_m3

    def validate(self) -> None:
        _validate_vector(self.anchor_pos_w, 3, "tether.anchor_pos_w")
        _validate_vector(self.attach_offset_b, 3, "tether.attach_offset_b")
        _validate_nonnegative(self.slack_length, "tether.slack_length")
        _validate_nonnegative(self.stiffness, "tether.stiffness")
        _validate_nonnegative(self.damping, "tether.damping")
        _validate_nonnegative(self.drag_coeff, "tether.drag_coeff")
        _validate_nonnegative(self.winch_target_length, "tether.winch_target_length")
        _validate_nonnegative(self.winch_reel_speed, "tether.winch_reel_speed")
        _validate_nonnegative(self.winch_min_length, "tether.winch_min_length")
        _validate_nonnegative(self.winch_max_length, "tether.winch_max_length")
        if float(self.winch_max_length) < float(self.winch_min_length):
            raise ValueError("tether.winch_max_length must be >= tether.winch_min_length.")
        if int(self.num_segments) != self.num_segments or self.num_segments < 1:
            raise ValueError("tether.num_segments must be a positive integer.")
        _validate_positive(self.segment_diameter, "tether.segment_diameter")
        _validate_nonnegative(self.segment_density, "tether.segment_density")
        _validate_nonnegative(self.segment_buoyancy_density, "tether.segment_buoyancy_density")

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "tether_enabled": self.enabled,
            "tether_anchor_pos_w": self.anchor_pos_w,
            "tether_attach_offset_b": self.attach_offset_b,
            "tether_slack_length": self.slack_length,
            "tether_stiffness": self.stiffness,
            "tether_damping": self.damping,
            "tether_drag_coeff": self.drag_coeff,
            "tether_winch_enabled": self.winch_enabled,
            "tether_winch_target_length": self.winch_target_length,
            "tether_winch_reel_speed": self.winch_reel_speed,
            "tether_winch_min_length": self.winch_min_length,
            "tether_winch_max_length": self.winch_max_length,
            "tether_num_segments": int(self.num_segments),
            "tether_segment_diameter": self.segment_diameter,
            "tether_segment_density": self.segment_density,
            "tether_segment_buoyancy_density": self.segment_buoyancy_density,
        }


@dataclass(frozen=True)
class ObservationProfile:
    noise_std: float | NumberSequence = 0.0
    bias_range: float | NumberSequence = 0.0
    delay_steps: int = 0
    update_period_steps: int = 1
    dropout_probability: float | NumberSequence = 0.0
    lowpass_alpha: float | NumberSequence = 1.0
    bias_drift_std: float | NumberSequence = 0.0

    def validate(self) -> None:
        _validate_scalar_or_vector_bounds(self.noise_std, "observation.noise_std", lower=0.0)
        _validate_scalar_or_vector_bounds(self.bias_range, "observation.bias_range", lower=0.0)
        if int(self.delay_steps) != self.delay_steps or self.delay_steps < 0:
            raise ValueError("observation.delay_steps must be a non-negative integer.")
        if int(self.update_period_steps) != self.update_period_steps or self.update_period_steps < 1:
            raise ValueError("observation.update_period_steps must be a positive integer.")
        _validate_scalar_or_vector_bounds(
            self.dropout_probability,
            "observation.dropout_probability",
            lower=0.0,
            upper=1.0,
        )
        _validate_scalar_or_vector_bounds(self.lowpass_alpha, "observation.lowpass_alpha", lower=0.0, upper=1.0)
        _validate_scalar_or_vector_bounds(self.bias_drift_std, "observation.bias_drift_std", lower=0.0)

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "observation_noise_std": self.noise_std,
            "observation_bias_range": self.bias_range,
            "observation_delay_steps": int(self.delay_steps),
            "observation_update_period_steps": int(self.update_period_steps),
            "observation_dropout_probability": self.dropout_probability,
            "observation_lowpass_alpha": self.lowpass_alpha,
            "observation_bias_drift_std": self.bias_drift_std,
        }




@dataclass(frozen=True)
class PoolProfileAuditOptions:
    near_boundaries_expected: bool = False
    near_surface_expected: bool = False
    sloshing_expected: bool = False
    tether_expected: bool = False
    spatial_current_expected: bool = False
    domain_randomization_expected: bool = True


@dataclass(frozen=True)
class PoolProfileAuditFinding:
    severity: AuditSeverity
    section: str
    message: str
    recommendation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "section": self.section,
            "message": self.message,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class PoolProfileAuditReport:
    profile_name: str
    findings: tuple[PoolProfileAuditFinding, ...]

    @property
    def counts_by_severity(self) -> dict[str, int]:
        counts = {"critical": 0, "warning": 0, "info": 0}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    @property
    def readiness_score(self) -> float:
        if not self.findings:
            return 1.0
        penalty = 0.0
        for finding in self.findings:
            if finding.severity == "critical":
                penalty += 0.25
            elif finding.severity == "warning":
                penalty += 0.10
            else:
                penalty += 0.03
        return max(0.0, 1.0 - penalty)

    def has_blocking_findings(self) -> bool:
        return any(finding.severity == "critical" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "readiness_score": self.readiness_score,
            "counts_by_severity": self.counts_by_severity,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class PoolCalibrationTask:
    priority: str
    section: str
    severity: AuditSeverity
    title: str
    reason: str
    experiment: str
    calibration_functions: tuple[str, ...]
    update_keys: tuple[str, ...]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "section": self.section,
            "severity": self.severity,
            "title": self.title,
            "reason": self.reason,
            "experiment": self.experiment,
            "calibration_functions": list(self.calibration_functions),
            "update_keys": list(self.update_keys),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class PoolCalibrationLogColumn:
    name: str
    units: str
    description: str
    required: bool = True
    value_type: str = "float"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.units,
            "description": self.description,
            "required": self.required,
            "value_type": self.value_type,
        }


@dataclass(frozen=True)
class PoolCalibrationLogSchema:
    section: str
    dataset_name: str
    filename: str
    description: str
    columns: tuple[PoolCalibrationLogColumn, ...]
    calibration_functions: tuple[str, ...]
    update_keys: tuple[str, ...]
    notes: str = ""
    minimum_rows: int = 1

    @property
    def csv_header(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "dataset_name": self.dataset_name,
            "filename": self.filename,
            "description": self.description,
            "columns": [column.to_dict() for column in self.columns],
            "calibration_functions": list(self.calibration_functions),
            "update_keys": list(self.update_keys),
            "notes": self.notes,
            "minimum_rows": self.minimum_rows,
        }


@dataclass(frozen=True)
class PoolCalibrationLogValidationIssue:
    severity: str
    filename: str
    message: str
    row_number: int | None = None
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "filename": self.filename,
            "message": self.message,
            "row_number": self.row_number,
            "column": self.column,
        }


@dataclass(frozen=True)
class PoolCalibrationLogValidationReport:
    directory: str
    expected_files: tuple[str, ...]
    row_counts: dict[str, int]
    issues: tuple[PoolCalibrationLogValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "expected_files": list(self.expected_files),
            "row_counts": dict(self.row_counts),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _validate_payload_scale(value: Any, name: str) -> None:
    if _is_sequence(value):
        _validate_length(value, 6, name)
        _validate_nonnegative_sequence(value, name)
    else:
        _validate_nonnegative(value, name)


def _validate_payload_samples(value: Sequence[Mapping[str, Any]], name: str) -> None:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a sequence of payload mappings.")
    allowed = {
        "name",
        "weight",
        "mass",
        "volume",
        "inertia",
        "center_of_mass_offset",
        "com_to_cob_offset",
        "linear_damping_scale",
        "quadratic_damping_scale",
        "added_mass_scale",
    }
    required = {
        "mass",
        "volume",
        "inertia",
        "center_of_mass_offset",
        "com_to_cob_offset",
    }
    names: set[str] = set()
    for index, sample in enumerate(value):
        sample_name = f"{name}[{index}]"
        if not isinstance(sample, Mapping):
            raise ValueError(f"{sample_name} must be a mapping.")
        unknown = sorted(set(sample) - allowed)
        if unknown:
            raise ValueError(f"{sample_name} contains unknown field(s): {', '.join(unknown)}.")
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"{sample_name} is missing field(s): {', '.join(missing)}.")
        label = str(sample.get("name", f"payload-{index}"))
        if not label.strip() or label in names:
            raise ValueError(f"{sample_name}.name must be non-empty and unique.")
        names.add(label)
        _validate_positive(sample.get("weight", 1.0), f"{sample_name}.weight")
        _validate_positive(sample["mass"], f"{sample_name}.mass")
        _validate_positive(sample["volume"], f"{sample_name}.volume")
        _validate_inertia_tensor(sample["inertia"], f"{sample_name}.inertia")
        _validate_vector(sample["center_of_mass_offset"], 3, f"{sample_name}.center_of_mass_offset")
        _validate_vector(sample["com_to_cob_offset"], 3, f"{sample_name}.com_to_cob_offset")
        for scale_name in ("linear_damping_scale", "quadratic_damping_scale", "added_mass_scale"):
            _validate_payload_scale(sample.get(scale_name, 1.0), f"{sample_name}.{scale_name}")


@dataclass(frozen=True)
class DomainRandomizationProfile:
    """Optional reset-time randomization ranges for calibrated uncertainty."""

    use_custom_randomization: bool | None = None
    # ``None`` preserves pre-feature-selection recipe behaviour: all runtime
    # DR feature groups are available. New recipes may pin a reproducible
    # subset, and an individual training request may override that subset.
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
    observation_noise_std_range: NumberSequence | None = None
    observation_bias_range: NumberSequence | None = None
    observation_delay_steps_range: NumberSequence | None = None
    observation_update_period_steps_range: NumberSequence | None = None
    observation_dropout_probability_range: NumberSequence | None = None
    observation_lowpass_alpha_range: NumberSequence | None = None
    observation_bias_drift_std_range: NumberSequence | None = None
    disturbance_curriculum: bool | None = None
    disturbance_curriculum_stage_steps: Sequence[int] | None = None
    water_current_smooth: bool | None = None
    water_current_tau_range: NumberSequence | None = None
    water_current_max_by_stage: NumberSequence | None = None
    water_current_vertical_max_by_stage: NumberSequence | None = None
    water_current_variation_std_by_stage: NumberSequence | None = None
    damping_scale_by_stage: NumberSequence | None = None
    # Standard deviation of a zero-mean Gaussian latent variable in log space.
    # Runtime scale = exp(sigma*z - sigma^2/2), so added mass stays positive
    # and has unit expectation before pool/payload multipliers are composed.
    added_mass_log_std_by_stage: NumberSequence | None = None
    thruster_scale_by_stage: NumberSequence | None = None
    thruster_tau_scale_by_stage: NumberSequence | None = None
    # 0 keeps the modeled periodic/boundary/free-surface/tether terms neutral;
    # 1 applies the complete deterministic profile.  Intermediate values let
    # the same disturbance curriculum introduce those terms gradually.
    additional_hydrodynamics_scale_by_stage: NumberSequence | None = None

    def validate(self) -> None:
        if self.enabled_features is not None:
            from .features import normalize_domain_randomization_features

            normalize_domain_randomization_features(self.enabled_features)
        for name in (
            "use_custom_randomization",
            "disturbance_curriculum",
            "water_current_smooth",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"domain_randomization.{name} must be boolean.")
        if self.payload_samples is not None:
            _validate_payload_samples(self.payload_samples, "domain_randomization.payload_samples")
        if self.com_to_cob_offset_radius is not None:
            _validate_nonnegative(self.com_to_cob_offset_radius, "domain_randomization.com_to_cob_offset_radius")
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
            "observation_noise_std_range",
            "observation_bias_range",
            "observation_bias_drift_std_range",
            "water_current_tau_range",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_range(value, f"domain_randomization.{name}")
                if name in {"mass_range", "volume_range", "battery_voltage_range", "water_current_tau_range"}:
                    if float(value[0]) <= 0.0:
                        raise ValueError(f"domain_randomization.{name} must be positive.")
                elif float(value[0]) < 0.0:
                    raise ValueError(f"domain_randomization.{name} must be non-negative.")
                if name == "thruster_command_dropout_probability_range" and (
                    float(value[0]) < 0.0 or float(value[1]) > 1.0
                ):
                    raise ValueError("domain_randomization.thruster_command_dropout_probability_range must be in [0, 1].")
                if name.endswith("_scale_range") and float(value[0]) < 0.0:
                    raise ValueError(f"domain_randomization.{name} must be non-negative.")
                if name == "observation_bias_drift_std_range" and float(value[0]) < 0.0:
                    raise ValueError("domain_randomization.observation_bias_drift_std_range must be non-negative.")
                if name == "water_current_tau_range" and float(value[0]) <= 0.0:
                    raise ValueError("domain_randomization.water_current_tau_range must be positive.")
        for name in (
            "observation_dropout_probability_range",
            "observation_lowpass_alpha_range",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_range(value, f"domain_randomization.{name}")
                if float(value[0]) < 0.0 or float(value[1]) > 1.0:
                    raise ValueError(f"domain_randomization.{name} must be in [0, 1].")
        if self.thruster_command_delay_steps_range is not None:
            _validate_range(
                self.thruster_command_delay_steps_range,
                "domain_randomization.thruster_command_delay_steps_range",
                integer=True,
            )
            if int(self.thruster_command_delay_steps_range[0]) < 0:
                raise ValueError("domain_randomization.thruster_command_delay_steps_range must be non-negative.")
        if self.observation_delay_steps_range is not None:
            _validate_range(
                self.observation_delay_steps_range,
                "domain_randomization.observation_delay_steps_range",
                integer=True,
            )
            if int(self.observation_delay_steps_range[0]) < 0:
                raise ValueError("domain_randomization.observation_delay_steps_range must be non-negative.")
        if self.observation_update_period_steps_range is not None:
            _validate_range(
                self.observation_update_period_steps_range,
                "domain_randomization.observation_update_period_steps_range",
                integer=True,
            )
            if int(self.observation_update_period_steps_range[0]) < 1:
                raise ValueError("domain_randomization.observation_update_period_steps_range must be positive.")
        for name in (
            "water_current_max_by_stage",
            "water_current_vertical_max_by_stage",
            "water_current_variation_std_by_stage",
            "damping_scale_by_stage",
            "added_mass_log_std_by_stage",
            "thruster_scale_by_stage",
            "thruster_tau_scale_by_stage",
            "additional_hydrodynamics_scale_by_stage",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_nonnegative_sequence(value, f"domain_randomization.{name}")
                if name in {
                    "damping_scale_by_stage",
                    "thruster_scale_by_stage",
                    "thruster_tau_scale_by_stage",
                    "additional_hydrodynamics_scale_by_stage",
                } and any(float(item) > 1.0 for item in value):
                    raise ValueError(
                        f"domain_randomization.{name} must not exceed 1.0 because it is used as a ± amplitude."
                    )
        disturbance_stage_lengths = [
            len(value)
            for value in (
                self.water_current_max_by_stage,
                self.water_current_vertical_max_by_stage,
                self.water_current_variation_std_by_stage,
                self.damping_scale_by_stage,
                self.added_mass_log_std_by_stage,
                self.thruster_scale_by_stage,
                self.thruster_tau_scale_by_stage,
                self.additional_hydrodynamics_scale_by_stage,
            )
            if value is not None
        ]
        if disturbance_stage_lengths and any(
            length != disturbance_stage_lengths[0] for length in disturbance_stage_lengths
        ):
            raise ValueError("disturbance by-stage arrays must have matching lengths.")
        if (
            self.disturbance_curriculum
            and disturbance_stage_lengths
            and self.disturbance_curriculum_stage_steps is None
        ):
            raise ValueError(
                "domain_randomization.disturbance_curriculum_stage_steps is required when curriculum is enabled."
            )
        if self.disturbance_curriculum_stage_steps is not None:
            _validate_integer_sequence(
                self.disturbance_curriculum_stage_steps,
                "domain_randomization.disturbance_curriculum_stage_steps",
                nonnegative=True,
            )
            if (
                self.disturbance_curriculum
                and disturbance_stage_lengths
                and len(self.disturbance_curriculum_stage_steps) != disturbance_stage_lengths[0] - 1
            ):
                raise ValueError(
                    "domain_randomization.disturbance_curriculum_stage_steps must have one fewer entry "
                    "than disturbance by-stage arrays."
                )

    def to_cfg_updates(self) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value is not None:
                updates[key] = value
        return updates


@dataclass(frozen=True)
class PoolDynamicsProfile:
    name: str = "auv-nominal-pool"
    description: str = "Calibration base with neutral hydrodynamics and an opt-in high-order residual interface."
    rigid_body: RigidBodyProfile = field(default_factory=RigidBodyProfile)
    hydrodynamics: HydrodynamicsProfile = field(default_factory=HydrodynamicsProfile)
    thrusters: ThrusterProfile = field(default_factory=ThrusterProfile)
    battery: BatteryProfile = field(default_factory=BatteryProfile)
    pool_boundary: PoolBoundaryProfile = field(default_factory=PoolBoundaryProfile)
    free_surface: FreeSurfaceProfile = field(default_factory=FreeSurfaceProfile)
    tether: TetherProfile = field(default_factory=TetherProfile)
    observation: ObservationProfile = field(default_factory=ObservationProfile)
    domain_randomization: DomainRandomizationProfile | None = None

    def validate(self) -> None:
        self.rigid_body.validate()
        self.hydrodynamics.validate()
        self.thrusters.validate()
        self.battery.validate()
        self.pool_boundary.validate()
        self.free_surface.validate()
        self.tether.validate()
        self.observation.validate()
        if self.domain_randomization is not None:
            self.domain_randomization.validate()


NOMINAL_POOL_DYNAMICS_PROFILE = PoolDynamicsProfile()


_CFG_UPDATE_TO_PROFILE_FIELD: dict[str, tuple[str, str]] = {
    "mass": ("rigid_body", "mass"),
    "volume": ("rigid_body", "volume"),
    "inertia_diag": ("rigid_body", "inertia_diag"),
    "center_of_mass_offset": ("rigid_body", "center_of_mass_offset"),
    "com_to_cob_offset": ("rigid_body", "com_to_cob_offset"),
    "water_rho": ("rigid_body", "water_rho"),
    "water_beta": ("rigid_body", "water_beta"),
    "linear_damping": ("hydrodynamics", "linear_damping"),
    "quadratic_damping": ("hydrodynamics", "quadratic_damping"),
    "speed_dependent_damping_enabled": ("hydrodynamics", "speed_dependent_damping_enabled"),
    "damping_speed_points": ("hydrodynamics", "damping_speed_points"),
    "linear_damping_speed_scales": ("hydrodynamics", "linear_damping_speed_scales"),
    "quadratic_damping_speed_scales": ("hydrodynamics", "quadratic_damping_speed_scales"),
    "added_mass_diag": ("hydrodynamics", "added_mass"),
    "added_mass_inertia_scale": ("hydrodynamics", "added_mass_inertia_scale"),
    "added_mass_accel_filter_alpha": ("hydrodynamics", "added_mass_accel_filter_alpha"),
    "high_order_residual_enabled": ("hydrodynamics", "high_order_residual_enabled"),
    "high_order_residual_added_mass_factor": ("hydrodynamics", "high_order_residual_added_mass_factor"),
    "high_order_residual_linear_damping_factor": ("hydrodynamics", "high_order_residual_linear_damping_factor"),
    "high_order_residual_quadratic_damping_factor": ("hydrodynamics", "high_order_residual_quadratic_damping_factor"),
    "high_order_residual_cubic_damping_factor": ("hydrodynamics", "high_order_residual_cubic_damping_factor"),
    "water_current_w": ("hydrodynamics", "water_current_w"),
    "water_current_periodic_enabled": ("hydrodynamics", "water_current_periodic_enabled"),
    "water_current_periodic_amplitude_w": ("hydrodynamics", "water_current_periodic_amplitude_w"),
    "water_current_periodic_period_s": ("hydrodynamics", "water_current_periodic_period_s"),
    "water_current_periodic_phase_rad": ("hydrodynamics", "water_current_periodic_phase_rad"),
    "water_current_field_enabled": ("hydrodynamics", "water_current_field_enabled"),
    "water_current_field_bounds": ("hydrodynamics", "water_current_field_bounds"),
    "water_current_field_shape": ("hydrodynamics", "water_current_field_shape"),
    "water_current_field_values": ("hydrodynamics", "water_current_field_values"),
    "dyn_time_constant": ("thrusters", "dyn_time_constant"),
    "thruster_command_delay_steps": ("thrusters", "command_delay_steps"),
    "thruster_max_command_rate": ("thrusters", "max_command_rate"),
    "thruster_command_resolution": ("thrusters", "command_resolution"),
    "thruster_command_dropout_probability": ("thrusters", "command_dropout_probability"),
    "thruster_inflow_loss_enabled": ("thrusters", "inflow_loss_enabled"),
    "thruster_inflow_loss_coefficient": ("thrusters", "inflow_loss_coefficient"),
    "thruster_inflow_reference_speed": ("thrusters", "inflow_reference_speed"),
    "thruster_inflow_min_scale": ("thrusters", "inflow_min_scale"),
    "thruster_wake_interaction_enabled": ("thrusters", "wake_interaction_enabled"),
    "thruster_wake_loss_coefficient": ("thrusters", "wake_loss_coefficient"),
    "thruster_wake_length": ("thrusters", "wake_length"),
    "thruster_wake_radius": ("thrusters", "wake_radius"),
    "thruster_wake_expansion_rate": ("thrusters", "wake_expansion_rate"),
    "thruster_wake_min_scale": ("thrusters", "wake_min_scale"),
    "thruster_reaction_torque_coeff": ("thrusters", "reaction_torque_coeff"),
    "thruster_spin_directions": ("thrusters", "spin_directions"),
    "battery_voltage_nominal": ("battery", "nominal_voltage"),
    "battery_voltage": ("battery", "initial_voltage"),
    "battery_min_voltage": ("battery", "min_voltage"),
    "battery_voltage_drop_per_s": ("battery", "voltage_drop_per_s"),
    "battery_voltage_thrust_exponent": ("battery", "thrust_exponent"),
    "pool_boundary_effects_enabled": ("pool_boundary", "enabled"),
    "pool_bounds": ("pool_boundary", "bounds"),
    "pool_boundary_effect_distance": ("pool_boundary", "effect_distance"),
    "pool_boundary_damping_scale": ("pool_boundary", "damping_scale_at_boundary"),
    "pool_boundary_added_mass_scale": ("pool_boundary", "added_mass_scale_at_boundary"),
    "pool_boundary_thrust_scale": ("pool_boundary", "thrust_scale_at_boundary"),
    "free_surface_effects_enabled": ("free_surface", "enabled"),
    "free_surface_z": ("free_surface", "surface_z"),
    "free_surface_effect_distance": ("free_surface", "effect_distance"),
    "free_surface_heave_damping_scale": ("free_surface", "heave_damping_scale"),
    "free_surface_roll_pitch_damping_scale": ("free_surface", "roll_pitch_damping_scale"),
    "free_surface_added_mass_scale": ("free_surface", "added_mass_scale"),
    "free_surface_buoyancy_scale": ("free_surface", "buoyancy_scale"),
    "free_surface_thrust_scale": ("free_surface", "thrust_scale"),
    "free_surface_sloshing_enabled": ("free_surface", "sloshing_enabled"),
    "free_surface_sloshing_pool_bounds": ("free_surface", "sloshing_pool_bounds"),
    "free_surface_sloshing_water_depth": ("free_surface", "sloshing_water_depth"),
    "free_surface_sloshing_mode_numbers": ("free_surface", "sloshing_mode_numbers"),
    "free_surface_sloshing_amplitudes_m": ("free_surface", "sloshing_amplitudes_m"),
    "free_surface_sloshing_phases_rad": ("free_surface", "sloshing_phases_rad"),
    "free_surface_sloshing_depth_axis_sign": ("free_surface", "sloshing_depth_axis_sign"),
    "tether_enabled": ("tether", "enabled"),
    "tether_anchor_pos_w": ("tether", "anchor_pos_w"),
    "tether_attach_offset_b": ("tether", "attach_offset_b"),
    "tether_slack_length": ("tether", "slack_length"),
    "tether_stiffness": ("tether", "stiffness"),
    "tether_damping": ("tether", "damping"),
    "tether_drag_coeff": ("tether", "drag_coeff"),
    "tether_winch_enabled": ("tether", "winch_enabled"),
    "tether_winch_target_length": ("tether", "winch_target_length"),
    "tether_winch_reel_speed": ("tether", "winch_reel_speed"),
    "tether_winch_min_length": ("tether", "winch_min_length"),
    "tether_winch_max_length": ("tether", "winch_max_length"),
    "tether_num_segments": ("tether", "num_segments"),
    "tether_segment_diameter": ("tether", "segment_diameter"),
    "tether_segment_density": ("tether", "segment_density"),
    "tether_segment_buoyancy_density": ("tether", "segment_buoyancy_density"),
    "observation_noise_std": ("observation", "noise_std"),
    "observation_bias_range": ("observation", "bias_range"),
    "observation_delay_steps": ("observation", "delay_steps"),
    "observation_update_period_steps": ("observation", "update_period_steps"),
    "observation_dropout_probability": ("observation", "dropout_probability"),
    "observation_lowpass_alpha": ("observation", "lowpass_alpha"),
    "observation_bias_drift_std": ("observation", "bias_drift_std"),
}


_CALIBRATION_TASK_DETAILS: dict[str, dict[str, Any]] = {
    "rigid_body.static_properties": {
        "priority": "P0",
        "title": "Measure rigid-body mass, buoyancy, COM, and COB.",
        "experiment": "Dry scale readings, displacement or neutral-buoyancy test, and static tilt/restoring-torque trials.",
        "calibration_functions": (
            "fit_mass_from_scale_readings",
            "fit_buoyancy_volume_from_forces",
            "fit_com_to_cob_offset_from_static_torques",
        ),
        "update_keys": ("mass", "volume", "center_of_mass_offset", "com_to_cob_offset", "water_rho"),
    },
    "rigid_body.inertia_diag": {
        "priority": "P0",
        "title": "Replace diagonal inertia with a measured full 3x3 tensor when needed.",
        "experiment": "CAD mass-property correction or compound-pendulum/torsion-pendulum tests about independent axes.",
        "calibration_functions": (
            "fit_inertia_tensor_from_axis_moments",
            "fit_inertia_tensor_from_compound_pendulum",
        ),
        "update_keys": ("inertia_diag",),
    },
    "hydrodynamics.added_mass": {
        "priority": "P1",
        "title": "Fit nonzero added mass.",
        "experiment": "Axis step, free-motion, or multi-axis acceleration trials with known applied wrench.",
        "calibration_functions": (
            "fit_diagonal_added_mass_linear_quadratic_damping",
            "fit_full_matrix_added_mass_linear_quadratic_damping",
            "project_added_mass_to_physical",
        ),
        "update_keys": ("added_mass_diag", "added_mass_inertia_scale", "added_mass_accel_filter_alpha"),
    },
    "hydrodynamics.linear_damping": {
        "priority": "P1",
        "title": "Upgrade damping to measured full 6x6 coefficients if coupled motion is visible.",
        "experiment": "Free-decay, constant-thrust, tow, or multi-axis excitation logs with relative velocity and applied wrench.",
        "calibration_functions": (
            "fit_full_matrix_linear_quadratic_damping",
            "project_linear_damping_to_dissipative",
            "damping_is_dissipative_for_samples",
        ),
        "update_keys": ("linear_damping", "quadratic_damping"),
    },
    "hydrodynamics.speed_dependent_damping": {
        "priority": "P2",
        "title": "Add speed-dependent damping scale curves when one constant fit leaves systematic residuals.",
        "experiment": "Repeat damping identification across low, medium, and high speed ranges.",
        "calibration_functions": ("fit_speed_dependent_damping_scales",),
        "update_keys": (
            "speed_dependent_damping_enabled",
            "damping_speed_points",
            "linear_damping_speed_scales",
            "quadratic_damping_speed_scales",
        ),
    },
    "hydrodynamics.water_current": {
        "priority": "P1",
        "title": "Measure mean and stochastic pool current.",
        "experiment": "ADV samples, neutral drift markers, or DVL water-track logs in the operating volume.",
        "calibration_functions": ("fit_water_current_process",),
        "update_keys": (
            "water_current_w",
            "water_current_smooth",
            "water_current_tau_range",
            "water_current_max_by_stage",
            "water_current_vertical_max_by_stage",
            "water_current_variation_std_by_stage",
        ),
    },
    "hydrodynamics.water_current_field": {
        "priority": "P1",
        "title": "Build a spatial water-current field.",
        "experiment": "Sample current at multiple pool positions and depths.",
        "calibration_functions": ("fit_water_current_field_grid",),
        "update_keys": (
            "water_current_field_enabled",
            "water_current_field_bounds",
            "water_current_field_shape",
            "water_current_field_values",
        ),
    },
    "thrusters.wake_interaction": {
        "priority": "P2",
        "title": "Calibrate inter-thruster wake loss.",
        "experiment": "Run paired-thruster tests across measured axial/radial offsets or use a PIV-derived wake envelope.",
        "calibration_functions": ("fit_thruster_wake_loss_coefficient",),
        "update_keys": (
            "thruster_wake_interaction_enabled",
            "thruster_wake_loss_coefficient",
            "thruster_wake_length",
            "thruster_wake_radius",
            "thruster_wake_expansion_rate",
            "thruster_wake_min_scale",
        ),
    },
    "thrusters.reaction_torque": {
        "priority": "P2",
        "title": "Calibrate propeller reaction torque.",
        "experiment": "Measure axial thrust and shaft/body reaction torque for both commands and documented propeller spin directions.",
        "calibration_functions": ("fit_thruster_reaction_torque_coefficient",),
        "update_keys": (
            "thruster_reaction_torque_coeff",
            "thruster_spin_directions",
        ),
    },
    "thrusters.command_chain": {
        "priority": "P1",
        "title": "Calibrate command transport and actuator dynamics.",
        "experiment": "Log sent command, received command, ESC output, and thrust/velocity step response.",
        "calibration_functions": ("fit_thruster_first_order_response",),
        "update_keys": (
            "dyn_time_constant",
            "thruster_command_delay_steps",
            "thruster_max_command_rate",
            "thruster_command_resolution",
            "thruster_command_dropout_probability",
        ),
    },
    "battery.voltage_drop_per_s": {
        "priority": "P2",
        "title": "Model battery voltage sag and thrust scaling.",
        "experiment": "Long bollard-pull or fixed-command runs logging battery voltage and measured thrust.",
        "calibration_functions": ("fit_battery_voltage_sag", "fit_thruster_voltage_exponent"),
        "update_keys": (
            "battery_voltage_nominal",
            "battery_voltage",
            "battery_min_voltage",
            "battery_voltage_drop_per_s",
            "battery_voltage_thrust_exponent",
        ),
    },
    "pool_boundary.enabled": {
        "priority": "P1",
        "title": "Calibrate near-wall and near-bottom hydrodynamic corrections.",
        "experiment": "Repeat the same maneuvers in open water and near walls/floor/corners.",
        "calibration_functions": ("fit_pool_boundary_effect_scales",),
        "update_keys": (
            "pool_boundary_effects_enabled",
            "pool_bounds",
            "pool_boundary_effect_distance",
            "pool_boundary_damping_scale",
            "pool_boundary_added_mass_scale",
            "pool_boundary_thrust_scale",
        ),
    },
    "free_surface.enabled": {
        "priority": "P1",
        "title": "Calibrate free-surface proximity effects.",
        "experiment": "Depth sweeps for heave/roll/pitch damping, static buoyancy, and thrust near the water surface.",
        "calibration_functions": ("fit_free_surface_effect_scales",),
        "update_keys": (
            "free_surface_effects_enabled",
            "free_surface_z",
            "free_surface_effect_distance",
            "free_surface_heave_damping_scale",
            "free_surface_roll_pitch_damping_scale",
            "free_surface_added_mass_scale",
            "free_surface_buoyancy_scale",
            "free_surface_thrust_scale",
        ),
    },
    "free_surface.sloshing": {
        "priority": "P3",
        "title": "Fit rectangular-pool free-surface sloshing modes.",
        "experiment": "Record synchronized wave gauges at several non-nodal pool positions after representative disturbances.",
        "calibration_functions": ("fit_rectangular_pool_sloshing_modes",),
        "update_keys": (
            "free_surface_sloshing_enabled",
            "free_surface_sloshing_pool_bounds",
            "free_surface_sloshing_water_depth",
            "free_surface_sloshing_mode_numbers",
            "free_surface_sloshing_amplitudes_m",
            "free_surface_sloshing_phases_rad",
            "free_surface_sloshing_depth_axis_sign",
        ),
    },
    "tether.enabled": {
        "priority": "P1",
        "title": "Enable and calibrate tether dynamics.",
        "experiment": "Measure cable anchor/attach geometry, pull-force extension, velocity damping, drag, diameter, and density.",
        "calibration_functions": ("fit_tether_spring_damper", "fit_tether_drag_coefficient"),
        "update_keys": (
            "tether_enabled",
            "tether_anchor_pos_w",
            "tether_attach_offset_b",
            "tether_slack_length",
            "tether_stiffness",
            "tether_damping",
            "tether_drag_coeff",
            "tether_winch_enabled",
            "tether_winch_target_length",
            "tether_winch_reel_speed",
            "tether_winch_min_length",
            "tether_winch_max_length",
            "tether_num_segments",
            "tether_segment_diameter",
            "tether_segment_density",
            "tether_segment_buoyancy_density",
        ),
    },
    "tether.num_segments": {
        "priority": "P2",
        "title": "Upgrade tether from one segment to quasi-static multi-segment cable.",
        "experiment": "Estimate cable sag, buoyancy/negative buoyancy, and distributed drag under representative currents.",
        "calibration_functions": ("fit_tether_drag_coefficient",),
        "update_keys": (
            "tether_num_segments",
            "tether_segment_diameter",
            "tether_segment_density",
            "tether_segment_buoyancy_density",
        ),
    },
    "observation": {
        "priority": "P2",
        "title": "Model policy-observation estimator errors.",
        "experiment": "Compare estimator/policy inputs against motion-capture or high-trust reference logs.",
        "calibration_functions": (),
        "update_keys": (
            "observation_noise_std",
            "observation_bias_range",
            "observation_delay_steps",
            "observation_update_period_steps",
            "observation_dropout_probability",
            "observation_lowpass_alpha",
            "observation_bias_drift_std",
        ),
    },
    "domain_randomization": {
        "priority": "P0",
        "title": "Convert measured residual uncertainty into reset-time randomization.",
        "experiment": "Use repeated-trial residuals and calibration confidence intervals from all previous experiments.",
        "calibration_functions": ("to_domain_randomization_updates",),
        "update_keys": (
            "mass_range",
            "volume_range",
            "com_to_cob_offset_radius",
            "thruster_command_delay_steps_range",
            "thruster_wake_loss_coefficient_scale_range",
            "thruster_reaction_torque_coeff_scale_range",
            "damping_speed_linear_scale_range",
            "damping_speed_quadratic_scale_range",
            "battery_voltage_range",
            "observation_noise_std_range",
            "water_current_max_by_stage",
            "water_current_vertical_max_by_stage",
            "water_current_variation_std_by_stage",
        ),
    },
}


def _column(
    name: str,
    units: str,
    description: str,
    required: bool = True,
    value_type: str | None = None,
) -> PoolCalibrationLogColumn:
    if value_type is None:
        if name in {
            "sample_id",
            "configuration",
            "thruster_index",
            "channel_name",
            "sensor_name",
            "axis",
            "parameter_name",
        }:
            value_type = "string"
        elif name == "valid":
            value_type = "boolean"
        else:
            value_type = "float"
    return PoolCalibrationLogColumn(name, units, description, required, value_type)


_CALIBRATION_LOG_SCHEMAS: dict[str, tuple[dict[str, Any], ...]] = {
    "rigid_body.static_properties": (
        {
            "dataset_name": "Mass scale readings",
            "filename": "rigid_body_mass_readings.csv",
            "description": "Repeated dry or wet scale readings for fit_mass_from_scale_readings(...).",
            "columns": (
                _column("sample_id", "-", "Unique reading identifier."),
                _column("mass_kg", "kg", "Measured vehicle mass for this reading."),
                _column("configuration", "-", "Battery/payload/ballast configuration label.", False),
            ),
            "notes": "Use the same vehicle configuration as the pool trial.",
        },
        {
            "dataset_name": "Buoyancy force samples",
            "filename": "rigid_body_buoyancy_forces.csv",
            "description": "World-frame buoyancy force samples for fit_buoyancy_volume_from_forces(...).",
            "columns": (
                _column("sample_id", "-", "Unique reading identifier."),
                _column("buoyancy_force_w_x_n", "N", "World-frame buoyancy force x component."),
                _column("buoyancy_force_w_y_n", "N", "World-frame buoyancy force y component."),
                _column("buoyancy_force_w_z_n", "N", "World-frame buoyancy force z component."),
                _column("water_density_kg_m3", "kg/m^3", "Measured pool water density."),
                _column("gravity_w_z_mps2", "m/s^2", "World-frame gravity z component."),
            ),
            "notes": "Samples must be fluid buoyancy force, not net buoyancy minus weight.",
        },
        {
            "dataset_name": "Static buoyancy restoring torques",
            "filename": "rigid_body_static_buoyancy_torques.csv",
            "description": "Static attitude and body-frame restoring torque samples for COB fitting.",
            "columns": (
                _column("sample_id", "-", "Unique static pose identifier."),
                _column("quat_w", "-", "World-frame root quaternion w component."),
                _column("quat_x", "-", "World-frame root quaternion x component."),
                _column("quat_y", "-", "World-frame root quaternion y component."),
                _column("quat_z", "-", "World-frame root quaternion z component."),
                _column("buoyancy_torque_b_x_nm", "N m", "Body-frame restoring torque x component."),
                _column("buoyancy_torque_b_y_nm", "N m", "Body-frame restoring torque y component."),
                _column("buoyancy_torque_b_z_nm", "N m", "Body-frame restoring torque z component."),
                _column("volume_m3", "m^3", "Displaced volume used to compute buoyancy force."),
                _column("water_density_kg_m3", "kg/m^3", "Measured pool water density."),
            ),
            "notes": "Use multiple non-coplanar attitudes so com_to_cob_offset is observable in 3D.",
            "minimum_rows": 3,
        },
    ),
    "rigid_body.inertia_diag": (
        {
            "dataset_name": "Axis moment measurements",
            "filename": "rigid_body_axis_moments.csv",
            "description": "Measured moments about body-frame axes for full 3x3 inertia fitting.",
            "columns": (
                _column("sample_id", "-", "Unique axis measurement identifier."),
                _column("axis_b_x", "-", "Body-frame unit axis x component."),
                _column("axis_b_y", "-", "Body-frame unit axis y component."),
                _column("axis_b_z", "-", "Body-frame unit axis z component."),
                _column("moment_kg_m2", "kg m^2", "Measured moment of inertia about the axis."),
            ),
            "notes": "At least six independent axes are needed for a full symmetric 3x3 tensor.",
            "minimum_rows": 6,
        },
        {
            "dataset_name": "Compound pendulum periods",
            "filename": "rigid_body_compound_pendulum_periods.csv",
            "description": "Small-angle compound-pendulum periods for inertia tensor fitting.",
            "columns": (
                _column("sample_id", "-", "Unique period measurement identifier."),
                _column("axis_b_x", "-", "Body-frame unit suspension axis x component."),
                _column("axis_b_y", "-", "Body-frame unit suspension axis y component."),
                _column("axis_b_z", "-", "Body-frame unit suspension axis z component."),
                _column("period_s", "s", "Average oscillation period."),
                _column("mass_kg", "kg", "Vehicle mass during the pendulum test."),
                _column("pivot_to_com_distance_m", "m", "Perpendicular distance from pivot axis to COM."),
            ),
            "notes": "Keep oscillations small and average over many cycles.",
            "minimum_rows": 6,
        },
    ),
    "hydrodynamics.added_mass": (
        {
            "dataset_name": "Hydrodynamic motion and wrench log",
            "filename": "hydrodynamics_motion_wrench_log.csv",
            "description": "Relative velocity, optional acceleration, and known applied wrench for added-mass/damping fits.",
            "columns": (
                _column("time_s", "s", "Sample timestamp."),
                _column("nu_r_u_mps", "m/s", "Body-frame relative surge velocity."),
                _column("nu_r_v_mps", "m/s", "Body-frame relative sway velocity."),
                _column("nu_r_w_mps", "m/s", "Body-frame relative heave velocity."),
                _column("nu_r_p_radps", "rad/s", "Body-frame relative roll rate."),
                _column("nu_r_q_radps", "rad/s", "Body-frame relative pitch rate."),
                _column("nu_r_r_radps", "rad/s", "Body-frame relative yaw rate."),
                _column("wrench_x_n", "N", "Known applied body-frame force x."),
                _column("wrench_y_n", "N", "Known applied body-frame force y."),
                _column("wrench_z_n", "N", "Known applied body-frame force z."),
                _column("wrench_k_nm", "N m", "Known applied body-frame roll torque."),
                _column("wrench_m_nm", "N m", "Known applied body-frame pitch torque."),
                _column("wrench_n_nm", "N m", "Known applied body-frame yaw torque."),
                _column("nu_r_dot_u_mps2", "m/s^2", "Optional measured surge acceleration.", False),
                _column("nu_r_dot_v_mps2", "m/s^2", "Optional measured sway acceleration.", False),
                _column("nu_r_dot_w_mps2", "m/s^2", "Optional measured heave acceleration.", False),
                _column("nu_r_dot_p_radps2", "rad/s^2", "Optional measured roll acceleration.", False),
                _column("nu_r_dot_q_radps2", "rad/s^2", "Optional measured pitch acceleration.", False),
                _column("nu_r_dot_r_radps2", "rad/s^2", "Optional measured yaw acceleration.", False),
            ),
            "notes": (
                "If acceleration columns are absent, calibration_tools.finite_difference(...) can estimate them from "
                "time_s and nu_r. For fit_pool_hydrodynamics_logs.py --fit-mode high-order-residual, wrench_* "
                "must instead contain CFD/experiment fluid-wrench residual after subtracting the nominal Fossen model."
            ),
            "minimum_rows": 8,
        },
    ),
    "hydrodynamics.linear_damping": (),
    "hydrodynamics.speed_dependent_damping": (),
    "hydrodynamics.water_current": (
        {
            "dataset_name": "Water current time series",
            "filename": "water_current_timeseries.csv",
            "description": "Measured world-frame current samples for mean/current-process fitting.",
            "columns": (
                _column("time_s", "s", "Sample timestamp."),
                _column("current_w_x_mps", "m/s", "World-frame current x component."),
                _column("current_w_y_mps", "m/s", "World-frame current y component."),
                _column("current_w_z_mps", "m/s", "World-frame current z component."),
            ),
            "notes": "Use ADV, neutral drift marker, or DVL water-track estimates in a stable pool state.",
            "minimum_rows": 2,
        },
    ),
    "hydrodynamics.water_current_field": (
        {
            "dataset_name": "Spatial water current samples",
            "filename": "water_current_field_samples.csv",
            "description": "Sparse position/current samples for grid reconstruction.",
            "columns": (
                _column("pos_x_m", "m", "Pool-local sample x position."),
                _column("pos_y_m", "m", "Pool-local sample y position."),
                _column("pos_z_m", "m", "Pool-local sample z position."),
                _column("current_w_x_mps", "m/s", "World-frame current x component."),
                _column("current_w_y_mps", "m/s", "World-frame current y component."),
                _column("current_w_z_mps", "m/s", "World-frame current z component."),
            ),
            "notes": "Cover walls, return flow, operating depth, and any pump/jet regions.",
        },
    ),
    "thrusters.wake_interaction": (
        {
            "dataset_name": "Paired-thruster wake interaction samples",
            "filename": "thruster_wake_interaction.csv",
            "description": "Target-thruster loss under a known source-thruster wake and measured pair geometry.",
            "columns": (
                _column("source_thruster_index", "-", "Source thruster index or stand identifier."),
                _column("target_thruster_index", "-", "Target thruster index or stand identifier."),
                _column("source_thrust_n", "N", "Signed source-thruster thrust."),
                _column("source_reference_thrust_n", "N", "Positive full-strength reference thrust for the source."),
                _column("axial_distance_m", "m", "Source-to-target distance along the signed source wake axis."),
                _column("radial_distance_m", "m", "Source-to-target distance perpendicular to the wake axis."),
                _column("measured_target_thrust_scale", "-", "Target thrust divided by its matched no-wake baseline."),
            ),
            "notes": "Use measured wake length/radius/expansion in the base profile; include unsaturated in-wake and out-of-wake controls.",
            "minimum_rows": 4,
        },
    ),
    "thrusters.reaction_torque": (
        {
            "dataset_name": "Thruster axial reaction torque samples",
            "filename": "thruster_reaction_torque.csv",
            "description": "Signed axial reaction torque paired with thrust and propeller spin direction.",
            "columns": (
                _column("thruster_index", "-", "Thruster index; use 0 through 7 to update vehicle spin directions."),
                _column("spin_direction", "-", "Propeller spin direction, exactly -1 or 1."),
                _column("thrust_n", "N", "Signed axial thrust."),
                _column("reaction_torque_nm", "N m", "Measured reaction torque about the positive thruster axis."),
            ),
            "notes": "Keep the torque sign convention consistent with Q = -k * spin_direction * thrust.",
            "minimum_rows": 4,
        },
    ),
    "thrusters.command_chain": (
        {
            "dataset_name": "Thruster command response",
            "filename": "thruster_step_response.csv",
            "description": "Command and measured thrust response for delay/time-constant fitting.",
            "columns": (
                _column("time_s", "s", "Sample timestamp."),
                _column("pwm_us", "us", "PWM command sent to the T60 actuator chain."),
                _column("measured_thrust_n", "N", "Measured thrust response."),
                _column("voltage_v", "V", "Supply voltage during response.", False),
            ),
            "notes": "Include pre-step baseline and enough post-step tail to estimate steady thrust.",
            "minimum_rows": 4,
        },
    ),
    "battery.voltage_drop_per_s": (
        {
            "dataset_name": "Battery voltage sag and thrust scaling",
            "filename": "battery_voltage_thrust_samples.csv",
            "description": "Time-voltage samples with optional normalized thrust scale for battery model fitting.",
            "columns": (
                _column("time_s", "s", "Sample timestamp."),
                _column("voltage_v", "V", "Measured battery voltage."),
                _column(
                    "thrust_scale",
                    "-",
                    "Measured thrust divided by nominal-voltage thrust at the same command.",
                    False,
                ),
            ),
            "notes": "Use a fixed command/load for sag tests; thrust_scale rows should span non-nominal voltages.",
            "minimum_rows": 2,
        },
    ),
    "pool_boundary.enabled": (
        {
            "dataset_name": "Boundary effect scale samples",
            "filename": "pool_boundary_effect_samples.csv",
            "description": "Near-wall/floor scale ratios relative to open-water baseline.",
            "columns": (
                _column("pos_x_m", "m", "Pool-local vehicle x position."),
                _column("pos_y_m", "m", "Pool-local vehicle y position."),
                _column("pos_z_m", "m", "Pool-local vehicle z position."),
                _column("damping_scale", "-", "Measured damping ratio relative to open water.", False),
                _column("added_mass_scale", "-", "Measured added-mass ratio relative to open water.", False),
                _column("thrust_scale", "-", "Measured thrust ratio relative to open water.", False),
            ),
            "notes": "Repeat the same motion/open-loop thrust trials at matched speeds near and away from boundaries.",
            "minimum_rows": 2,
        },
    ),
    "free_surface.enabled": (
        {
            "dataset_name": "Free-surface effect scale samples",
            "filename": "free_surface_effect_samples.csv",
            "description": "Depth-dependent scale ratios near the water surface.",
            "columns": (
                _column("pos_z_m", "m", "Vehicle z position in the pool/world convention used by cfg."),
                _column("heave_damping_scale", "-", "Measured heave damping ratio.", False),
                _column("roll_pitch_damping_scale", "-", "Measured roll/pitch damping ratio.", False),
                _column("added_mass_scale", "-", "Measured heave/roll/pitch added-mass ratio.", False),
                _column("buoyancy_scale", "-", "Measured effective buoyancy ratio.", False),
                _column("thrust_scale", "-", "Measured thrust ratio near the surface.", False),
            ),
            "notes": "Record surface_z and keep command/speed envelopes consistent across depths.",
            "minimum_rows": 2,
        },
    ),
    "free_surface.sloshing": (
        {
            "dataset_name": "Rectangular-pool sloshing wave gauges",
            "filename": "free_surface_wave_gauge.csv",
            "description": "Timestamped surface coordinate at one or more fixed wave-gauge positions.",
            "columns": (
                _column("time_s", "s", "Synchronized gauge timestamp."),
                _column("gauge_x_m", "m", "Pool-local gauge x coordinate."),
                _column("gauge_y_m", "m", "Pool-local gauge y coordinate."),
                _column("surface_z_m", "m", "Measured world-coordinate free-surface z."),
            ),
            "notes": "Use multiple non-nodal gauge locations and enough periods to identify every requested mode.",
            "minimum_rows": 8,
        },
    ),
    "tether.enabled": (
        {
            "dataset_name": "Tether spring-damper samples",
            "filename": "tether_tension_samples.csv",
            "description": "Cable extension, velocity, and tension samples.",
            "columns": (
                _column("length_m", "m", "Anchor-to-attach cable length."),
                _column("tension_n", "N", "Measured cable tension."),
                _column("velocity_along_tether_mps", "m/s", "Body velocity dotted with direction to anchor."),
            ),
            "notes": "Velocity is negative when the robot moves away from the anchor in the simulator convention.",
            "minimum_rows": 2,
        },
        {
            "dataset_name": "Tether drag samples",
            "filename": "tether_drag_samples.csv",
            "description": "Relative water/cable velocity and measured drag force samples.",
            "columns": (
                _column("relative_velocity_x_mps", "m/s", "Relative flow velocity x."),
                _column("relative_velocity_y_mps", "m/s", "Relative flow velocity y."),
                _column("relative_velocity_z_mps", "m/s", "Relative flow velocity z."),
                _column("drag_force_x_n", "N", "Measured drag force x."),
                _column("drag_force_y_n", "N", "Measured drag force y."),
                _column("drag_force_z_n", "N", "Measured drag force z."),
            ),
            "notes": "Use representative cable length, diameter, and orientation.",
        },
    ),
    "tether.num_segments": (),
    "observation": (
        {
            "dataset_name": "Observation estimator reference log",
            "filename": "observation_reference_log.csv",
            "description": "Policy-observation or estimator outputs aligned to a high-trust reference.",
            "columns": (
                _column("time_s", "s", "Sample timestamp."),
                _column("channel_name", "-", "Observation channel name, such as linear_velocity_b."),
                _column("measured_value", "channel units", "Estimator/policy-input value."),
                _column("reference_value", "channel units", "Trusted reference value."),
            ),
            "notes": "Use repeated rows for vector channels or expand channel_name with axis suffixes.",
            "minimum_rows": 2,
        },
    ),
    "domain_randomization": (
        {
            "dataset_name": "Calibration residual and uncertainty summary",
            "filename": "calibration_residual_summary.csv",
            "description": "Per-parameter uncertainty ranges for DomainRandomizationProfile.",
            "columns": (
                _column("parameter_name", "-", "Target cfg/domain parameter name."),
                _column("nominal_value", "parameter units", "Nominal calibrated value."),
                _column("lower_bound", "parameter units", "Lower randomization bound."),
                _column("upper_bound", "parameter units", "Upper randomization bound."),
                _column("residual_std", "parameter units", "Observed residual standard deviation.", False),
            ),
            "notes": "Use repeated-trial statistics and confidence intervals, not arbitrary wide ranges.",
        },
    ),
}


_CALIBRATION_LOG_SCHEMAS["hydrodynamics.linear_damping"] = _CALIBRATION_LOG_SCHEMAS["hydrodynamics.added_mass"]
_CALIBRATION_LOG_SCHEMAS["hydrodynamics.speed_dependent_damping"] = _CALIBRATION_LOG_SCHEMAS[
    "hydrodynamics.added_mass"
]
_CALIBRATION_LOG_SCHEMAS["tether.num_segments"] = _CALIBRATION_LOG_SCHEMAS["tether.enabled"]


def merge_pool_dynamics_cfg_updates(
    base_profile: PoolDynamicsProfile | None = None,
    cfg_updates: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    domain_randomization_updates: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
) -> PoolDynamicsProfile:
    """Merge flat calibration ``to_cfg_updates()`` dictionaries into a profile.

    Later mappings override earlier mappings.  Unknown keys raise by default so
    unit mistakes or environment-only settings do not silently disappear.
    """

    base = copy.deepcopy(NOMINAL_POOL_DYNAMICS_PROFILE if base_profile is None else base_profile)
    base.validate()
    flat_updates = _combine_update_mappings(cfg_updates, "cfg_updates")
    randomization_updates = _combine_update_mappings(
        domain_randomization_updates,
        "domain_randomization_updates",
    )

    section_updates: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    randomization_fields = {field.name for field in fields(DomainRandomizationProfile)}
    for key, value in flat_updates.items():
        target = _CFG_UPDATE_TO_PROFILE_FIELD.get(key)
        if target is not None:
            section_name, field_name = target
            section_updates.setdefault(section_name, {})[field_name] = copy.deepcopy(value)
        elif key in randomization_fields:
            randomization_updates[key] = copy.deepcopy(value)
        else:
            unknown.append(key)

    valid_randomization_updates: dict[str, Any] = {}
    for key, value in randomization_updates.items():
        if key in randomization_fields:
            valid_randomization_updates[key] = copy.deepcopy(value)
        else:
            unknown.append(key)
    if unknown and strict:
        raise ValueError(f"Unknown pool dynamics cfg update key(s): {', '.join(sorted(set(unknown)))}.")

    profile = base
    for section_name in (
        "rigid_body",
        "hydrodynamics",
        "thrusters",
        "battery",
        "pool_boundary",
        "free_surface",
        "tether",
        "observation",
    ):
        changes = section_updates.get(section_name)
        if changes:
            profile = replace(profile, **{section_name: replace(getattr(profile, section_name), **changes)})

    if valid_randomization_updates:
        base_randomization = profile.domain_randomization or DomainRandomizationProfile()
        profile = replace(
            profile,
            domain_randomization=replace(base_randomization, **valid_randomization_updates),
        )

    if name is not None:
        profile = replace(profile, name=name)
    if description is not None:
        profile = replace(profile, description=description)

    profile.validate()
    return profile


def audit_pool_dynamics_profile(
    profile: PoolDynamicsProfile,
    options: PoolProfileAuditOptions | None = None,
) -> PoolProfileAuditReport:
    """Return a structured checklist for high-fidelity pool simulation readiness."""

    profile.validate()
    options = PoolProfileAuditOptions() if options is None else options
    findings: list[PoolProfileAuditFinding] = []

    def add(severity: AuditSeverity, section: str, message: str, recommendation: str) -> None:
        findings.append(PoolProfileAuditFinding(severity, section, message, recommendation))

    if _rigid_body_static_properties_are_default(profile.rigid_body):
        add(
            "info",
            "rigid_body.static_properties",
            "Mass, volume, COM, COB, and water density still match the nominal AUV defaults.",
            "Verify or replace them with measured scale, displacement, static buoyancy, and tilt-restoring data.",
        )
    if not _is_full_matrix_like(profile.rigid_body.inertia_diag):
        add(
            "info",
            "rigid_body.inertia_diag",
            "Rigid-body inertia is still represented as a diagonal 3-vector.",
            "Use a measured/CAD-corrected full 3x3 inertia tensor when asymmetric payloads or off-axis fixtures matter.",
        )

    if _all_numeric_close_to_zero(profile.hydrodynamics.added_mass):
        add(
            "warning",
            "hydrodynamics.added_mass",
            "Added mass is zero, so acceleration-dependent fluid inertia is not represented.",
            "Fit nonzero added mass with fit_diagonal_added_mass_linear_quadratic_damping(...) or fit_full_matrix_added_mass_linear_quadratic_damping(...).",
        )
    elif not _is_full_6d_matrix_like(profile.hydrodynamics.added_mass):
        add(
            "info",
            "hydrodynamics.added_mass",
            "Added mass is diagonal only.",
            "Upgrade to a symmetric full 6x6 matrix when multi-axis excitation shows surge/sway/yaw or heave/pitch coupling.",
        )

    if not _is_full_6d_matrix_like(profile.hydrodynamics.linear_damping):
        add(
            "info",
            "hydrodynamics.linear_damping",
            "Linear damping is diagonal only.",
            "Use fit_full_matrix_linear_quadratic_damping(...) when coupled sway-yaw, heave-pitch, or roll-yaw damping is visible.",
        )
    if not profile.hydrodynamics.speed_dependent_damping_enabled:
        add(
            "info",
            "hydrodynamics.speed_dependent_damping",
            "Speed-dependent damping curves are disabled.",
            "Enable speed-dependent damping if one constant damping fit cannot match both low-speed and high-speed trials.",
        )

    current_is_static_zero = (
        _all_numeric_close_to_zero(profile.hydrodynamics.water_current_w)
        and not profile.hydrodynamics.water_current_field_enabled
        and not _domain_randomization_has_current(profile.domain_randomization)
    )
    if current_is_static_zero:
        add(
            "warning",
            "hydrodynamics.water_current",
            "No mean, stochastic, or spatial water-current model is configured.",
            "Estimate current with fit_water_current_process(...) or fit_water_current_field_grid(...) when pump return flow or drift is measurable.",
        )
    if options.spatial_current_expected and not profile.hydrodynamics.water_current_field_enabled:
        add(
            "warning",
            "hydrodynamics.water_current_field",
            "Spatial current field is expected but disabled.",
            "Populate water_current_field_* from ADV, drift marker, or DVL water-track samples.",
        )

    if not profile.thrusters.wake_interaction_enabled:
        add(
            "info",
            "thrusters.wake_interaction",
            "Inter-thruster wake interaction is disabled.",
            "Fit wake loss from paired-thruster tests when one thruster operates inside another thruster's wake.",
        )
    if profile.thrusters.reaction_torque_coeff == 0.0:
        add(
            "info",
            "thrusters.reaction_torque",
            "Propeller reaction torque is disabled.",
            "Fit axial reaction torque and verify all eight propeller spin directions when roll/yaw coupling matters.",
        )
    if (
        profile.thrusters.command_delay_steps == 0
        and profile.thrusters.max_command_rate == 0.0
        and profile.thrusters.command_resolution == 0.0
        and profile.thrusters.command_dropout_probability == 0.0
    ):
        add(
            "info",
            "thrusters.command_chain",
            "Command transport effects are all disabled.",
            "Calibrate delay, rate limit, quantization, and dropout when the controller path differs from ideal direct actuation.",
        )

    if profile.battery.voltage_drop_per_s == 0.0:
        add(
            "info",
            "battery.voltage_drop_per_s",
            "Episode-level battery voltage sag is disabled.",
            "Fit voltage sag and voltage-thrust exponent when long pool runs show thrust reduction over time.",
        )

    if not profile.pool_boundary.enabled:
        add(
            "warning" if options.near_boundaries_expected else "info",
            "pool_boundary.enabled",
            "Near-wall/bottom boundary effects are disabled.",
            "Enable and calibrate pool boundary scales if trajectories approach walls, floor, or tank corners.",
        )
    if not profile.free_surface.enabled:
        add(
            "warning" if options.near_surface_expected else "info",
            "free_surface.enabled",
            "Free-surface proximity effects are disabled.",
            "Enable free-surface scales for shallow-depth heave, partial surfacing, or thruster ventilation trials.",
        )
    if not profile.free_surface.sloshing_enabled:
        add(
            "warning" if options.sloshing_expected else "info",
            "free_surface.sloshing",
            "Rectangular-pool sloshing and wave orbital velocity are disabled.",
            "Fit standing-wave modes from synchronized wave gauges when surface motion is measurable.",
        )

    if options.tether_expected and not profile.tether.enabled:
        add(
            "warning",
            "tether.enabled",
            "A tether is expected but the tether model is disabled.",
            "Enable tether dynamics and calibrate slack length, anchor, stiffness, damping, drag, and cable buoyancy.",
        )
    elif profile.tether.enabled and profile.tether.num_segments == 1:
        add(
            "info",
            "tether.num_segments",
            "Tether is enabled with a single segment.",
            "Use multiple quasi-static segments when cable buoyancy, sag, or distributed drag is visible.",
        )

    if _observation_profile_is_default(profile.observation):
        add(
            "info",
            "observation",
            "Policy observation noise, delay, dropout, low-pass, and bias drift are all disabled.",
            "Add observation effects or domain randomization when the real policy input comes from sensors/estimators.",
        )
    if options.domain_randomization_expected and profile.domain_randomization is None:
        add(
            "warning",
            "domain_randomization",
            "No reset-time uncertainty ranges are configured.",
            "Add measured uncertainty ranges for mass, buoyancy, current, thrusters, battery, and sensors before robust RL training.",
        )

    return PoolProfileAuditReport(profile.name, tuple(findings))


def pool_profile_calibration_tasks(
    profile: PoolDynamicsProfile,
    options: PoolProfileAuditOptions | None = None,
) -> tuple[PoolCalibrationTask, ...]:
    """Return experiment-oriented tasks required to close current profile gaps."""

    report = audit_pool_dynamics_profile(profile, options)
    tasks: list[PoolCalibrationTask] = []
    for finding in report.findings:
        details = _CALIBRATION_TASK_DETAILS.get(finding.section)
        if details is None:
            details = {
                "priority": _priority_from_severity(finding.severity),
                "title": f"Calibrate {finding.section}.",
                "experiment": finding.recommendation,
                "calibration_functions": (),
                "update_keys": (),
            }
        tasks.append(
            PoolCalibrationTask(
                priority=str(details["priority"]),
                section=finding.section,
                severity=finding.severity,
                title=str(details["title"]),
                reason=finding.message,
                experiment=str(details["experiment"]),
                calibration_functions=tuple(details["calibration_functions"]),
                update_keys=tuple(details["update_keys"]),
                recommendation=finding.recommendation,
            )
        )
    return tuple(tasks)


def pool_profile_calibration_update_template(
    profile: PoolDynamicsProfile,
    options: PoolProfileAuditOptions | None = None,
    placeholder: Any = None,
) -> dict[str, Any]:
    """Return a JSON-friendly skeleton for missing calibration update values.

    The payload is intentionally nested under ``update_payload`` so the whole
    template is not accidentally accepted as a ready-to-merge update file.
    """

    profile.validate()
    tasks = pool_profile_calibration_tasks(profile, options)
    randomization_fields = {field.name for field in fields(DomainRandomizationProfile)}
    cfg_updates: dict[str, Any] = {}
    randomization_updates: dict[str, Any] = {}
    unmapped_keys: list[str] = []
    for task in tasks:
        for key in task.update_keys:
            if key in _CFG_UPDATE_TO_PROFILE_FIELD:
                cfg_updates.setdefault(key, copy.deepcopy(placeholder))
            elif key in randomization_fields:
                randomization_updates.setdefault(key, copy.deepcopy(placeholder))
            elif key not in unmapped_keys:
                unmapped_keys.append(key)

    return {
        "template_type": "pool_calibration_update_template",
        "profile_name": profile.name,
        "instructions": (
            "Fill null values from the listed experiments, then copy update_payload to a separate "
            "updates JSON file before using build_pool_profile_from_calibration.py."
        ),
        "update_payload": {
            "cfg_updates": cfg_updates,
            "domain_randomization_updates": randomization_updates,
        },
        "unmapped_update_keys": unmapped_keys,
        "tasks": [task.to_dict() for task in tasks],
    }


def pool_profile_calibration_log_schemas(
    profile: PoolDynamicsProfile,
    options: PoolProfileAuditOptions | None = None,
) -> tuple[PoolCalibrationLogSchema, ...]:
    """Return CSV-style log schemas for the experiments required by a profile."""

    profile.validate()
    tasks = pool_profile_calibration_tasks(profile, options)
    schemas: list[PoolCalibrationLogSchema] = []
    seen_filenames: set[str] = set()
    for task in tasks:
        for schema_data in _CALIBRATION_LOG_SCHEMAS.get(task.section, ()):
            filename = str(schema_data["filename"])
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            schemas.append(
                PoolCalibrationLogSchema(
                    section=task.section,
                    dataset_name=str(schema_data["dataset_name"]),
                    filename=filename,
                    description=str(schema_data["description"]),
                    columns=tuple(schema_data["columns"]),
                    calibration_functions=task.calibration_functions,
                    update_keys=task.update_keys,
                    notes=str(schema_data.get("notes", "")),
                    minimum_rows=int(schema_data.get("minimum_rows", 1)),
                )
            )
    return tuple(schemas)


def validate_pool_calibration_log_directory(
    directory: str | Path,
    schemas: Sequence[PoolCalibrationLogSchema],
    max_issues_per_file: int = 50,
) -> PoolCalibrationLogValidationReport:
    """Validate experiment CSV files against generated calibration schemas."""

    if int(max_issues_per_file) != max_issues_per_file or int(max_issues_per_file) < 1:
        raise ValueError("max_issues_per_file must be a positive integer.")
    root = Path(directory)
    issues: list[PoolCalibrationLogValidationIssue] = []
    row_counts: dict[str, int] = {}

    for schema in schemas:
        path = root / schema.filename
        row_counts[schema.filename] = 0
        if not path.exists():
            issues.append(
                PoolCalibrationLogValidationIssue(
                    "error",
                    schema.filename,
                    "Expected calibration log file is missing.",
                )
            )
            continue
        if not path.is_file():
            issues.append(
                PoolCalibrationLogValidationIssue(
                    "error",
                    schema.filename,
                    "Expected calibration log path is not a regular file.",
                )
            )
            continue

        file_issue_count = 0
        truncated = False

        def add_issue(message: str, row_number: int | None = None, column: str | None = None) -> None:
            nonlocal file_issue_count, truncated
            if file_issue_count >= int(max_issues_per_file):
                truncated = True
                return
            issues.append(
                PoolCalibrationLogValidationIssue(
                    "error",
                    schema.filename,
                    message,
                    row_number=row_number,
                    column=column,
                )
            )
            file_issue_count += 1

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                if fieldnames is None:
                    add_issue("CSV header is missing.")
                    continue
                if len(fieldnames) != len(set(fieldnames)):
                    add_issue("CSV header contains duplicate column names.")

                present_columns = set(fieldnames)
                for column in schema.columns:
                    if column.required and column.name not in present_columns:
                        add_issue("Required column is missing from the CSV header.", column=column.name)

                row_count = 0
                for row_number, row in enumerate(reader, start=2):
                    if not any(value is not None and value.strip() for value in row.values() if isinstance(value, str)):
                        continue
                    row_count += 1
                    for column in schema.columns:
                        if column.name not in present_columns:
                            continue
                        raw_value = row.get(column.name)
                        value = "" if raw_value is None else raw_value.strip()
                        if value == "":
                            if column.required:
                                add_issue("Required value is empty.", row_number=row_number, column=column.name)
                            continue
                        if not _calibration_log_value_is_valid(value, column.value_type):
                            add_issue(
                                f"Value is not a valid {column.value_type}.",
                                row_number=row_number,
                                column=column.name,
                            )
                row_counts[schema.filename] = row_count
        except (OSError, csv.Error, UnicodeError) as exc:
            add_issue(f"Failed to read CSV: {exc}")
            continue

        if row_counts[schema.filename] < int(schema.minimum_rows):
            add_issue(
                f"CSV has {row_counts[schema.filename]} data rows; at least {schema.minimum_rows} are required."
            )
        if truncated:
            issues.append(
                PoolCalibrationLogValidationIssue(
                    "warning",
                    schema.filename,
                    f"Additional issues were omitted after {max_issues_per_file} errors.",
                )
            )

    return PoolCalibrationLogValidationReport(
        directory=str(root),
        expected_files=tuple(schema.filename for schema in schemas),
        row_counts=row_counts,
        issues=tuple(issues),
    )


def _calibration_log_value_is_valid(value: str, value_type: str) -> bool:
    if value_type == "string":
        return value != ""
    if value_type == "boolean":
        return value.lower() in {"0", "1", "true", "false", "yes", "no"}
    if value_type == "float":
        try:
            return math.isfinite(float(value))
        except ValueError:
            return False
    raise ValueError(f"Unknown calibration log value_type: {value_type}.")


def _validate_shared_or_per_thruster_curve(
    values: Sequence[Any],
    axis_length: int,
    name: str,
    required: bool,
) -> None:
    if not _is_sequence(values):
        raise ValueError(f"{name} must be a sequence.")
    if not required and len(values) == 0:
        return
    if len(values) == 0:
        raise ValueError(f"{name} must be provided.")
    first = values[0]
    if _is_sequence(first):
        if len(values) != 8:
            raise ValueError(f"{name} must have 8 rows for per-thruster curves.")
        for row_index, row in enumerate(values):
            _validate_vector(row, axis_length, f"{name}[{row_index}]")
    else:
        _validate_vector(values, axis_length, name)


def _validate_damping_speed_scale_curve(
    speed_points: Sequence[float],
    scale_points: Sequence[Any],
    name: str,
) -> None:
    _validate_increasing_axis(speed_points, "hydrodynamics.damping_speed_points")

    if not _is_sequence(scale_points) or len(scale_points) == 0:
        return

    if len(scale_points) != len(speed_points):
        raise ValueError(f"{name} must have one sample per damping_speed_points entry.")

    first = scale_points[0]
    if _is_sequence(first):
        for row_index, row in enumerate(scale_points):
            _validate_vector(row, 6, f"{name}[{row_index}]")
            for col_index, item in enumerate(row):
                _validate_nonnegative(float(item), f"{name}[{row_index}][{col_index}]")
    else:
        _validate_vector(scale_points, len(speed_points), name)
        for index, item in enumerate(scale_points):
            _validate_nonnegative(float(item), f"{name}[{index}]")


def _validate_increasing_axis(points: Sequence[float], name: str) -> None:
    if not _is_sequence(points) or len(points) < 2:
        raise ValueError(f"{name} must contain at least two points.")
    previous = float(points[0])
    for index, point in enumerate(points[1:], start=1):
        point_value = float(point)
        if point_value <= previous:
            raise ValueError(f"{name} must be strictly increasing at index {index}.")
        previous = point_value


def _validate_2d_lookup_grid(
    table: Sequence[Any],
    num_commands: int,
    num_inflow_points: int,
    name: str,
) -> None:
    if not _is_sequence(table) or len(table) != num_commands:
        raise ValueError(f"{name} must contain {num_commands} command rows.")
    for row_index, row in enumerate(table):
        if not _is_sequence(row):
            raise ValueError(f"{name}[{row_index}] must be a row of inflow samples.")
        _validate_vector(row, num_inflow_points, f"{name}[{row_index}]")


def _iter_numeric_values(value: Any):
    if value is None:
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_numeric_values(item)
        return
    if _is_sequence(value):
        for item in value:
            yield from _iter_numeric_values(item)
        return
    yield float(value)


def _all_numeric_close_to(value: Any, target: float, tolerance: float = 1.0e-8) -> bool:
    values = list(_iter_numeric_values(value))
    if len(values) == 0:
        return True
    return all(abs(item - float(target)) <= tolerance for item in values)


def _all_numeric_close_to_zero(value: Any, tolerance: float = 1.0e-8) -> bool:
    return _all_numeric_close_to(value, 0.0, tolerance)


def _is_full_matrix_like(value: Any) -> bool:
    return _is_sequence(value) and len(value) > 0 and _is_sequence(value[0])


def _is_full_6d_matrix_like(value: Any) -> bool:
    return _is_sequence(value) and len(value) == 6 and _is_sequence(value[0])


def _numeric_values_close(left: Any, right: Any, tolerance: float = 1.0e-8) -> bool:
    left_values = list(_iter_numeric_values(left))
    right_values = list(_iter_numeric_values(right))
    return len(left_values) == len(right_values) and all(
        abs(left_item - right_item) <= tolerance
        for left_item, right_item in zip(left_values, right_values)
    )


def _rigid_body_static_properties_are_default(rigid_body: RigidBodyProfile) -> bool:
    nominal = RigidBodyProfile()
    return (
        abs(float(rigid_body.mass) - float(nominal.mass)) <= 1.0e-8
        and abs(float(rigid_body.volume) - float(nominal.volume)) <= 1.0e-8
        and _numeric_values_close(rigid_body.center_of_mass_offset, nominal.center_of_mass_offset)
        and _numeric_values_close(rigid_body.com_to_cob_offset, nominal.com_to_cob_offset)
        and abs(float(rigid_body.water_rho) - float(nominal.water_rho)) <= 1.0e-8
    )


def _priority_from_severity(severity: AuditSeverity) -> str:
    if severity == "critical":
        return "P0"
    if severity == "warning":
        return "P1"
    return "P2"


def _domain_randomization_has_current(domain_randomization: DomainRandomizationProfile | None) -> bool:
    if domain_randomization is None:
        return False
    for value in (
        domain_randomization.water_current_max_by_stage,
        domain_randomization.water_current_vertical_max_by_stage,
        domain_randomization.water_current_variation_std_by_stage,
    ):
        if value is not None and not _all_numeric_close_to_zero(value):
            return True
    return False


def _observation_profile_is_default(observation: ObservationProfile) -> bool:
    return (
        _all_numeric_close_to_zero(observation.noise_std)
        and _all_numeric_close_to_zero(observation.bias_range)
        and observation.delay_steps == 0
        and observation.update_period_steps == 1
        and _all_numeric_close_to_zero(observation.dropout_probability)
        and _all_numeric_close_to(observation.lowpass_alpha, 1.0)
        and _all_numeric_close_to_zero(observation.bias_drift_std)
    )


def pool_dynamics_profile_to_cfg_updates(profile: PoolDynamicsProfile) -> dict[str, Any]:
    """Return top-level AUV config updates for a validated profile."""

    profile.validate()
    updates: dict[str, Any] = {}
    for section in (
        profile.rigid_body,
        profile.hydrodynamics,
        profile.thrusters,
        profile.battery,
        profile.pool_boundary,
        profile.free_surface,
        profile.tether,
        profile.observation,
    ):
        updates.update(section.to_cfg_updates())
    return {key: _as_plain_value(value) for key, value in updates.items()}


def pool_dynamics_domain_randomization_updates(profile: PoolDynamicsProfile) -> dict[str, Any]:
    """Return nested ``cfg.domain_randomization`` updates for a profile."""

    profile.validate()
    if profile.domain_randomization is None:
        return {}
    return {
        key: _as_plain_value(value)
        for key, value in profile.domain_randomization.to_cfg_updates().items()
    }


def pool_dynamics_profile_to_dict(profile: PoolDynamicsProfile) -> dict[str, Any]:
    """Return a JSON-friendly dictionary for a validated profile."""

    profile.validate()
    return _as_plain_value(asdict(profile))


def pool_dynamics_profile_from_dict(data: Mapping[str, Any]) -> PoolDynamicsProfile:
    """Build and validate a pool dynamics profile from a nested mapping."""

    if not isinstance(data, Mapping):
        raise TypeError("Pool dynamics profile data must be a mapping.")

    section_types = {
        "rigid_body": RigidBodyProfile,
        "hydrodynamics": HydrodynamicsProfile,
        "thrusters": ThrusterProfile,
        "battery": BatteryProfile,
        "pool_boundary": PoolBoundaryProfile,
        "free_surface": FreeSurfaceProfile,
        "tether": TetherProfile,
        "observation": ObservationProfile,
        "domain_randomization": DomainRandomizationProfile,
    }
    allowed = {"name", "description", *section_types}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown pool dynamics profile field(s): {', '.join(unknown)}.")

    kwargs: dict[str, Any] = {}
    if "name" in data:
        kwargs["name"] = data["name"]
    if "description" in data:
        kwargs["description"] = data["description"]

    for section_name, section_type in section_types.items():
        if section_name not in data:
            continue
        section_data = data[section_name]
        if section_name == "domain_randomization" and section_data is None:
            kwargs[section_name] = None
        else:
            kwargs[section_name] = _dataclass_from_mapping(section_type, section_data, section_name)

    profile = PoolDynamicsProfile(**kwargs)
    profile.validate()
    return profile


def load_pool_dynamics_profile_json(path: str | Path) -> PoolDynamicsProfile:
    """Load a pool dynamics profile from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream, parse_constant=_reject_json_constant)
    return pool_dynamics_profile_from_dict(data)


def resolve_pool_dynamics_profile(value: PoolDynamicsProfile | str | Path) -> PoolDynamicsProfile:
    """Resolve a profile object or JSON path through the same validation path."""

    if isinstance(value, PoolDynamicsProfile):
        value.validate()
        return value
    return load_pool_dynamics_profile_json(value)


def write_pool_dynamics_profile_json(profile: PoolDynamicsProfile, path: str | Path, indent: int = 2) -> None:
    """Write a validated pool dynamics profile to a JSON file."""

    data = pool_dynamics_profile_to_dict(profile)
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, indent=indent, sort_keys=True)
        stream.write("\n")


def _dataclass_from_mapping(cls: type, data: Mapping[str, Any], section_name: str) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError(f"{section_name} must be a mapping.")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section_name} field(s): {', '.join(unknown)}.")
    return cls(**{field.name: data[field.name] for field in fields(cls) if field.name in data})


def _combine_update_mappings(
    updates: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    name: str,
) -> dict[str, Any]:
    if updates is None:
        return {}
    if isinstance(updates, Mapping):
        mappings = (updates,)
    elif _is_sequence(updates):
        mappings = updates
    else:
        raise TypeError(f"{name} must be a mapping or sequence of mappings.")

    combined: dict[str, Any] = {}
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name}[{index}] must be a mapping.")
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise ValueError(f"{name}[{index}] keys must be strings.")
            combined[key] = copy.deepcopy(value)
    return combined


def apply_pool_dynamics_profile(
    cfg: Any,
    profile: PoolDynamicsProfile,
    *,
    include_legacy_domain_randomization: bool = True,
) -> Any:
    """Apply a pool dynamics profile to an AUV-style config object.

    The function mutates and returns ``cfg`` so callers can write:

    ``cfg = apply_pool_dynamics_profile(AUVTrajEnvCfg(), measured_profile)``

    ``include_legacy_domain_randomization`` defaults to ``True`` for existing
    calibration callers. Training environments pass ``False`` so their only
    randomization source is the independently versioned recipe.
    """

    for key, value in pool_dynamics_profile_to_cfg_updates(profile).items():
        setattr(cfg, key, copy.deepcopy(value))

    domain_updates = (
        pool_dynamics_domain_randomization_updates(profile)
        if include_legacy_domain_randomization
        else {}
    )
    if domain_updates:
        if not hasattr(cfg, "domain_randomization"):
            raise AttributeError("cfg must define domain_randomization to apply randomization updates.")
        for key, value in domain_updates.items():
            setattr(cfg.domain_randomization, key, copy.deepcopy(value))
    return cfg

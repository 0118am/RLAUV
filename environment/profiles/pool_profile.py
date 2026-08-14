"""Structured environment profiles for high-fidelity pool simulations.

The profile objects in this module are intentionally independent from
IsaacLab. They load versioned parameters from the repository, validate the
shapes that the AUV environment expects, and apply those values to any config
object with matching attributes.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

from robot.dynamics.parameters import AUV


NumberSequence = Sequence[float]
HydroCoefficients = Sequence[float] | Sequence[Sequence[float]]
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
    # ``None`` is valid only while assembling a partial randomization overlay;
    # complete recipes always materialize an explicit selection.
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
                if name == "water_current_tau_range" and float(value[0]) <= 0.0:
                    raise ValueError("domain_randomization.water_current_tau_range must be positive.")
        if self.thruster_command_delay_steps_range is not None:
            _validate_range(
                self.thruster_command_delay_steps_range,
                "domain_randomization.thruster_command_delay_steps_range",
                integer=True,
            )
            if int(self.thruster_command_delay_steps_range[0]) < 0:
                raise ValueError("domain_randomization.thruster_command_delay_steps_range must be non-negative.")
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
    description: str = "Nominal pool profile with neutral hydrodynamics and an opt-in high-order residual interface."
    rigid_body: RigidBodyProfile = field(default_factory=RigidBodyProfile)
    hydrodynamics: HydrodynamicsProfile = field(default_factory=HydrodynamicsProfile)
    thrusters: ThrusterProfile = field(default_factory=ThrusterProfile)
    battery: BatteryProfile = field(default_factory=BatteryProfile)
    pool_boundary: PoolBoundaryProfile = field(default_factory=PoolBoundaryProfile)
    free_surface: FreeSurfaceProfile = field(default_factory=FreeSurfaceProfile)
    tether: TetherProfile = field(default_factory=TetherProfile)

    def validate(self) -> None:
        self.rigid_body.validate()
        self.hydrodynamics.validate()
        self.thrusters.validate()
        self.battery.validate()
        self.pool_boundary.validate()
        self.free_surface.validate()
        self.tether.validate()


NOMINAL_POOL_DYNAMICS_PROFILE = PoolDynamicsProfile()



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
    ):
        updates.update(section.to_cfg_updates())
    return {key: _as_plain_value(value) for key, value in updates.items()}


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



def apply_pool_dynamics_profile(cfg: Any, profile: PoolDynamicsProfile) -> Any:
    """Apply a pool dynamics profile to an AUV-style config object.

    The function mutates and returns ``cfg`` so callers can write:

    ``cfg = apply_pool_dynamics_profile(AUVTrajEnvCfg(), measured_profile)``

    Domain randomization is applied separately from a versioned
    :class:`DomainRandomizationSpec`.
    """

    for key, value in pool_dynamics_profile_to_cfg_updates(profile).items():
        setattr(cfg, key, copy.deepcopy(value))

    return cfg

"""Locked-rotor envelope reconstruction."""

from __future__ import annotations

import math

from environment.openfoam.step_repair.core import _closed_solid, _common_volume, _solid_volume
from environment.openfoam.step_repair.pressure_boundary import (
    _axisymmetric_profile_solid,
    _body_direction_from_step,
    _body_point_from_step,
    _coaxial_cylinder_records,
    _longest_radius_interval,
)

def _measure_rotor_interfaces(
    motor_shell,
    hub_solid,
    propeller_solid,
    axis_point: list[float],
    axis_direction: list[float],
    config: dict,
) -> tuple[dict, dict, float, float]:
    tolerance = float(config["signature_tolerance_mm"])
    radius = float(config["nominal_motor_shaft_radius_mm"])
    motor_shaft = _longest_radius_interval(
        _coaxial_cylinder_records(motor_shell, axis_point, axis_direction),
        radius,
        tolerance,
        "motor shaft",
    )
    propeller_bore = _longest_radius_interval(
        _coaxial_cylinder_records(propeller_solid, axis_point, axis_direction),
        radius,
        tolerance,
        "propeller bore",
    )
    hub_records = _coaxial_cylinder_records(hub_solid, axis_point, axis_direction)
    if not hub_records:
        raise RuntimeError("Reviewed locked-propeller hub has no coaxial cylinders")
    return (
        motor_shaft,
        propeller_bore,
        min(item["start_mm"] for item in hub_records),
        max(item["end_mm"] for item in hub_records),
    )


def _validate_rotor_signatures(
    motor_shaft: dict,
    propeller_bore: dict,
    hub_start: float,
    hub_end: float,
    config: dict,
) -> None:
    tolerance = float(config["signature_tolerance_mm"])
    signatures = (
        (
            motor_shaft["end_mm"] - motor_shaft["start_mm"],
            float(config["expected_motor_shaft_length_mm"]),
            "motor shaft",
        ),
        (
            propeller_bore["end_mm"] - propeller_bore["start_mm"],
            float(config["expected_propeller_hub_length_mm"]),
            "propeller hub",
        ),
        (
            hub_end - hub_start,
            float(config["expected_separate_hub_length_mm"]),
            "separate hub",
        ),
    )
    for measured, expected, description in signatures:
        if abs(measured - expected) > tolerance:
            raise RuntimeError(
                f"Reviewed {description} length changed: expected {expected:g} mm, got {measured:.12g} mm"
            )
    if abs(hub_end - propeller_bore["start_mm"]) > tolerance:
        raise RuntimeError("Reviewed hub and propeller no longer meet at one axial plane")


def _motor_profile(
    axis_point: list[float],
    axis_direction: list[float],
    hub_start: float,
    motor_metadata: dict,
    config: dict,
):
    points = [(float(point[0]), float(point[1])) for point in config["axisymmetric_profile_mm"]]
    start = points[0][0]
    end = points[-1][0]
    tolerance = float(config["signature_tolerance_mm"])
    expected_start = hub_start - float(config["shaft_tip_extension_mm"])
    if abs(start - expected_start) > tolerance:
        raise RuntimeError("Reviewed motor profile no longer starts at the expected hub-tip overlap")
    if abs(end - float(motor_metadata["height_mm"])) > tolerance:
        raise RuntimeError("Reviewed motor profile no longer ends at the mount-side casing extension")
    return _axisymmetric_profile_solid(axis_point, axis_direction, points), points, start, end


def _overlap_evidence(
    envelope,
    mount_reference,
    hub_solid,
    propeller_solid,
    config: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    common = {
        "mount": _common_volume(envelope, mount_reference),
        "hub": _common_volume(envelope, hub_solid),
        "propeller": _common_volume(envelope, propeller_solid),
    }
    minimum = {name: float(value) for name, value in config["minimum_common_volume_mm3"].items()}
    failures = [
        f"{name} {common[name]:.6g} < {required:.6g} mm^3"
        for name, required in minimum.items()
        if common.get(name, -math.inf) < required
    ]
    if failures:
        raise RuntimeError("Locked motor-envelope overlap failed: " + "; ".join(failures))
    return common, minimum


def _volume_evidence(motor_shell, envelope, config: dict) -> tuple[float, float, float, float]:
    source_motor = _closed_solid(motor_shell)
    if source_motor is None:
        raise RuntimeError("Reviewed detailed motor shell is no longer a closed solid")
    source_volume = _solid_volume(source_motor)
    envelope_volume = _solid_volume(envelope)
    relative_error = abs(envelope_volume - source_volume) / source_volume
    maximum_error = float(config["maximum_source_volume_relative_error"])
    if relative_error > maximum_error:
        raise RuntimeError(
            "Smooth motor-envelope volume changed too much relative to the source motor: "
            f"{relative_error:.6%} > {maximum_error:.6%}"
        )
    return source_volume, envelope_volume, relative_error, maximum_error


def _axis_landmarks(
    axis_point: list[float],
    axis_direction: list[float],
    profile_start: float,
    profile_end: float,
    propeller_bore: dict,
    config: dict,
) -> tuple[list[float], list[float], list[float], list[float]]:
    from OCP.gp import gp_Dir, gp_Pnt

    direction = gp_Dir(*axis_direction).XYZ()
    origin = gp_Pnt(*axis_point).XYZ()
    shaft_end = float(config["shaft_refinement_end_mm"])
    coordinates = (
        origin + direction * profile_start,
        origin + direction * shaft_end,
        origin + direction * profile_end,
        origin + direction * (0.5 * (propeller_bore["start_mm"] + propeller_bore["end_mm"])),
    )
    return tuple([point.X(), point.Y(), point.Z()] for point in coordinates)


def _locked_rotor_envelope(
    motor_shell,
    mount_reference,
    hub_solid,
    propeller_solid,
    motor_metadata: dict,
    config: dict,
    translation_body_mm,
) -> tuple[object, dict]:
    """Build and validate one smooth casing/nose/locked-shaft motor solid."""

    axis_point = [float(value) for value in motor_metadata["start_step_mm"]]
    axis_direction = [float(value) for value in motor_metadata["direction_step"]]
    motor_shaft, propeller_bore, hub_start, hub_end = _measure_rotor_interfaces(
        motor_shell,
        hub_solid,
        propeller_solid,
        axis_point,
        axis_direction,
        config,
    )
    _validate_rotor_signatures(motor_shaft, propeller_bore, hub_start, hub_end, config)
    envelope, profile_points, profile_start, profile_end = _motor_profile(
        axis_point,
        axis_direction,
        hub_start,
        motor_metadata,
        config,
    )
    common, minimum_common = _overlap_evidence(
        envelope,
        mount_reference,
        hub_solid,
        propeller_solid,
        config,
    )
    source_volume, envelope_volume, volume_error, maximum_volume_error = _volume_evidence(
        motor_shell,
        envelope,
        config,
    )
    start_step, end_step, profile_end_step, propeller_centre_step = _axis_landmarks(
        axis_point,
        axis_direction,
        profile_start,
        profile_end,
        propeller_bore,
        config,
    )
    return envelope, {
        "condition": "static_locked",
        "representation": "single_axisymmetric_smooth_motor_envelope",
        "nominal_motor_shaft_radius_mm": float(config["nominal_motor_shaft_radius_mm"]),
        "connector_radius_mm": float(config["connector_radius_mm"]),
        "connector_length_mm": float(config["shaft_refinement_end_mm"]) - profile_start,
        "connector_axis_start_step_mm": start_step,
        "connector_axis_end_step_mm": end_step,
        "connector_axis_start_body_mm": _body_point_from_step(start_step, translation_body_mm),
        "connector_axis_end_body_mm": _body_point_from_step(end_step, translation_body_mm),
        "motor_profile_axis_end_step_mm": profile_end_step,
        "motor_profile_axis_end_body_mm": _body_point_from_step(profile_end_step, translation_body_mm),
        "axis_direction_step": axis_direction,
        "axis_direction_body": _body_direction_from_step(axis_direction),
        "propeller_centre_step_mm": propeller_centre_step,
        "propeller_centre_body_mm": _body_point_from_step(propeller_centre_step, translation_body_mm),
        "measured_motor_shaft_interval_from_casing_start_mm": [
            motor_shaft["start_mm"],
            motor_shaft["end_mm"],
        ],
        "measured_propeller_bore_interval_from_casing_start_mm": [
            propeller_bore["start_mm"],
            propeller_bore["end_mm"],
        ],
        "measured_hub_interval_from_casing_start_mm": [hub_start, hub_end],
        "axisymmetric_profile_mm": [list(point) for point in profile_points],
        "source_detailed_motor_volume_mm3": source_volume,
        "smooth_envelope_volume_mm3": envelope_volume,
        "source_volume_relative_error": volume_error,
        "maximum_source_volume_relative_error": maximum_volume_error,
        "common_volume_mm3": common,
        "minimum_common_volume_mm3": minimum_common,
    }

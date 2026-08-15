"""Validate the reviewed STEP repair configuration."""

from __future__ import annotations

import math
from pathlib import Path

from environment.openfoam.step_repair.core import (
    EXPECTED_BODY_TRANSLATION_MM,
    EXPECTED_SOURCE_COM_BODY_MM,
)

def _validate_source_and_frame(config: dict, source: Path):
    if not isinstance(config, dict):
        raise ValueError("Repair configuration root must be a JSON object")
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported repair configuration schema")
    source_config = config.get("source", {})
    if source_config.get("basename") != source.name:
        raise ValueError("Repair configuration basename does not match the input STEP")
    if source_config.get("units") != "mm":
        raise ValueError("This repair workflow requires a millimetre STEP source")
    selection = config.get("selection", {})
    if selection.get("default_closed_shell_action") != "keep":
        raise ValueError("Closed-shell default must remain 'keep'")
    if selection.get("open_shell_action") != "remove":
        raise ValueError("Open-shell action must remain 'remove'")
    if selection.get("replace_groups") != {
        "thruster_motor_with_cable": "smooth_axisymmetric_envelope"
    }:
        raise ValueError("Motor replacement contract changed unexpectedly")
    if not selection.get("hole_policy"):
        raise ValueError("Retained-hole policy must be explicit")
    
    output_frame = config.get("output_frame", {})
    if output_frame.get("name") != "body_flu_com":
        raise ValueError("Output frame must remain body_flu_com")
    if output_frame.get("mapping") != {
        "x_body": "z_step",
        "y_body": "x_step",
        "z_body": "y_step",
    }:
        raise ValueError("Output-frame mapping must remain (z_step,x_step,y_step)")
    if output_frame.get("source_com_body_mm") != list(EXPECTED_SOURCE_COM_BODY_MM):
        raise ValueError("Source COM must match the reviewed SolidWorks mass report")
    if output_frame.get("translation_mm") != list(EXPECTED_BODY_TRANSLATION_MM):
        raise ValueError("Output translation must move the reviewed source COM to zero")
    if not output_frame.get("reference_assumption"):
        raise ValueError("STEP/Coordinate System1 origin assumption must be explicit")
    return selection, output_frame


def _validate_shell_groups(config: dict, selection: dict):
    groups = config.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Repair configuration contains no reviewed shell groups")
    seen: dict[int, str] = {}
    for name, values in groups.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Reviewed shell group {name!r} must be non-empty")
        for index in values:
            if not isinstance(index, int) or index < 1:
                raise ValueError(f"Invalid shell index {index!r} in group {name!r}")
            if index in seen:
                raise ValueError(
                    f"Shell {index} occurs in both {seen[index]!r} and {name!r}"
                )
            seen[index] = name
    
    remove_names = selection.get("remove_groups", [])
    preserve_names = selection.get("preserve_groups", [])
    if len(remove_names) != len(set(remove_names)) or len(preserve_names) != len(
        set(preserve_names)
    ):
        raise ValueError("Remove/preserve group lists must not contain duplicates")
    unknown = (set(remove_names) | set(preserve_names)) - set(groups)
    if unknown:
        raise ValueError("Unknown reviewed shell group(s): " + ", ".join(sorted(unknown)))
    remove_indices = {index for name in remove_names for index in groups[name]}
    preserve_indices = {index for name in preserve_names for index in groups[name]}
    overlap = remove_indices & preserve_indices
    if overlap:
        raise ValueError(f"Shells cannot be both removed and preserved: {sorted(overlap)}")
    pressure_hull = set(groups.get("main_pressure_hull", []))
    if pressure_hull != {30} or not pressure_hull <= preserve_indices:
        raise ValueError("Reviewed main pressure hull shell 30 must be explicitly preserved")
    pressure_endcaps = set(groups.get("pressure_hull_endcaps", []))
    if pressure_endcaps != {42, 43} or not pressure_endcaps <= preserve_indices:
        raise ValueError("Reviewed pressure-hull endcap shells 42/43 must be explicitly preserved")
    buoyancy_material = set(groups.get("main_closed_cell_buoyancy_material", []))
    if buoyancy_material != {257} or not buoyancy_material <= preserve_indices:
        raise ValueError("Reviewed closed-cell buoyancy-material shell 257 must be preserved")
    motors = set(groups.get("thruster_motor_with_cable", []))
    if not motors or not motors <= remove_indices:
        raise ValueError("Every detailed motor must be removed before replacement")
    propellers = set(groups.get("propeller_3blade", [])) | set(
        groups.get("propeller_4blade", [])
    )
    hubs = set(groups.get("propeller_hub_or_nut", []))
    if len(propellers) != 8 or len(hubs) != 8:
        raise ValueError("Locked-rotor geometry requires eight propellers and eight hubs")
    if not propellers <= preserve_indices or not hubs <= preserve_indices:
        raise ValueError("Every locked propeller and hub must be explicitly preserved")
    return pressure_hull, pressure_endcaps, buoyancy_material, motors, propellers, hubs, preserve_indices


def _validate_motor_replacement(config: dict, motors: set[int], preserve_indices: set[int]) -> None:
    replacement = config.get("motor_replacement", {})
    if replacement.get("shape") != "smooth_axisymmetric_envelope":
        raise ValueError("Only the reviewed smooth axisymmetric motor envelope is supported")
    side_keys = {int(value) for value in replacement.get("mount_side_by_shell", {})}
    mount_keys = {int(value) for value in replacement.get("mount_shells_by_motor", {})}
    if side_keys != motors or mount_keys != motors:
        raise ValueError("Motor side/mount maps must exactly cover the detailed motors")
    if any(
        side not in {"vmin", "vmax"}
        for side in replacement["mount_side_by_shell"].values()
    ):
        raise ValueError("Motor mount sides must be vmin or vmax")
    if any(
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) for value in values)
        for values in replacement["mount_shells_by_motor"].values()
    ):
        raise ValueError("Every motor mount validation set must be a non-empty integer list")
    mount_targets = {
        int(index)
        for values in replacement["mount_shells_by_motor"].values()
        for index in values
    }
    if not mount_targets <= preserve_indices:
        raise ValueError("Every motor validation mount must be explicitly preserved")
    extension = replacement.get("mount_extension_mm")
    minimum_common = replacement.get("minimum_common_volume_mm3")
    if not isinstance(extension, (int, float)) or not math.isfinite(extension) or not (
        0.0 < extension <= 10.0
    ):
        raise ValueError("Motor mount extension must be finite and in (0, 10] mm")
    if (
        not isinstance(minimum_common, (int, float))
        or not math.isfinite(minimum_common)
        or minimum_common <= 0.0
    ):
        raise ValueError("Minimum motor/mount common volume must be finite and positive")


def _validate_locked_rotors(
    config: dict,
    motors: set[int],
    propellers: set[int],
    hubs: set[int],
) -> None:
    locked = config.get("locked_propeller", {})
    if locked.get("condition") != "fully_assembled_static_locked":
        raise ValueError("Production rotor condition must remain fully assembled and static locked")
    assemblies = locked.get("assemblies_by_motor", {})
    try:
        assembly_motor_indices = {int(index) for index in assemblies}
    except (TypeError, ValueError) as exc:
        raise ValueError("Locked-rotor motor keys must be integer strings") from exc
    if assembly_motor_indices != motors:
        raise ValueError("Locked-rotor assemblies must exactly cover all replacement motors")
    assembly_propellers = {int(item.get("propeller_shell", -1)) for item in assemblies.values()}
    assembly_hubs = {int(item.get("hub_shell", -1)) for item in assemblies.values()}
    assembly_labels = [item.get("label") for item in assemblies.values()]
    if assembly_propellers != propellers or assembly_hubs != hubs:
        raise ValueError("Locked-rotor assemblies must exactly cover reviewed propellers and hubs")
    if set(assembly_labels) != {f"T{index}" for index in range(1, 9)}:
        raise ValueError("Locked-rotor assemblies must contain unique T1--T8 labels")
    locked_numeric = (
        "nominal_motor_shaft_radius_mm",
        "connector_radius_mm",
        "shaft_tip_extension_mm",
        "expected_motor_shaft_length_mm",
        "expected_propeller_hub_length_mm",
        "expected_separate_hub_length_mm",
        "signature_tolerance_mm",
    )
    for name in locked_numeric:
        value = locked.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Locked-rotor {name} must be finite and positive")
    nominal_shaft_radius = float(locked["nominal_motor_shaft_radius_mm"])
    connector_radius = float(locked["connector_radius_mm"])
    if not nominal_shaft_radius < connector_radius <= nominal_shaft_radius + 0.1:
        raise ValueError("Locked-rotor connector radius must use only the reviewed bore overlap")
    refinement_end = locked.get("shaft_refinement_end_mm")
    if not isinstance(refinement_end, (int, float)) or not math.isfinite(refinement_end):
        raise ValueError("Locked-rotor shaft refinement end must be finite")
    maximum_volume_error = locked.get("maximum_source_volume_relative_error")
    if (
        not isinstance(maximum_volume_error, (int, float))
        or not math.isfinite(maximum_volume_error)
        or not 0.0 < maximum_volume_error <= 0.05
    ):
        raise ValueError("Locked motor-envelope source-volume tolerance must lie in (0, 0.05]")
    profile = locked.get("axisymmetric_profile_mm")
    if (
        not isinstance(profile, list)
        or len(profile) < 4
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in point
            )
            for point in profile
        )
    ):
        raise ValueError("Locked motor axisymmetric profile must contain finite [axial,radius] points")
    if profile[0][1] != 0.0 or profile[-1][1] != 0.0:
        raise ValueError("Locked motor profile must start and end on its axis")
    if any(float(point[1]) < 0.0 for point in profile):
        raise ValueError("Locked motor profile radii cannot be negative")
    if any(
        float(second[0]) < float(first[0])
        for first, second in zip(profile, profile[1:])
    ):
        raise ValueError("Locked motor profile axial coordinates must be nondecreasing")
    if not float(profile[0][0]) < float(refinement_end) < float(profile[-1][0]):
        raise ValueError("Shaft refinement endpoint must lie inside the motor profile")
    locked_common = locked.get("minimum_common_volume_mm3", {})
    if set(locked_common) != {"mount", "hub", "propeller"} or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in locked_common.values()
    ):
        raise ValueError("Locked-rotor common-volume gates are incomplete or invalid")


def _validate_sealed_boundary(
    config: dict,
    pressure_hull: set[int],
    pressure_endcaps: set[int],
) -> None:
    sealed = config.get("sealed_pressure_boundary", {})
    if sealed.get("condition") != "waterproof_assembled_vehicle":
        raise ValueError(
            "Pressure boundary requires the waterproof assembled-vehicle condition"
        )
    if sealed.get("representation") != "two_tube_opening_sealing_disks":
        raise ValueError("Pressure boundary must use two tube-opening sealing disks")
    if int(sealed.get("hull_shell", -1)) not in pressure_hull:
        raise ValueError("Pressure boundary must derive from the reviewed hull shell")
    if set(sealed.get("endcap_shells", [])) != pressure_endcaps:
        raise ValueError("Pressure boundary must reference both reviewed end fittings")
    endcap_by_end = sealed.get("endcap_by_hull_end", {})
    if set(endcap_by_end) != {"start", "end"} or set(endcap_by_end.values()) != pressure_endcaps:
        raise ValueError(
            "Pressure boundary must map one reviewed end fitting to each hull end"
        )
    sealed_numeric = (
        "expected_inner_radius_mm",
        "expected_outer_radius_mm",
        "expected_hull_length_mm",
        "radial_wall_overlap_mm",
        "disk_half_thickness_mm",
        "signature_tolerance_mm",
    )
    for name in sealed_numeric:
        value = sealed.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Sealed pressure-boundary {name} must be finite and positive")
    inner_radius = float(sealed["expected_inner_radius_mm"])
    outer_radius = float(sealed["expected_outer_radius_mm"])
    radial_overlap = float(sealed["radial_wall_overlap_mm"])
    if not inner_radius < inner_radius + radial_overlap < outer_radius:
        raise ValueError(
            "Sealed pressure-boundary radial overlap must remain inside the hull wall"
        )
    if 2.0 * float(sealed["disk_half_thickness_mm"]) > 2.0:
        raise ValueError("Pressure-boundary disks must remain no thicker than 2 mm")
    expected_sealed_common = {
        "start_disk_hull",
        "start_disk_endcap",
        "end_disk_hull",
        "end_disk_endcap",
    }
    sealed_common = sealed.get("minimum_common_volume_mm3", {})
    if set(sealed_common) != expected_sealed_common or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in sealed_common.values()
    ):
        raise ValueError(
            "Sealed pressure-boundary common-volume gates are incomplete or invalid"
        )


def _validate_buoyancy(config: dict, buoyancy_material: set[int]) -> None:
    buoyancy = config.get("buoyancy_material_validation", {})
    if int(buoyancy.get("shell", -1)) not in buoyancy_material:
        raise ValueError("Buoyancy-material validation must reference shell 257")
    if buoyancy.get("role") != "waterproof closed-cell main buoyancy material":
        raise ValueError("Shell 257 must remain identified as the main buoyancy material")
    if buoyancy.get("condition") != "waterproof_closed_cell":
        raise ValueError("Main buoyancy material must remain waterproof and closed-cell")
    if buoyancy.get("identification_status") != "high-confidence geometry/placement inference":
        raise ValueError("Buoyancy-material identification confidence must remain explicit")
    expected_buoyancy_volume = buoyancy.get("expected_closed_solid_volume_mm3")
    buoyancy_volume_tolerance = buoyancy.get("volume_relative_tolerance")
    buoyancy_bbox_tolerance = buoyancy.get("bbox_absolute_tolerance_mm")
    if (
        not isinstance(expected_buoyancy_volume, (int, float))
        or not math.isfinite(expected_buoyancy_volume)
        or expected_buoyancy_volume <= 0.0
        or not isinstance(buoyancy_volume_tolerance, (int, float))
        or not math.isfinite(buoyancy_volume_tolerance)
        or not 0.0 < buoyancy_volume_tolerance < 0.05
        or not isinstance(buoyancy_bbox_tolerance, (int, float))
        or not math.isfinite(buoyancy_bbox_tolerance)
        or buoyancy_bbox_tolerance <= 0.0
    ):
        raise ValueError("Buoyancy-material volume gate is invalid")
    buoyancy_bbox = buoyancy.get("expected_bbox_step_mm")
    if (
        not isinstance(buoyancy_bbox, list)
        or len(buoyancy_bbox) != 6
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in buoyancy_bbox
        )
    ):
        raise ValueError("Buoyancy-material expected STEP bbox must contain six finite values")
    if not buoyancy.get("hydrodynamic_accounting"):
        raise ValueError("Buoyancy-material anti-double-counting policy must be explicit")


def _validate_volume(config: dict) -> None:
    volume_validation = config.get("volume_validation", {})
    target_volume = volume_validation.get("target_displaced_volume_mm3")
    wrapped_tolerance = volume_validation.get("wrapped_surface_relative_tolerance")
    mesh_tolerance = volume_validation.get("snappy_excluded_volume_relative_tolerance")
    if (
        not isinstance(target_volume, (int, float))
        or not math.isfinite(target_volume)
        or target_volume <= 0.0
    ):
        raise ValueError("Target displaced volume must be a positive finite mm^3 value")
    for name, value in (
        ("wrapped surface", wrapped_tolerance),
        ("snappy excluded volume", mesh_tolerance),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
        ):
            raise ValueError(f"{name} relative tolerance must lie in (0, 1)")


def _validate_frame_landmarks(output_frame: dict, propellers: set[int]) -> None:
    frame_validation = output_frame.get("validation", {})
    landmarks = frame_validation.get(
        "propeller_landmarks", []
    )
    landmark_indices = {int(item["shell_index"]) for item in landmarks}
    propeller_indices = propellers
    if landmark_indices != propeller_indices or len(landmarks) != len(
        landmark_indices
    ):
        raise ValueError("Labelled frame landmarks must exactly cover all propellers")
    labels = [item.get("label") for item in landmarks]
    if set(labels) != {f"T{index}" for index in range(1, 9)}:
        raise ValueError("Frame landmarks must contain unique T1--T8 labels")
    for item in landmarks:
        point = item.get("expected_body_mm")
        if (
            not isinstance(point, list)
            or len(point) != 3
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in point
            )
        ):
            raise ValueError("Frame landmark coordinates must be finite XYZ triples")
    frame_tolerance = frame_validation.get("max_nearest_error_mm")
    if (
        not isinstance(frame_tolerance, (int, float))
        or not math.isfinite(frame_tolerance)
        or frame_tolerance <= 0.0
    ):
        raise ValueError("Frame landmark tolerance must be finite and positive")


def _validate_output_parameters(config: dict) -> None:
    triangulation = config.get("triangulation", {})
    for name in ("linear_deflection_mm", "angular_deflection_rad"):
        value = triangulation.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"Configured {name} must be finite and positive")
    crosscheck = config.get("entity_crosscheck", {})
    for name in ("main_outer_fairing", "thruster_support_complex", "propellers", "motors"):
        values = crosscheck.get(name)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, int) or value < 1 for value in values
        ):
            raise ValueError(f"Missing reviewed STEP entity cross-check group {name!r}")


def _validate_config(config: dict, source: Path) -> None:
    """Reject semantic drift before loading the expensive CAD model."""

    selection, output_frame = _validate_source_and_frame(config, source)
    (
        pressure_hull,
        pressure_endcaps,
        buoyancy_material,
        motors,
        propellers,
        hubs,
        preserve_indices,
    ) = _validate_shell_groups(config, selection)
    _validate_motor_replacement(config, motors, preserve_indices)
    _validate_locked_rotors(config, motors, propellers, hubs)
    _validate_sealed_boundary(config, pressure_hull, pressure_endcaps)
    _validate_buoyancy(config, buoyancy_material)
    _validate_volume(config)
    _validate_frame_landmarks(output_frame, propellers)
    _validate_output_parameters(config)

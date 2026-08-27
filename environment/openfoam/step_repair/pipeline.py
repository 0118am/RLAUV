"""Orchestrate shell selection, reconstructed solids, and report output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from environment.openfoam.step_repair.core import (
    DEFAULT_OCP_SITE,
    EXPECTED_BODY_TRANSLATION_MM,
    _bbox,
    _closed_solid,
    _dominant_motor_cylinder,
    _load_ocp,
    _make_compound,
    _read_step,
    _shell_map,
    _to_body_flu,
)
from environment.openfoam.step_repair.output import (
    _audit_binary_stl,
    _preflight_paths,
    _validate_body_frame,
    _write_stl,
)
from environment.openfoam.step_repair.pressure_boundary import (
    _sealed_pressure_boundary,
    _validate_buoyancy_material,
)
from environment.openfoam.step_repair.rotor import _locked_rotor_envelope
from environment.openfoam.step_repair.validation import _validate_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class SelectionPlan:
    remove_indices: set[int]
    preserve_indices: set[int]
    motor_indices: list[int]
    motor_extension_mm: float
    motor_mount_sides: dict[int, str]
    motor_mount_shells: dict[int, list[int]]
    locked_config: dict
    locked_assemblies: dict[int, dict[str, Any]]


def _load_source(args: argparse.Namespace):
    source, config_path, output, report_path = _preflight_paths(
        args.input, args.config, args.output, args.report, args.force
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, source)
    ocp_version = _load_ocp()
    shells = _shell_map(_read_step(source))
    configured_indices = [index for values in config["groups"].values() for index in values]
    if configured_indices and max(configured_indices) > shells.Extent():
        raise RuntimeError(
            "Repair configuration references shell "
            f"{max(configured_indices)}, but the source contains only {shells.Extent()} shells"
        )
    return source, config_path, output, report_path, config, ocp_version, shells


def _frame_and_mesh_settings(config: dict, shells):
    frame_config = config["output_frame"]
    translation = [float(value) for value in frame_config["translation_mm"]]
    if translation != list(EXPECTED_BODY_TRANSLATION_MM):
        raise RuntimeError("Configured body translation does not centre the reviewed COM")
    frame_validation = _validate_body_frame(shells, frame_config["validation"], translation)
    mesh_config = config["triangulation"]
    deflection = float(mesh_config["linear_deflection_mm"])
    angular_deflection = float(mesh_config["angular_deflection_rad"])
    if not math.isfinite(deflection) or deflection <= 0:
        raise RuntimeError("Configured linear triangulation deflection must be positive")
    if not math.isfinite(angular_deflection) or angular_deflection <= 0:
        raise RuntimeError("Configured angular triangulation deflection must be positive")
    return frame_config, translation, frame_validation, deflection, angular_deflection


def _selection_plan(config: dict) -> SelectionPlan:
    groups = config["groups"]
    motor_indices = groups["thruster_motor_with_cable"]
    replacement = config["motor_replacement"]
    motor_mount_sides = {int(index): side for index, side in replacement["mount_side_by_shell"].items()}
    motor_mount_shells = {
        int(index): [int(value) for value in shell_indices]
        for index, shell_indices in replacement["mount_shells_by_motor"].items()
    }
    if set(motor_indices) != set(motor_mount_sides) or set(motor_indices) != set(motor_mount_shells):
        raise RuntimeError("Every reviewed motor shell must have side and mount decisions")
    locked_config = config["locked_propeller"]
    locked_assemblies = {
        int(index): {
            "label": item["label"],
            "propeller_shell": int(item["propeller_shell"]),
            "hub_shell": int(item["hub_shell"]),
        }
        for index, item in locked_config["assemblies_by_motor"].items()
    }
    return SelectionPlan(
        remove_indices={
            index
            for name in config["selection"]["remove_groups"]
            for index in groups.get(name, [])
        },
        preserve_indices={
            index
            for name in config["selection"]["preserve_groups"]
            for index in groups[name]
        },
        motor_indices=motor_indices,
        motor_extension_mm=float(replacement["mount_extension_mm"]),
        motor_mount_sides=motor_mount_sides,
        motor_mount_shells=motor_mount_shells,
        locked_config=locked_config,
        locked_assemblies=locked_assemblies,
    )


def _retain_closed_shells(shells, remove_indices: set[int]):
    from OCP.TopoDS import TopoDS

    kept = []
    open_removed = []
    explicit_removed = []
    for index in range(1, shells.Extent() + 1):
        shell = TopoDS.Shell_s(shells.FindKey(index))
        if index in remove_indices:
            explicit_removed.append(index)
            continue
        solid = _closed_solid(shell)
        if solid is None:
            open_removed.append(index)
        else:
            kept.append((index, solid))
    return kept, open_removed, explicit_removed


def _buoyancy_metadata(config: dict, kept_by_index: dict[int, Any]) -> dict:
    buoyancy_config = config["buoyancy_material_validation"]
    shell_index = int(buoyancy_config["shell"])
    if shell_index not in kept_by_index:
        raise RuntimeError(f"Main buoyancy-material shell {shell_index} was not retained")
    return _validate_buoyancy_material(kept_by_index[shell_index], buoyancy_config)


def _append_motor_envelopes(
    shells,
    kept: list,
    kept_by_index: dict[int, Any],
    plan: SelectionPlan,
    translation: list[float],
) -> tuple[list[dict], list[dict]]:
    from OCP.TopoDS import TopoDS

    replacements: list[dict] = []
    assemblies: list[dict] = []
    for index in plan.motor_indices:
        shell = TopoDS.Shell_s(shells.FindKey(index))
        _, metadata = _dominant_motor_cylinder(
            shell,
            plan.motor_extension_mm,
            plan.motor_mount_sides[index],
        )
        mounts = plan.motor_mount_shells[index]
        missing_mounts = [mount for mount in mounts if mount not in kept_by_index]
        if missing_mounts:
            raise RuntimeError(f"Motor {index} validation mounts were not retained: {missing_mounts}")
        assembly = plan.locked_assemblies[index]
        propeller_index = assembly["propeller_shell"]
        hub_index = assembly["hub_shell"]
        missing_rotor_shells = [
            shell_index
            for shell_index in (propeller_index, hub_index)
            if shell_index not in kept_by_index
        ]
        if missing_rotor_shells:
            raise RuntimeError(
                f"Locked rotor {assembly['label']} shells were not retained: {missing_rotor_shells}"
            )
        metadata["source_shell_index"] = index
        metadata["validated_mount_shell_indices"] = mounts
        envelope, envelope_metadata = _locked_rotor_envelope(
            shell,
            _make_compound([kept_by_index[mount] for mount in mounts]),
            kept_by_index[hub_index],
            kept_by_index[propeller_index],
            metadata,
            plan.locked_config,
            translation,
        )
        envelope_metadata.update(
            {
                "label": assembly["label"],
                "source_motor_shell_index": index,
                "source_propeller_shell_index": propeller_index,
                "source_hub_shell_index": hub_index,
            }
        )
        metadata["locked_rotor_envelope"] = envelope_metadata
        replacements.append(metadata)
        assemblies.append(envelope_metadata)
        kept.append((f"motor_envelope_{index}", envelope))
    return replacements, assemblies


def _append_sealed_boundary(
    shells,
    kept: list,
    kept_by_index: dict[int, Any],
    config: dict,
    translation: list[float],
) -> dict:
    from OCP.TopoDS import TopoDS

    sealed = config["sealed_pressure_boundary"]
    hull_index = int(sealed["hull_shell"])
    endcap_indices = [int(value) for value in sealed["endcap_shells"]]
    missing = [index for index in [hull_index, *endcap_indices] if index not in kept_by_index]
    if missing:
        raise RuntimeError(f"Sealed pressure-boundary source shells were not retained: {missing}")
    boundary, metadata = _sealed_pressure_boundary(
        TopoDS.Shell_s(shells.FindKey(hull_index)),
        kept_by_index[hull_index],
        {index: kept_by_index[index] for index in endcap_indices},
        sealed,
        translation,
    )
    kept.append(("sealed_pressure_boundary", boundary))
    return metadata


def _report_payload(
    *,
    source: Path,
    config_path: Path,
    output: Path,
    config: dict,
    ocp_version: str,
    frame_config: dict,
    translation: list[float],
    frame_validation: dict,
    shells,
    kept_indices: set[int],
    explicit_removed: list[int],
    open_removed: list[int],
    replacements: list[dict],
    assemblies: list[dict],
    buoyancy: dict,
    sealed_boundary: dict,
    compound_body,
    temporary_output: Path,
    deflection: float,
    angular_deflection: float,
    triangulation: dict,
    audit: dict,
) -> dict:
    return {
        "schema_version": 2,
        "source": str(source),
        "source_sha256": _sha256(source),
        "cad_backend": {
            "name": "cadquery-ocp-novtk",
            "ocp_version": ocp_version,
            "module_path": str(DEFAULT_OCP_SITE.resolve()),
        },
        "selection_config": str(config_path),
        "selection_config_sha256": _sha256(config_path),
        "source_units": "mm",
        "output_units": "mm",
        "output_frame": "body_flu_com",
        "axis_mapping": ["x_body=z_step", "y_body=x_step", "z_body=y_step"],
        "source_com_body_mm": frame_config["source_com_body_mm"],
        "translation_body_mm": translation,
        "reference_assumption": frame_config["reference_assumption"],
        "body_frame_validation": frame_validation,
        "volume_validation": config["volume_validation"],
        "reviewed_entity_crosscheck": config["entity_crosscheck"],
        "shell_count": shells.Extent(),
        "closed_shells_kept": len(kept_indices),
        "explicit_shells_removed": sorted(explicit_removed),
        "open_shells_removed": sorted(open_removed),
        "smooth_motor_replacements": replacements,
        "locked_rotor_condition": "fully_assembled_static_locked",
        "locked_rotor_assemblies": sorted(assemblies, key=lambda item: item["label"]),
        "main_buoyancy_material": buoyancy,
        "sealed_pressure_boundary": sealed_boundary,
        "output": str(output),
        "output_sha256": _sha256(temporary_output),
        "output_size_bytes": temporary_output.stat().st_size,
        "output_bbox_body_mm": _bbox(compound_body),
        "triangulation": {
            "linear_deflection_mm": deflection,
            "angular_deflection_rad": angular_deflection,
            **triangulation,
            **audit,
        },
        "warning": (
            "This STL contains intersecting retained solids and is an intermediate "
            "input to voxel_wrap.py, not the final CFD wetted surface."
        ),
    }


def _write_outputs(
    *,
    output: Path,
    report_path: Path,
    compound_body,
    report_context: dict,
    deflection: float,
    angular_deflection: float,
) -> dict:
    output_fd, output_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".stl",
    )
    os.close(output_fd)
    report_fd, report_name = tempfile.mkstemp(
        dir=report_path.parent,
        prefix=f".{report_path.name}.",
        suffix=".json",
    )
    os.close(report_fd)
    temporary_output = Path(output_name)
    temporary_report = Path(report_name)
    try:
        triangulation = _write_stl(compound_body, temporary_output, deflection, angular_deflection)
        audit = _audit_binary_stl(temporary_output)
        report = _report_payload(
            **report_context,
            temporary_output=temporary_output,
            deflection=deflection,
            angular_deflection=angular_deflection,
            triangulation=triangulation,
            audit=audit,
        )
        temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_output, output)
        os.replace(temporary_report, report_path)
        return report
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)


def prepare(args: argparse.Namespace) -> dict:
    source, config_path, output, report_path, config, ocp_version, shells = _load_source(args)
    frame_config, translation, frame_validation, deflection, angular_deflection = (
        _frame_and_mesh_settings(config, shells)
    )
    plan = _selection_plan(config)
    kept, open_removed, explicit_removed = _retain_closed_shells(shells, plan.remove_indices)
    kept_by_index = dict(kept)
    buoyancy = _buoyancy_metadata(config, kept_by_index)
    replacements, assemblies = _append_motor_envelopes(
        shells,
        kept,
        kept_by_index,
        plan,
        translation,
    )
    sealed_boundary = _append_sealed_boundary(
        shells,
        kept,
        kept_by_index,
        config,
        translation,
    )
    kept_indices = {index for index, _ in kept if isinstance(index, int)}
    missing_preserved = sorted(plan.preserve_indices - kept_indices)
    if missing_preserved:
        raise RuntimeError(f"Required preserved shells were not retained: {missing_preserved}")
    compound_body = _to_body_flu(_make_compound([shape for _, shape in kept]), translation)
    context = {
        "source": source,
        "config_path": config_path,
        "output": output,
        "config": config,
        "ocp_version": ocp_version,
        "frame_config": frame_config,
        "translation": translation,
        "frame_validation": frame_validation,
        "shells": shells,
        "kept_indices": kept_indices,
        "explicit_removed": explicit_removed,
        "open_removed": open_removed,
        "replacements": replacements,
        "assemblies": assemblies,
        "buoyancy": buoyancy,
        "sealed_boundary": sealed_boundary,
        "compound_body": compound_body,
    }
    return _write_outputs(
        output=output,
        report_path=report_path,
        compound_body=compound_body,
        report_context=context,
        deflection=deflection,
        angular_deflection=angular_deflection,
    )

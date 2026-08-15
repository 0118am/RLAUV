"""Per-run environment profiles and the default trajectory curriculum."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

from environment.profiles.domain_randomization import (
    domain_randomization_parameters_requiring_sources,
    load_domain_randomization_spec_json,
    write_domain_randomization_spec_json,
)
from environment.profiles.environment_profile import (
    load_environment_profile_json,
    write_environment_profile_json,
)
from simulation.isaac.trajectory.experiment_models import TrajectoryCurriculumRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT_PROFILE = (
    PROJECT_ROOT / "environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json"
)
DEFAULT_DOMAIN_RANDOMIZATION_SPEC = (
    PROJECT_ROOT / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
)
_PROFILE_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TrainingProfilePaths:
    """Materialized environment and DR inputs selected by ``train.ipynb``."""

    environment: Path
    randomization: Path


def materialize_training_profiles(
    run_name: str,
    *,
    environment_base: str | Path = DEFAULT_ENVIRONMENT_PROFILE,
    randomization_base: str | Path = DEFAULT_DOMAIN_RANDOMIZATION_SPEC,
    output_root: str | Path = PROJECT_ROOT / "simulation/isaac/rlpolicy/_configs",
    hydrodynamics: dict[str, Any] | None = None,
    pool_boundary: dict[str, Any] | None = None,
    free_surface: dict[str, Any] | None = None,
    randomization: dict[str, Any] | None = None,
) -> TrainingProfilePaths:
    """Write immutable per-run inputs from notebook-selected numeric values.

    The notebook owns experiment choices; the source profiles remain reviewed
    baselines. Generated files live beside ignored policy outputs so changing
    one run never mutates the repository's physical source data.
    """

    slug = _PROFILE_NAME_SAFE.sub("-", run_name.strip()).strip("-._")
    if not slug:
        raise ValueError("run_name must contain at least one filename-safe character.")
    output_dir = Path(output_root).expanduser().resolve() / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    base_environment = load_environment_profile_json(environment_base)
    resolved_environment = replace(
        base_environment,
        name=f"{base_environment.name}--{slug}",
        description=f"{base_environment.description} Per-run values selected by train.ipynb ({slug}).",
        hydrodynamics=replace(base_environment.hydrodynamics, **(hydrodynamics or {})),
        pool_boundary=replace(base_environment.pool_boundary, **(pool_boundary or {})),
        free_surface=replace(base_environment.free_surface, **(free_surface or {})),
    )
    environment_path = output_dir / "environment.json"
    write_environment_profile_json(resolved_environment, environment_path)

    base_randomization = load_domain_randomization_spec_json(randomization_base)
    resolved_parameters = replace(base_randomization.parameters, **(randomization or {}))
    sources = dict(base_randomization.parameter_sources)
    for parameter in domain_randomization_parameters_requiring_sources(resolved_parameters):
        sources.setdefault(parameter, f"Numeric value selected in train.ipynb for run {slug}.")
    metadata = dict(base_randomization.metadata)
    metadata["configured_by"] = "train.ipynb"
    metadata["run_name"] = slug
    resolved_randomization = replace(
        base_randomization,
        name=f"{base_randomization.name}--{slug}",
        description=f"{base_randomization.description} Per-run values selected by train.ipynb ({slug}).",
        base_profile_name=resolved_environment.name,
        parameters=resolved_parameters,
        parameter_sources=sources,
        metadata=metadata,
    )
    randomization_path = output_dir / "domain_randomization.json"
    write_domain_randomization_spec_json(resolved_randomization, randomization_path)
    return TrainingProfilePaths(environment=environment_path, randomization=randomization_path)


def default_trajectory_curriculum() -> TrajectoryCurriculumRequest:
    """Return the version-controlled T60 speed and amplitude curriculum."""

    return TrajectoryCurriculumRequest(
        enabled=True,
        amplitude_x_range=(0.60, 0.78),
        amplitude_y_range=(0.55, 0.75),
        amplitude_z_range=(0.08, 0.20),
        period_range=(10.0, 20.0),
        speed_levels_mps=(0.1, 0.2, 0.3, 0.4),
        stage_steps=(9_750, 22_500, 40_500),
        stage_0_types=(8, 9, 10),
        stage_1_types=(8, 9, 10),
        stage_2_types=(8, 9, 10),
        stage_3_types=(8, 9, 10),
        amplitude_scales=(0.55, 0.75, 0.90, 1.0),
        vertical_amplitude_scales=(0.25, 0.50, 0.75, 1.0),
        period_min_by_stage=(20.0, 10.0, 10.0, 10.0),
        period_max_by_stage=(20.0, 10.0, 10.0, 10.0),
    )

"""Strict simulator-independent profile for water and pool physics only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .pool_profile import FreeSurfaceProfile, HydrodynamicsProfile, PoolBoundaryProfile


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

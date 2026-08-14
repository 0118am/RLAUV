"""Build a measured PoolDynamicsProfile JSON from calibration update files."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.profiles.pool_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolDynamicsProfile,
    load_pool_dynamics_profile_json,
    merge_pool_dynamics_cfg_updates,
    write_pool_dynamics_profile_json,
)
from environment.profiles.domain_randomization import (  # noqa: E402
    DomainRandomizationSpec,
    complete_domain_randomization_profile,
    domain_randomization_parameters_requiring_sources,
    domain_randomization_spec_from_pool_profile,
    write_domain_randomization_spec_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge calibration to_cfg_updates() JSON files into a PoolDynamicsProfile JSON.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        help="Optional existing PoolDynamicsProfile JSON to update. Defaults to the nominal profile.",
    )
    parser.add_argument(
        "--updates",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON file containing flat cfg updates, or a wrapper with cfg_updates and "
            "domain_randomization_updates. May be repeated; later files override earlier ones."
        ),
    )
    parser.add_argument(
        "--domain-randomization-updates",
        type=Path,
        action="append",
        default=[],
        help="JSON file containing flat DomainRandomizationProfile updates. May be repeated.",
    )
    parser.add_argument("--name", help="Override the output profile name.")
    parser.add_argument("--description", help="Override the output profile description.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the merged profile JSON.")
    parser.add_argument(
        "--domain-randomization-output",
        type=Path,
        help="Optionally export the merged uncertainty ranges as a separate versioned recipe JSON.",
    )
    parser.add_argument(
        "--domain-randomization-name",
        help="Optional name for --domain-randomization-output.",
    )
    parser.add_argument(
        "--domain-randomization-sources",
        type=Path,
        help="Optional JSON mapping from randomization parameter names to provenance strings.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Ignore unknown update keys instead of failing.",
    )
    return parser


def load_update_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either a flat cfg update mapping or a cfg/domain wrapper mapping."""

    data = _load_json_mapping(path)
    if "cfg_updates" not in data and "domain_randomization_updates" not in data:
        return dict(data), {}

    allowed = {"cfg_updates", "domain_randomization_updates"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown wrapped update field(s): {', '.join(unknown)}.")

    cfg_updates = data.get("cfg_updates", {})
    domain_updates = data.get("domain_randomization_updates", {})
    if not isinstance(cfg_updates, Mapping):
        raise TypeError(f"{path}: cfg_updates must be a mapping.")
    if not isinstance(domain_updates, Mapping):
        raise TypeError(f"{path}: domain_randomization_updates must be a mapping.")
    return dict(cfg_updates), dict(domain_updates)


def build_profile_from_files(
    *,
    base_profile_path: Path | None,
    update_paths: list[Path],
    domain_randomization_update_paths: list[Path] | None = None,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
) -> PoolDynamicsProfile:
    base_profile = (
        load_pool_dynamics_profile_json(base_profile_path)
        if base_profile_path is not None
        else NOMINAL_POOL_DYNAMICS_PROFILE
    )
    cfg_updates: list[dict[str, Any]] = []
    domain_updates: list[dict[str, Any]] = []

    for path in update_paths:
        cfg_update, domain_update = load_update_payload(path)
        cfg_updates.append(cfg_update)
        if domain_update:
            domain_updates.append(domain_update)

    for path in domain_randomization_update_paths or []:
        domain_updates.append(dict(_load_json_mapping(path)))

    return merge_pool_dynamics_cfg_updates(
        base_profile,
        cfg_updates=cfg_updates,
        domain_randomization_updates=domain_updates,
        name=name,
        description=description,
        strict=strict,
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{path}: non-finite JSON constant {value!r} is not allowed.")
            ),
        )
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a JSON object.")
    for key in data:
        if not isinstance(key, str):
            raise ValueError(f"{path} must contain string keys.")
    return data


def collect_domain_randomization_sources(
    update_paths: list[Path],
    domain_randomization_update_paths: list[Path],
) -> dict[str, str]:
    """Record which calibration update file supplied each uncertainty field."""

    sources: dict[str, str] = {}
    for path in update_paths:
        _, domain_updates = load_update_payload(path)
        for name in domain_updates:
            sources[name] = f"Calibration update file: {path.name}"
    for path in domain_randomization_update_paths:
        for name in _load_json_mapping(path):
            sources[name] = f"Domain-randomization update file: {path.name}"
    return sources


def write_profile_and_randomization_spec_atomically(
    profile: PoolDynamicsProfile,
    profile_path: Path,
    spec: DomainRandomizationSpec,
    spec_path: Path,
) -> None:
    """Commit paired JSON artifacts only after both serialize successfully."""

    profile_path = profile_path.resolve()
    spec_path = spec_path.resolve()
    if profile_path == spec_path:
        raise ValueError("Profile and domain-randomization outputs must be different files.")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    profile_tmp = profile_path.with_name(f".{profile_path.name}.{token}.tmp")
    spec_tmp = spec_path.with_name(f".{spec_path.name}.{token}.tmp")
    profile_backup = profile_path.with_name(f".{profile_path.name}.{token}.bak")
    spec_backup = spec_path.with_name(f".{spec_path.name}.{token}.bak")
    had_profile = profile_path.exists()
    had_spec = spec_path.exists()
    try:
        write_pool_dynamics_profile_json(profile, profile_tmp)
        write_domain_randomization_spec_json(spec, spec_tmp)
        if had_profile:
            os.replace(profile_path, profile_backup)
        if had_spec:
            os.replace(spec_path, spec_backup)
        os.replace(profile_tmp, profile_path)
        os.replace(spec_tmp, spec_path)
    except Exception:
        if profile_path.exists() and (not had_profile or profile_backup.exists()):
            profile_path.unlink()
        if spec_path.exists() and (not had_spec or spec_backup.exists()):
            spec_path.unlink()
        if profile_backup.exists():
            os.replace(profile_backup, profile_path)
        if spec_backup.exists():
            os.replace(spec_backup, spec_path)
        raise
    finally:
        for path in (profile_tmp, spec_tmp, profile_backup, spec_backup):
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        profile = build_profile_from_files(
            base_profile_path=args.base_profile,
            update_paths=args.updates,
            domain_randomization_update_paths=args.domain_randomization_updates,
            name=args.name,
            description=args.description,
            strict=not bool(args.allow_unknown),
        )
        if args.domain_randomization_output is not None:
            if profile.domain_randomization is None:
                raise ValueError(
                    f"Pool profile {profile.name!r} has no domain_randomization section to export."
                )
            parameter_sources = collect_domain_randomization_sources(
                args.updates,
                args.domain_randomization_updates,
            )
            if args.domain_randomization_sources is not None:
                parameter_sources.update(dict(_load_json_mapping(args.domain_randomization_sources)))
            completed_parameters = complete_domain_randomization_profile(
                profile.domain_randomization,
                profile,
            )
            missing_sources = sorted(
                domain_randomization_parameters_requiring_sources(completed_parameters)
                - set(parameter_sources)
            )
            if missing_sources:
                raise ValueError(
                    "Missing provenance for inherited domain-randomization fields: "
                    + ", ".join(missing_sources)
                    + ". Provide --domain-randomization-sources."
                )
            randomization_spec = domain_randomization_spec_from_pool_profile(
                profile,
                name=args.domain_randomization_name,
                description=(
                    f"Calibration-derived uncertainty recipe bound to measured profile {profile.name}."
                ),
                parameter_sources=parameter_sources,
                metadata={
                    "source": "build_pool_profile_from_calibration.py",
                    "update_files": [path.name for path in args.updates],
                    "domain_randomization_update_files": [
                        path.name
                        for path in args.domain_randomization_updates
                    ],
                },
            )
            deterministic_profile = replace(profile, domain_randomization=None)
            write_profile_and_randomization_spec_atomically(
                deterministic_profile,
                args.output,
                randomization_spec,
                args.domain_randomization_output,
            )
        else:
            write_pool_dynamics_profile_json(profile, args.output)
    except Exception as exc:
        print(f"Failed to build pool profile: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

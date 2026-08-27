#!/usr/bin/env python3
"""Report whether a loaded shell provides the required OpenCFD toolchain."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

try:  # Support both direct execution and namespace-package imports.
    from .inspect_stl import write_json_report
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from inspect_stl import write_json_report


REPORT_SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STRICT_FAILURE = 2

# Commands needed by the geometry, mesh, transient-run, parallel, and extraction
# stages.  foamVersion is intentionally absent: OpenCFD defines it as a shell
# function, not a standalone executable.  Version/API come from the exported
# environment and foamEtcFile.
DEFAULT_REQUIRED_COMMANDS = (
    "foamEtcFile",
    "blockMesh",
    "surfaceTransformPoints",
    "surfaceCheck",
    "snappyHexMesh",
    "checkMesh",
    "decomposePar",
    "reconstructPar",
    "potentialFoam",
    "pimpleFoam",
    "postProcess",
    "mpirun",
)


def _run_command(command: Sequence[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _numeric_api(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", value)
    return int(match.group(1)) if match else None


def _version_from_path(command_paths: Mapping[str, str | None]) -> str | None:
    for path in command_paths.values():
        if not path:
            continue
        match = re.search(r"openfoam[-_/]?(v?\d{4})", path, flags=re.IGNORECASE)
        if match:
            version = match.group(1)
            return version if version.lower().startswith("v") else f"v{version}"
    return None


def _project_identifies_opencfd(project_dir: str | None) -> bool:
    if not project_dir:
        return False
    bashrc = Path(project_dir).expanduser() / "etc" / "bashrc"
    try:
        text = bashrc.read_text(encoding="utf-8", errors="ignore")[:65536]
    except OSError:
        return False
    return "openfoam.com" in text.lower() or "opencfd ltd" in text.lower()


def inspect_environment(
    *,
    required_commands: Sequence[str] = DEFAULT_REQUIRED_COMMANDS,
    environ: Mapping[str, str] | None = None,
    minimum_api: int | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable OpenFOAM/OpenCFD environment report."""

    env = os.environ if environ is None else environ
    path_value = env.get("PATH")
    commands: dict[str, dict[str, Any]] = {}
    command_paths: dict[str, str | None] = {}
    for command in dict.fromkeys(required_commands):
        resolved = shutil.which(command, path=path_value)
        command_paths[command] = resolved
        commands[command] = {
            "found": resolved is not None,
            "path": str(Path(resolved).resolve()) if resolved else None,
        }

    project_version = env.get("WM_PROJECT_VERSION")
    version_source: str | None = "WM_PROJECT_VERSION" if project_version else None
    if not project_version:
        project_version = _version_from_path(command_paths)
        if project_version:
            version_source = "command_path"

    api = _numeric_api(env.get("FOAM_API"))
    api_source: str | None = "FOAM_API" if api is not None else None
    foam_etc_result: dict[str, Any] | None = None
    foam_etc = command_paths.get("foamEtcFile") or shutil.which("foamEtcFile", path=path_value)
    if api is None and foam_etc:
        foam_etc_result = _run_command([foam_etc, "-show-api"])
        if foam_etc_result["ok"]:
            api = _numeric_api(foam_etc_result["stdout"] or foam_etc_result["stderr"])
            if api is not None:
                api_source = "foamEtcFile -show-api"
    if api is None:
        api = _numeric_api(project_version)
        if api is not None:
            api_source = version_source

    version_api_shape = bool(project_version and re.fullmatch(r"v?\d{4}", project_version))
    project_dir = env.get("WM_PROJECT_DIR")
    opencfd_evidence = {
        "version_uses_release_api_shape": version_api_shape,
        "project_bashrc_mentions_openfoam_com_or_opencfd": _project_identifies_opencfd(project_dir),
        "foam_api_exported": _numeric_api(env.get("FOAM_API")) is not None,
    }
    is_opencfd = version_api_shape and api is not None and (
        opencfd_evidence["project_bashrc_mentions_openfoam_com_or_opencfd"]
        or opencfd_evidence["foam_api_exported"]
        or (api >= 1606 and project_version is not None and project_version.startswith("v"))
    )
    distribution = "OpenCFD" if is_opencfd else "unknown"

    missing = [command for command, status in commands.items() if not status["found"]]
    failures: list[str] = []
    if distribution != "OpenCFD":
        failures.append("loaded environment is not positively identified as OpenCFD")
    if api is None:
        failures.append("OpenFOAM API could not be determined")
    if minimum_api is not None and (api is None or api < minimum_api):
        failures.append(f"OpenFOAM API {api!r} is below required API {minimum_api}")
    if missing:
        failures.append("missing required commands: " + ", ".join(missing))

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ready": not failures,
        "openfoam": {
            "distribution": distribution,
            "version": project_version,
            "version_source": version_source,
            "api": api,
            "api_source": api_source,
            "minimum_api": minimum_api,
            "environment_loaded": bool(project_dir and project_version),
            "WM_PROJECT": env.get("WM_PROJECT"),
            "WM_PROJECT_DIR": project_dir,
            "WM_OPTIONS": env.get("WM_OPTIONS"),
            "FOAM_APPBIN": env.get("FOAM_APPBIN"),
            "FOAM_LIBBIN": env.get("FOAM_LIBBIN"),
            "opencfd_evidence": opencfd_evidence,
            "foam_etc_file_query": foam_etc_result,
        },
        "required_commands": list(dict.fromkeys(required_commands)),
        "commands": commands,
        "missing_required_commands": missing,
        "failures": failures,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        dest="json_path",
        default="-",
        metavar="PATH",
        help="JSON report path, or '-' for stdout (default: '-')",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 unless OpenCFD/API and every required command are confirmed",
    )
    parser.add_argument("--min-api", type=int, help="minimum acceptable four-digit OpenCFD API")
    parser.add_argument(
        "--require",
        action="append",
        metavar="COMMAND",
        help="override the default command set; repeat once per required command",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    required = tuple(args.require) if args.require else DEFAULT_REQUIRED_COMMANDS
    try:
        report = inspect_environment(required_commands=required, minimum_api=args.min_api)
        write_json_report(report, args.json_path)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"environment inspection failed: {exc}\n")
        return EXIT_ERROR
    if args.strict and not report["ready"]:
        return EXIT_STRICT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

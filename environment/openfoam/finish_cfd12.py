#!/usr/bin/env python3
"""Wait for the 12-case 1.5 Hz campaign, fit it, and publish it fail-closed."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

from environment.openfoam import run_cases  # noqa: E402
from environment.openfoam.analysis.motion import DOF_NAMES, MotionSpec  # noqa: E402


_EXPECTED_FREQUENCY_HZ = 1.5
_EXPECTED_CASE_COUNT = 12
_EXPECTED_CASES_PER_DOF = 2
_EXPECTED_COMPLETE_CYCLES = 4
_RUNNER_FAILURE_RE = re.compile(
    r"^\s*\[fail\]|Traceback \(most recent call last\)|FOAM\s+FATAL|"
    r"\bMPI_ABORT\b|segmentation fault|floating point exception|"
    r"No runnable cases matched|Missing commands:|requires OpenCFD API|"
    r"run_cases\.py:\s*error:",
    re.IGNORECASE,
)


class FinishFailure(RuntimeError):
    """A terminal condition that must not publish coefficient files."""

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details or {})


@dataclass(frozen=True)
class Campaign:
    cases_dir: Path
    config_path: Path
    solver: str
    case_dirs: tuple[Path, ...]

    @property
    def case_names(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.case_dirs)


@dataclass(frozen=True)
class CompletionSnapshot:
    valid: tuple[Path, ...]
    pending: Mapping[str, str]

    @property
    def complete(self) -> bool:
        return not self.pending


class RunnerLogWatcher:
    """Incrementally inspect a runner log, including content present at startup."""

    def __init__(self, path: Path):
        self.path = path
        self._identity: tuple[int, int] | None = None
        self._offset = 0

    def failure(self) -> str | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise FinishFailure(
                f"cannot inspect runner log {self.path}: {exc}",
                details={"runner_log": str(self.path)},
            ) from exc

        identity = (stat.st_dev, stat.st_ino)
        if identity != self._identity or stat.st_size < self._offset:
            self._identity = identity
            self._offset = 0
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self._offset)
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    if _RUNNER_FAILURE_RE.search(line):
                        excerpt = line.strip()[:500]
                        self._offset = stream.tell()
                        return f"{self.path}: {excerpt}"
                self._offset = stream.tell()
        except OSError as exc:
            raise FinishFailure(
                f"cannot read runner log {self.path}: {exc}",
                details={"runner_log": str(self.path)},
            ) from exc
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=30.0,
        help="Seconds between completion checks (default: 30).",
    )
    parser.add_argument(
        "--runner-log",
        type=Path,
        action="append",
        default=[],
        help="run_cases.py console log to monitor for terminal failures; repeatable.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--passivity-samples", type=int, default=10000)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinishFailure(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinishFailure(f"{label} {path} must contain a JSON object")
    return value


def _finite_positive_sequence(config: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise FinishFailure(f"config {key} must be a non-empty JSON array")
    if any(isinstance(item, bool) for item in value):
        raise FinishFailure(f"config {key} must contain finite positive numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise FinishFailure(f"config {key} must contain finite positive numbers") from exc
    if not all(math.isfinite(item) and item > 0.0 for item in result):
        raise FinishFailure(f"config {key} must contain finite positive numbers")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-10, abs_tol=1.0e-12)


def _load_campaign(cases_dir: Path, config_path: Path) -> Campaign:
    cases_dir = cases_dir.resolve()
    config_path = config_path.resolve()
    if not cases_dir.is_dir():
        raise FinishFailure(f"cases directory does not exist: {cases_dir}")
    config = _read_json_object(config_path, "config")
    if config.get("openfoam_version") != "v2512" or config.get("solver") != "pimpleFoam":
        raise FinishFailure("campaign config must target OpenCFD v2512 pimpleFoam")

    frequencies = _finite_positive_sequence(config, "frequencies_hz")
    translations = _finite_positive_sequence(config, "translation_amplitudes_m")
    rotations_deg = _finite_positive_sequence(config, "rotation_amplitudes_deg")
    if len(frequencies) != 1 or not _close(frequencies[0], _EXPECTED_FREQUENCY_HZ):
        raise FinishFailure("campaign config must contain exactly frequencies_hz=[1.5]")
    if len(set(translations)) != _EXPECTED_CASES_PER_DOF or len(translations) != 2:
        raise FinishFailure("campaign config must contain exactly two distinct translation amplitudes")
    if len(set(rotations_deg)) != _EXPECTED_CASES_PER_DOF or len(rotations_deg) != 2:
        raise FinishFailure("campaign config must contain exactly two distinct rotation amplitudes")
    try:
        configured_cycles = float(config["sample_cycles"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinishFailure("campaign config sample_cycles is invalid") from exc
    if not _close(configured_cycles, _EXPECTED_COMPLETE_CYCLES):
        raise FinishFailure("campaign config must request exactly four sample cycles")

    candidates: list[tuple[Path, MotionSpec, dict[str, Any]]] = []
    for motion_path in sorted(cases_dir.glob("*/motion.json")):
        metadata = _read_json_object(motion_path, "motion metadata")
        if (
            metadata.get("purpose") == "shared_mesh"
            or metadata.get("include_in_fit") is False
            or metadata.get("dof") is None
        ):
            continue
        try:
            motion = MotionSpec.from_mapping(metadata, source_path=str(motion_path))
        except (TypeError, ValueError) as exc:
            raise FinishFailure(f"invalid motion metadata {motion_path}: {exc}") from exc
        candidates.append((motion_path.parent.resolve(), motion, metadata))

    if len(candidates) != _EXPECTED_CASE_COUNT:
        raise FinishFailure(
            f"expected exactly {_EXPECTED_CASE_COUNT} fitted cases, found {len(candidates)}",
            details={"case_names": [case.name for case, _, _ in candidates]},
        )

    expected_amplitudes = {
        dof: (translations if index < 3 else tuple(math.radians(value) for value in rotations_deg))
        for index, dof in enumerate(DOF_NAMES)
    }
    found_by_dof: dict[str, list[float]] = {dof: [] for dof in DOF_NAMES}
    for case, motion, metadata in candidates:
        if metadata.get("schema_version") != 1:
            raise FinishFailure(f"{case.name}: motion.json schema_version must be 1")
        if metadata.get("openfoam_version") != "v2512" or metadata.get("solver") != "pimpleFoam":
            raise FinishFailure(f"{case.name}: motion.json must target v2512 pimpleFoam")
        if metadata.get("purpose") != "identification" or metadata.get("include_in_fit") is not True:
            raise FinishFailure(f"{case.name}: case must be an enabled identification case")
        if motion.case_name != case.name:
            raise FinishFailure(f"{case.name}: motion case_name does not match its directory")
        frequency = metadata.get("frequency_hz")
        if isinstance(frequency, bool):
            raise FinishFailure(f"{case.name}: frequency_hz must be 1.5")
        try:
            frequency_hz = float(frequency)
        except (TypeError, ValueError) as exc:
            raise FinishFailure(f"{case.name}: frequency_hz must be 1.5") from exc
        if not _close(frequency_hz, _EXPECTED_FREQUENCY_HZ):
            raise FinishFailure(f"{case.name}: frequency_hz must be 1.5")
        if not _close(motion.omega_rad_s, 2.0 * math.pi * _EXPECTED_FREQUENCY_HZ):
            raise FinishFailure(f"{case.name}: omega_rad_s is inconsistent with 1.5 Hz")
        if not _close(motion.sample_cycles or -1.0, _EXPECTED_COMPLETE_CYCLES):
            raise FinishFailure(f"{case.name}: motion.json must request four sample cycles")
        try:
            settle_cycles = float(config["settle_cycles"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinishFailure("campaign config settle_cycles is invalid") from exc
        if not _close(motion.settle_cycles, settle_cycles):
            raise FinishFailure(f"{case.name}: settle_cycles does not match the campaign config")
        if not any(_close(motion.amplitude_si, expected) for expected in expected_amplitudes[motion.dof]):
            raise FinishFailure(f"{case.name}: amplitude does not match the campaign config")
        found_by_dof[motion.dof].append(motion.amplitude_si)

    for dof in DOF_NAMES:
        found = sorted(found_by_dof[dof])
        expected = sorted(expected_amplitudes[dof])
        if len(found) != _EXPECTED_CASES_PER_DOF or any(
            not _close(left, right) for left, right in zip(found, expected, strict=True)
        ):
            raise FinishFailure(
                f"DOF {dof} must have exactly the two amplitudes configured for the campaign"
            )

    ordered = tuple(
        case
        for case, _, _ in sorted(
            candidates,
            key=lambda item: (DOF_NAMES.index(item[1].dof), item[1].amplitude_si),
        )
    )
    return Campaign(cases_dir, config_path, "pimpleFoam", ordered)


def _completion_snapshot(campaign: Campaign) -> CompletionSnapshot:
    valid: list[Path] = []
    pending: dict[str, str] = {}
    for case in campaign.case_dirs:
        marker_path = case / ".completed"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pending[case.name] = "missing .completed marker"
            continue
        except (OSError, json.JSONDecodeError) as exc:
            pending[case.name] = f"unreadable .completed marker: {exc}"
            continue
        if not isinstance(marker, dict) or marker.get("schema_version") != 2:
            pending[case.name] = ".completed marker is not schema version 2"
            continue
        try:
            completed, reason = run_cases._validated_completion(case, campaign.solver)
        except Exception as exc:  # keep the monitor fail-closed if validation itself rejects input
            completed, reason = False, f"completion validation raised {type(exc).__name__}: {exc}"
        if completed:
            valid.append(case)
        else:
            pending[case.name] = reason
    return CompletionSnapshot(tuple(valid), pending)


def _resolve_process_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def _find_runner_processes(cases_dir: Path, proc_root: Path = Path("/proc")) -> set[int]:
    """Return run_cases.py PIDs explicitly targeting ``cases_dir``."""

    target = cases_dir.resolve()
    matches: set[int] = set()
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return matches
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            tokens = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
            cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, OSError):
            continue
        is_runner = any(Path(token).name == "run_cases.py" for token in tokens)
        is_module = any(
            tokens[index : index + 2] == ["-m", "openfoam.run_cases"]
            for index in range(max(0, len(tokens) - 1))
        )
        if not is_runner and not is_module:
            continue
        supplied: str | None = None
        for index, token in enumerate(tokens):
            if token == "--cases-dir" and index + 1 < len(tokens):
                supplied = tokens[index + 1]
                break
            if token.startswith("--cases-dir="):
                supplied = token.split("=", 1)[1]
                break
        if supplied is not None and _resolve_process_path(supplied, cwd) == target:
            matches.add(int(entry.name))
    return matches


def _wait_for_completion(
    campaign: Campaign,
    runner_logs: Sequence[Path],
    wait_seconds: float,
    *,
    process_probe: Callable[[Path], set[int]] = _find_runner_processes,
    sleep: Callable[[float], None] = time.sleep,
) -> CompletionSnapshot:
    watchers = [RunnerLogWatcher(path.resolve()) for path in runner_logs]
    saw_runner = False
    observed_pids: set[int] = set()
    last_progress: tuple[int, tuple[str, ...]] | None = None

    while True:
        snapshot = _completion_snapshot(campaign)
        if snapshot.complete:
            return snapshot

        for watcher in watchers:
            failure = watcher.failure()
            if failure is not None:
                raise FinishFailure(
                    "a specified runner log reports failure",
                    details={
                        "runner_log_failure": failure,
                        "completed_count": len(snapshot.valid),
                        "pending": dict(snapshot.pending),
                    },
                )

        current_pids = process_probe(campaign.cases_dir)
        if current_pids:
            saw_runner = True
            observed_pids.update(current_pids)
        elif saw_runner:
            # Recheck after observing the process transition.  run_cases writes
            # its marker atomically before the process exits, so a remaining
            # invalid case is terminal evidence rather than a marker race.
            final_snapshot = _completion_snapshot(campaign)
            if final_snapshot.complete:
                return final_snapshot
            raise FinishFailure(
                "all observed run_cases.py processes exited before the campaign completed",
                details={
                    "observed_runner_pids": sorted(observed_pids),
                    "completed_count": len(final_snapshot.valid),
                    "pending": dict(final_snapshot.pending),
                },
            )

        progress = (len(snapshot.valid), tuple(sorted(snapshot.pending)))
        if progress != last_progress:
            print(
                f"[wait] {len(snapshot.valid)}/{_EXPECTED_CASE_COUNT} validated; "
                f"pending: {', '.join(sorted(snapshot.pending))}",
                flush=True,
            )
            last_progress = progress
        sleep(wait_seconds)


def _as_matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FinishFailure(f"{label} is not a numeric 6x6 matrix") from exc
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise FinishFailure(f"{label} must be a finite 6x6 matrix")
    return matrix


def _require_zero_fraction(passivity: Mapping[str, Any], key: str) -> None:
    value = passivity.get(key)
    if isinstance(value, bool):
        raise FinishFailure(f"diagnostics.passivity.{key} is invalid")
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise FinishFailure(f"diagnostics.passivity.{key} is invalid") from exc
    if not math.isfinite(fraction) or fraction != 0.0:
        raise FinishFailure(f"diagnostics.passivity.{key} must be exactly zero, got {value!r}")


def _validate_fit_outputs(
    staging: Path,
    campaign: Campaign,
    bootstrap_samples: int,
    passivity_samples: int,
) -> dict[str, Any]:
    report = _read_json_object(staging / "hydrodynamic_fit.json", "fit report")
    updates_file = _read_json_object(staging / "config_updates.json", "config updates")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != _EXPECTED_CASE_COUNT:
        raise FinishFailure("fit report must contain exactly 12 case summaries")
    if any(not isinstance(item, dict) for item in cases):
        raise FinishFailure("fit report case summaries must be JSON objects")
    report_names = [str(item.get("case_name", "")) for item in cases]
    if len(set(report_names)) != _EXPECTED_CASE_COUNT or set(report_names) != set(campaign.case_names):
        raise FinishFailure("fit report case identities do not match the validated campaign")
    dof_counts = Counter(str(item.get("dof", "")) for item in cases)
    if dof_counts != Counter({dof: _EXPECTED_CASES_PER_DOF for dof in DOF_NAMES}):
        raise FinishFailure("fit report must contain exactly two cases for every DOF")
    for item in cases:
        try:
            sample_cycles = float(item["sample_cycles"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinishFailure(f"{item.get('case_name')}: invalid sample_cycles in fit report") from exc
        if not _close(sample_cycles, _EXPECTED_COMPLETE_CYCLES):
            raise FinishFailure(f"{item.get('case_name')}: fit report does not request four cycles")

    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise FinishFailure("fit report diagnostics are missing")
    fits = diagnostics.get("fit_by_excited_dof")
    if not isinstance(fits, dict) or set(fits) != set(DOF_NAMES):
        raise FinishFailure("fit report must contain diagnostics for all six DOFs")
    for dof in DOF_NAMES:
        fit = fits[dof]
        if not isinstance(fit, dict) or fit.get("rank") != 3:
            raise FinishFailure(f"DOF {dof} aggregate regression rank must equal 3")
        case_names = fit.get("case_names")
        if not isinstance(case_names, list) or len(case_names) != 2:
            raise FinishFailure(f"DOF {dof} fit must use exactly two cases")
        cycles = fit.get("complete_cycles_by_case")
        if not isinstance(cycles, dict) or set(cycles) != set(case_names):
            raise FinishFailure(f"DOF {dof} complete-cycle evidence is incomplete")
        if any(value != _EXPECTED_COMPLETE_CYCLES for value in cycles.values()):
            raise FinishFailure(f"DOF {dof} did not contribute four complete cycles per case")

    convergence = diagnostics.get("cycle_convergence_by_case")
    if not isinstance(convergence, list) or len(convergence) != _EXPECTED_CASE_COUNT:
        raise FinishFailure("cycle convergence diagnostics must cover all 12 cases")
    convergence_names: set[str] = set()
    for item in convergence:
        if not isinstance(item, dict):
            raise FinishFailure("cycle convergence diagnostics contain a malformed item")
        name = str(item.get("case_name", ""))
        convergence_names.add(name)
        cycles = item.get("cycles")
        if not isinstance(cycles, list) or len(cycles) != _EXPECTED_COMPLETE_CYCLES:
            raise FinishFailure(f"{name}: exactly four complete cycle fits are required")
        if any(not isinstance(cycle, dict) or cycle.get("fit", {}).get("rank") != 3 for cycle in cycles):
            raise FinishFailure(f"{name}: every per-cycle regression rank must equal 3")
        comparison = item.get("last_two_cycle_comparison")
        if not isinstance(comparison, dict) or comparison.get("available") is not True:
            raise FinishFailure(f"{name}: last-two-cycle comparison is unavailable")
    if convergence_names != set(campaign.case_names):
        raise FinishFailure("cycle convergence case identities do not match the campaign")

    matrices = report.get("matrices")
    if not isinstance(matrices, dict):
        raise FinishFailure("fit report matrices are missing")
    matrix_values = {
        name: _as_matrix(matrices.get(name), f"matrices.{name}")
        for name in (
            "added_mass_raw",
            "added_mass",
            "linear_damping",
            "quadratic_damping",
        )
    }
    added_mass = matrix_values["added_mass"]
    scale = max(1.0, float(np.linalg.norm(added_mass, ord=2)))
    symmetry_tolerance = 1.0e-9 * scale
    if not np.allclose(added_mass, added_mass.T, rtol=1.0e-9, atol=symmetry_tolerance):
        raise FinishFailure("projected added-mass matrix is not symmetric")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * (added_mass + added_mass.T))))
    if minimum_eigenvalue < -symmetry_tolerance:
        raise FinishFailure(
            f"projected added-mass matrix is not PSD: minimum eigenvalue {minimum_eigenvalue:.12g}"
        )
    projection = diagnostics.get("added_mass_projection")
    if not isinstance(projection, dict) or projection.get("enabled") is not True:
        raise FinishFailure("added-mass PSD projection was not enabled")

    expected_updates = {
        "added_mass_diag": matrix_values["added_mass"],
        "linear_damping": matrix_values["linear_damping"],
        "quadratic_damping": matrix_values["quadratic_damping"],
    }
    report_updates = report.get("config_updates")
    if not isinstance(report_updates, dict) or set(report_updates) != set(expected_updates):
        raise FinishFailure("fit report config_updates keys are incomplete or unexpected")
    if set(updates_file) != set(expected_updates):
        raise FinishFailure("config_updates.json keys are incomplete or unexpected")
    for key, expected in expected_updates.items():
        report_matrix = _as_matrix(report_updates[key], f"report config_updates.{key}")
        file_matrix = _as_matrix(updates_file[key], f"config_updates.json {key}")
        if not np.array_equal(report_matrix, expected) or not np.array_equal(file_matrix, expected):
            raise FinishFailure(f"config update {key} is inconsistent with the fitted matrix")

    passivity = diagnostics.get("passivity")
    if not isinstance(passivity, dict):
        raise FinishFailure("passivity diagnostics are missing")
    _require_zero_fraction(passivity, "observed_negative_fraction")
    _require_zero_fraction(passivity, "random_negative_fraction")
    if passivity.get("random_sample_count") != passivity_samples:
        raise FinishFailure("passivity random sample count does not match the requested value")

    options = report.get("options")
    if not isinstance(options, dict):
        raise FinishFailure("fit report options are missing")
    if options.get("project_added_mass_psd") is not True:
        raise FinishFailure("fit report did not record added-mass PSD projection")
    if options.get("bootstrap_samples") != bootstrap_samples:
        raise FinishFailure("fit report bootstrap sample count does not match the request")
    if options.get("passivity_samples") != passivity_samples:
        raise FinishFailure("fit report passivity sample count does not match the request")
    confidence = report.get("confidence_intervals")
    if bootstrap_samples > 0:
        if not isinstance(confidence, dict) or confidence.get("samples") != bootstrap_samples:
            raise FinishFailure("bootstrap confidence intervals are incomplete")

    required_files = {
        "hydrodynamic_fit.json",
        "config_updates.json",
        "added_mass.csv",
        "added_mass_raw.csv",
        "linear_damping.csv",
        "quadratic_damping.csv",
    }
    missing_files = sorted(name for name in required_files if not (staging / name).is_file())
    if missing_files:
        raise FinishFailure(f"analysis did not produce required file(s): {', '.join(missing_files)}")

    return {
        "case_count": _EXPECTED_CASE_COUNT,
        "cases_per_dof": {dof: _EXPECTED_CASES_PER_DOF for dof in DOF_NAMES},
        "aggregate_rank_by_dof": {dof: 3 for dof in DOF_NAMES},
        "complete_cycles_per_case": _EXPECTED_COMPLETE_CYCLES,
        "cycle_comparisons_available": True,
        "finite_matrix_shape": [6, 6],
        "added_mass_psd": True,
        "added_mass_minimum_eigenvalue": minimum_eigenvalue,
        "config_updates_consistent": True,
        "observed_passivity_negative_fraction": 0.0,
        "random_passivity_negative_fraction": 0.0,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _failure_path(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.failure.json")


def _run_analysis(
    campaign: Campaign,
    staging: Path,
    bootstrap_samples: int,
    passivity_samples: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "environment.openfoam.analysis",
        "--cases-root",
        str(campaign.cases_dir),
        "--config",
        str(campaign.config_path),
        "--output-dir",
        str(staging),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--passivity-samples",
        str(passivity_samples),
        "--project-added-mass-psd",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_text = completed.stdout or ""
    (staging / "analysis.log").write_text(log_text, encoding="utf-8")
    if completed.returncode:
        raise FinishFailure(
            f"matrix analysis exited with status {completed.returncode}",
            details={"analysis_log_tail": log_text[-4000:]},
        )


def _published_manifest_matches(
    output_dir: Path,
    campaign: Campaign,
    bootstrap_samples: int,
    passivity_samples: int,
) -> bool:
    try:
        manifest = json.loads((output_dir / "finish_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(manifest, dict)
        and manifest.get("status") == "completed"
        and manifest.get("cases_dir") == str(campaign.cases_dir)
        and manifest.get("config") == str(campaign.config_path)
        and manifest.get("case_names") == list(campaign.case_names)
        and manifest.get("bootstrap_samples") == bootstrap_samples
        and manifest.get("passivity_samples") == passivity_samples
    )


def _fit_and_publish(
    campaign: Campaign,
    output_dir: Path,
    bootstrap_samples: int,
    passivity_samples: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or not _published_manifest_matches(
            output_dir, campaign, bootstrap_samples, passivity_samples
        ):
            raise FinishFailure(f"refusing to overwrite existing output path: {output_dir}")
        acceptance = _validate_fit_outputs(
            output_dir, campaign, bootstrap_samples, passivity_samples
        )
        print(f"[done] validated existing publication: {output_dir}", flush=True)
        return acceptance

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        print("[fit] all 12 cases validated; fitting matrices in staging", flush=True)
        _run_analysis(campaign, staging, bootstrap_samples, passivity_samples)
        acceptance = _validate_fit_outputs(
            staging, campaign, bootstrap_samples, passivity_samples
        )
        final_snapshot = _completion_snapshot(campaign)
        if not final_snapshot.complete:
            raise FinishFailure(
                "case completion evidence changed while analysis was running",
                details={"pending": dict(final_snapshot.pending)},
            )
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "completed_at": _utc_now(),
            "cases_dir": str(campaign.cases_dir),
            "config": str(campaign.config_path),
            "case_names": list(campaign.case_names),
            "bootstrap_samples": bootstrap_samples,
            "passivity_samples": passivity_samples,
            "acceptance": acceptance,
        }
        _atomic_write_json(staging / "finish_status.json", manifest)
        try:
            os.replace(staging, output_dir)
        except OSError as exc:
            if output_dir.is_dir() and _published_manifest_matches(
                output_dir, campaign, bootstrap_samples, passivity_samples
            ):
                shutil.rmtree(staging)
                acceptance = _validate_fit_outputs(
                    output_dir, campaign, bootstrap_samples, passivity_samples
                )
            else:
                raise FinishFailure(f"cannot atomically publish {output_dir}: {exc}") from exc
        failure_path = _failure_path(output_dir)
        failure_path.unlink(missing_ok=True)
        print(f"[done] atomically published accepted matrices: {output_dir}", flush=True)
        return acceptance
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def finish(
    *,
    cases_dir: Path,
    config_path: Path,
    output_dir: Path,
    wait_seconds: float,
    runner_logs: Sequence[Path],
    bootstrap_samples: int,
    passivity_samples: int,
) -> dict[str, Any]:
    if not math.isfinite(wait_seconds) or wait_seconds <= 0.0:
        raise FinishFailure("wait_seconds must be finite and positive")
    if bootstrap_samples < 0:
        raise FinishFailure("bootstrap_samples must be non-negative")
    if passivity_samples <= 0:
        raise FinishFailure("passivity_samples must be positive so random passivity can be accepted")
    campaign = _load_campaign(cases_dir, config_path)
    _wait_for_completion(campaign, runner_logs, wait_seconds)
    return _fit_and_publish(
        campaign,
        output_dir,
        bootstrap_samples,
        passivity_samples,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        acceptance = finish(
            cases_dir=args.cases_dir,
            config_path=args.config,
            output_dir=output_dir,
            wait_seconds=args.wait_seconds,
            runner_logs=args.runner_log,
            bootstrap_samples=args.bootstrap_samples,
            passivity_samples=args.passivity_samples,
        )
    except KeyboardInterrupt:
        failure = FinishFailure("finish monitor was interrupted")
    except FinishFailure as exc:
        failure = exc
    except Exception as exc:  # retain a durable terminal record for unexpected failures
        failure = FinishFailure(f"unexpected {type(exc).__name__}: {exc}")
    else:
        print(json.dumps({"status": "completed", "output_dir": str(output_dir), **acceptance}, indent=2))
        return 0

    payload = {
        "schema_version": 1,
        "status": "failed",
        "failed_at": _utc_now(),
        "reason": failure.reason,
        "details": failure.details,
        "cases_dir": str(args.cases_dir.resolve()),
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
    }
    failure_path = _failure_path(output_dir)
    try:
        _atomic_write_json(failure_path, payload)
    except Exception as write_error:
        print(f"[fail] {failure.reason}; also could not write {failure_path}: {write_error}", file=sys.stderr)
        return 1
    print(f"[fail] {failure.reason}; durable record: {failure_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

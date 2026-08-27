"""Read OpenCFD v2512 split force-function output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
import warnings

import numpy as np


_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|(?:nan|inf))"
_NUMBER_RE = re.compile(_NUMBER, re.IGNORECASE)
_FORCE_SPLIT_RE = re.compile(r"^force(?:_(\d+))?\.dat$")
_MOMENT_SPLIT_RE = re.compile(r"^moment(?:_(\d+))?\.dat$")


@dataclass(frozen=True)
class ForceSeries:
    """Time history of global force and moment about the configured CofR."""

    time_s: np.ndarray
    force_global: np.ndarray
    moment_global: np.ndarray
    source_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        force = np.asarray(self.force_global, dtype=float)
        moment = np.asarray(self.moment_global, dtype=float)
        if time.ndim != 1:
            raise ValueError(f"time_s must be one-dimensional, got {time.shape}")
        if force.shape != (time.size, 3) or moment.shape != (time.size, 3):
            raise ValueError(
                "force_global and moment_global must both have shape "
                f"({time.size}, 3), got {force.shape} and {moment.shape}"
            )
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "force_global", force)
        object.__setattr__(self, "moment_global", moment)

    @property
    def wrench_global(self) -> np.ndarray:
        return np.concatenate((self.force_global, self.moment_global), axis=1)


@dataclass(frozen=True)
class _VectorSeries:
    time_s: np.ndarray
    values: np.ndarray
    source_files: tuple[str, ...]

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if time.ndim != 1 or values.shape != (time.size, 3):
            raise ValueError(
                f"Vector series must have shapes (N,) and (N,3), got {time.shape} and {values.shape}"
            )
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "values", values)


def parse_total_vector_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a v2512 ``force.dat`` or ``moment.dat`` total-vector history.

    The v2512 header is ``Time total_* pressure_* viscous_*``.  The total is
    read directly from the first three data columns.
    """

    source = Path(path)
    times: list[float] = []
    vectors: list[np.ndarray] = []
    header_has_total = False
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                lowered = stripped.lower()
                if "time" in lowered and "total_x" in lowered:
                    header_has_total = True
                continue
            time_match = _NUMBER_RE.match(stripped)
            if time_match is None:
                continue
            time_s = float(time_match.group(0))
            remainder = stripped[time_match.end() :].split("#", 1)[0]
            if not header_has_total:
                raise ValueError(f"{source}:{line_number}: missing OpenCFD v2512 total-vector header")
            values = [float(token) for token in _NUMBER_RE.findall(remainder)]
            if len(values) not in (9, 12):
                raise ValueError(
                    f"{source}:{line_number}: expected 9 or 12 v2512 "
                    "total/pressure/viscous[/porous] values, "
                    f"got {len(values)}"
                )
            vector = np.asarray(values[:3], dtype=float)
            times.append(time_s)
            vectors.append(np.asarray(vector, dtype=float))
    if not times:
        raise ValueError(f"No vector samples found in {source}")
    return np.asarray(times, dtype=float), np.vstack(vectors)


def _numeric_path_key(path: Path) -> tuple[tuple[int, float | str], ...]:
    key: list[tuple[int, float | str]] = []
    for part in path.parts:
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part))
    return tuple(key)


def discover_force_moment_files(case_dir: str | Path) -> list[tuple[Path, Path]]:
    """Return paired v2512 split-vector restart segments.

    OpenCFD avoids overwriting an existing function-object file by appending
    numeric suffixes such as ``force_0.dat``/``moment_0.dat``.  Within one
    output directory, the unsuffixed pair is oldest, followed by ``_0``,
    ``_1``, and so on; this order is also the overwrite priority used when
    merging duplicate restart times.
    """

    root = Path(case_dir)
    force_files = list(root.glob("postProcessing/forces/**/force*.dat"))
    moment_files = list(root.glob("postProcessing/forces/**/moment*.dat"))
    if not force_files and not moment_files:
        force_files = list(root.glob("postProcessing/*/**/force*.dat"))
        moment_files = list(root.glob("postProcessing/*/**/moment*.dat"))

    def keyed(paths: Iterable[Path], pattern: re.Pattern[str]) -> dict[tuple[Path, int], Path]:
        result: dict[tuple[Path, int], Path] = {}
        for path in paths:
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            suffix_priority = -1 if match.group(1) is None else int(match.group(1))
            result[(path.parent, suffix_priority)] = path
        return result

    forces = keyed(force_files, _FORCE_SPLIT_RE)
    moments = keyed(moment_files, _MOMENT_SPLIT_RE)
    unpaired = set(forces) ^ set(moments)
    if unpaired:
        paths = [forces.get(key, moments.get(key)) for key in unpaired]
        raise FileNotFoundError(
            "Unpaired v2512 force/moment restart segment(s): "
            + ", ".join(str(path) for path in sorted(paths) if path is not None)
        )

    ordered = sorted(
        forces,
        key=lambda key: (
            _numeric_path_key(key[0].relative_to(root)),
            key[1],
        ),
    )
    return [(forces[key], moments[key]) for key in ordered]


def _time_tolerance(first: float, second: float) -> float:
    return 1.0e-9 * max(1.0, abs(float(first)), abs(float(second)))


def _merge_vector_series(chunks: Iterable[_VectorSeries]) -> _VectorSeries:
    """Merge restart chunks, with each later chunk superseding its time range."""

    series = list(chunks)
    if not series:
        raise ValueError("At least one vector series is required")
    times = np.empty(0, dtype=float)
    values = np.empty((0, 3), dtype=float)
    for item in series:
        chunk_start = float(np.min(item.time_s))
        tolerance = _time_tolerance(chunk_start, chunk_start)
        # A restarted function object can use slightly different adjustable
        # timestamps.  Its first time therefore invalidates every older
        # sample at or after that time, not only bitwise duplicate timestamps.
        keep = times < chunk_start - tolerance
        times = np.concatenate((times[keep], item.time_s))
        values = np.concatenate((values[keep], item.values), axis=0)

    serial = np.arange(times.size)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    serial = serial[order]
    kept_times: list[float] = []
    kept_values: list[np.ndarray] = []
    start = 0
    while start < times.size:
        stop = start + 1
        while stop < times.size and abs(float(times[stop]) - float(times[stop - 1])) <= _time_tolerance(
            times[stop], times[stop - 1]
        ):
            stop += 1
        cluster = np.arange(start, stop)
        winner = max(cluster, key=lambda index: int(serial[index]))
        kept_times.append(float(times[winner]))
        kept_values.append(values[winner])
        start = stop
    sources = tuple(source for item in series for source in item.source_files)
    return _VectorSeries(np.asarray(kept_times), np.vstack(kept_values), sources)


def _align_force_moment(force: _VectorSeries, moment: _VectorSeries) -> ForceSeries:
    """Inner-align independently merged force/moment histories by timestamp."""

    force_index = 0
    moment_index = 0
    times: list[float] = []
    forces: list[np.ndarray] = []
    moments: list[np.ndarray] = []
    unmatched_force = 0
    unmatched_moment = 0
    while force_index < force.time_s.size and moment_index < moment.time_s.size:
        force_time = float(force.time_s[force_index])
        moment_time = float(moment.time_s[moment_index])
        tolerance = _time_tolerance(force_time, moment_time)
        difference = force_time - moment_time
        if abs(difference) <= tolerance:
            # Preserve the force timestamp; the two files normally print the
            # exact same value and the tolerance only covers formatting noise.
            times.append(force_time)
            forces.append(force.values[force_index])
            moments.append(moment.values[moment_index])
            force_index += 1
            moment_index += 1
        elif difference < 0.0:
            unmatched_force += 1
            force_index += 1
        else:
            unmatched_moment += 1
            moment_index += 1
    unmatched_force += force.time_s.size - force_index
    unmatched_moment += moment.time_s.size - moment_index
    if not times:
        raise ValueError("force.dat and moment.dat contain no matching timestamps")
    if unmatched_force or unmatched_moment:
        warnings.warn(
            "Dropped unmatched split force-function samples while aligning timestamps: "
            f"force={unmatched_force}, moment={unmatched_moment}",
            RuntimeWarning,
            stacklevel=2,
        )
    return ForceSeries(
        np.asarray(times),
        np.vstack(forces),
        np.vstack(moments),
        force.source_files + moment.source_files,
    )


def load_case_forces(case_dir: str | Path) -> ForceSeries:
    """Read and merge all force restart segments for a generated case."""

    root = Path(case_dir)
    split_pairs = discover_force_moment_files(root)
    if not split_pairs:
        raise FileNotFoundError(
            f"No OpenCFD v2512 postProcessing/forces/**/{{force,moment}}.dat below {root}"
        )
    force_chunks: list[_VectorSeries] = []
    moment_chunks: list[_VectorSeries] = []
    for force_path, moment_path in split_pairs:
        force_time, force_values = parse_total_vector_file(force_path)
        moment_time, moment_values = parse_total_vector_file(moment_path)
        force_chunks.append(_VectorSeries(force_time, force_values, (str(force_path),)))
        moment_chunks.append(_VectorSeries(moment_time, moment_values, (str(moment_path),)))
    return _align_force_moment(
        _merge_vector_series(force_chunks),
        _merge_vector_series(moment_chunks),
    )

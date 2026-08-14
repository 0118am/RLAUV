"""Assign the shared regression cases to source-aligned pytest modules."""

from __future__ import annotations

from collections.abc import Callable
import inspect

from .integration import dynamics_cases


Case = Callable[[], None]
CATEGORIES = (
    "physics",
    "thrusters",
    "environment",
    "sensing",
    "profiles",
    "identification",
    "validation",
    "tether",
)


def _category_for(case_name: str) -> str:
    """Map a regression case to the module that owns the exercised behavior."""

    if "replay" in case_name or "validation" in case_name:
        return "validation"
    if "tether" in case_name or "winch" in case_name or "multisegment" in case_name:
        return "tether"
    if any(
        token in case_name
        for token in (
            "sensor",
            "observation",
            "measurement_delay",
        )
    ):
        return "sensing"
    if any(token in case_name for token in ("calibration", "_fit_", "_fits_", "_pipeline_", "builder_cli")):
        return "identification"
    if "profile" in case_name:
        return "profiles"
    if any(token in case_name for token in ("water_current", "pool_boundary", "free_surface", "sloshing", "environment")):
        return "environment"
    if any(
        token in case_name
        for token in (
            "thruster",
            "thrust",
            "pwm",
            "voltage",
            "battery",
            "inflow",
            "wake",
            "reaction_torque",
            "allocation",
        )
    ):
        return "thrusters"
    return "physics"


def all_cases() -> tuple[Case, ...]:
    """Return every test case defined in the shared regression suite."""

    return tuple(
        function
        for name, function in inspect.getmembers(dynamics_cases, inspect.isfunction)
        if name.startswith("test_") and function.__module__ == dynamics_cases.__name__
    )


def cases_for(category: str) -> tuple[Case, ...]:
    """Return the exhaustive, non-overlapping cases assigned to ``category``."""

    if category not in CATEGORIES:
        raise ValueError(f"Unknown test category {category!r}; expected one of {CATEGORIES}.")
    selected = tuple(case for case in all_cases() if _category_for(case.__name__) == category)
    if not selected:
        raise RuntimeError(f"Test category {category!r} is empty.")
    return selected


def assert_catalog_is_complete() -> None:
    """Fail collection if a case is dropped or assigned more than once."""

    complete = all_cases()
    assigned = tuple(case for category in CATEGORIES for case in cases_for(category))
    if len(complete) != len(assigned) or set(complete) != set(assigned):
        raise AssertionError("Dynamics test catalog is incomplete or contains duplicate assignments.")


assert_catalog_is_complete()

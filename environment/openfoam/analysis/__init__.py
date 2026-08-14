"""Post-process prescribed-motion OpenFOAM cases into 6-DOF coefficients.

The public API deliberately depends only on :mod:`numpy` and the Python
standard library so it can run beside a stock OpenFOAM installation.
"""

from .fit import FitOptions, HydroFitResult, analyze_cases, fit_case_data
from .forces import (
    ForceSeries,
    discover_force_moment_files,
    discover_forces_files,
    load_case_forces,
    parse_forces_file,
    parse_total_vector_file,
)
from .motion import CaseData, MotionSpec, load_case_data

__all__ = [
    "CaseData",
    "FitOptions",
    "ForceSeries",
    "HydroFitResult",
    "MotionSpec",
    "analyze_cases",
    "discover_force_moment_files",
    "discover_forces_files",
    "fit_case_data",
    "load_case_data",
    "load_case_forces",
    "parse_forces_file",
    "parse_total_vector_file",
]

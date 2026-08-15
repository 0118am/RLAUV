"""Configuration and result records for hydrodynamic matrix fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .motion import DOF_NAMES, WRENCH_NAMES


@dataclass(frozen=True)
class FitOptions:
    """Numerical options for the coefficient fit."""

    project_added_mass_psd: bool = True
    min_added_mass_eigenvalue: float = 0.0
    bootstrap_samples: int = 200
    bootstrap_seed: int = 20260810
    passivity_samples: int = 10000
    passivity_tolerance: float = 1.0e-10
    minimum_samples_per_dof: int = 12
    phase_samples_per_cycle: int = 256
    diagonal_only: bool = False
    port_starboard_symmetry: bool = False
    include_rotation_attitude_term: bool = True
    include_roll_attitude_term: bool = True

    def __post_init__(self) -> None:
        if self.min_added_mass_eigenvalue < 0.0:
            raise ValueError("min_added_mass_eigenvalue must be non-negative")
        if self.bootstrap_samples < 0 or self.passivity_samples < 0:
            raise ValueError("bootstrap_samples and passivity_samples must be non-negative")
        if self.diagonal_only and self.port_starboard_symmetry:
            raise ValueError(
                "diagonal_only and port_starboard_symmetry are mutually exclusive"
            )
        if self.minimum_samples_per_dof < 3:
            raise ValueError("minimum_samples_per_dof must be at least 3")
        if (
            type(self.phase_samples_per_cycle) is not int
            or self.phase_samples_per_cycle < 8
            or self.phase_samples_per_cycle % 2
        ):
            raise ValueError("phase_samples_per_cycle must be an even integer of at least 8")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FitOptions":
        if not value:
            return cls()
        valid = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value[key] for key in valid if key in value})



@dataclass
class HydroFitResult:
    added_mass_raw: np.ndarray
    added_mass: np.ndarray
    linear_damping: np.ndarray
    quadratic_damping: np.ndarray
    diagnostics: dict[str, Any]
    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    case_summaries: list[dict[str, Any]] = field(default_factory=list)
    options: FitOptions = field(default_factory=FitOptions)

    def config_updates(self) -> dict[str, list[list[float]]]:
        return {
            # The runtime config key accepts the fitted full 6x6 matrix.
            "added_mass_diag": self.added_mass.tolist(),
            "linear_damping": self.linear_damping.tolist(),
            "quadratic_damping": self.quadratic_damping.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "coordinate_convention": {
                "frame": "body FLU at moving COM",
                "dof_order": list(DOF_NAMES),
                "wrench_order": list(WRENCH_NAMES),
                "force_sign": "fluid-on-body",
                "model": (
                    "tau = -M_A*nudot - C_A(nu,M_A)*nu - D_L*nu "
                    "- D_Q*(abs(nu)*nu)"
                ),
            },
            "matrices": {
                "added_mass_raw": self.added_mass_raw.tolist(),
                "added_mass": self.added_mass.tolist(),
                "linear_damping": self.linear_damping.tolist(),
                "quadratic_damping": self.quadratic_damping.tolist(),
            },
            "config_updates": self.config_updates(),
            "diagnostics": self.diagnostics,
            "confidence_intervals": self.confidence_intervals,
            "cases": self.case_summaries,
            "options": asdict(self.options),
        }


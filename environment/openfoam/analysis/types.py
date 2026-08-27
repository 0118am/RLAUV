"""Records for the preliminary full-response hydrodynamic fit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .motion import DOF_NAMES, WRENCH_NAMES


@dataclass(frozen=True)
class FitOptions:
    phase_samples_per_cycle: int = 128

    def __post_init__(self) -> None:
        if (
            type(self.phase_samples_per_cycle) is not int
            or self.phase_samples_per_cycle < 8
            or self.phase_samples_per_cycle % 2
        ):
            raise ValueError("phase_samples_per_cycle must be an even integer >= 8")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FitOptions":
        if value is None:
            return cls()
        valid = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ValueError(f"Unknown analysis option(s): {', '.join(unknown)}")
        return cls(**dict(value))


@dataclass
class HydroFitResult:
    added_mass: np.ndarray
    linear_damping: np.ndarray
    quadratic_damping: np.ndarray
    diagnostics: dict[str, Any]
    case_summaries: list[dict[str, Any]] = field(default_factory=list)
    options: FitOptions = field(default_factory=FitOptions)

    def config_updates(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "added_mass": self.added_mass.tolist(),
            "linear_damping": self.linear_damping.tolist(),
            "quadratic_damping": self.quadratic_damping.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "status": "preliminary_full_response_open_water_cfd",
            "coordinate_convention": {
                "frame": "body FLU at moving COM",
                "dof_order": list(DOF_NAMES),
                "wrench_order": list(WRENCH_NAMES),
                "force_sign": "fluid-on-body",
                "model": (
                    "tau_h=-M_A*nudot-C_A(nu,M_A)*nu-D_L*nu"
                    "-D_Q*(abs(nu)*nu)"
                ),
            },
            "identification_design": {
                "case_count": 24,
                "matrix_structure": (
                    "full response under exact port-starboard reflection symmetry; "
                    "allowed cross-axis coefficients are retained"
                ),
                "added_mass": (
                    "translation from one low-amplitude oscillatory case per DOF; "
                    "rotation jointly identified with damping from two rate amplitudes"
                ),
                "translation_damping": (
                    "two positive/negative steady speeds per translational DOF"
                ),
                "rotation_damping": (
                    "added mass, linear damping, and quadratic damping jointly fitted "
                    "from two rate amplitudes at one fixed frequency per rotational DOF"
                ),
                "rotation_low_amplitude_cases": (
                    "retained as an amplitude-dependence cross-check, not used to fix "
                    "the published rotational added mass"
                ),
                "coefficient_projection": (
                    "reflection mask for all matrices; symmetric reciprocity average "
                    "for added mass"
                ),
            },
            "matrices": {
                "added_mass": self.added_mass.tolist(),
                "linear_damping": self.linear_damping.tolist(),
                "quadratic_damping": self.quadratic_damping.tolist(),
            },
            "config_updates": self.config_updates(),
            "diagnostics": self.diagnostics,
            "cases": self.case_summaries,
            "options": asdict(self.options),
        }

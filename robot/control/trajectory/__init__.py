"""Deployable T60 trajectory guidance and reference generation."""

from typing import TYPE_CHECKING

from .catalog import (
    AXIS_SINE,
    BREATHING_LOOP,
    CHIRP,
    CIRCLE,
    EVALUATION_TRAJECTORY_NAMES,
    LATERAL_SINE,
    LISSAJOUS,
    RACETRACK,
    RANDOM_SMOOTH,
    SPATIAL_HELIX,
    SPEED_CONTROLLED_TYPES,
    TRAJECTORY_GENERATOR_VERSION,
    TRAJECTORY_TYPE_IDS,
    VERTICAL_SINE,
    WAVY_LOOP,
)
from .observation_contract import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_FIELD_DIMENSIONS,
    OBSERVATION_FIELD_SLICES,
    OBSERVATION_NORMALIZATION_SCALES,
    TRAJECTORY_OBSERVATION,
)
if TYPE_CHECKING:
    from .geometry import evaluate_geometry
    from .retiming import (
        RetimedTrajectoryTables,
        TrajectoryKinematicLimits,
        build_retimed_tables,
        evaluate_retimed_reference,
        sample_retimed_phase,
        smooth_startup_time,
    )

__all__ = [
    "ACTION_DIM",
    "AXIS_SINE",
    "BASE_OBSERVATION_DIM",
    "BREATHING_LOOP",
    "CHIRP",
    "CIRCLE",
    "EVALUATION_TRAJECTORY_NAMES",
    "LATERAL_SINE",
    "LISSAJOUS",
    "OBSERVATION_CONTRACT_VERSION",
    "OBSERVATION_FIELD_DIMENSIONS",
    "OBSERVATION_FIELD_SLICES",
    "OBSERVATION_NORMALIZATION_SCALES",
    "RACETRACK",
    "RANDOM_SMOOTH",
    "RetimedTrajectoryTables",
    "SPATIAL_HELIX",
    "SPEED_CONTROLLED_TYPES",
    "TRAJECTORY_OBSERVATION",
    "TRAJECTORY_GENERATOR_VERSION",
    "TRAJECTORY_TYPE_IDS",
    "TrajectoryKinematicLimits",
    "VERTICAL_SINE",
    "WAVY_LOOP",
    "build_retimed_tables",
    "evaluate_geometry",
    "evaluate_retimed_reference",
    "sample_retimed_phase",
    "smooth_startup_time",
]


def __getattr__(name: str):
    if name == "evaluate_geometry":
        from .geometry import evaluate_geometry

        return evaluate_geometry
    if name in {
        "RetimedTrajectoryTables",
        "TrajectoryKinematicLimits",
        "build_retimed_tables",
        "evaluate_retimed_reference",
        "sample_retimed_phase",
        "smooth_startup_time",
    }:
        from . import retiming

        return getattr(retiming, name)
    raise AttributeError(name)

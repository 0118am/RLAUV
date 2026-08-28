"""Single deployable observation contract shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


OBSERVATION_CONTRACT_VERSION = "t60_trajectory_obs_v8"


@dataclass(frozen=True)
class ObservationField:
    name: str
    width: int
    physical_scale: float = 1.0


@dataclass(frozen=True)
class TrajectoryObservationContract:
    fields: tuple[ObservationField, ...]
    action_dim: int

    @property
    def dimension(self) -> int:
        return sum(field.width for field in self.fields)

    @property
    def slices(self) -> Mapping[str, slice]:
        start = 0
        result: dict[str, slice] = {}
        for field in self.fields:
            result[field.name] = slice(start, start + field.width)
            start += field.width
        return MappingProxyType(result)

    @property
    def field_dimensions(self) -> Mapping[str, int]:
        return MappingProxyType({field.name: field.width for field in self.fields})

    @property
    def normalization_scales(self) -> tuple[float, ...]:
        return tuple(
            scale
            for field in self.fields
            for scale in (float(field.physical_scale),) * field.width
        )

    def field(self, name: str) -> ObservationField:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)


TRAJECTORY_OBSERVATION = TrajectoryObservationContract(
    fields=(
        ObservationField("position_error_b", 3, 0.25),
        ObservationField("target_linear_velocity_b", 3, 0.50),
        ObservationField("linear_velocity_error_b", 3, 0.20),
        ObservationField("attitude_error_quat", 4, 1.0),
        ObservationField("projected_gravity_b", 3, 1.0),
        ObservationField("angular_velocity_b", 3, 0.80),
        ObservationField("target_angular_velocity_b", 3, 0.80),
        ObservationField("target_linear_acceleration_b", 3, 0.45),
        ObservationField("motor_command", 8, 1.0),
    ),
    action_dim=8,
)

BASE_OBSERVATION_DIM = TRAJECTORY_OBSERVATION.dimension
ACTION_DIM = TRAJECTORY_OBSERVATION.action_dim
OBSERVATION_FIELD_SLICES = TRAJECTORY_OBSERVATION.slices
OBSERVATION_FIELD_DIMENSIONS = TRAJECTORY_OBSERVATION.field_dimensions
OBSERVATION_NORMALIZATION_SCALES = TRAJECTORY_OBSERVATION.normalization_scales

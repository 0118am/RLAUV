"""Single deployable observation contract shared by training and inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch


OBSERVATION_CONTRACT_VERSION = "t60_trajectory_obs_v11"


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
        ObservationField("angular_velocity_b", 3, 0.80),
        ObservationField("target_angular_velocity_b", 3, 0.80),
        ObservationField("target_linear_acceleration_b", 3, 0.45),
        ObservationField("previous_motor_command", 8, 1.0),
    ),
    action_dim=8,
)

BASE_OBSERVATION_DIM = TRAJECTORY_OBSERVATION.dimension
ACTION_DIM = TRAJECTORY_OBSERVATION.action_dim
OBSERVATION_FIELD_SLICES = TRAJECTORY_OBSERVATION.slices
OBSERVATION_FIELD_DIMENSIONS = TRAJECTORY_OBSERVATION.field_dimensions
OBSERVATION_NORMALIZATION_SCALES = TRAJECTORY_OBSERVATION.normalization_scales

# History profiles may select base fields or deployable quantities derived from
# fields in the same normalized current sample. Angular velocity and its target
# share the same physical scale, so their normalized difference is the
# normalized body-frame tracking error without any simulator-only input.
HISTORY_OBSERVATION_FIELD_DIMENSIONS = MappingProxyType(
    {
        **dict(OBSERVATION_FIELD_DIMENSIONS),
        "angular_velocity_error_b": 3,
    }
)


def normalized_history_observation(
    normalized_current_observation: torch.Tensor,
    fields: Sequence[str],
) -> torch.Tensor:
    """Return selected normalized history features in declared field order."""

    terms: list[torch.Tensor] = []
    for name in fields:
        if name == "angular_velocity_error_b":
            target = normalized_current_observation[
                ..., OBSERVATION_FIELD_SLICES["target_angular_velocity_b"]
            ]
            measured = normalized_current_observation[
                ..., OBSERVATION_FIELD_SLICES["angular_velocity_b"]
            ]
            terms.append(target - measured)
            continue
        try:
            field_slice = OBSERVATION_FIELD_SLICES[name]
        except KeyError as error:
            raise KeyError(f"Unknown history observation field {name!r}.") from error
        terms.append(normalized_current_observation[..., field_slice])

    if not terms:
        return normalized_current_observation[..., :0]
    return torch.cat(terms, dim=-1)

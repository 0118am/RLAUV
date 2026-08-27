"""Shared strict types for versioned JSON inputs."""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
)
from typing_extensions import TypeAliasType


def _reject_boolean(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric value")
    return value


class StrictFrozenModel(BaseModel):
    """Immutable input model that rejects misspelled fields and non-finite floats."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


StrictBoolean = StrictBool
FiniteJsonValue = TypeAliasType(
    "FiniteJsonValue",
    dict[str, "FiniteJsonValue"]
    | list["FiniteJsonValue"]
    | str
    | bool
    | int
    | FiniteFloat
    | None,
)
FiniteNumber = Annotated[FiniteFloat, BeforeValidator(_reject_boolean)]
PositiveFloat = Annotated[FiniteNumber, Field(gt=0.0)]
NonNegativeFloat = Annotated[FiniteNumber, Field(ge=0.0)]
Probability = Annotated[FiniteNumber, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, BeforeValidator(_reject_boolean), Field(gt=0)]
NonNegativeInt = Annotated[int, BeforeValidator(_reject_boolean), Field(ge=0)]

Vector2 = Annotated[tuple[FiniteNumber, ...], Field(min_length=2, max_length=2)]
Vector3 = Annotated[tuple[FiniteNumber, ...], Field(min_length=3, max_length=3)]
Vector4 = Annotated[tuple[FiniteNumber, ...], Field(min_length=4, max_length=4)]
Vector6 = Annotated[tuple[FiniteNumber, ...], Field(min_length=6, max_length=6)]

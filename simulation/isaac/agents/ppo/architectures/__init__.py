"""Versioned feed-forward trajectory-policy architecture profiles."""

from .base import (
    BASE_OBSERVATION_DIM,
    CRITIC_PRIVILEGED_FIELD_DIMENSIONS,
    TRAJECTORY_CRITIC_PRIVILEGED_FIELDS,
    MlpArchitecture,
)
from .registry import MLP_ARCHITECTURES, available_mlp_architectures, get_mlp_architecture

__all__ = [
    "BASE_OBSERVATION_DIM",
    "CRITIC_PRIVILEGED_FIELD_DIMENSIONS",
    "MLP_ARCHITECTURES",
    "MlpArchitecture",
    "TRAJECTORY_CRITIC_PRIVILEGED_FIELDS",
    "available_mlp_architectures",
    "get_mlp_architecture",
]

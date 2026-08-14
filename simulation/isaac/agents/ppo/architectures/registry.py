"""Named PPO architecture registry used by notebook-facing workflow code."""

from __future__ import annotations

from .base import MlpArchitecture
from .mlp.mlp_30d import ARCHITECTURE as MLP_30D
from .mlp.mlp_history_5 import ARCHITECTURE as MLP_HISTORY_5


MLP_ARCHITECTURES: dict[str, MlpArchitecture] = {
    architecture.name: architecture for architecture in (MLP_30D, MLP_HISTORY_5)
}


def available_mlp_architectures() -> tuple[str, ...]:
    return tuple(MLP_ARCHITECTURES)


def get_mlp_architecture(name: str) -> MlpArchitecture:
    try:
        return MLP_ARCHITECTURES[name]
    except KeyError as error:
        available = ", ".join(available_mlp_architectures())
        raise ValueError(f"Unknown MLP architecture {name!r}. Available: {available}.") from error

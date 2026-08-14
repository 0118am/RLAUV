"""Feed-forward PPO policy architecture profiles."""

from .mlp_30d import ARCHITECTURE as MLP_30D
from .mlp_history_5 import ARCHITECTURE as MLP_HISTORY_5

__all__ = ["MLP_30D", "MLP_HISTORY_5"]


"""Classical trajectory controllers using the same actuator interface as PPO."""

from .pid import PIDGains, PIDTrajectoryController

__all__ = ["PIDGains", "PIDTrajectoryController"]

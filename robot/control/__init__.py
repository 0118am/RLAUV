"""Robot-owned classical controllers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .allocation import NonlinearThrusterAllocator
    from .pid import PIDGains, PIDTrajectoryController

__all__ = ["NonlinearThrusterAllocator", "PIDGains", "PIDTrajectoryController"]


def __getattr__(name: str):
    if name == "NonlinearThrusterAllocator":
        from .allocation import NonlinearThrusterAllocator

        return NonlinearThrusterAllocator
    if name in {"PIDGains", "PIDTrajectoryController"}:
        from .pid import PIDGains, PIDTrajectoryController

        return {"PIDGains": PIDGains, "PIDTrajectoryController": PIDTrajectoryController}[name]
    raise AttributeError(name)

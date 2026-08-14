"""Vehicle geometry, mass properties, and inertia data.

``parameters.py`` is the authoritative source for the physical AUV contract.
The propulsion package consumes its T1--T8 geometry and measured force-curve
coefficients without duplicating them in a simulator.
"""

from .parameters import AUV

__all__ = ["AUV"]

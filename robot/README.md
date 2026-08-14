# Robot

This domain is the shared T60 AUV definition. `dynamics/parameters.py` is the
authoritative physical contract; `propulsion/thrusters.py` evaluates the
measured T1--T8 force curves; and `assets/` stores the simulator-specific body
representations. Simulator adapters import these files instead of duplicating
mass, inertia, thruster, or geometry constants.

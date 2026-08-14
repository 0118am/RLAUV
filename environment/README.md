# Environment

This domain owns the water around the AUV and the evidence used to identify
its hydrodynamics. OpenFOAM and PMM keep their raw inputs, generated cases, and
results in self-contained subdirectories. Runtime simulators consume validated
profiles from `hydrodynamics/coefficients/` and reusable equations from the
Python packages here.

Nothing in this directory may depend on Isaac Lab or MuJoCo.

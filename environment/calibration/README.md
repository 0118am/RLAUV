# Calibration workflows

Run these scripts to transform validated experimental CSV logs into profile
updates, then merge them with `build_pool_profile_from_calibration.py` and
check the result with `audit_pool_profile.py`.

`fit_pool_static_logs.py` fits rigid-body/hydrostatic properties;
`fit_pool_thruster_logs.py` fits actuator timing, voltage, wake, and reaction-torque effects;
the other fitters cover hydrodynamics and pool effects.

The measured T1--T8 three-axis PWM-to-force polynomials in
`robot/dynamics/parameters.py` are the only static thrust mapping. Calibration workflows do not
generate an alternate lookup table or select a static-mapping backend.

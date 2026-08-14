# Test organization

The test tree mirrors production ownership:

- `environment/` covers water effects, hydrodynamics, profiles, identification, PMM, and calibration.
- `robot/` covers rigid-body, tether, and measured thruster behavior.
- `simulation/isaac/` covers PPO, rewards, PID, PhysX integration, sensors, trajectories, and workflows.
- `simulation/mujoco/` covers the independent cross-simulator policy bridge and model contract.
- `integration/dynamics_cases.py` retains shared case implementations while domain modules provide exhaustive pytest collection.

Run the fast checks before a long policy-training job:

```bash
conda run -n env_isaaclab python tests/run_policy_preflight.py
```

Run all simulator-independent regression tests after changing physics, sensors,
profiles, calibration, or replay code:

```bash
conda run -n env_isaaclab python -m pytest -q
```

These tests intentionally avoid launching Isaac Sim. A short one-iteration
IsaacLab smoke training remains the final check before a full GPU training run.

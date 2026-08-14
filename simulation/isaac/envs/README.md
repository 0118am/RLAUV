# Environment integration

`auv/` contains the Isaac-facing environment implementation. Its `env.py`
facade composes the shared `environment/` and `robot/` packages, exposes
policy observations/actions, and applies resulting wrenches to PhysX.

Put simulator-independent water equations in `environment/`, robot equations
in `robot/`, and only the PhysX adapter or lifecycle integration in `auv/`.
This keeps unit tests independent of IsaacLab startup and avoids duplicate
physics formulas.

## AUV package ownership

| Module | Owns | Must not own |
| --- | --- | --- |
| `auv/config.py` | Gym/IsaacLab configuration and policy-space contract | Runtime state or force equations |
| `auv/env.py` | Isaac lifecycle, scene setup, reset/step/reward orchestration | Sensor, trajectory, or fluid-model implementations |
| `auv/observations.py` | Policy observation delay, noise, link jitter, and MLP history | Force or actuator equations |
| `auv/trajectory/` | Trajectory command distribution, guidance, and curricula | Fluid or actuator equations |
| `auv/dynamics.py` | Reading Isaac state and composing the final wrench | Duplicate environment or robot equations |
| `auv/visualization.py` | Debug-only markers and UI | Simulation state changes |

The package exposes one Gym task: `Isaac-AUV-Traj-Direct-v1`.

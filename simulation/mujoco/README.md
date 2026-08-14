# MuJoCo policy validation

This workflow runs an exported trajectory Actor in MuJoCo as an independent
cross-simulator check. It supports ONNX policies and RSL-RL `.pt` checkpoints.

The bridge preserves the deployed policy contract:

- 30-D current observation, including its physical normalization;
- optional `mlp_history_5` causal history, producing 135 Actor inputs;
- eight normalized thruster commands;
- command delay/rate limiting, the measured T1--T8 three-component PWM force
  polynomials clamped to 1300--1700 us with the 1475--1525 us zero dead zone,
  and first-order force response;
- buoyancy, relative-velocity damping, added-mass Coriolis terms, periodic
  current, and the selected profile's full 6x6 OpenFOAM hydrodynamic matrices.
  The added-mass diagonal is integrated as a stable generalized-axis armature
  approximation rather than an unstable delayed acceleration force.

## Setup

Activate the project environment, then install only the MuJoCo validation
extras:

```bash
conda activate env_isaaclab
python -m pip install -r simulation/mujoco/requirements.txt
```

Export a checkpoint when ONNX is preferred:

```bash
python simulation/isaac/rlpolicy/export_onnx.py \
  --checkpoint /path/to/model_480.pt \
  --mlp_architecture mlp_history_5
```

## Run

```bash
python simulation/mujoco/validate_policy.py \
  --policy simulation/isaac/rlpolicy/<architecture>/<run>/exports/<policy>.onnx \
  --mlp-architecture mlp_history_5
```

A checkpoint can be used directly by passing `model_*.pt`. Add `--render` for
the passive viewer.

Results are written to `results/mujoco/<policy>_<architecture>_<trajectory>/`:

- `rollout.csv`: target, actual position, raw actions and applied actions;
- `summary.json`: simulator/profile contract, RMSE and pass/fail gates.

The command exits with status 2 if post-settling position RMSE exceeds
`--max-position-rmse` or too many raw actions exceed `[-1, 1]`. These gates are
cross-simulator regression checks, not a substitute for IsaacLab evaluation or
pool trials.

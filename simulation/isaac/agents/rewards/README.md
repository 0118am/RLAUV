# AUV reward policies

- `policy_0.py` keeps the original pose/velocity balance and adds mandatory
  body +X/actual-motion alignment.
- `policy_1.py` balances command-direction and actual-motion alignment.
  Its maximum positive per-step reward remains 4.87, matching `policy_0`.
- `policy_2.py` replaces the long-tailed tracking kernels with Gaussian kernels
  while retaining command and actual-motion alignment.
- `policy_3.py` couples command-direction alignment with actual body-x motion
  alignment; a stationary or backward-moving vehicle cannot collect the full
  heading reward for a moving target.
- `policy_4.py` uses normalized L1 action and action-rate penalties to suppress
  persistent low-amplitude chatter, while retaining actual-motion alignment.
  Its penalty at the action bounds matches `policy_1` despite the different norm.
- `base.py` contains shared immutable types, quaternion math, and reusable
  tolerance/alignment kernels.
- `registry.py` selects a policy and applies the coefficients owned by that file.
- `__init__.py` exposes the stable API imported by the environment.

Each policy file owns both its reward equation and all of its coefficients.
Never silently modify a policy used to train a checkpoint. Add the next file
(`policy_5.py`, `policy_6.py`, then `policy_7.py`, and so on), register it in `registry.py`,
and select the same policy in the train and eval notebooks.

The legacy names `baseline` and `heading_v1` remain readable as aliases for
`policy_0` and `policy_1`. New experiments should use `policy_N` names only.
Use `tracking_reward_profile="custom"` only for temporary Hydra coefficient
sweeps.

## Contract revision: actual-motion alignment

As of 2026-07-16, every `policy_0` through `policy_4` contains an explicit
positive term for alignment between the vehicle's +X nose and its actual
body-frame linear velocity. The score is 1 for forward motion, 0 for reverse
motion, and 0 when actual speed is too small to define a direction.

This intentional revision changes the reward contract of `policy_0` and
`policy_1`. Checkpoints trained before this revision may be evaluated, but must
not be resumed under the revised reward or compared by raw reward value.

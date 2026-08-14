# AUV reward policies

Each `policy_N.py` file owns a complete, immutable reward equation and its
coefficients. `registry.py` is the only selector used by the environment,
Python training manager, and evaluation entry point.

- `policy_0`: pose/velocity tracking with body-forward motion alignment.
- `policy_1`: command-direction and actual-motion alignment.
- `policy_2`: Gaussian tracking kernels.
- `policy_3`: coupled commanded-heading and actual forward-motion alignment.
- `policy_4`: normalized L1 action and action-rate regularization.
- `policy_5`: compact applied-action-rate objective.
- `policy_6`: Huber tracking residuals, applied-action regularization, and true termination cost.

`base.py` contains shared immutable types and reward math. Add a new numbered
file when the equation or coefficients change, register it, and select the
same name in the repository-root `train.ipynb` and `eval.ipynb`. Use
`tracking_reward_profile="custom"` only for an explicitly configured Hydra
coefficient sweep.

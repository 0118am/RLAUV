# MLP architecture profiles

`train.ipynb` and `eval.ipynb` expose one architecture
field: `MLP_ARCHITECTURE`.  Its value resolves through `registry.py`; no
separate history length, history-field list, or layer-width switch should be
added to a notebook.

Each profile declares:

- the feed-forward Actor/Critic layer widths;
- the 30-D current-sample history fields and number of past samples; and
- the simulator-only privileged fields appended to the Critic; and
- an isolated RSL-RL experiment namespace.

`mlp_history_5` uses the current 30-D deployable observation plus five prior
samples of position error, linear-velocity error, attitude error, angular
velocity, and the actual rate-limited actuator command. This is 135 inputs.
The Critic receives those same 135 inputs plus 77-D exact simulator state:
true navigation state, instantaneous water current, effective damping/added
mass/buoyancy, sampled rigid-body properties, realized actuator force, and
actuator/battery/tether state. The environment builds and resets the causal
buffer; train, evaluation, and ONNX export resolve the same profile so their
Actor shapes cannot drift. Evaluation and ONNX never consume the Critic group.

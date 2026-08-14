# Training agents

`simulation/isaac/agents/` owns everything that defines how a policy learns:

- `ppo/` contains the PPO configuration, algorithm, runner, evaluation loader,
  and feed-forward MLP architecture profiles.
- `rewards/` contains task reward equations and versioned reward profiles.

Simulation code must not import PPO internals. The IsaacLab environment may
call the reward package with explicit state tensors, while reward modules must
not import the environment.

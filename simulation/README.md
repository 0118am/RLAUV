# Simulation

`isaac/` owns training, evaluation, policy code, controllers, and the PhysX
adapter. `mujoco/` is an independent validation backend. Both consume the same
`environment/` and `robot/` contracts.

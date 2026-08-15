# Simulation

`simulation/` 只管理模拟器接线和训练流程，不拥有机器人或水环境的物理参数。

- `assembly.py`：唯一 Isaac Lab/PhysX 组装文件；读取模拟状态、调用 `robot/` 和
  `environment/` 的模型，并只提交一次最终 wrench。
- `training/`：版本化 recipe、run manifest、网络、PPO、奖励、观测、训练轨迹、评估、导出和进程管理。
- `rlpolicy/`：仅保存 checkpoint、日志、评估和导出产物，不包含功能代码。

机器人数据来自 `robot/`；水流、水动力和池体效应来自 `environment/`。本目录没有
`isaac/` 中间层，也不保留旧路径的兼容转发模块。

Assembly 从 PhysX 构造只读 `BodyKinematics`，分别传给 `EnvironmentRuntimeState` 和
`RobotRuntimeState`。两个 runtime 不访问 PhysX；reset 后的质量、惯量和 COM，以及每个物理步
唯一一次合成 wrench，均由 Assembly 写回。

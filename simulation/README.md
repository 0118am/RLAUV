# Simulation

`simulation/` 只管理模拟器接线和训练流程，不拥有机器人或水环境的物理参数。

- `assembly.py`：唯一 Isaac Lab/PhysX 组装文件；读取模拟状态、调用 `robot/` 和
  `environment/` 的模型，并只提交一次最终 wrench。
- `assets.py`：从 robot-owned USD 和刚体事实构造 Isaac asset 配置。
- `composition.py`：组合确定性 environment、robot runtime 和可选跨域 DR。
- `domain_randomization.py`：Pydantic 2 跨域 DR schema、feature 选择和 JSON I/O。
- `dynamics.py`：把总惯量求解映射为 PhysX 外部 wrench 的纯张量桥。
- `training/`：Pydantic 2 recipe、网络、PPO、奖励、观测、训练轨迹、评估、
  PyTorch ONNX 导出和基于 `psutil` 的进程管理。
- `rlpolicy/`：仅保存 checkpoint、日志、评估和导出产物，不包含功能代码。

机器人数据来自 `robot/`；水流、水动力和池体效应来自 `environment/`。本目录没有
`isaac/` 中间层，也不保留旧路径的兼容转发模块。

Assembly 从 PhysX 构造只读 `BodyKinematics`，分别传给 `EnvironmentRuntimeState` 和
`RobotRuntimeState`。两个 runtime 不访问 PhysX；reset 后的质量、惯量和 COM，以及每个物理步
唯一一次合成 wrench，均由 Assembly 写回。

当前训练轨迹契约为 `curve_v5` / `t60_trajectory_obs_v7`：目标姿态始终水平，只生成 yaw
角速度；姿态 reward 对实际 roll、pitch 与目标 yaw 使用三轴独立误差。评估日志 schema v7 分别记录
目标 yaw 角速度、艇首到目标 heading 的水平夹角和艇首到实际运动 heading 的水平夹角。

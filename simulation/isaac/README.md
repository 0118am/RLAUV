# Isaac Lab 合成与训练层

本目录负责把 `robot/` 与 `environment/` 的共享定义接入 Isaac Lab/PhysX，并实现任务、学习
和实验生命周期。它不是机器人或水动力参数的数据源。

## 文件职责

- `config.py`：Direct Task 的策略空间、仿真频率、任务和可序列化适配字段。
- `env.py`：`DirectRLEnv` reset/step 生命周期、状态张量和奖励调用。
- `composition.py`：加载并校验一个环境 profile、T60 runtime 和可选 DR spec。
- `physics_adapter.py`：计算水流、相对速度、池体缩放和有效水动力状态。
- `force_composition.py`：计算推进器与流体 wrench，并归并可选系缆项。
- `robot_asset.py`：绑定 T60 USD，并显式关闭 PhysX 自带线性/角阻尼。
- `observations.py`：30-D 当前观测的归一化，以及架构声明的因果历史堆叠。
- `visualization.py`、`visualization_geometry.py`：Isaac 调试可视化。
- `ppo/`：PPO 算法、runner、Actor 加载和命名 MLP 架构。
- `rewards/`：版本化轨迹奖励及注册表。
- `training.py`：供根目录 `train.ipynb` 调用的 profile 快照和 campaign 进程管理。
- `trajectory/`：Isaac task mixin、隔离的 train/eval worker、能力门控与报告工具。
- `rlpolicy/`：Git 忽略的运行产物、评估结果和 ONNX 导出工具。

## 外部依赖边界

机器人质量、惯量、资产、推进器、执行器、电池、系缆、PID 和通用轨迹代码来自 `robot/`。
水流、阻尼、附加质量、池壁、自由液面及对应随机化来自 `environment/`。

本目录允许保存 Isaac 所需的字段名和坐标/张量适配，但不允许维护第二套物理默认值。如果一个
函数不依赖 Isaac API，并且表达的是机器人性质或水物理，应分别迁移到 `robot/` 或
`environment/`。

最终只有一个机体系质心 wrench 入口。推进器力来自 `robot/propulsion/curves.py` 的完整
FLU 三分量曲线；水动力来自 `environment/hydrodynamics/models.py` 的完整 `6×6` 矩阵。
本层不重建推进器方向，也不叠加反扭矩或高阶残差 wrench。

## 训练入口

人工训练入口只有仓库根目录的 `train.ipynb`。它把流速、水池效应、完整 DR 强度和训练规模
传给 `training.py`；后者生成 `rlpolicy/_configs/<RUN_NAME>/` 快照，并管理 preview/start/
status/stop。`trajectory/train.py` 是 Isaac Sim worker，不是人工配置文件。

## 观测与 sensing 边界

模拟环境只组装当前模拟状态及 MLP 所需的因果历史，不注入观测延迟、滤波、丢包或传感器
噪声。滤波算法应由真实设备数据验证并放在实际测量/部署链，不在这里建立模拟 sensing
子系统。

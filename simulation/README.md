# Isaac simulation

`simulation/` 只管理 Isaac 适配和实验流程，不拥有机器人本体或水动力物理参数。

- `isaac/`：Isaac Lab/PhysX Direct Task、共享模型适配、观测、奖励、PPO、训练和评估。

水动力状态计算位于 `isaac/physics_adapter.py`，推进器、缆绳和最终 wrench 合成位于
`isaac/force_composition.py`。训练和评估命令分别位于
`isaac/trajectory/training_commands.py` 与 `evaluation_commands.py`，旧的公共导入入口保持兼容。

Isaac 后端必须从 `robot/` 读取 T60 本体、推进器和运行参数，从 `environment/` 读取水流、
水动力与池体效应。模拟器目录可以保存坐标转换、状态读取、外力注入和运行生命周期，但不能
复制质量、惯量、推力曲线、阻尼矩阵或水流模型。

训练的人工入口是仓库根目录的 `train.ipynb`，评估入口是 `eval.ipynb`。机器人 PID 与轨迹
几何位于 `robot/control/`；机器人随机化位于 `robot/randomization/`；水流和水动力随机化
位于 `environment/randomization/`。

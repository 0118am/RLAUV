# Training

训练树只保留三个需要分组的功能目录：

| 路径 | 职责 |
| --- | --- |
| `recipes/` | 版本化训练输入 |
| `ppo/` | 网络、算法、runner 和 PPO 配置 |
| `evaluation/` | 评估配置、执行、指标、调度和 ONNX 导出 |

根目录中的单文件模块各自对应一个完整职责：

| 文件 | 职责 |
| --- | --- |
| `recipe.py`、`manifest.py` | 训练输入、请求和 run-local 输出契约 |
| `campaign.py` | 命令构造、进程管理、run/checkpoint 查询 |
| `train.py` | IsaacLab 训练入口及 worker 生命周期 |
| `config.py` | policy-facing Direct Task 配置 |
| `observations.py` | Actor 历史观测和 Critic 特权观测 |
| `rewards.py` | 不可变奖励 profile、选择和唯一张量实现 |
| `trajectory.py` | 训练轨迹、课程和 reference runtime |
| `visualization.py` | 环境调试显示 |

奖励的 `policy_0`–`policy_6` 参数和统一计算函数都在 `rewards.py`。新增奖励时扩展该文件中的
profile 和明确的 variant，不复制 `policy_N.py`。

`train.py` 是 IsaacLab worker，人工入口为仓库根目录 `train.ipynb`。训练模块不保存机器人或
水动力参数，也不直接写 PhysX；最终接线只在 `simulation/assembly.py`。

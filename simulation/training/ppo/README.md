# PPO and network profiles

`networks.py` 是 Actor/Critic 网络结构的唯一注册表：

| profile | Actor 输入 | Actor/Critic 隐藏层 |
| --- | ---: | --- |
| `mlp_30d` | 30 | `512,256,128` |
| `mlp_history_5` | 135 | `512,384,256,128` |

`mlp_history_5` 在当前 30-D 可部署观测后追加五个历史样本中的位置误差、线速度误差、姿态
误差、角速度和实际 applied action。两种 Critic 都可再使用 76-D 模拟器特权状态；Actor、
评估 Actor 和 ONNX 不使用特权字段。

`config.py` 保存 PPO 超参数，`algorithm.py` 实现按 rollout 固定学习率、KL early-stop 和下一
rollout 学习率调整，`runner.py` 批量收集 GPU episode 统计。训练、评估和导出都按同一个
network profile 解析输入维度，不能在 notebook 中分别设置历史长度或隐藏层。

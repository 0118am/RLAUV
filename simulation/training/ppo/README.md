# PPO and network profiles

`networks.py` 是 Actor/Critic 网络结构的唯一注册表：

| profile | Actor 输入 | Actor/Critic 隐藏层 |
| --- | ---: | --- |
| `mlp_33d` | 33 | `512,256,128` |
| `mlp_history_8` | 201 | `512,384,256,128` |

`mlp_history_8` 在当前 33-D 可部署观测后追加八个历史样本中的位置误差、线速度误差、完整
四元数姿态误差、角速度和处理后推进器指令，覆盖 320 ms 动态窗口。该姿态误差同时包含
roll、pitch 和 yaw；两种 Critic 都可再使用 73-D 模拟器特权状态；Actor、
评估 Actor 和 ONNX 不使用特权字段。

`config.py` 直接配置 RSL-RL 原生 `PPO`，使用 `schedule="adaptive"` 和
`desired_kl=0.01`。每个 minibatch 更新前计算新旧高斯策略的平均 KL；KL 大于 `0.02` 时共享
Adam 学习率除以 `1.5`，KL 位于 `(0, 0.005)` 时乘以 `1.5`。原生实现不回滚参数、不重试
minibatch，也不分别维护 Actor/Critic 学习率。策略使用 log-standard-deviation 参数化，初始
标准差为 `0.5`，不再强制投影最小标准差。训练、评估和导出都按同一个 network profile 解析
输入维度，不能在 notebook 中分别设置历史长度或隐藏层。

TensorBoard 中主要查看 `Loss/value_function`、`Loss/surrogate`、`Loss/entropy`、
`Loss/learning_rate` 和 `Policy/mean_noise_std`。原生版本不产生自定义的 rollout KL、回滚次数
或独立 Actor/Critic 更新计数。

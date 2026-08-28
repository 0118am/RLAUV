# PPO and network profiles

`networks.py` 是 Actor/Critic 网络结构的唯一注册表：

| profile | Actor 输入 | Actor/Critic 隐藏层 |
| --- | ---: | --- |
| `mlp_33d` | 33 | `512,256,128` |
| `mlp_history_8` | 201 | `512,256,128` |

`mlp_history_8` 在当前 33-D 可部署观测后追加八个历史样本中的位置误差、线速度误差、完整
四元数姿态误差、角速度和处理后推进器指令，覆盖 320 ms 动态窗口。该姿态误差同时包含
roll、pitch 和 yaw；两种 Critic 都可再使用 63-D 模拟器特权状态；Actor、
评估 Actor 和 ONNX 不使用特权字段。

`smooth_ppo.py` 保留当前 RSL-RL PPO 的 clipped surrogate 和 clipped value loss，并在 Actor
loss 中增加确定性 Actor 均值曲率：

`L_curvature = 2.5 · mean(||μ_t - 2μ_{t-1} + μ_{t-2}||₂ / √8)`。

完整形式的分母是 `dt² · 625 action/s²`；当前 `dt=0.04 s`，所以该分母恰为 1。rollout 的
扁平索引用于取回 `t-1/t-2` 观测，不复制 201-D 观测缓存；只使用 `t≥2` 且两处分界均未 done
的同 episode 三连帧。三个 Actor 前向都参与反向传播，存储的旧策略均值不用于该 loss。
该公式对应 Grad-CAPS 的未位移归一化二阶差分核心；这里使用固定物理尺度，不包含完整
Grad-CAPS 中按总动作位移再乘 `tanh` 的归一化，因此不把当前实现标作完整 Grad-CAPS 复现。

Actor 和 Critic 的隐藏线性层使用 gain `√2` 的正交权重初始化，Actor 输出层 gain 为 `0.01`，
Critic 输出层 gain 为 `1.0`，所有线性层 bias 初始化为零但保持可训练。该初始化只决定从头训练
的起点，不改写 checkpoint；状态无关的 `log_std` 仍由 `init_noise_std=0.5` 单独初始化。

PPO 使用 `schedule="fixed"` 和一个包含两个参数组的 Adam：Actor（含 `log_std`）固定学习率为
`3e-5`，Critic 固定学习率为 `3e-4`，两组梯度分别裁剪到 `1.0`。rollout 保存旧策略的高斯均值与标准差；
每个 minibatch 在反向传播前计算对角高斯的解析 `KL(old || new)`，对八个动作维求和并对样本
取平均。KL 超过 `1.5 × desired_kl = 0.015` 时，只停止本轮剩余 Actor 更新，不回滚已经完成的
前一步；Critic 仍完成全部 `5 × 32 = 160` 个 minibatch 更新。因此 PPO surrogate、Actor 曲率项
和标准差参数共同造成的策略位移都受同一个 KL 停止条件约束，Critic 则不会因 Actor 越界而欠训练，
也不会再由低 KL 触发学习率连续放大。策略使用 log-standard-deviation 参数化，初始标准差为
`0.5`。训练、评估和导出都按同一个 network profile 解析输入维度，不能在 notebook 中分别设置
历史长度或隐藏层。

TensorBoard 中主要查看 `Loss/value_function`、`Loss/surrogate`、`Loss/entropy`、
`Loss/action_curvature`、`Loss/kl`、`Loss/kl_max`、`Loss/actor_update_fraction`、
`Loss/critic_update_fraction`、`Loss/learning_rate` 和
`Policy/mean_noise_std`。`Loss/action_curvature` 是已乘 `2.5` 的 Actor loss 贡献；发生 KL early
stopping 时，`Loss/kl_max` 会超过 `0.015`，`Loss/actor_update_fraction` 会小于 `1.0`，而
`Loss/critic_update_fraction` 应保持 `1.0`。`Loss/learning_rate` 记录 Actor 学习率 `3e-5`。

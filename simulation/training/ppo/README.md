# PPO and network profiles

`networks.py` 是 Actor/Critic 网络结构的唯一注册表：

| profile | Actor 输入 | Actor/Critic 隐藏层 |
| --- | ---: | --- |
| `mlp_30d` | 30 | `512,256,128` |
| `mlp_history_8` | 198 | `512,256,128` |

`mlp_history_8` 在当前 30-D 可部署观测后追加八个历史样本中的位置误差、线速度误差、完整
四元数姿态误差、角速度误差和上一周期有界电机指令，覆盖 320 ms 动态窗口。角速度误差由同一
时刻、同一机体系的目标角速度减去实测角速度得到；该姿态误差同时包含
roll、pitch 和 yaw；两种 Critic 都可再使用 60-D 模拟器特权状态；Actor、
评估 Actor 和 ONNX 不使用特权字段。

`squashed_actor_critic.py` 用 PyTorch 原生 `TanhTransform(cache_size=1)` 和
`TransformedDistribution` 将 RSL-RL 3.1.2 的 latent Gaussian 变成有界电机指令：
`z ~ Normal(μ, σ)`、`a = tanh(z) ∈ (-1, 1)`。rollout 同时保存实际执行的 `a` 和精确的
pre-tanh `z`；PPO log-prob 由 transformed distribution 计算，包含 tanh Jacobian。确定性训练、
评估和 checkpoint 导出均使用 `tanh(μ)`。训练 wrapper、环境和评估器不再二次裁剪动作。
checkpoint 必须包含 `action_squash_version`，评估加载器通过严格 state-dict 加载执行这一唯一格式。

`smooth_ppo.py` 保留当前 RSL-RL PPO 的 clipped surrogate 和 clipped value loss，并在 Actor
loss 中增加确定性有界电机指令曲率：

`L_curvature = 2.5 · mean(||a_t - 2a_{t-1} + a_{t-2}||₂ / √8)`，其中
`a_t = tanh(μ_t)`。

T1–T4 四个垂向推进器通道另加：

`L_vertical = 0.5 · mean(||Δ²a_t[T1:T4]||₂ / √4)`。

该项直接对四个通道各自收费，不做前后带符号求和，因此共同升沉和差动姿态振荡都不会抵消。

完整形式的分母是 `dt² · 625 action/s²`；当前 `dt=0.04 s`，所以该分母恰为 1。rollout 的
扁平索引用于取回 `t-1/t-2` 观测，不复制 198-D 观测缓存；只使用 `t≥2` 且两处分界均未 done
的同 episode 三连帧。三个 Actor 前向都参与反向传播，存储的旧策略均值不用于该 loss。
该公式对应 Grad-CAPS 的未位移归一化二阶差分核心；这里使用固定物理尺度，不包含完整
Grad-CAPS 中按总动作位移再乘 `tanh` 的归一化，因此不把当前实现标作完整 Grad-CAPS 复现。

Actor 和 Critic 的隐藏线性层使用 gain `√2` 的正交权重初始化，Actor 输出层 gain 为 `0.01`，
Critic 输出层 gain 为 `1.0`，所有线性层 bias 初始化为零但保持可训练。该初始化只决定从头训练
的起点，不改写 checkpoint；latent Gaussian 的状态无关 `log_std` 仍由
`init_noise_std=0.5` 单独初始化。

PPO 使用 `schedule="fixed"` 和一个包含两个参数组的 Adam：Actor（含 `log_std`）固定学习率为
`3e-5`，Critic 固定学习率为 `3e-4`，两组梯度分别裁剪到 `1.0`。rollout 保存旧策略的 latent
Gaussian 均值与标准差；每个 minibatch 在反向传播前用 PyTorch distribution API 计算其解析
`KL(old || new)`，对八个动作维求和并对样本取平均。相同可逆 tanh 变换不改变两分布的 KL，
所以该值也正是 squashed policy 的 KL。KL 超过 `1.5 × desired_kl = 0.015` 时，只停止本轮剩余 Actor 更新，不回滚已经完成的
前一步；Critic 仍完成全部 `5 × 32 = 160` 个 minibatch 更新。因此 PPO surrogate、Actor 曲率项
和标准差参数共同造成的策略位移都受同一个 KL 停止条件约束，Critic 不受 Actor early stopping 影响，
也不会再由低 KL 触发学习率连续放大。latent policy 使用 log-standard-deviation 参数化，初始标准差为
`0.5`。训练、评估和导出都按同一个 network profile 解析输入维度，不能在 notebook 中分别设置
历史长度或隐藏层。

TensorBoard 中主要查看 `Loss/value_function`、`Loss/surrogate`、`Loss/entropy`、
`Loss/action_curvature`、`Loss/vertical_action_curvature`、`Loss/kl`、`Loss/kl_max`、`Loss/actor_update_fraction`、
`Loss/critic_update_fraction`、`Loss/learning_rate` 和
`Policy/mean_noise_std`（即 latent Gaussian 的平均 `σ`）。两个曲率指标分别是已乘 `2.5` 和 `0.5` 的 Actor loss 贡献；发生 KL early
stopping 时，`Loss/kl_max` 会超过 `0.015`，`Loss/actor_update_fraction` 会小于 `1.0`，而
`Loss/critic_update_fraction` 应保持 `1.0`。`Loss/learning_rate` 记录 Actor 学习率 `3e-5`。

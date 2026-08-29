# 最终 V16 Policy 训练与评估记录

本目录是 `t60-trajectory-precision-v16` 的最终生产归档。训练开始于 2026-08-29 19:02:41
（Asia/Shanghai），`model_499.pt` 与 TensorBoard 事件结束于 21:02:33，ONNX 于 21:07:36
导出，最终名义评估于 21:13:53 完成。

## 归档内容和完整性

- 最终 checkpoint：`model_499.pt`，6,833,450 字节，训练状态中的 `iter=499`。
- 部署导出：`exports/auv_traj_policy_mlp_history_8_2026-08-29_model_499.onnx`，输入
  `obs[1,201]`、输出 `actions[1,8]`，ONNX IR 10、opset 18。
- 不可变输入：`params/inputs/{training_recipe,environment,domain_randomization}.json`。
- 训练曲线：一个 TensorBoard event 文件，覆盖全部 500 次迭代。
- 最终评估：summary、domain samples、逐步 `logs.csv` 和两张图。这里保留逐步日志，因为它是
  最终 V16 的唯一完整评估；“旧 IsaacLab 评估不保存 logs.csv”的过滤规则只作用于旧归档。
- `git/isaac-auv-env.diff` 是训练启动器留下的原始差异证据，但 `git apply --check` 报
  `corrupt patch at line 302`。它在一个 hunk 中间结束，不能用于重建源码。
- `SHA256SUMS` 给出所有上述原始载荷的哈希。

原始载荷（不含本文、哈希清单和损坏差异快照）共 14,593,481 字节。中间
`model_0.pt` 到 `model_450.pt` 没有进入 Git。

## 源代码沿革

训练启动时仓库 HEAD 是 `f93e51fef4e464cf60e4844fe82b92cf5957ddac`（`v6`），工作树有
未提交修改。训练结束后这些训练相关修改随 V7 提交为
`9e86ab0cc9a613783bb88c3fb78ac56d85239f6d`，提交时间 21:36:22。由于启动时保存的 diff
被截断，不能声称训练工作树的每个字节都能从 Git 精确复原；可复现的权威输入是本目录的三个
run-local JSON，训练实现以 V7 代码为最接近的完整版本。

## 策略和观测

- 并行环境数 2,048，随机种子 42，策略频率 25 Hz，物理仿真频率 100 Hz。
- 每次迭代每环境 128 步，共 500 次迭代；采样规模为
  `500 × 128 × 2048 = 131,072,000` 个环境转移。
- Actor 输入维数为 `33 + 8 × 21 = 201`。基础观测 33 维；8 个历史帧各含艇体坐标位置误差
  3、速度误差 3、姿态误差四元数 4、角速度 3、电机命令 8，共 21 维。
- 八个历史采样覆盖 320 ms，包含约 50 ms 传感延迟和超过三个 80 ms 推进器时间常数。
- Actor 为 `201 → 512 → 256 → 128 → 8`，ELU，输出潜在对角高斯分布并用 `tanh` 映射到
  八个有界电机命令。Critic 使用同样宽度，并额外读取仿真器特权状态；部署 ONNX 只含 Actor。
- Actor/critic 学习率分别为 `3e-5` / `3e-4`；PPO clip `0.2`，每轮 5 epochs、32
  minibatches，`gamma=0.994009`，`lambda=0.9604`，目标 KL `0.01`，超过 `0.015` 后停止
  当轮后续 Actor 更新，梯度范数上限 1.0。

PPO 使用

```text
delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
A_t = sum_l (gamma lambda)^l delta_{t+l}
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))
L_clip = E[min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)]
```

训练代码最小化其负号，并加 value loss。平滑项对确定性有界动作均值
`a_t=tanh(mu_theta(s_t))` 使用二阶差分：

```text
kappa_t = (a_t - 2 a_{t-1} + a_{t-2}) / (Delta t^2 a_scale)
L_smooth = 2.5 * RMS_8(kappa_t) + 0.5 * RMS_T1:T4(kappa_t)
```

其中 `Delta t=0.04 s`，`a_scale=4/tau_thruster^2`，只对没有跨 episode 边界的三连帧计算。

## 课程和随机化

训练 recipe 使用最终 OpenFOAM 全矩阵系数和 DR v9。轨迹课程在 global step
0 / 6,400 / 19,200 / 38,400 进入四个阶段，最终请求速度上限 0.5 m/s；DR 在
0 / 12,800 / 25,600 / 44,800 逐级增强。episode 为 30 s，参考启动段为 4 s。完整轨迹族、
初始状态范围、奖励和随机化参数以 `params/inputs/*.json` 为准。

TensorBoard 最终批次显示的随机化抽样包括：水流模长均值 0.0285 m/s、最大 0.0608 m/s，
线性/二次阻尼相对标准差约 0.0986 / 0.1525，附加质量相对标准差约 0.0998，单推进器推力
尺度标准差约 0.0497，共同衰减尺度均值 0.926、最小 0.852。

## 训练结果

| 指标 | 起始 | 最终/最好 |
|---|---:|---:|
| mean reward | 36.56 | 727.95（最大 728.24） |
| running reward | — | 0.9707（最大 0.9755） |
| position RMSE | 1.090 m | 0.02871 m（最小 0.02168 m） |
| velocity RMSE | — | 0.02353 m/s（最小 0.02027 m/s） |
| attitude error mean | — | 0.612°（最小 0.566°） |
| action saturation fraction | — | 0.00159 |
| value loss | 34.74 | 0.03081 |
| Gaussian noise std | 0.4997 | 0.1304 |
| mean KL | — | 0.00520；该轮最大 0.01178 |
| simulation throughput | 24,870 FPS | 17,841 FPS |

训练 position RMSE 是并行课程分布上的即时统计，不应与下面单轨迹、确定性评估直接等同。

## 最终评估结果

唯一归档评估为 seed 42、无水流和无参数扰动、1 条 5 m × 3 m 平面 Lissajous 轨迹、目标
`vmax=0.25 m/s`、时长 60 s。参考曲线有效且位于运动学包络内。

| 指标 | 全时段 | 4.29 s 后稳态 |
|---|---:|---:|
| position RMSE | 0.010621 m | 0.010826 m |
| position error p95 | 0.020372 m | 0.020422 m |
| maximum position error | 0.026920 m | 0.026920 m |
| cross-track RMSE | 0.009506 m | 0.009685 m |
| velocity RMSE | 0.006469 m/s | 0.006362 m/s |
| attitude RMSE | 0.5619° | 0.5633° |
| mean reward / step | 0.987785 | 0.987618 |

动作 RMS 为 0.19898，动作饱和率为 0；平均绝对实际推力 0.6967 N，最大 5.0914 N；最小
边界间隙 0.5013 m，无边界违规、终止或失败，survival rate 为 1.0。完整字段和值位于
`summary_metrics.csv`。

该结果只证明一个 seed 下的一条名义轨迹，不证明跨 seed、随机水流、推力退化或参数失配下的
鲁棒性；旧评估目录中的其他实验也不能替代针对 V16 的扰动评估。

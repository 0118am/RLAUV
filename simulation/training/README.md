# Training

训练树只保留三个需要分组的功能目录：

| 路径 | 职责 |
| --- | --- |
| `recipes/` | 版本化训练 recipe 与跨域 DR 输入 |
| `ppo/` | 网络 profile、有界动作分布、PPO 配置、Actor 曲率 loss 与 runner |
| `evaluation/` | 评估配置、执行、指标、调度和 ONNX 导出 |

根目录中的单文件模块各自对应一个完整职责：

| 文件 | 职责 |
| --- | --- |
| `recipe.py` | 完整训练输入与 run-local 输入路径 |
| `campaign.py` | 前台命令执行、run/checkpoint 查询 |
| `train.py` | IsaacLab 训练入口及 worker 生命周期 |
| `config.py` | policy-facing Direct Task 配置 |
| `observations.py` | Actor 历史观测和 Critic 特权观测 |
| `rewards.py` | 不可变奖励 profile、选择和唯一张量实现 |
| `trajectory.py` | 训练轨迹、课程和 reference runtime |
| `visualization.py` | 环境调试显示 |

默认奖励契约 `precision_v9` 和各分量的统一张量实现都在 `rewards.py`。姿态 reward 以
`roll=pitch=0` 和轨迹 yaw 分别计算三轴误差，三轴等权汇总；恢复项使用转折点
`π/18 rad`、在 `π/3 rad` 过零的 Huber，精度 Cauchy 的每轴半宽为 `2.5°`；
角速度也拆为宽范围与精度 Cauchy 项；reward 对八路有界电机指令的平方和收费，并按真实
policy 周期计算、以 `25 action/s` 归一化的 `du/dt` 平方均值收费；两项均不对 T60 死区豁免。
位置精度和明确上下界不会由 notebook 或 Hydra 参数另行覆盖。
Actor 使用 `z ~ Normal(μ, σ)`、`a = tanh(z)` 的有界动作分布，训练 rollout 保存执行动作和
精确 pre-tanh 样本，并由 PyTorch `TransformedDistribution` 计算 Jacobian 修正后的 log-prob。
二阶变化在 `ppo/smooth_ppo.py` 中直接约束确定性有界指令 `a = tanh(μ)`：同一 episode 的三连帧先以
`625 action/s²` 归一化，再对八台推进器取 RMS，loss 系数为 `2.5`；同时以系数 `0.5`
对 T1–T4 四个垂向推进器通道的二阶变化 RMS 额外收费。TensorBoard 中用
`Loss/action_curvature` 和 `Loss/vertical_action_curvature` 查看两个已加权优化项，用
`action_acceleration_rms_per_s2` 查看电机指令的物理加速度诊断；后者不参与
reward。PPO 使用同一个 Adam 的两个参数组：
Actor 固定学习率为 `3e-5`，Critic 为 `3e-4`。每个 minibatch 更新前计算新旧 latent Gaussian 的
解析 KL；固定可逆 tanh 不改变 KL。当 `KL(old || new) > 0.015` 时只停止本轮剩余 Actor 更新，Critic 仍完成全部更新。
`Loss/kl_max`、`Loss/actor_update_fraction` 和 `Loss/critic_update_fraction` 分别显示最大 KL、
Actor 实际更新比例和 Critic 实际更新比例。

`t60_trajectory_precision_v16.json` 明确选择 `tanh_gaussian_v1` 动作分布，使用最近一次完整
6×6 CFD 响应矩阵，并从头训练 500
个迭代。所有轨迹目标都严格
保持 `roll=pitch=0`；闭合/前进曲线只让 yaw 跟随水平速度方向，升沉由 heave 推进器完成。
三轴纯正弦使用明确的
峰值速度—幅值组合训练加减速、停止与反向，并在课程中覆盖 `0.1/0.2/0.3/0.4/0.5 m/s`；
横向和垂向前进正弦使用波数、纵向尺度、横向尺度及路径速度的显式可实现组合训练曲率。
每个阶段只声明新命令，runtime 累积并均衡采样此前全部命令，不再生成速度与曲率的笛卡尔积。
Lissajous 和正反空间 Helix 只用于未见几何评估。对应生成器、Actor 观测和评估日志契约分别为
`curve_v5`、`t60_trajectory_obs_v11` 和 schema v9。

`auv_open_water_openfoam_hydrodynamics_dr_v9.json` 在每次环境 reset 时分别抽取线性阻尼、
二次阻尼和流体附加质量的均值为 1 的对数正态倍率。阻尼使用整矩阵标量倍率以保留完整 6×6
矩阵内部的耦合比例和零元素；线性与二次阻尼不再共享随机倍率。流体附加质量使用六自由度正
倍率的 `S M_A S` 合同变换，保留对称性与正定性。TensorBoard 中搜索
`domain_randomization/` 可直接查看三类倍率和最终矩阵元素的 mean/std/min/max。

推进器 DR 分成两个互不混淆的量：八台推进器各自的对称增益失配，以及每个环境只有一个、
由八台推进器共同使用的弱化系数。后者只在 `[1-reduction, 1]` 内采样，最后阶段为
`[0.85, 1.0]`，不会产生全体推进器共同增强，也不引入额外的能源状态模型。

刚体质量、排水体积、完整惯量、COM 和 CoB 使用固定实测值，不进入 DR。流体附加质量
`M_A` 独立表示周围流体的惯性效应；其四阶段对数标准差为 `0/0.025/0.05/0.10`，不会改写
刚体参数。

`train.py` 是 IsaacLab worker，人工入口为仓库根目录 `train.ipynb`。worker 的生命周期由
notebook 最后一个单元格直接管理，中断单元格即终止训练，不创建 launcher PID/日志文件。
训练模块不保存机器人或水动力参数，也不直接写 PhysX；最终接线只在
`simulation/assembly.py`。

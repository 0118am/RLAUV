# T60 AUV 水动力、轨迹控制与强化学习

本仓库面向 T60 AUV，统一管理机器人本体、实测推进器曲线、水池水动力、Isaac Lab
强化学习训练和策略评估。Gym 任务为 `Isaac-AUV-Traj-Direct-v1`。

## 代码边界

| 目录 | 唯一职责 |
| --- | --- |
| `robot/` | 机器人质量、惯量、浮心、推进器、执行器、系缆、PID 和可部署轨迹几何 |
| `environment/` | 水流、完整水动力矩阵、池壁/自由液面、确定性环境 profile 和逐步流体力计算 |
| `simulation/` | environment+robot 组合、跨域 DR、PhysX 动力学桥和 Isaac 场景组装 |
| `simulation/training/` | 仅训练相关：网络、PPO、奖励、观测、训练轨迹接入、评估、可视化、导出和进程管理 |
| `common/` | 三域共同使用的四元数、采样和数据 profile 基础工具 |
| `train.ipynb` / `eval.ipynb` | 人工选择一次运行的参数并调用上述 API |

`simulation/` 不再设置 `isaac/` 中间层：根目录只保留跨域组合、DR、动力学桥和
`assembly.py`，训练实现全部位于唯一的 `training/` 树。物理参数不在模拟目录复制，
训练目录也不直接向 PhysX 写力。

```text
robot/ 机器人事实 ───────────┐
                             ├─ simulation/composition.py
environment/ 环境事实 ──────┘                 │
                                               ▼
                         simulation/assembly.py
                 ┌───────────────┼──────────────────┐
                 ▼               ▼                  ▼
       environment/runtime   robot 力模型   training 观测/奖励/轨迹
                 └───────────────┼──────────────────┘
                                 ▼
                         单次 PhysX wrench 提交
```

## 目录结构

```text
environment/
├── hydrodynamics/             水流、阻尼、附加质量、池壁和自由液面方程
├── profile.py                 仅确定性水体/池体 profile
├── randomization/             水流与水动力随机化
├── runtime/                   有效系数、相对状态和流体 wrench 计算
├── openfoam/                  OpenFOAM 工程、工具和结果
└── pmm/                       PMM 历史数据与旧辨识报告（不作为生产矩阵来源）

robot/
├── assets/isaac/              T60 的 Isaac USD 表示
├── dynamics/                  刚体参数、变换和系缆
├── propulsion/                T1–T8 实测 FLU 曲线、执行器动态和安装点合成
├── control/                   PID、分配器和可部署轨迹生成
├── randomization/             刚体与推进器随机化
├── runtime.py                 执行器、传感器和系缆名义运行参数
└── runtime_state.py           刚体、执行器和系缆的显式逐环境状态

simulation/
├── assets.py                  robot 数据到 Isaac asset 配置的适配
├── composition.py             组合 environment、robot 和跨域 DR
├── domain_randomization.py    跨域 DR schema、选择和 JSON I/O
├── dynamics.py                总惯量模型到 PhysX wrench 的纯张量桥
├── assembly.py                唯一 Isaac/PhysX 场景组装文件
├── training/
│   ├── recipes/               版本化训练 recipe JSON
│   ├── ppo/                   网络 profile、PPO 配置和 rollout 级 KL 控制
│   ├── evaluation/            评估配置、运行、指标、调度与导出
│   ├── recipe.py              recipe、训练/评估请求和 run-local 输入
│   ├── campaign.py            前台命令执行和 run/checkpoint 管理
│   ├── config.py              Direct Task 配置
│   ├── observations.py        Actor 与 Critic 观测
│   ├── rewards.py             奖励 profile 与唯一张量实现
│   ├── trajectory.py          训练轨迹、课程和 reference runtime
│   ├── visualization.py       环境调试显示
│   └── train.py               Isaac Sim 训练入口与 worker
└── rlpolicy/                  仅本地 checkpoint、日志、评估和导出产物
```

## 物理基准

长期有效的数据只从下列位置读取：

- T60 刚体参数：`robot/dynamics/parameters.py`
- T1–T8 安装位置/轴向：`robot/dynamics/parameters.py`
- T1–T8 实测三分量推力曲线：`robot/propulsion/curves.py`
- 执行器响应、传感器和系缆：`robot/runtime.py`
- 最近一次完整开水域 CFD 水动力矩阵：
  `environment/hydrodynamics/coefficients/auv_open_water_openfoam_full_hydrodynamics_v2.json`
- DR 基准：`simulation/training/recipes/auv_open_water_openfoam_hydrodynamics_dr_v8.json`

名义质量为 `11.301 kg`。实测浮力比重力多 `0.24 kg` 等效质量，因此在
`ρ=1000 kg/m³` 时排水体积为 `0.011541 m³`，净上浮力约 `2.3544 N`。

局部坐标使用 FLU：`+X` 前、`+Y` 左、`+Z` 上。当前 open-water 训练/评估安全盒为：

```text
x ∈ [-0.75, 5.75] m, y ∈ [-0.75, 3.75] m, z ∈ [-0.60, 1.60] m
```

中心为 `(2.5, 1.5, 0.5) m`。安全盒给 `5 × 3 × 1 m` 曲线、艇体包围盒和 `0.20 m`
初始位置扰动留出独立余量；它不是未经测量的池壁/自由液面水动力模型。

## 推力与水动力链路

每个 `100 Hz` 物理步执行一次：

1. 策略输出限制到 `[-1, 1]`，再按 `PWM_model = 1500 + 200·command` 映射为物理
   `1300–1700 µs`；
2. 施加可选量化和丢指令，再将归一化指令饱和到 `[-1,1]`，不添加命令传输延迟；
3. 用正/反 PWM 分支计算每台推进器实测 FLU `(Fx,Fy,Fz)`；
4. 对三分量目标力施加名义 `0.08 s` 一阶响应；
5. 在八个安装点计算 `ΣFᵢ` 和 `Σ(rᵢ×Fᵢ)`；
6. 叠加浮力、流体力和可选系缆力；
7. `assembly.py` 只向 PhysX 提交一次机体系合力与合矩。

`1475–1525 µs` 是闭区间零推力死区。正 PWM 时 T5/T6 正转，但安装朝 `F−`，所以
`Fx<0`；T7/T8 朝 `F+`，所以 `Fx>0`。三个分量的符号均来自实测曲线，不存在第二套
polarity、spin direction、固定轴重建、反扭矩或高阶残差逻辑。

水动力使用机体系相对速度。非惯性流体载荷与总惯性方程为：

```text
νr = [v_b - Rᵀv_current_w, ω_b]
τfluid = τbuoyancy - DLνr - DQ(|νr|⊙νr) - CA(νr;MA)νr
(MRB + MA)ν̇ = τexternal + τgravity - CRB(ν)ν + MAν̇current
```

`MA`、`DL`、`DQ` 接受完整 `6×6` 矩阵并保留允许的非对角耦合。代码先解
`MRB+MA` 的六自由度总惯性，再映射为 PhysX 应接收的等效 wrench；不再用艇体速度有限差分
反馈 `-MAν̇r`，也没有旧的 0.35 滤波。只有规定的背景流加速度需要显式求导。PhysX 自带
线性和角阻尼为零，避免重复阻力。当前 recipe 确定性使用最近一次已完成的 24 工况 CFD
全部六分量响应：左右镜像对称禁止的耦合为零，允许的非对角项保留，附加质量按互易性取
对称平均。`enabled_features` 未列出 `hydrodynamics` 只关闭其 DR，并不关闭名义水动力。
该矩阵尚未通过多轴实物辨识；原始完整线性阻尼的对称部分最小特征值为 `-0.1924`，因此
不能宣称在任意组合速度下全局被动。

## 频率与观测

- PhysX：`100 Hz` (`dt=0.01 s`)
- Policy、融合状态观测和 8 路动作：`25 Hz` (`decimation=4`)
- 真机 PWM 脉冲帧可继续保持 `50 Hz`，每个 policy 指令保持两个 PWM 周期
- 融合状态延迟：`0.05 s`，由 100 Hz 真值环形缓冲精确实现为 5 个物理步
- 推力建立：`0.08 s` 一阶时间常数；实测 PWM `1475–1525 µs` 为零推力死区

Actor 当前样本为 33 维：位置误差 3、目标线速度 3、线速度误差 3、姿态误差四元数 4、
机体系重力方向 3、角速度 3、目标角速度 3、目标线加速度 3、实际应用动作 8。机体系重力
方向由同一份延迟、无噪声的姿态测量计算，不向 Actor 泄漏模拟器未来状态。

网络只在 `training/ppo/networks.py` 定义：

| 网络 | Actor 输入 | Actor/Critic 隐藏层 |
| --- | ---: | --- |
| `mlp_33d` | 当前 33 维 | `512,256,128` |
| `mlp_history_8` | 当前 33 维 + 8 个历史样本，共 201 维 | `512,384,256,128` |

8 个先前样本覆盖 320 ms，超过 50 ms 状态延迟加三个 80 ms 推进器时间常数。每个样本包含
位置误差、线速度误差、姿态误差、角速度和实际应用动作。Actor 的艇体状态全部直接取自
50 ms 延迟环形缓冲，不再按 40 ms 观测周期重抽位置或姿态白噪声。Critic 额外接收 74 维
模拟器特权状态，并保留当前真值；奖励也始终使用
当前真值。Actor 和导出的 ONNX 不使用特权字段。

## 训练轨迹与课程

训练分成两个相互独立的技能族：

- `surge_sine / sway_sine / heave_sine`：三轴直线往复，固定目标姿态，显式训练加速、减速、停止和反向；
- `lateral_wave / vertical_wave`：沿 `X` 方向前进的有界正弦，目标始终保持 `roll=pitch=0`，仅让 yaw 跟随水平速度方向，训练不同曲率。

前进正弦每个半周期从水池一端走到另一端。`n=1/2/3` 分别完成一、二、三次正弦；25% 横向
余量构成回程通道，使出程和回程分离，同时避免端点零切向和垂直航向奇点。基础幅值独立采样
`x=2.25–2.50 m`、`y=1.35–1.50 m`、`z=0.40–0.50 m`，横向幅值另取 `0.5/1.0` 档。

Recipe 直接列出可实现的命令，不再构造“类型 × 速度 × 统一尺度”的笛卡尔积。后续阶段累积保留
早期命令：

| 阶段起点（policy step） | 累积命令数 | 新增训练重点 |
| ---: | ---: | --- |
| 0 | 3 | 三轴 `0.10 m/s` 纯正弦 |
| 6,400 | 13 | 更短幅值、`0.20 m/s` 纯正弦及 `n=1` 前进正弦 |
| 19,200 | 28 | `0.30 m/s` 三轴与 Surge/Sway `0.40 m/s` 纯正弦、较快 `n=1` 与中曲率 `n=2` |
| 38,400 | 37 | Surge/Sway `0.50 m/s` 纯正弦及满足水平航向角速度上限的 `n=2/3` 高曲率命令 |

纯轴正弦明确覆盖 `0.10/0.20/0.30/0.40/0.50 m/s` 五档；Heave 因较短垂向幅值只到
`0.30 m/s`，`0.40/0.50 m/s` 由 Surge/Sway 覆盖。重定时器使用解析三阶导数、区间探测
和 Simpson 时间积分，逐项满足 `0.50 m/s`、
`0.45 m/s²`、`0.36 m/s³` 和 `0.80 rad/s` yaw 上限；速度和曲率在 recipe 中预先配对，
高曲率区间仍会由重定时器主动降速，基础幅值端点上的最低局部速度保持率约为 `68.4%`，
不能把请求速度理解为全周期恒定实速。Lissajous 与正反空间 Helix 不参与训练，
仅作为多轴组合泛化评估。DR 扰动阶段位于
`12,800 / 25,600 / 44,800`，与轨迹难度切换交错，避免同时提高运动和模型不确定性。

训练和评估都启用上述六面安全盒；越界是安全终止并扣 1。评估还按艇体当前姿态计算包围盒
到边界的最小净空，自动重置后的状态不会混入终止前轨迹日志。

评估还可选择 `wavy_loop`、`breathing_loop`、chirp、G²
racetrack 和 random smooth，专门测试未作为训练课程主体的几何泛化。

## 奖励与 PPO

默认奖励 `precision_v6` 位于 `training/rewards.py`。位置项是权重 `0.35`、半奖励宽度
`0.10 m` 的 Cauchy；姿态误差使用完整四元数，因此同时约束 roll、pitch 和 yaw，计算全程
使用 rad，由权重 `0.25`、转折点 `π/18 rad`、在
`π/3 rad` 过零的带符号 Huber 恢复项和权重 `0.25`、半奖励宽度 `π/180 rad` 的 Cauchy
精度项组成。线速度权重为 `0.10`
（`0.08 m/s`），角速度
同时使用权重 `0.03` 的 `0.30 rad/s` 宽项与权重 `0.02` 的 `0.15 rad/s` 精度项。垂向运动
由 heave 推进器完成，不通过动态 pitch 指令倾斜艇体。控制代价只计算 T60
死区之外的 processed command；变化率 `du/dt` 以 `25 action/s` 归一化，加速度
`d²u/dt²` 以 `625 action/s²` 归一化，权重分别为 `0.010` 和 `1.200`。这两个尺度由
`0.08 s` 推进器时间常数定义，不随 policy 步长改变。正常单步奖励位于约
`[-2.18745, 1.0]`，安全终止另减 1；TensorBoard 中
`tracking/*` 与 `reward/*` 分别显示误差、Huber/Cauchy 姿态分量、宽/精度角速度分量及两个
动作平滑分量。

PPO 当前默认值：

| 参数 | 值 |
| --- | ---: |
| rollout | 128 steps/env = 5.12 s |
| notebook 环境数 | 2048 |
| 最大迭代 | 500 |
| 学习轮数 / mini-batches | 5 / 32 |
| learning rate | `3e-4` 初值，由 RSL-RL 原生 adaptive KL 调整 |
| clip / value loss coef | `0.2 / 1.0` |
| gamma / lambda | `0.994009 / 0.9604` |
| desired KL | `0.01` |
| entropy / max grad norm | `0.0 / 1.0` |
| 初始动作标准差 | `0.5` |

PPO 使用 RSL-RL 原生 adaptive KL 调整共享 Adam 学习率，不保存 Actor 快照，不回滚参数，
也不重试 minibatch。

## 训练与评估

Python 运行时直接依赖 Pydantic 2（版本化 JSON）和 `psutil`（训练进程生命周期）；ONNX
导出使用 PyTorch 2 的 dynamo exporter，并需要 `onnx`/`onnxscript`。这些依赖必须存在，代码
不保留手写解析、`/proc` 扫描或旧 exporter 兜底。

从仓库根目录启动 Jupyter：

```bash
conda activate env_isaaclab
jupyter lab train.ipynb
```

`train.ipynb` 只选择版本化 recipe、运行名、seed、环境数和启动开关，并在最后一个单元格中
直接管理训练 worker；停止或中断该单元格会同时终止训练。worker 创建 run 后，把解析完成的
recipe 和两个 profile 固化在该 run 内：

```text
simulation/rlpolicy/<architecture>/<timestamp>_<RUN_NAME>/params/
└── inputs/
    ├── training_recipe.json
    ├── environment.json
    └── domain_randomization.json
```

训练 worker 是 `simulation/training/train.py`。评估入口是 `eval.ipynb`，它按 checkpoint 生成时间自动
选择最新 policy，并直接读取 checkpoint 同目录下的 run-local recipe 和两个 profile，结果写入该运行的
`evaluation/`。ONNX 导出使用同一份 run-local recipe：

```bash
python simulation/training/evaluation/export.py \
  --checkpoint simulation/rlpolicy/auv_traj_mlp_history_8/<run>/model_450.pt
```

本次迁移和文档整理不会自动启动训练。

## 测试

```bash
conda run -n env_isaaclab python -m pytest -q tests
```

完整训练前还应在 Isaac Lab 中运行短迭代 smoke test。`run_campaign.sh` 只生成不可变 CFD
拟合；三张生产矩阵只有在拟合与 48 份数值验证报告都通过后才由
`environment/openfoam/publish_results.py` 更新。工程说明见
`environment/openfoam/README.md`；`environment/pmm/` 仅保留模型试验历史档案。

# T60 AUV 水动力、轨迹控制与强化学习

本仓库面向 T60 AUV，统一管理机器人本体、实测推进器曲线、水池水动力、Isaac Lab
强化学习训练和策略评估。Gym 任务为 `Isaac-AUV-Traj-Direct-v1`。

## 代码边界

| 目录 | 唯一职责 |
| --- | --- |
| `robot/` | 机器人质量、惯量、浮心、推进器、执行器、电池、系缆、PID 和可部署轨迹几何 |
| `environment/` | 水流、完整水动力矩阵、池壁/自由液面、环境 profile、DR 和逐步流体力计算 |
| `simulation/assembly.py` | 唯一 Isaac/PhysX 组装层：场景生命周期、状态接线、最终 wrench 提交和 Gym 注册 |
| `simulation/training/` | 仅训练相关：网络、PPO、奖励、观测、训练轨迹接入、评估、可视化、导出和进程管理 |
| `train.ipynb` / `eval.ipynb` | 人工选择一次运行的参数并调用上述 API |

`simulation/` 不再设置 `isaac/` 中间层：根目录只有包入口和 `assembly.py`，训练实现全部
位于唯一的 `training/` 树。物理参数不在模拟目录复制，训练目录也不直接向 PhysX 写力。

```text
robot/ 机器人事实 ───────────┐
                             ├─ environment/profiles/composition.py
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
├── profiles/                  环境/DR 配方、运行组合和评估覆盖
├── randomization/             水流与水动力随机化
├── runtime/                   有效系数、相对状态和流体 wrench 计算
├── openfoam/                  OpenFOAM 工程、工具和结果
└── pmm/                       PMM 数据与六自由度辨识

robot/
├── assets/isaac/              T60 USD 与 Isaac 资产生成配置
├── dynamics/                  刚体参数、变换和系缆
├── propulsion/                T1–T8 实测 FLU 曲线、执行器动态和安装点合成
├── control/                   PID、分配器和可部署轨迹生成
├── randomization/             刚体、推进器和电池随机化
├── runtime.py                 执行器、电池、系缆名义运行参数
└── runtime_state.py           刚体、执行器、电池、系缆的显式逐环境状态

simulation/
├── assembly.py                唯一 Isaac/PhysX 组装文件
├── training/
│   ├── recipes/               版本化训练 recipe JSON
│   ├── ppo/                   网络、算法、runner 和配置
│   ├── evaluation/            评估配置、运行、指标、调度与导出
│   ├── recipe.py              recipe、训练/评估请求和 run-local 输入
│   ├── manifest.py            run 输出契约
│   ├── campaign.py            命令、进程和 run/checkpoint 管理
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
- T1–T8 安装位置和实测三分量推力曲线：`robot/propulsion/curves.py`
- 执行器响应、延迟、电池和系缆：`robot/runtime.py`
- 名义水池和水动力矩阵：
  `environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json`
- DR 基准：`environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json`

名义质量为 `11.301 kg`。实测浮力比重力多 `0.24 kg` 等效质量，因此在
`ρ=1000 kg/m³` 时排水体积为 `0.011541 m³`，净上浮力约 `2.3544 N`。

水池局部坐标使用 FLU：`+X` 前、`+Y` 左、`+Z` 上；下角为原点。真实边界严格为：

```text
x ∈ [0, 5.0] m, y ∈ [0, 3.5] m, z ∈ [0, 0.75] m
```

池底 `z=0`，水面 `z=0.75 m`，名义中心为 `(2.5, 1.75, 0.375) m`。水流场、池壁、
自由液面和晃荡共享这套范围。

## 推力与水动力链路

每个 `100 Hz` 物理步执行一次：

1. 策略输出限制到 `[-1, 1]`，再映射为物理 `1300–1700 µs` PWM；
2. 显式施加 `0.13 s` 命令延迟、可选限速/量化/丢指令；
3. 用正/反 PWM 分支计算每台推进器实测 FLU `(Fx,Fy,Fz)`；
4. 对三分量目标力施加名义 `0.08 s` 一阶响应；
5. 在八个安装点计算 `ΣFᵢ` 和 `Σ(rᵢ×Fᵢ)`；
6. 叠加浮力、流体力和可选系缆力；
7. `assembly.py` 只向 PhysX 提交一次机体系合力与合矩。

`1475–1525 µs` 是闭区间零推力死区。正 PWM 时 T5/T6 正转，但安装朝 `F−`，所以
`Fx<0`；T7/T8 朝 `F+`，所以 `Fx>0`。三个分量的符号均来自实测曲线，不存在第二套
polarity、spin direction、固定轴重建、反扭矩或高阶残差逻辑。

水动力使用机体系相对速度：

```text
νr = [v_b - Rᵀv_current_w, ω_b]
τh = τbuoyancy - DLνr - DQ(|νr|⊙νr) - CA(νr)νr - MAν̇r
```

`MA`、`DL`、`DQ` 使用测得/拟合的完整 `6×6` 矩阵，保留非对角耦合项。PhysX 自带线性和
角阻尼设为零，避免重复阻力。`ν̇r` 在 `100 Hz` 上有限差分并使用现有 `0.35` 滤波。

## 频率与观测

- PhysX：`100 Hz` (`dt=0.01 s`)
- 控制：`50 Hz` (`decimation=2`)
- 融合状态观测：`50 Hz`
- `0.13 s` 执行器延迟：量化为 13 个物理步

Actor 当前样本为 30 维：位置误差 3、目标线速度 3、线速度误差 3、姿态误差四元数 4、
角速度 3、目标角速度 3、目标线加速度 3、实际应用动作 8。

网络只在 `training/ppo/networks.py` 定义：

| 网络 | Actor 输入 | Actor/Critic 隐藏层 |
| --- | ---: | --- |
| `mlp_30d` | 当前 30 维 | `512,256,128` |
| `mlp_history_5` | 当前 30 维 + 5 个历史样本，共 135 维 | `512,384,256,128` |

历史样本包含位置误差、线速度误差、姿态误差、角速度和实际应用动作。Critic 额外接收 76 维
模拟器特权状态；Actor 和导出的 ONNX 不使用这些字段。没有额外的模拟传感器延迟、滤波、
丢包或噪声链。

## 训练轨迹与课程

训练只使用三种互补命令：

- `lateral_sine`：沿 `Y` 轴往返，覆盖横移和正反航向；
- `vertical_sine`：沿 `Z` 轴往返，覆盖升沉和俯仰；
- `spatial_helix`：闭合三维双层螺旋，同时激励三轴平移和姿态变化。

每种类型与 `0.1/0.2/0.3/0.4 m/s` 四档速度均衡组合。四阶段课程在全局策略步
`9,750 / 22,500 / 40,500` 切换；幅值比例为 `0.55/0.75/0.90/1.0`，垂向比例为
`0.25/0.50/0.75/1.0`。轨迹经过 `curve_v2` 局部重定时，限制速度 `0.60 m/s`、加速度
`0.45 m/s²`、姿态角速度 `0.80 rad/s` 和 jerk `0.36 m/s³`。

训练不会因越过水池盒而终止，以免短暂误差切碎长 episode。验收测试使用
`keep_boundaries=True`，按真实水池六面边界判定越界；默认评估几何及最终验收采样范围均以
池中心为基准并限制在真实空间内。

评估还可选择 Lissajous、wavy loop (`helix`)、breathing loop (`spiral`)、chirp、G²
racetrack 和 random smooth，专门测试未作为训练课程主体的几何泛化。

## 奖励与 PPO

奖励 profile、选择逻辑和张量实现在 `training/rewards.py`，公共误差只计算一次。
`policy_0`–`policy_6` 的不可变权重和公式变体也保存在该文件。当前训练默认 `policy_6`：Huber 位置/姿态/
线速度/角速度跟踪，叠加 applied-action 能量与按实际变化率归一化的平滑惩罚，并对真实越界
终止扣分。

PPO 当前默认值：

| 参数 | 值 |
| --- | ---: |
| rollout | 256 steps/env = 5.12 s |
| notebook 环境数 | 1024 |
| 最大迭代 | 500 |
| 学习轮数 / mini-batches | 5 / 32 |
| learning rate | `3e-4`，按 rollout 在 `[1e-4,5e-4]` 内调整 |
| clip / value loss coef | `0.2 / 1.0` |
| gamma / lambda | `0.997 / 0.98` |
| desired KL / early stop | `0.01 / 0.015` |
| entropy / max grad norm | `0.0 / 1.0` |
| 初始动作标准差 | `0.5` |

学习率在一个 rollout 的所有更新中保持不变，rollout 结束后根据 KL 调整下一轮；KL 超过
`0.015` 时提前停止本轮剩余更新。

## 训练与评估

从仓库根目录启动 Jupyter：

```bash
conda activate env_isaaclab
jupyter lab train.ipynb
```

`train.ipynb` 只选择版本化 recipe、运行名、seed、环境数和启动开关。训练 worker 创建 run 后，
把解析完成的 recipe 和两个 profile 固化在该 run 内：

```text
simulation/rlpolicy/<architecture>/<timestamp>_<RUN_NAME>/params/
├── run_manifest.json
└── inputs/
    ├── training_recipe.json
    ├── environment.json
    └── domain_randomization.json
```

训练 worker 是 `simulation/training/train.py`。评估入口是 `eval.ipynb`，它要求显式设置
`POLICY_RUN_DIR`，从 `run_manifest.json` 恢复网络、奖励和 run-local profile，结果写入该运行的
`evaluation/`。ONNX 导出同样从 manifest 恢复网络契约：

```bash
python simulation/training/evaluation/export.py \
  --checkpoint simulation/rlpolicy/auv_traj_mlp_history_5/<run>/model_450.pt
```

本次迁移和文档整理不会自动启动训练。

## 测试

```bash
conda run -n env_isaaclab python -m pytest -q tests
```

完整训练前还应在 Isaac Lab 中运行短迭代 smoke test。OpenFOAM 工程说明见
`environment/openfoam/README.md`，PMM 六自由度辨识入口为
`environment/pmm/six_dof_identification.py`。

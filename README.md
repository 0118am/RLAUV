# T60 AUV 水动力、轨迹控制与强化学习

本仓库只面向 T60 AUV，统一管理机器人本体、水池与水动力模型、Isaac Lab
强化学习训练和策略评估。当前 Isaac Lab Direct Task 为
`Isaac-AUV-Traj-Direct-v1`。

项目采用明确的领域边界：机器人自身性质归 `robot/`，水和水动力物理归
`environment/`，`simulation/isaac/` 只把两者合成为 Isaac/PhysX 任务。训练人员只需在
仓库根目录的 [`train.ipynb`](train.ipynb) 中选择流速、Domain Randomization（DR）强度和
训练规模，不应在 Isaac 目录里维护第二套物理参数。

## 设计原则

| 领域 | 负责内容 | 不应放入的内容 |
| --- | --- | --- |
| `robot/` | T60 质量、惯量、浮力中心、推进器、执行器、电池、系缆、PID 和可部署轨迹几何 | 水流、水池边界、Isaac 生命周期 |
| `environment/` | 水流场、水动力方程、池壁/自由液面效应、OpenFOAM/PMM 数据和环境随机化 | 机器人控制器、Isaac 场景代码 |
| `simulation/isaac/` | Isaac 配置、场景生命周期、PhysX 适配、观测、奖励、PPO、训练与评估 worker | 机器人或水动力参数副本、模拟传感器滤波链 |
| `train.ipynb` | 单次训练的人工数值选择和启动动作 | 可复用函数、物理实现、进程实现 |

核心数据流如下：

```text
environment/ 中的水动力基准 ─┐
                              ├─ simulation/isaac/composition.py
robot/ 中的 T60 本体基准 ─────┘                │
                                              ▼
train.ipynb ──生成本次运行的环境/DR 快照──> Isaac 任务配置
                                              │
                         environment/ 力模型 ─┤
                         robot/ 推进器模型 ────┤
                                              ▼
                 physics_adapter.py + force_composition.py → PhysX
                                              │
                                              ▼
                               PPO checkpoint / 评估结果
```

`composition.py` 只解析、校验和组合外部数据源；`physics_adapter.py` 计算 Isaac 水动力状态，
`force_composition.py` 合成推进器、缆绳与流体 wrench。它们都不是物理参数的数据源。

## 目录结构

```text
train.ipynb                         训练的唯一人工数值配置与进程操作入口
eval.ipynb                          指定策略运行、评估和绘图入口

environment/                        水与水动力领域
├── hydrodynamics/
│   ├── coefficients/               经确认、版本化的水动力系数 JSON
│   ├── current_fields.py           常值、周期和空间水流场
│   ├── models.py                   阻尼、附加质量等水动力方程
│   └── pool_effects.py             池壁和自由液面效应
├── profiles/
│   ├── environment_profile.py      严格的环境配置边界
│   ├── domain_randomization.py     DR 配方结构、校验和序列化
│   └── configs/                    经确认的 DR 基准配方
├── randomization/                  水流与水动力随机化执行函数
├── openfoam/                       OpenFOAM v2512 几何、算例、工具和结果
└── pmm/                            PMM 原始数据与六自由度辨识

robot/                              T60 机器人领域
├── assets/isaac/t60_auv.usd        Isaac 资产
├── dynamics/                       本体参数、刚体变换和系缆模型
├── propulsion/                     T1–T8 实测三分量推力曲线与推力合成
├── control/
│   ├── pid.py                      六自由度 PID 控制器
│   └── trajectory/                 可部署的轨迹运动学、重定时与航向生成
├── randomization/                  刚体、推进器和电池随机化执行函数
└── runtime.py                      执行器、电池和系缆运行参数

simulation/
└── isaac/
    ├── config.py                   策略空间、任务参数和 Isaac 配置契约
    ├── env.py                      DirectRLEnv 生命周期和状态推进
    ├── composition.py              环境与机器人数据源的显式合成
    ├── physics_adapter.py          水流、池体效应与有效水动力状态
    ├── force_composition.py        推进器、缆绳和流体 wrench 合成
    ├── observations.py             当前观测、归一化和因果历史
    ├── robot_asset.py              T60 USD 与 Isaac prim 的绑定
    ├── visualization*.py           Isaac 调试可视化
    ├── ppo/                        PPO 算法、runner 和 MLP 架构注册表
    ├── rewards/                    版本化奖励函数
    ├── training.py                 快照生成与训练进程管理
    ├── trajectory/                 Isaac 任务 mixin、train/eval worker 和报告工具
    └── rlpolicy/                   本地运行、评估和导出产物

tests/                               与生产代码相同的领域归属
```

## 配置的三个层次

### 1. 经确认的物理基准

长期有效、需要评审或来自实验/CFD 的参数保存在领域目录：

- T60 质量、惯量、排水体积、质心和浮心：
  [`robot/dynamics/parameters.py`](robot/dynamics/parameters.py)
- T1–T8 安装位置、PWM 映射和实测三分量推力曲线：
  [`robot/propulsion/curves.py`](robot/propulsion/curves.py)
- 执行器、电池和系缆名义运行参数：[`robot/runtime.py`](robot/runtime.py)
- 水池名义水动力配置：
  [`environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json`](environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json)
- DR 基准配方：
  [`environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json`](environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json)

这些文件是物理事实或经确认配方的单一来源。更新机器人参数时修改 `robot/`；更新水流、
阻尼、附加质量或池体效应时修改 `environment/`。不要把数值复制进
`simulation/isaac/config.py`。

### 2. 单次训练选择

[`train.ipynb`](train.ipynb) 是训练人员唯一需要编辑的入口。它显式列出本次运行所需的：

- 训练身份：`RUN_NAME`、`SEED`、`MLP_ARCHITECTURE`、`REWARD_PROFILE`；
- 训练规模：`NUM_ENVS`、`MAX_ITERATIONS`、`ROLLOUT_STEPS`；
- 确定性水流：世界坐标系常值流、周期流，以及可选空间流场；
- 水池效应：池壁边界、自由液面和晃荡参数；
- 全部 DR 数值：刚体、水流、水动力、推进器和电池的范围与分阶段强度；
- 进程动作：`preview`、`start`、`status` 或 `stop`。

Notebook 不定义 `def` 或 `class`。校验、序列化、命令构造、进程发现和停止逻辑全部位于
Python 模块中，避免交互式单元成为隐藏实现。

### 3. 运行快照

`materialize_training_profiles()` 读取上述基准并应用 Notebook 选择，生成：

```text
simulation/isaac/rlpolicy/_configs/<RUN_NAME>/
├── environment.json
└── domain_randomization.json
```

生成过程不会修改 `environment/` 或 `robot/` 的基准文件。快照路径传给训练 worker，DR
快照同时记录绑定的环境名称、运行名称和活动参数来源。`_configs/` 被 Git 忽略。

同一 `RUN_NAME` 会使用同一个快照目录，因此不同参数实验应使用不同名称；不要在一个仍在
运行或需要审计的实验上复用名称。

## 使用 `train.ipynb`

### 环境准备

当前代码按以下组合开发：

- Isaac Sim `4.5.0`
- Isaac Lab `2.2.0`
- 已验证 Isaac Lab 提交 `0d520b2`
- Conda 环境 `env_isaaclab`

让仓库在 Isaac Lab Direct Task 路径中可见，推荐使用软链接：

```bash
cd <IsaacLab_Path>/source/isaaclab_tasks/isaaclab_tasks/direct
ln -s <isaac-auv-env_Path> isaac-auv-env
```

随后从仓库根目录启动 Jupyter，并打开 `train.ipynb`：

```bash
conda activate env_isaaclab
cd <isaac-auv-env_Path>
jupyter lab train.ipynb
```

Notebook 会验证当前工作目录是否为仓库根目录。默认 `ISAACLAB_ROOT` 为
`Path.home() / "IsaacLab"`；安装位置不同时应先修改该变量。

### 确定性环境参数

`HYDRODYNAMICS`、`POOL_BOUNDARY` 和 `FREE_SURFACE` 描述每个环境都要应用的确定性物理：

- `water_current_w`：世界坐标系中的常值流速；
- `water_current_periodic_*`：周期流的三轴幅值、周期和相位；
- `water_current_field_*`：可选规则网格流场及其边界、形状和值；
- `POOL_BOUNDARY`：接近池壁时的阻尼、附加质量和推力缩放；
- `FREE_SURFACE`：接近水面时的阻尼、附加质量、浮力、推力和晃荡效应。

这些值与 DR 不同：即使关闭随机化，确定性环境仍然生效。

### Domain Randomization

`DR` 字典完整列出当前配方的所有训练随机化字段：

| DR 组 | 执行函数归属 | 主要内容 |
| --- | --- | --- |
| `rigid_body` | `robot/randomization/rigid_body.py` | 质量、排水体积、负载、COM/COB 偏移 |
| `actuators` | `robot/randomization/actuators.py` | 延迟、变化率、分辨率、丢指令、推力与时间常数缩放 |
| `battery` | `robot/randomization/battery.py` | 初始电压和电压下降 |
| `current` | `environment/randomization/current.py` | 平滑随机流、时间常数、水平/垂向幅值和变化量 |
| `hydrodynamics` | `environment/randomization/hydrodynamics.py` | 阻尼、附加质量及附加水动力项 |

使用规则：

- `enabled_features` 决定哪些组可以执行；
- `[a, b]` 表示采样范围，`a == b` 表示固定值；
- `disturbance_curriculum_stage_steps` 是各阶段切换的全局策略步；
- `*_by_stage` 数组按阶段给出扰动上限或缩放强度；
- 启用的非固定随机参数必须具有来源说明，快照生成器会补充本次 Notebook 选择的来源；
- 关闭某个组时，执行层不会因为配置中仍有字段而偷偷应用它。

### 训练动作

最后一个单元的 `ACTION` 控制训练生命周期：

| 值 | 行为 |
| --- | --- |
| `preview` | 生成并校验快照，打印完整 worker 命令，不启动训练 |
| `start` | 以独立进程直接启动训练 worker |
| `status` | 查找属于该 campaign 的训练和评估进程，并显示日志尾部 |
| `stop` | 只终止该 campaign 的匹配进程，并清理失效 PID 记录 |

建议先运行 `preview`，确认环境/DR 快照和打印命令，再改为 `start`。

`simulation/isaac/trajectory/train.py` 是 Isaac Sim 隔离 worker，不是人工配置入口；不应手工
在其中维护实验数值。`simulation/isaac/training.py` 是 Notebook 调用的无 Isaac Sim 导入的
管理 API。

## 训练运行与文件布局

PPO 架构决定独立的 experiment 目录：

- `mlp_30d` → `auv_traj_mlp/`
- `mlp_history_5` → `auv_traj_mlp_history_5/`

RSL-RL 的运行文件全部留在仓库内，不再写入外部 IsaacLab 的 `logs/rsl_rl`：

```text
simulation/isaac/rlpolicy/
├── _configs/<RUN_NAME>/                 本次环境与 DR 快照
└── <architecture experiment>/
    ├── _launcher/                       PID 和启动日志
    └── <timestamp>_<RUN_NAME>/
        ├── model_*.pt                   checkpoint
        ├── params/                      RSL-RL/环境解析配置
        ├── evaluation/                  CSV、汇总表和图片
        ├── exports/                     ONNX
        └── TensorBoard 等运行文件
```

运行产物由 [`simulation/isaac/rlpolicy/.gitignore`](simulation/isaac/rlpolicy/.gitignore) 排除；
目录说明与导出工具仍受版本控制。

## 策略、观测与控制频率

Isaac/PhysX 以 `200 Hz` 推进，策略每四个物理步执行一次，即 `50 Hz`。机器人使用
body `+X` 为前向；Isaac 世界坐标为 z-up，当前水面默认位于 `z=-1 m`，所以正深度对应负的
世界 z 坐标。

Actor 的当前样本为 30 维可部署观测：

```text
位置误差_b(3)
+ 目标线速度_b(3)
+ 线速度误差_b(3)
+ 姿态误差四元数(4)
+ 角速度_b(3)
+ 目标角速度_b(3)
+ 目标线加速度_b(3)
+ 实际应用的八维动作(8)
= 30
```

`mlp_history_5` 在当前样本后追加过去五个 50 Hz 样本中的位置误差、线速度误差、姿态误差、
角速度和实际应用动作，Actor 输入共 135 维。Critic 可在训练时额外使用 77 维精确模拟器
状态；这些 privileged state 不进入 Actor、ONNX 或部署输入。

八维策略动作是归一化推进器命令。运行时只映射一次到 `1300–1700 µs`；实测拟合已经包含
推力符号和离轴分量，`1475–1525 µs` 为闭区间零推力死区。

仿真不注入观测延迟、滤波、丢包或传感器噪声。滤波属于真实测量链，只有通过硬件/水池数据
证明有效后才应进入实际部署代码，而不是在 Isaac 中维护一套未经验证的 sensing 模块。

## 奖励和轨迹

奖励函数以不可变版本保存在 `simulation/isaac/rewards/policy_N.py`，并由
`rewards/registry.py` 统一选择。改变奖励公式或系数时新增版本，不要原地改变既有策略定义。
当前训练入口默认选择 `policy_6`。

可部署的轨迹运动学和航向生成位于 `robot/control/trajectory/`；Isaac 的
`trajectory/mixin.py` 只负责把轨迹状态接入任务 reset/step。训练、评估命令和报表工具位于
`simulation/isaac/trajectory/`，因为它们属于模拟实验流程而不是机器人本体。

## 评估

从仓库根目录打开 [`eval.ipynb`](eval.ipynb)，并把 `POLICY_RUN_DIR` 设置为训练生成的完整
运行目录。评估不会猜测最新 checkpoint 所属运行，也不会搜索外部 IsaacLab 日志；
`FINAL_CHECKPOINT="latest"` 只在已经明确选择的运行内解析。

Notebook 支持：

- 名义环境和完整 Stage-4 DR 的最终验收；
- 多轨迹鲁棒泛化矩阵；
- 保持随机种子和曲线不变的固定流速 sweep；
- checkpoint 汇总、RMSE 图、热图和单条轨迹细节图。

结果写入 `<POLICY_RUN_DIR>/evaluation/`。训练和评估必须选择相同的 MLP 架构、奖励版本以及
相容的环境/DR 配方。

## ONNX 导出

ONNX 默认写入 checkpoint 所在运行的 `exports/`：

```bash
python simulation/isaac/rlpolicy/export_onnx.py \
  --checkpoint simulation/isaac/rlpolicy/auv_traj_mlp_history_5/<run>/model_480.pt \
  --mlp_architecture mlp_history_5
```

## OpenFOAM 与 PMM

OpenFOAM 工程锁定 OpenCFD OpenFOAM `v2512`。几何门禁、算例生成、批量运行和结果拟合见
[`environment/openfoam/README.md`](environment/openfoam/README.md)。PMM 六自由度辨识入口为：

```bash
python environment/pmm/six_dof_identification.py
```

OpenFOAM/PMM 结果经过检查后，应更新 `environment/hydrodynamics/coefficients/` 或相应环境
配置；运行时不会从临时结果目录自动挑选“最新”文件。

## 测试

精简测试集不启动 Isaac Sim，只保留实验主链路的数值与接口回归：PMM 拟合、共享物理、
PID/轨迹、PPO、训练快照和 Isaac 适配。

```bash
conda run -n env_isaaclab python -m pytest -q tests
```

完整 GPU 训练前仍应执行一次 Isaac Lab 短迭代 smoke test；OpenFOAM 的几何与网格质量由
实际工具链输出判定，不再维护一套独立的 Python 门禁测试。

## 修改内容应放在哪里

- 修改机器人质量、惯量、浮力或推进器实测曲线：`robot/dynamics/`、`robot/propulsion/`。
- 修改执行器、电池、系缆、PID 或部署轨迹：`robot/runtime.py`、`robot/control/`。
- 修改水流、阻尼、附加质量、池壁或自由液面模型：`environment/hydrodynamics/`。
- 修改水流或水动力随机化的执行方式：`environment/randomization/`。
- 修改机器人属性随机化的执行方式：`robot/randomization/`。
- 修改单次训练的流速、DR 数值或训练规模：`train.ipynb`。
- 修改 Isaac 状态读取、wrench 注入、观测组装、奖励或 PPO：`simulation/isaac/`。
- 修改真实传感器滤波：应进入实际机器人测量/部署工程，并由真实测试数据支撑；不放回本模拟目录。

仓库目前没有 URDF。以后新增 URDF/xacro 时放入 `robot/assets/`，不要复制到模拟器目录。

# T60 AUV 仿真与水动力项目

本仓库用于 T60 水下机器人的水动力辨识、动力学建模、Isaac Lab 训练/评估和
MuJoCo 跨仿真验证。项目按物理职责分为 `environment`、`robot` 和
`simulation`：环境与机器人定义是共享的数据源，模拟器只负责适配和运行，不在各自目录里
复制质量、惯量、推进器或水动力参数。

当前 Isaac Lab 任务名为 `Isaac-AUV-Traj-Direct-v1`。

## 项目结构

```text
environment/                         环境、实验数据与水动力模型
├── openfoam/                        CFD 几何、算例、工具、本地运行时与结果
├── pmm/                             PMM 原始数据与六自由度参数辨识
├── hydrodynamics/                   运行时水动力模型与版本化系数
├── water/                           流场、边界和池壁/自由液面影响
├── profiles/                        已校验的环境配置契约
├── identification/                  通用参数拟合代码
└── calibration/                     水池标定工作流

robot/                               与模拟器无关的 T60 机器人定义
├── assets/isaac/                    Isaac USD 资产
├── assets/mujoco/                   MuJoCo XML 资产
├── dynamics/                        几何、质量、惯量、浮力和系缆模型
└── propulsion/                      T1–T8 实测推进器曲线模型

simulation/
├── isaac/                           Isaac Lab 训练、评估与 PhysX 适配
│   ├── notebooks/                   train.ipynb、evaluate.ipynb
│   ├── agents/                      PPO 配置、观测和奖励
│   ├── controllers/                 PID 基线
│   ├── configs/                     Domain Randomization 等仿真配置
│   ├── envs/auv/                    Isaac/PhysX 任务适配层
│   └── workflows/                   训练、评估、导出和回放脚本
└── mujoco/                          独立的策略验证后端

tests/                               按上述三层组织的测试
archives/                            历史实验快照，不作为运行时依赖
```

依赖方向固定为：

```text
environment ──┐
              ├──> simulation/isaac
robot ────────┤
              └──> simulation/mujoco
```

`environment/` 和 `robot/` 不应依赖 Isaac Lab 或 MuJoCo。Isaac 与 MuJoCo 也不应
互相引用实现细节。

## 数据单一来源

- 质量、主惯量/惯性轴、质心、浮心和推进器布置：
  [`robot/dynamics/parameters.py`](robot/dynamics/parameters.py)
- T1–T8 三分量实测推力曲线及 PWM 映射：
  [`robot/propulsion/thrusters.py`](robot/propulsion/thrusters.py)
- 水池名义水动力配置：
  [`environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json`](environment/hydrodynamics/coefficients/auv_pool_openfoam_hydrodynamics_v1.json)
- Domain Randomization 配置：
  [`simulation/isaac/configs/domain_randomization/auv_pool_openfoam_hydrodynamics_dr_v1.json`](simulation/isaac/configs/domain_randomization/auv_pool_openfoam_hydrodynamics_dr_v1.json)
- Isaac 生成与资产绑定契约：
  [`simulation/isaac/envs/auv/robot_asset.py`](simulation/isaac/envs/auv/robot_asset.py)
- Isaac USD：[`robot/assets/isaac/`](robot/assets/isaac/)
- MuJoCo XML：[`robot/assets/mujoco/auv.xml`](robot/assets/mujoco/auv.xml)

仓库当前没有可直接使用的 URDF。以后新增的 URDF/xacro 应放在 `robot/assets/` 下，
转换出的 USD 仍放在 `robot/assets/isaac/`，而不是放入某个模拟器的工作流目录。

推进器输入是八维归一化动作，运行时只映射一次到 `1300–1700 µs`。拟合曲线本身已包含
符号和离轴力，`1475–1525 µs` 为闭区间零推力死区，因此模拟器适配层不再叠加安装极性
或固定推力轴映射。

## 主要入口

| 工作 | 入口 |
|---|---|
| 配置并启动训练 | [`simulation/isaac/notebooks/train.ipynb`](simulation/isaac/notebooks/train.ipynb) |
| 配置、运行并分析评估 | [`simulation/isaac/notebooks/evaluate.ipynb`](simulation/isaac/notebooks/evaluate.ipynb) |
| Isaac 脚本训练 | `simulation/isaac/workflows/train/trajectory.py` |
| Isaac 脚本评估 | `simulation/isaac/workflows/evaluate/trajectory.py` |
| PPO 导出 ONNX | `simulation/isaac/workflows/export/trajectory_onnx.py` |
| MuJoCo 策略验证 | `simulation/mujoco/validate_policy.py` |
| PMM 六自由度辨识 | `environment/pmm/six_dof_identification.py` |
| OpenFOAM 工作流 | [`environment/openfoam/README.md`](environment/openfoam/README.md) |
| 水池标定工作流 | [`environment/calibration/README.md`](environment/calibration/README.md) |

Notebook 是面向实验者的配置与报告层；可复用逻辑应保留在
`simulation/isaac/workflows/`，不要复制进 Notebook 单元格。

## Isaac Lab 环境

当前代码按以下组合开发和验证：

- Isaac Sim `4.5.0`
- Isaac Lab `2.2.0`
- 已验证的 Isaac Lab 提交：`0d520b2`
- 项目 Conda 环境名：`env_isaaclab`

先按照对应版本的 Isaac Sim 与 Isaac Lab 官方文档完成安装，并确认 Isaac Lab 自带示例
能够启动。然后让本仓库在 Isaac Lab 任务路径下可见；本地开发建议使用软链接：

```bash
cd <IsaacLab_Path>/source/isaaclab_tasks/isaaclab_tasks/direct
ln -s <isaac-auv-env_Path> isaac-auv-env
```

启动项目前：

```bash
conda activate env_isaaclab
cd <IsaacLab_Path>
```

训练：

```bash
./isaaclab.sh -p \
  source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/simulation/isaac/workflows/train/trajectory.py \
  --task Isaac-AUV-Traj-Direct-v1 \
  --num_envs 2048
```

无 checkpoint 的 PID 轨迹基线：

```bash
./isaaclab.sh -p \
  source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/simulation/isaac/workflows/evaluate/trajectory.py \
  --task Isaac-AUV-Traj-Direct-v1 \
  --controller pid \
  --num_envs 8 \
  --headless
```

训练与评估的可选参数以入口脚本的 `--help` 及两个 Notebook 为准。

## OpenFOAM、PMM 与标定

OpenFOAM 工程锁定 OpenCFD OpenFOAM `v2512`，其几何门禁、24 个强制振荡算例、矩阵
定义和本地运行时安装方法都在
[`environment/openfoam/README.md`](environment/openfoam/README.md) 中维护。已有本地运行时
时可先验证环境：

```bash
source environment/openfoam/env.sh
python3 environment/openfoam/tools/check_environment.py --strict --min-api 2512
environment/openfoam/verify_local_install.sh
```

PMM 默认输入与输出都相对于 `environment/pmm/` 解析：

```bash
python environment/pmm/six_dof_identification.py
```

OpenFOAM/PMM 结果在进入模拟器前，应通过 `environment/identification/` 和
`environment/calibration/` 的拟合、审计流程生成版本化配置；模拟器不直接读取未经校验的
原始结果。

## MuJoCo 跨仿真验证

安装独立验证所需的附加依赖：

```bash
conda activate env_isaaclab
python -m pip install -r simulation/mujoco/requirements.txt
```

可先导出 ONNX：

```bash
python simulation/isaac/workflows/export/trajectory_onnx.py \
  --checkpoint /path/to/model_480.pt \
  --mlp_architecture mlp_history_5
```

再运行 MuJoCo；也可以把 `--policy` 直接指向 `model_*.pt`：

```bash
python simulation/mujoco/validate_policy.py \
  --policy /path/to/policy.onnx \
  --mlp-architecture mlp_history_5
```

结果写入 `results/mujoco/`。MuJoCo 的 RMSE 门禁用于跨模拟器回归检查，不能替代 Isaac
Lab 评估或真实水池实验。完整说明见
[`simulation/mujoco/README.md`](simulation/mujoco/README.md)。

## 测试与静态检查

在项目 Conda 环境中运行主测试集：

```bash
conda run -n env_isaaclab python -m pytest -q --ignore=tests/environment/jn
```

OpenFOAM 自带测试不在根目录 `pytest` 的默认发现范围内，需要单独运行：

```bash
conda run -n env_isaaclab python -m pytest -q environment/openfoam/tests
```

策略导出/部署前置检查：

```bash
conda run -n env_isaaclab python simulation/isaac/agents/rsl_rl/preflight.py
```

`tests/environment/jn/` 是尚未接入当前生产结构的旧 `jn2` 契约测试；仓库当前不包含它所
依赖的 `jn2.six_dof_identification`，所以它不属于主测试集，也没有被静默重定向到 PMM
实现。

## 机器人资产约束

替换刚体资产时必须保持米制、Z-up、FLU、质心原点，并且只包含一个动态刚体。URDF 转换
时应合并固定关节，并分离 visual 与 collision 几何。仅替换网格不会改变
`robot/dynamics/parameters.py` 中的质量、惯量、浮力或推进器参数。

示例转换流程：

```bash
rosrun xacro xacro --inorder -o <output.urdf> <input.xacro>
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  <input.urdf> <output.usd> --merge-joints
```

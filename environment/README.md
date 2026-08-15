# 水环境与水动力领域

`environment/` 管理 AUV 周围的水、池体效应以及水动力数据，所有运行时模型应与具体模拟器
解耦。Isaac 只通过适配层消费这里经确认的 profile 和方程。

名义水池尺寸为 `5.0 × 3.5 × 0.75 m`。池内局部坐标采用 FLU，并以水池下角为原点，
边界为 `[0,5.0] × [0,3.5] × [0,0.75] m`；自由液面、晃荡和空间水流场使用同一范围。

## 内容

- `hydrodynamics/coefficients/`：经人工确认、版本化的水动力系数和名义水池环境。
- `hydrodynamics/current_fields.py`：常值、周期和空间水流场。
- `hydrodynamics/models.py`：阻尼、附加质量和相关水动力方程。
- `hydrodynamics/pool_effects.py`：池壁和自由液面缩放。
- `profiles/environment_profile.py`：严格环境 profile，只接受 hydrodynamics、pool boundary 和
  free surface 三类章节。
- `profiles/domain_randomization.py`：完整 DR 配方的结构、来源审计、JSON 读写和基准绑定。
- `profiles/composition.py`：组合环境 profile、机器人 runtime profile 和可选 DR 配方。
- `profiles/evaluation.py`：在名义 profile 和 DR 之后应用显式评估覆盖。
- `profiles/configs/`：经确认的 DR 基准配方。
- `randomization/`：水流和水动力的运行时随机化函数。
- `runtime/`：显式 `EnvironmentRuntimeState`，管理水流、池体/自由液面、水动力矩阵、相对
  加速度滤波和有效水动力缓存，并计算流体 wrench。
- `openfoam/`：CFD 几何、算例、运行工具和结果。
- `pmm/`：PMM 数据与六自由度辨识。

## 边界

环境 profile 会拒绝机器人、执行器、电池、系缆、传感器和任务字段。机器人本体参数属于
`robot/`。`runtime/` 不调用 PhysX；Isaac 状态接线和最终外力提交只属于
`simulation/assembly.py`。

训练中的确定性流速、池体效应和水动力 DR 强度由 `simulation/training/recipes/` 的版本化
recipe 选择；长期物理基准仍在本目录维护，run-local 输入不会反向修改基准文件。

环境随机化函数只接收环境 runtime、DR 配置、env IDs 和 stage。payload reset 由机器人侧返回
水动力缩放值，再由环境 runtime 应用；环境 runtime 不导入机器人模块。

## 运行时方程

运行时使用 `νr=[v_b-Rᵀv_current_w,ω_b]`，并施加浮力、完整 `6×6` 一次/二次阻力、附加
质量 Coriolis 项和 `-MA ν̇r`。名义 `MA`、`DL`、`DQ` 与版本化 OpenFOAM 拟合输出保持逐项
一致；这里不提供额外的高阶残差模型。池壁、自由液面和 DR 是独立的显式缩放，不能与
名义测量矩阵混为一谈。

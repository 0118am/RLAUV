# 水环境与水动力领域

`environment/` 管理 AUV 周围的水、池体效应以及水动力数据，所有运行时模型应与具体模拟器
解耦。Isaac 只通过适配层消费这里经确认的 profile 和方程。

实物名义水池尺寸仍为 `5.0 × 3.5 × 0.75 m`。当前
`auv_open_water_openfoam_full_hydrodynamics_v2.json` 是开水域训练 profile：池壁和自由液面模型
关闭，并使用 `[-0.75,5.75] × [-0.75,3.75] × [-0.60,1.60] m` 的虚拟安全盒容纳
`5 × 3 × 1 m` 命令、艇体和初始误差。该安全盒不是实物水池声明；完整 1 m 垂向曲线不能在
0.75 m 深的实物水池中原样验收，部署测试必须另行缩放垂向幅值。

## 内容

- `hydrodynamics/coefficients/`：版本化运行时水动力输入；每个文件单独声明 provisional、
  数值验证或实物验证状态，不能由“在此目录中”推断已经验证。
- `hydrodynamics/current_fields.py`：周期水流与基于 PyTorch `grid_sample` 的规则网格水流场。
- `hydrodynamics/models.py`：阻尼、附加质量和相关水动力方程。
- `hydrodynamics/pool_effects.py`：池壁和自由液面缩放。
- `profile.py`：Pydantic 2 环境输入模型，只接受 hydrodynamics、pool boundary 和
  free surface 三类章节，并直接负责 JSON 读写。
- `randomization/`：水流和水动力的运行时随机化函数。
- `runtime/`：显式 `EnvironmentRuntimeState`，管理水流、池体/自由液面、水动力矩阵、规定
  水流的机体系加速度和有效水动力缓存，并计算流体 wrench；不对艇体加速度做有限差分反馈。
- `openfoam/`：CFD 几何、算例、运行工具和结果。
- `pmm/`：旧 PMM 数据与不完整的固定航速辨识档案，不作为生产矩阵来源。

## 边界

环境 profile 会拒绝机器人、执行器、系缆、传感器和任务字段。机器人本体参数属于
`robot/`。`runtime/` 不调用 PhysX；Isaac 状态接线和最终外力提交只属于
`simulation/assembly.py`。跨 environment/robot 的组合与 DR schema 属于 `simulation/`；
评估覆盖属于 `simulation/training/evaluation/`。

训练中的确定性流速、池体效应和水动力 DR 强度由 `simulation/training/recipes/` 的版本化
recipe 选择；长期物理基准仍在本目录维护，run-local 输入不会反向修改基准文件。

环境随机化函数只接收环境 runtime、DR 配置、env IDs 和 stage；环境 runtime 不导入机器人
模块。质量、惯量、COM、CoB 和排水体积使用固定实测值，不参与 DR。

## 运行时方程

运行时使用 `νr=[v_b-Rᵀv_current_w,ω_b]`，并施加浮力、完整 `6×6` 一次/二次阻力和附加
质量 Coriolis 项。附加质量惯性不再作为有限差分外力 `-MA ν̇r` 回馈；组装层直接解
`(MRB+MA)ν̇`，再映射为 PhysX 的等效 wrench。名义 `MA`、`DL`、`DQ` 与当前选定的版本化
OpenFOAM 输入保持逐项一致。当前输入由最近完成的 24 个单轴工况全部六分量载荷重新拟合：
镜像对称允许的非对角项被保留，附加质量执行互易对称化。它仍缺少多轴组合工况和实物验证，
而且完整线性阻尼尚不满足全局被动性，不能称为最终标定。这里不提供额外的高阶残差模型。
池壁、自由液面和 DR 是独立的显式缩放，不能与名义矩阵混为一谈。

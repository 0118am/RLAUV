# 水环境与水动力领域

`environment/` 管理 AUV 周围的水、池体效应以及水动力数据，所有运行时模型应与具体模拟器
解耦。Isaac 只通过适配层消费这里经确认的 profile 和方程。

## 内容

- `hydrodynamics/coefficients/`：经人工确认、版本化的水动力系数和名义水池环境。
- `hydrodynamics/current_fields.py`：常值、周期和空间水流场。
- `hydrodynamics/models.py`：阻尼、附加质量和相关水动力方程。
- `hydrodynamics/pool_effects.py`：池壁和自由液面缩放。
- `profiles/environment_profile.py`：严格环境 profile，只接受 hydrodynamics、pool boundary 和
  free surface 三类章节。
- `profiles/domain_randomization.py`：完整 DR 配方的结构、来源审计、JSON 读写和基准绑定。
- `profiles/configs/`：经确认的 DR 基准配方。
- `randomization/`：水流和水动力的运行时随机化函数。
- `openfoam/`：CFD 几何、算例、运行工具和结果。
- `pmm/`：PMM 数据与六自由度辨识。

## 边界

环境 profile 会拒绝机器人、执行器、电池、系缆、传感器和任务字段。机器人本体参数属于
`robot/`；Isaac 状态读取和 PhysX 外力施加属于 `simulation/isaac/`。

训练中的确定性流速、池体效应和水动力 DR 强度由根目录 `train.ipynb` 为单次运行选择；长期
物理基准仍在本目录维护，Notebook 生成的快照不会反向修改基准文件。

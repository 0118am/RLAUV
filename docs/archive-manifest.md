# 服务器回收前归档清单

捕获日期：2026-08-29。归档目标是让最终 V16 策略、训练输入、最终 OpenFOAM 系数来源和旧
评估结论能从 Git 中审计，同时不把可再生成的数十 GB 网格、时序日志和中间 checkpoint
塞进普通 Git 历史。

## 已纳入

| 内容 | Git 路径 | 归档口径 |
|---|---|---|
| 最终 V16 Policy | `simulation/rlpolicy/...t60_precision_v16/` | `model_499.pt`、ONNX、run-local 输入、TensorBoard、最终完整评估；原始载荷 14,593,481 字节 |
| 训练环境 | `environment-training-versions.yml`、`docs/training-environment.md` | 实机读取的关键版本和已知缺口 |
| OpenFOAM 几何溯源 | `environment/openfoam/geometry/validated_locked_rotor_v1/*.json` | 小型选择/修复参数、拓扑审计和大型几何哈希 |
| OpenFOAM 最终拟合 | `environment/openfoam/results/fit_20260825_005547_340708/` | 最终三个 CSV、完整诊断 JSON、发布输入和数学报告 |
| 最终采用系数 | `environment/hydrodynamics/coefficients/auv_open_water_openfoam_full_hydrodynamics_v2.json` | 已验证与 `config_updates.json` 三个矩阵逐值相同 |
| 旧 IsaacLab 评估 | `experiments/legacy_isaaclab_evaluations/` | 315 个非 `logs.csv` CSV + 61 张 PNG，共 18,943,405 字节 |

## 明确未纳入 Git

- V16 的 `model_0.pt` 至 `model_450.pt`：约 61.5 MB，只保留最终 `model_499.pt`。
- 旧 IsaacLab 评估的 206 个 `logs.csv`：约 5.54 GB；同时不保存 39 个控制台日志。
- OpenFOAM `cases*`、网格、`postProcessing` 和 `.runtime`：约 22 GB + 4.4 GB，可由脚本和
  已归档参数再生成。
- 修复后的 `wetted_body_m.obj`：586,262,341 字节；Git 中保留 SHA-256
  `51cc600c37216dc2a2032b3bfd0d3dfc8e630ce30b7bf58d99710d1d6f5fc224`。
- STEP 选择输出 `selected_body_mm.stl`：61,010,134 字节；Git 中保留 SHA-256
  `65c131e5f14b14366b2f765cf93ab2c376e63d94e7cbd8f796d6b36f97282f9d`。

以上大文件若还要求“原样恢复而不是再生成”，必须在服务器回收前另存对象存储或离线盘；
普通 Git 归档不承担这部分备份。

## 仍需单独决定的服务器内容

审计时发现 `/home/jining_yang/workspace/RLUnderwater` 是另一个独立仓库，约 2.2 MB，包含
37 个已跟踪修改和 10 个未跟踪文件，且其远端当时无法访问。它不属于本次用户指定的 V16 /
OpenFOAM / 旧评估归档范围，因此未修改、未提交。若该目录仍有价值，应在服务器回收前在它
自己的仓库中单独审阅和提交。

## 归档验证

- V16 与 OpenFOAM `SHA256SUMS` 全部通过；旧评估 376 个文件的逐文件 SHA-256 全部通过。
- `model_499.pt` 可由 PyTorch 以 `weights_only=True` 加载，包含 `iter=499` 和完整模型/优化器状态。
- ONNX checker 通过：IR 10，输入 `obs[1,201]`，输出 `actions[1,8]`。
- 修复 OBJ 和中间 STL 的实物哈希与小型溯源 JSON 一致。
- `config_updates.json` 的 `M_A/D_L/D_Q` 与实际训练环境 coefficient JSON 逐值相同。
- `conda run -n env_isaaclab python -m pytest -q tests`：84 passed，1 个 CUDA 驱动不可用警告。

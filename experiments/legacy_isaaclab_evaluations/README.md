# 旧 IsaacLab 评估摘要归档

源目录：`/home/jining_yang/IsaacLab/results/rsl_rl`。2026-08-29 在服务器回收前复制，保持原有
实验/run/evaluation 相对目录结构。

## 过滤口径

- 保留所有非 `logs.csv` 的 CSV（summary metrics、domain samples、checkpoint 汇总等）和 PNG。
- 删除 206 个逐步 `logs.csv`，源文件合计约 5.54 GB。
- 删除 39 个 `evaluation_console.log`，因为用户指定只保留摘要和图片。
- 源目录本身未发现 `.pt`、`.onnx` 或 TensorBoard event，所以本归档不是旧策略 checkpoint
  备份，不能从这些评估文件重新部署旧策略。

最终为 315 个 CSV + 61 张 PNG，共 376 个文件、18,943,405 字节（十进制约 18.94 MB）。
`SHA256SUMS` 可用于检查复制和后续迁移完整性。

## 按实验统计

| 实验 | 文件数 | 字节数 |
|---|---:|---:|
| `auv_traj_mlp_history_5` | 2 | 4,317 |
| `eupauv_traj_direct` | 49 | 2,837,212 |
| `eupauv_traj_gru` | 9 | 705,208 |
| `eupauv_traj_mlp_history_5` | 164 | 3,990,553 |
| `eupauv_traj_transformer_history_256` | 8 | 112,513 |
| `eupauv_traj_transformer_history_64` | 16 | 225,019 |
| `warpauv_direct` | 6 | 1,694,516 |
| `warpauv_traj_direct` | 84 | 6,114,887 |
| `warpauv_traj_heavy2_direct` | 38 | 3,259,180 |

## 使用限制

摘要适合比较各次评估已经计算出的聚合指标；图片适合人工检查轨迹和推进器行为。由于逐步日志
已按要求删除，无法只靠本归档重新计算新的时间窗指标、频谱、失败瞬间或不同聚合公式。不同旧
实验的 schema 和评估场景也可能不一致，跨目录比较前必须先核对 CSV 表头、seed、扰动、轨迹和
checkpoint 名称，不能只按同名 metric 排序。

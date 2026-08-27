# RL policy 运行目录

根目录 `train.ipynb` 通过 `simulation/training/` 公共 API 把所有 RSL-RL 运行写入本目录，并按
命名 MLP 架构隔离。训练 worker 位于 `simulation/training/train.py`。

```text
rlpolicy/
└── <architecture experiment>/
    └── <timestamp>_<RUN_NAME>/
        ├── model_*.pt
        ├── params/
        │   └── inputs/
        │       ├── training_recipe.json
        │       ├── environment.json
        │       └── domain_randomization.json
        ├── evaluation/
        └── exports/
```

训练标准输出由 `train.ipynb` 最后一个单元格直接显示，不在这里创建 launcher PID/日志文件。
`eval.ipynb` 只在仓库内 `auv_traj_mlp_history_8` 目录自动选择最新完成训练的 run/checkpoint。
评估、续训和 ONNX 导出均从 checkpoint 同目录的 training recipe 恢复网络、奖励和输入文件，不使用独立 `_configs`
目录。checkpoint、快照、日志、评估和 ONNX 文件由 `.gitignore` 排除。导出工具位于
`simulation/training/evaluation/export.py`。

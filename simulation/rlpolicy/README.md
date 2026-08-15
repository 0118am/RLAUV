# RL policy 运行目录

根目录 `train.ipynb` 通过 `simulation/training/` 公共 API 把所有 RSL-RL 运行写入本目录，并按
命名 MLP 架构隔离。训练 worker 位于 `simulation/training/train.py`。

```text
rlpolicy/
└── <architecture experiment>/
    ├── _launcher/
    └── <timestamp>_<RUN_NAME>/
        ├── model_*.pt
        ├── params/
        │   ├── run_manifest.json
        │   └── inputs/
        │       ├── training_recipe.json
        │       ├── environment.json
        │       └── domain_randomization.json
        ├── evaluation/
        └── exports/
```

`eval.ipynb` 要求显式设置 `POLICY_RUN_DIR`，不会搜索 IsaacLab 外部日志或猜测最新运行。
评估、续训和 ONNX 导出均从 run manifest 恢复网络、奖励和输入文件，不使用独立 `_configs`
目录。checkpoint、快照、日志、评估和 ONNX 文件由 `.gitignore` 排除。导出工具位于
`simulation/training/evaluation/export.py`。

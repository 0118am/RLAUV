# RL policy 运行目录

根目录 `train.ipynb` 通过 `simulation/isaac/training.py` 把所有 RSL-RL 运行写入本目录，并按
命名 MLP 架构隔离。这里不再提供 `simulation.isaac.train` 命令入口。

```text
rlpolicy/
├── _configs/<RUN_NAME>/
│   ├── environment.json
│   └── domain_randomization.json
└── <architecture experiment>/
    ├── _launcher/
    └── <timestamp>_<RUN_NAME>/
        ├── model_*.pt
        ├── params/
        ├── evaluation/
        └── exports/
```

`eval.ipynb` 要求显式设置 `POLICY_RUN_DIR`，不会搜索 IsaacLab 外部日志或猜测最新运行。
checkpoint、快照、日志、评估和 ONNX 文件由 `.gitignore` 排除；本说明和
`export_onnx.py` 继续受版本控制。

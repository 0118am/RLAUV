# 测试组织

测试目录与生产代码的领域归属保持一致：

- `environment/`：水动力、水流、池体效应、profiles 和 PMM。
- `robot/`：刚体、系缆、推进器、PID 和通用轨迹控制。
- `simulation/isaac/`：配置合成、PhysX 适配、观测/PPO、奖励、训练与评估流程。
- `simulation/mujoco/`：独立策略桥接和模型契约。
- `integration/dynamics_cases.py`：跨领域共享的动力学用例集合。

机器人 PID、guidance、kinematics 和机器人 DR 测试不应放回 Isaac 测试目录；水流、水动力和
池体函数测试归 `tests/environment/`。只有真正依赖 Isaac 适配契约的测试才归
`tests/simulation/isaac/`。

全量非 Isaac Sim 回归：

```bash
conda run -n env_isaaclab python -m pytest -q tests
```

共享动力学用例：

```bash
conda run -n env_isaaclab python -m pytest -q tests/integration/dynamics_cases.py
```

长训练前快速策略检查：

```bash
conda run -n env_isaaclab python tests/run_policy_preflight.py
```

以上测试不会启动 Isaac Sim。完整 GPU 训练前仍需一次 Isaac Lab 短迭代 smoke test。

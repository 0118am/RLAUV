# RL 训练环境版本清单

本清单在 2026-08-29 服务器回收前从实际使用的 `env_isaaclab` 环境读取。机器可读版本位于
[`environment-training-versions.yml`](../environment-training-versions.yml)。它记录直接影响
训练、导出和评估的组件；不是包含所有间接依赖构建号的二进制锁文件。

## 核心版本

| 层 | 版本 |
|---|---|
| 系统 | Linux `5.15.0-164-generic`，glibc `2.35` |
| Conda / Python / pip | `26.3.2` / `3.11.15` / `26.1.2` |
| Isaac Lab 源码 | tag `v2.3.2`，commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Isaac Sim | `5.1.0.0` |
| Isaac Lab Python 包 | `isaaclab 0.54.2`，`assets 0.2.4`，`rl 0.4.7`，`tasks 0.11.12` |
| PyTorch | `2.7.0+cu128`，Torch CUDA `12.8`，cuDNN `9.7.1.26` |
| RL | `rsl-rl-lib 3.1.2` |
| 数值/导出 | NumPy `1.26.0`，SciPy `1.15.3`，ONNX `1.21.0`，Warp `1.14.0` |
| 记录/评估 | TensorBoard `2.20.0`，pandas `3.0.3`，Matplotlib `3.10.3` |

所有已确认的 CUDA wheel 和辅助库版本均在 YAML 清单中。Isaac Sim 的各个已安装
`isaacsim-*` 元包均为 `5.1.0.0`。

## 重建顺序

1. 以 Python `3.11.15` 新建 Conda 环境。
2. 检出 Isaac Lab `37ddf626871758333d6ed89cf64ad702aef127d0`，使用该版本自带的安装流程安装
   Isaac Sim `5.1.0.0` 和 Isaac Lab 扩展，避免把新版本 Isaac Lab 混入。
3. 按 YAML 核对 PyTorch、CUDA runtime wheel、RSL-RL、ONNX 和科学计算包。
4. 检出本仓库中包含本归档的提交，运行测试后再加载 V16 checkpoint。
5. 用最终 run 中 `params/inputs/*.json` 恢复训练输入；不要用随后可能修改的全局 recipe 替代它们。

## 无法恢复的硬件信息

捕获时 `nvidia-smi` 已报“无法与 NVIDIA driver 通信”，`torch.cuda.is_available()` 也为
`False`。因此不能诚实地写出训练 GPU 型号、显存和驱动版本；YAML 中将三项明确记为
`unknown`。wheel 只能证明训练环境使用 CUDA 12.8 用户态栈，不能反推出当时的驱动版本。

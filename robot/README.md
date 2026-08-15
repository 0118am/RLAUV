# T60 AUV 机器人领域

`robot/` 是 T60 本体性质的单一来源。这里的模型应尽量不依赖具体模拟器，以便 Isaac 和
实际部署代码共享。

## 内容

- `assets/isaac/t60_auv.usd`：Isaac 几何、碰撞体和材质资产。
- `dynamics/parameters.py`：质量、惯量、排水体积、质心、浮心、推进器安装与标签。
- `dynamics/rigid_body.py`：惯量和坐标变换。
- `dynamics/tether.py`：T60 系绳受力模型。
- `propulsion/curves.py`：T1–T8 物理 PWM 安装映射、实测 FLU 三轴力曲线与力矩归并。
- `propulsion/dynamics.py`：命令延迟、限速、量化和一阶执行器响应。
- `propulsion/effects.py`：电压、入流和尾流干扰修正。
- `runtime.py`：带来源的执行器响应/延迟、电池和系缆名义运行配置。
- `control/pid.py`：只负责从跟踪误差生成六自由度目标力矩。
- `control/allocation.py`：把目标力矩分配到实测非线性推进器曲线。
- `control/trajectory/`：分离的轨迹目录、几何、重定时和航向生成。
- `randomization/`：刚体、负载、推进器和电池随机化执行函数。

## 边界

机器人性质发生变化时在这里更新，再由 Isaac 适配器读取。`simulation/isaac/` 不应保存
质量、惯量、浮力、推进器曲线、执行器、电池或系缆参数副本。

水流、阻尼、附加质量、池壁和自由液面属于 `environment/`。训练中机器人 DR 的具体数值由
根目录 `train.ipynb` 选择，但采样与应用函数仍归本目录管理。

## 推进器约定

策略命令在命令处理器中限制为 `[-1, 1]`，再映射为 `1300–1700 µs`。曲线按物理 PWM 的
负/正分支保存完整 FLU `(Fx, Fy, Fz)`，不经过标量推力、独立 polarity、spin direction 或
固定轴重建。正 PWM 时 T5/T6 的 `Fx < 0`，T7/T8 的 `Fx > 0`。

名义一阶响应时间常数为 `0.08 s`，命令延迟为 `0.13 s`；它们是带来源的外部硬件参考，
不是 T60 本机测量。运行链路不施加未经测量的反扭矩。

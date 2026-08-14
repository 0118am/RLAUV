# T60 AUV 机器人领域

`robot/` 是 T60 本体性质的单一来源。这里的模型应尽量不依赖具体模拟器，以便 Isaac、
MuJoCo 和实际部署代码共享。

## 内容

- `assets/isaac/t60_auv.usd`：Isaac 几何、碰撞体和材质资产。
- `assets/mujoco/t60_auv.xml`：MuJoCo 刚体与 T1–T8 安装点。
- `dynamics/parameters.py`：质量、惯量、排水体积、质心、浮心、推进器安装与标签。
- `dynamics/rigid_body.py`：惯量和坐标变换。
- `dynamics/tether.py`：T60 系绳受力模型。
- `propulsion/thrusters.py`：T1–T8 实测 PWM/三轴力曲线、执行器响应与推力合成。
- `runtime.py`：执行器动态、电池和系缆的名义运行配置。
- `control/pid.py`：使用 T60 推进器模型的六自由度 PID 控制器。
- `control/trajectory/`：可部署的轨迹运动学、重定时和航向生成。
- `randomization/`：刚体、负载、推进器和电池随机化执行函数。

## 边界

机器人性质发生变化时在这里更新，再由模拟器适配器读取。Isaac 与 MuJoCo 不应保存质量、
惯量、浮力、推进器曲线、执行器、电池或系缆参数副本。

水流、阻尼、附加质量、池壁和自由液面属于 `environment/`。训练中机器人 DR 的具体数值由
根目录 `train.ipynb` 选择，但采样与应用函数仍归本目录管理。

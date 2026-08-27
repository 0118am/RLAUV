# T60 AUV 初版 CFD 水动力矩阵

本目录用 OpenCFD OpenFOAM v2512 生成一组可用于 PhysX/RL 的初版水动力参数。目标不是
发表级完整水动力数据库，而是在机器人相对水速低于 `0.4 m/s` 时从全部六分量载荷拟合三个
完整响应 `6×6` 矩阵：

```text
nu    = [u, v, w, p, q, r]^T
tau_h = [X, Y, Z, K, M, N]^T

tau_h = -M_A nu_dot - C_A(nu; M_A) nu
        -D_L nu - D_Q (abs(nu) .* nu)
```

坐标是以移动 COM 为原点的 body FLU。每个单轴激励都记录 `[X,Y,Z,K,M,N]` 六个响应，因此
形成相应矩阵的完整一列。body-FLU 左右反射对称只允许 `[u,w,q]` 和 `[v,p,r]` 两个块内
耦合；块间项清零，块内非对角项保留。附加质量独立列按互易性取对称平均。

## 24 个工况

- 平移阻尼 12 个：`u/v/w × ±0.08, ±0.40 m/s`。每个工况先计算 0.5 个艇长，再对后
  0.5 个艇长取时间加权平均。正负载荷半差去除静态偏置，两档速度解出该激励列全部六个响应的
  `D_L` 和 `D_Q`。
- 转动阻尼 6 个：`p/q/r × 0.2, 0.8 rad/s`，频率固定 `1.0 Hz`。固定频率避免把
  频率效应误认成幅值非线性。
- 附加质量 6 个：六自由度各一个 `1.0 Hz` 工况；平移峰值速度 `0.04 m/s`，转动峰值
  角速度 `0.08 rad/s`。一个周期平滑增幅、半个周期稳定、两个周期取样。

两类振荡工况使用同一频率并直接指定峰值速度，而不是固定位移后随频率改变速度。这样既
避免用不同频率的附加质量去扣除转动阻尼工况，也把最大转角限制在约 `7.3°`。

附加质量工况的周期波形同时回归 `-nu_dot`、`-nu` 和 `-|nu|nu`。加速度与速度相差
90°，而 `|nu|nu` 又有不同波形，因此完整周期可以为六个响应通道分别分离附加质量和两个
干扰阻尼系数。
最终阻尼仍来自专门的两档阻尼工况。完整线性阻尼的对称部分目前存在负特征值，因此这套
原始 CFD 拟合尚不能证明对任意多轴组合速度全局被动。

## 流场和网格

- 被动、静态锁桨、单相、静水参考、无重力、开水域；
- `rho=1000 kg/m³`、`nu=1e-6 m²/s`；
- 完全湍流 `kOmegaSST` URANS，1% 名义远场湍流强度；
- 外域 `[-3,3] × [-1.5,1.5] × [-1.5,1.5] m`，基础网格 `90×45×45`；
- 湿表面 level 4，桨盘、轴和电机 level 5，艇体背景近区 level 1，桨侧局部近场 level 4；
- 只有一个 `auv` 壁面，一次 `snappyHexMesh`，不生成棱柱层；
- 无棱柱层时使用 all-y+ SST 壁面函数；`y+` 被记录但不冒充低雷诺数解析结果；
- 二阶 backward 时间格式、PIMPLE 一次 outer corrector、`maxCo=0.7`、每周期
  100 个最大时间步，受力每个时间步输出。动态网格仍会在 Courant 数需要时自动缩小步长。

这个网格删除了原先失败率最高的双壁面分区和第二次棱柱层挤出。代价是黏性阻力精度低于
经过网格收敛且解析边界层的方案，因此结果只能标记为 preliminary。

## 运行

先编译运动边界并跑四个代表性工况：

```bash
environment/openfoam/run_campaign.sh --pilot
```

确认内存、单步耗时、网格单元数和载荷曲线可接受后，复用同一网格完成 24 个工况：

```bash
environment/openfoam/run_campaign.sh
```

可通过环境变量设置并行：

```bash
AUV_CFD_NP=8 AUV_CFD_JOBS=1 environment/openfoam/run_campaign.sh
```

拟合结果位于 `environment/openfoam/results/fit_<timestamp>/`。选择需要使用的结果并
把其中三个矩阵写入 PhysX 环境配置：

```bash
python3 environment/openfoam/publish_results.py \
  environment/openfoam/results/fit_<timestamp>/hydrodynamic_fit.json
```

发布器只要求三个矩阵是有限的 `6×6` 数组，不根据拟合误差、横轴响应、工况哈希或
源码版本拒绝写入。当前仓库 JSON 已经是 PhysX 直接使用的矩阵输入。

## 可信范围

此方案适合先跑通低速 RL，并不验证：

- 水池壁面、池底、自由液面和晃荡；
- 工作螺旋桨尾流与艇体相互作用；
- 多轴同时运动产生的混合二次阻力；
- 非对角附加质量/阻尼的多轴组合与重复工况验证；
- 实物表面粗糙度、实测湍流和硬件载荷。

理论和试验设计依据分别可查 Fossen 的海洋航行器模型
<https://fossen.biz/html/marineCraftModel.html>、ITTC Captive Model Test
<https://www.ittc.info/media/11876/75-02-06-07.pdf> 和 ITTC RANS Maneuvering
<https://ittc.info/media/11970/75-03-04-02.pdf>。本方案借用其符号、强迫运动和
正负/多幅值辨识思想，但没有声称达到这些规程要求的完整不确定度验证等级。

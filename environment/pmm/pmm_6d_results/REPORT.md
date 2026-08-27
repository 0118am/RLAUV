# PMM 6×6 水动力辨识结果

本报告只保留为水池模型试验档案，不是模拟器生产系数源；运行时三张矩阵只由目标尺度 OpenFOAM 工况发布。

辨识式为 `tau_h=-M_A@nu_dot-D_1_eff@nu-D_2@(abs(nu)*nu)-K_nuisance@xi`；`K_nuisance` 只用于隔离恢复项。顺序为 wrench `[X,Y,Z,K,M,N]`、velocity `[u,v,w,p,q,r]`，参考点为质心，坐标为艇体 FLU。

## 台架到辨识坐标的转换

用户确认原始 PMM 台架轴标签为 `(+X前,+Y左,+Z下)`。数据读入后先按 `[x,y,z] -> [x,y,-z]` 转成右手 FLU，艏向编码器同样取负号；之后才执行 vertical 装配旋转、力矩平移、Newton--Euler 叉乘和系数回归。台架左手轴标签不直接用作刚体计算坐标。

## 附加质量：直接对角项

| wrench \ velocity | u | v | w | p | q | r |
|---|---:|---:|---:|---:|---:|---:|
| X | NaN | NaN | NaN | NaN | NaN | NaN |
| Y | NaN | 6.413126 | NaN | NaN | NaN | NaN |
| Z | NaN | NaN | 11.56946 | NaN | NaN | NaN |
| K | NaN | NaN | NaN | NaN | NaN | NaN |
| M | NaN | NaN | NaN | NaN | 0.05772615 | NaN |
| N | NaN | NaN | NaN | NaN | NaN | 0.0373619 |

## 测力计反力符号

测力计给出艇体对测力架的反力 `R=-F_support`，因此应使用 `tau_h=R+tau_rigid`。原脚本的减法会重复计入刚体惯性。

## 一次阻力（固定 U0 直接对角项）

| wrench \ velocity | u | v | w | p | q | r |
|---|---:|---:|---:|---:|---:|---:|
| X | NaN | NaN | NaN | NaN | NaN | NaN |
| Y | NaN | -1.296268 | NaN | NaN | NaN | NaN |
| Z | NaN | NaN | 8.502781 | NaN | NaN | NaN |
| K | NaN | NaN | NaN | NaN | NaN | NaN |
| M | NaN | NaN | NaN | NaN | 0.1068126 | NaN |
| N | NaN | NaN | NaN | NaN | NaN | 0.07006275 |

## 二次阻力（固定 U0 直接对角项）

| wrench \ velocity | u | v | w | p | q | r |
|---|---:|---:|---:|---:|---:|---:|
| X | NaN | NaN | NaN | NaN | NaN | NaN |
| Y | NaN | -8.635178 | NaN | NaN | NaN | NaN |
| Z | NaN | NaN | -23.41755 | NaN | NaN | NaN |
| K | NaN | NaN | NaN | NaN | NaN | NaN |
| M | NaN | NaN | NaN | NaN | -0.3380768 | NaN |
| N | NaN | NaN | NaN | NaN | NaN | -0.2974209 |

## 固定 U0 的二次独占直航闭合

为当前 Isaac 接口提供的可选闭合取 `D1_uu=0`、`D2_uu=R(U0)/U0²=8.886724 N·s²/m²`。它在参考航速精确复现直航总阻力，但不代表已经测得前进速度律；`p,p` 和所有非对角项继续保留 NaN。

一次矩阵：

| wrench \ velocity | u | v | w | p | q | r |
|---|---:|---:|---:|---:|---:|---:|
| X | 0 | NaN | NaN | NaN | NaN | NaN |
| Y | NaN | -1.296268 | NaN | NaN | NaN | NaN |
| Z | NaN | NaN | 8.502781 | NaN | NaN | NaN |
| K | NaN | NaN | NaN | NaN | NaN | NaN |
| M | NaN | NaN | NaN | NaN | 0.1068126 | NaN |
| N | NaN | NaN | NaN | NaN | NaN | 0.07006275 |

二次矩阵：

| wrench \ velocity | u | v | w | p | q | r |
|---|---:|---:|---:|---:|---:|---:|
| X | 8.886724 | NaN | NaN | NaN | NaN | NaN |
| Y | NaN | -8.635178 | NaN | NaN | NaN | NaN |
| Z | NaN | NaN | -23.41755 | NaN | NaN | NaN |
| K | NaN | NaN | NaN | NaN | NaN | NaN |
| M | NaN | NaN | NaN | NaN | -0.3380768 | NaN |
| N | NaN | NaN | NaN | NaN | NaN | -0.2974209 |

## ITTC 固定航速 surge 偶次耦合项

力导数为 fluid-on-body 符号；阻力幅值是其相反数。`v²/w²/q²/r²` 是偶函数，不能写入现有 `D_2@(abs(nu)*nu)`。

| term | campaign | cycle-mean derivative | 2f derivative audit | resistance magnitude | units | DC R² | 2f R² | leave-one-repeat resistance range |
|---|---|---:|---:|---:|---|---:|---:|---:|
| X_vv | sway | -9.983294 | -4.297684 | 9.983294 | N*s^2/m^2 | 0.9223 | 0.3981 | [9.872291, 10.07282] |
| X_ww | heave | -5.918122 | 2.042828 | 5.918122 | N*s^2/m^2 | 0.1374 | -0.0098 | [5.674646, 6.193654] |
| X_qq | pitch | -0.4187482 | -0.2469578 | 0.4187482 | N*s^2 | 0.3683 | 0.0767 | [0.3850108, 0.4527584] |
| X_rr | yaw | -0.2206902 | -0.2070446 | 0.2206902 | N*s^2 | 0.2174 | -0.0125 | [0.1817576, 0.2380889] |

`cycle-mean derivative` 来自 Fourier 还原后的完整周期 DC 趋势；`2f derivative audit` 是独立的二倍频检查。理想无记忆平方项应使两者接近；表中的差异（尤其 heave 二倍频反号）说明这些 surge 耦合值可以作为固定航速的经验趋势，但尚不是已验证的全动态 Isaac 系数。

normal/pure 装配在参考航速 0.2015936 m/s 的直航 surge 力锚点为 -0.3611564 N，两个 campaign 截距范围为 [-0.3834793, -0.3388335] N。若仿真必须在当前接口中闭合 surge，可互斥选择 `D1_uu=1.791507 N·s/m` 或 `D2_uu=8.886724 N·s²/m²`；单一航速不能同时辨识这两个值，因此不能把它们相加。

## 固定 U0 基频同相诊断

| DOF | trials | median equivalent direct D | trial range | units |
|---|---:|---:|---:|---|
| sway | 21 | -1.911052 | [-14.10566, 1.676394] | N*s/m |
| heave | 21 | 4.545854 | [-23.82448, 16.39166] | N*s/m |
| pitch | 21 | -0.001750869 | [-0.1373719, 0.1785723] | N*m*s |
| yaw | 21 | -0.02355582 | [-0.15814, 0.06021163] | N*m*s |

这里用 Fourier 正弦/余弦基频向量做完整周期内积，避免非整周期时间窗的惯性边界能量混入。它显示直接通道随频率和振幅的有效同相导数，但不是把每个通道单独判为被动/非被动的门槛；固定航速操纵模型还包含 surge 偶次项和耦合项。

## 结论与限制

- 用户确认 `gather` 为电机记录（100 Hz），`sensor_` 为六分力记录（500 Hz）；每 5 个传感器点块平均后对齐到 100 Hz。
- 用户确认台架原始方向为前 `+X`、左 `+Y`、下 `+Z`；六分力和艏向角已在读入端转到项目 FLU。
- 用户确认两路采集硬同步；零时差是实验设置，不再视为待估计参数。
- 2–10 s 运动拟合及 2.5–9.5 s 载荷硬裁剪均直接沿用 `Downloads/jn` 原脚本。
- 已把原脚本的 `mapped_balance-rigid` 修正为反力关系 `R+rigid`，避免重复计入刚体惯性。
- 三张 PMM 矩阵只发布 `v/Y、w/Z、q/M、r/N` 直接对角回归；`u、p` 对角项和所有非对角项都保留 `NaN`。
- 没有对负系数取绝对值，没有对阻力矩阵强制对称/正定，也没有 PSD 投影。
- 不用左右对称将未辨识项填 0，不用附加质量互易性补齐他列。
- `D_1_eff` 含平均拖航速度下的交叉流与未单独分离的附加质量 Coriolis 同相项。
- `timing_sensitivity.csv` 的 ±50 ms 是人工相位扰动诊断，不代表真实同步误差。
- `v,w,q,r` 阻力对角项是约 0.202 m/s 的局部固定航速导数；不能外推成任意前进速度下不变的全局阻力矩阵。
- 七个工况同时改变强迫频率和振荡速度幅值；逐频率同相导数会变号，因此全局 `D1/D2` 是 0.1–0.7 Hz 扫频上的经验闭合，不能声称已独立分离固有频率效应与幅值非线性。
- `X_vv/X_ww/X_qq/X_rr` 已按 ITTC 的偶次 surge 模型辨识并单独导出；它们不是当前 Isaac 二次阻力矩阵的 X 行元素。
- 约 0.202 m/s 的直航 surge 总阻力已有锚点，但单一航速不能同时拆分 `D1_uu` 与 `D2_uu`。
- 若要三张无 NaN 且一次/二次项都由试验独立辨识的纯 PMM 全数值矩阵，必须补做多个直航速度的 surge 试验与 roll 独立强迫振荡。

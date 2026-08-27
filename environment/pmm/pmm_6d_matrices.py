# %% [markdown]
# # PMM 6×6 附加质量、一次阻力和二次阻力辨识
#
# 本文件只保留为水池模型试验的历史分析记录；模拟器三张生产矩阵只由 OpenFOAM
# 目标尺度速度包络工况发布，不读取这里的结果。
#
# 本 notebook 的统一模型为
#
# \[
# \tau_h=-M_A\dot\nu-D_1\nu-D_2(|\nu|\odot\nu)-K_\mathrm{nuisance}\xi,
# \]
#
# 其中广义力行顺序为 `[X,Y,Z,K,M,N]`，广义速度列顺序为
# `[u,v,w,p,q,r]`，坐标系是艇体质心处的右手 FLU（前、左、上）。
# `K_nuisance` 只用于隔离支架/静水恢复项，防止它混入附加质量；请求的三个
# 最终矩阵是 `M_A`、`D_1` 和 `D_2`。另外，固定前进速度 PMM 的 surge
# 偶次耦合项单独写成
#
# \[
# X_{\mathrm{even}}=X_{0,c}+X_{vv}v^2+X_{ww}w^2+X_{qq}q^2+X_{rr}r^2,
# \]
#
# 其中 `c` 表示试验 campaign。它们不能写入 `D_2` 的 X 行：`v²` 是偶函数，
# 而 `|v|v` 是奇函数，两者在一个对称振荡周期内的均值完全不同。
#
# 六自由度矩阵是分块量纲矩阵（弧度按无量纲处理）：
#
# | 矩阵 | 力行×线速度列 | 力行×角速度列 | 力矩行×线速度列 | 力矩行×角速度列 |
# |---|---|---|---|---|
# | `M_A` | kg | kg·m | kg·m | kg·m² |
# | `D_1` | N·s/m | N·s | N·s | N·m·s |
# | `D_2` | N·s²/m² | N·s² | N·s²/m | N·m·s² |
#
# 重要限制：现有池架只在池坐标 FL 平面内运动。normal/pure 装配可激励
# `v、r`，绕 F 轴顺时针（右手 `+90°`）安装后的 vertical 装配可激励
# `w、q`。按用户确认的台架辨识范围，三张 6×6 矩阵只发布直接对角项
# `(v,Y)、(w,Z)、(q,M)、(r,N)`。`u、p` 未独立激励，所有非对角项也不从
# 六分力交叉响应或互易性补齐；这些位置统一保留 `NaN`，而不是虚构成 0。

# %% [markdown]
# ## 原实验脚本的处理逻辑与本 notebook 的修正
#
# `Downloads/jn` 中的脚本给出了原始处理链：
#
# 1. `gather_data.py` 从运动控制器记录中抽取位移；
# 2. `gather_smooth_sway_yaw.py` 用正弦拟合运动，避免直接差分放大噪声；
# 3. `sensor_data_filter_frequency.py` 处理六分力；
# 4. `numerical_sway.py` / `numerical_yaw.py` 把六分力减去刚体项，再回归导数、
#    一次和二次速度项。
#
# 本 notebook 保留这条物理链，但直接从原始 CSV 开始并处理以下问题：
#
# - `pure_sway1/2/3` 是同一套 pure-sway 工况的 3 次重复，不是三种振幅；其余
#   `pure_yaw1/2/3`、`vertical_sway1/2/3`、`vertical_yaw1/2/3` 同理。文件夹内
#   同名 `gather_*` 与 `sensor_*` 是一一配对的电机/六分力记录；
# - 每套重复内的 7 个编号是 0.1--0.7 Hz 的 7 个强迫频率。sway 位移幅值约
#   0.101 m、yaw 角幅值约 10° 基本不变，因此改变的是横荡/升沉速度或角速度幅值；
#   直航前进速度不是这 7 组的自变量，84 个试次的平均值约为 0.202 m/s。由于
#   强迫频率和振荡速度幅值在现有设计中同步变化，把全局回归解释为常数 `D_1/D_2`
#   仍隐含“0.1--0.7 Hz 内导数不另随频率变化”的假设；逐频率诊断保留该假设的偏差；
# - raw `gather` 实际是 16 列；关闭的 U 轴组三列为 0，艏向编码器在零基第 10 列；
# - 位移的两次 `/1000` 等价于 raw count `/1e6` m；角度换算等价于
#   raw count `/18300` degree；
# - 用户确认：`gather` 是电机运动控制器记录，采样率 100 Hz；`sensor_` 是
#   六分力传感器记录，采样率 500 Hz。因此每 5 个 `sensor_` 点块平均后与
#   `gather` 同为 100 Hz。原传感器处理脚本中的 1000 Hz 常量不适用于本批数据；
# - 原 `gather_smooth_sway_yaw.py` 对运动硬裁剪 2--10 s，原
#   `numerical_sway.py` / `numerical_yaw.py` 对载荷回归硬裁剪 2.5--9.5 s；
#   本 notebook 原样沿用这两个时间窗。约 0--2 s 的启动/前进加速和 9.5 s 后的
#   减速、停止及横向/角度回中均不进入载荷回归；返回 F 轴起点发生在记录之后，
#   也不属于任何拟合样本；
# - 台架原始方向由用户确认为 `(+X前,+Y左,+Z下)`。这是左手的轴标签约定，
#   读入后立即用 `diag(1,1,-1)` 转成项目的右手 FLU，再进行旋转、叉乘和回归；
# - 艏向编码器绕台架 `+Z下` 记号，因此转到 FLU `+Z上` 时取负号。这个符号
#   同时使 pure-yaw 的残余横荡速度接近 0；反号会产生明显的伪横荡；
# - pure 力通道沿用原脚本的 `Y=-FY`。vertical 原始力通道的惯性相位与 pure
#   相反，因此 vertical 力通道单独反转极性。这两个力通道极性与上述轴系转换分开；
#   原脚本的 `N=TZ` 在台架轴下无额外通道极性，转到 FLU 后由 z 轴映射自然变成 `N=-TZ`；
# - 运动用 1--3 次 Fourier 项拟合并解析求导；载荷只投影到模型需要的 1、3
#   次谐波，再跨 7 个频率和 3 次重复做 Huber 回归；
# - 原 `numerical_sway.py` / `numerical_yaw.py` 同时回归了 `Y/N` 交叉响应，但按本次
#   确认的范围，当前矩阵只使用每个强迫自由度的同向力/力矩，不发布交叉导数；
# - 原脚本把映射后的测力计值当作“支架施加给艇体的力”并直接减去刚体项；但本批
#   数据在四个主通道中都显示 `measured/qdot<0`，即记录的是艇体施加给测力架的反力
#   `R=-F_support`。由 `m*a=F_support+tau_h` 得
#   `tau_h=R+tau_rigid`，所以这里修正为相加，而不是再次减去刚体惯性。使用用户给出
#   的两套质量、质心和质心惯量计算完整 Newton--Euler 刚体项，并把六分力矩从
#   坐标系原点平移到质心；
# - 依据 ITTC 7.5-02-06-07，先用 Fourier 模型把残缺稳定窗还原为全周期均值，
#   再把 surge 周期均值对 `v²/w²/q²/r²` 的周期均值回归，得到
#   `X_vv/X_ww/X_qq/X_rr`；每个 campaign 保留自己的零振幅截距，避免把
#   装配/清零差异误当成耦合导数，并用独立的 2 倍频估计检查无记忆二次模型。
# - 用完整周期 Fourier 基频系数计算流体力相对速度的同相导数。它不受 2.5--9.5 s
#   非整周期窗的惯性边界能量污染；该量用于检查频率/振幅趋势，不把固定航速操纵
#   导数误判成单自由度静水阻力，也不据此裁剪系数或投影到正定矩阵。
#
# 原始字段（零基索引）：
#
# | 文件 | 字段 | 列 | 换算/单位 |
# |---|---|---:|---|
# | gather | F 位置 | 1 | `/1e6` m |
# | gather | L 位置 | 4 | `/1e6` m |
# | gather | 已关闭 U 轴 | 7 | 全零，不使用 |
# | gather | 绕台架 `+Z下` 的角位置 | 10 | `(value-value[0])/18300` degree |
# | sensor_ | `TX,TY,TZ` | 0:3 | N·m |
# | sensor_ | `FX,FY,FZ` | 3:6 | N |

# %% [markdown]
# ## 伪代码
#
# ```text
# 设置台架 FLD 轴标签到 FLU 的映射、采样率、单位、时间窗、通道极性和 vertical 的 Rx(+90°)
# 读取 pure/vertical × sway/yaw × 3 repeats × 7 frequencies 的 84 组记录
#
# 对每一组试次：
#     读取电机 gather 与六分力 sensor_
#     从 gather[:,1], gather[:,4], gather[:,10] 取得 F、L、yaw
#     在名义频率附近估计实测频率
#     在原脚本的 2--10 s 硬裁剪窗内，用含漂移项的 3 阶 Fourier 模型拟合运动
#     并解析求速度/加速度
#     将 500 Hz 六分力每 5 点平均到 100 Hz，并硬裁剪到 2.5--9.5 s
#     pure: R_HB = I；vertical: R_HB = Rx(+90°)
#     先将原始力/力矩从台架 `(前,左,下)` 轴标签转成池 FLU
#     再将运动、力和力矩从池 FLU 转到艇体 FLU
#     将运动原点速度/加速度与力矩平移到用户给出的质心
#     用 m、I_G 计算刚体 Newton--Euler 左端项 tau_rigid
#     测力计为艇体对支架的反力 R=-F_support
#     tau_h = R + tau_rigid
#     去掉常量和线性漂移，投影到实测频率的 1、3 次谐波
#
# 对每个已激励直接自由度 j ∈ {v,w,q,r}：
#     联合 21 个试次，Huber 回归
#       tau_h[j] = -M_A[j,j] qdot
#                  -D_1[j,j] q
#                  -D_2[j,j] |q|q
#                  -K_nuisance[j,j] displacement
#     只把直接系数写入矩阵对角位置 [j,j]
#
# 对 surge 输出 X：
#     对每个 campaign c ∈ {sway, heave, pitch, yaw}：
#         用含漂移项的 Fourier 拟合取得完整周期 mean(X_h) 与 mean(nu_j²)
#         Huber 回归 cycle_mean(X_h) = X0_c + X_jj * cycle_mean(nu_j²)
#         独立用 X_h 与 nu_j² 的 2 倍频系数再估一次 X_jj，作为模型一致性审计
#         留一整个重复组重拟合，报告系数范围
#     只用 normal/pure 的 sway、yaw 截距估计 U_ref 下直航阻力锚点
#     分别给出“一次项独占”和“二次项独占”的互斥闭合值，不把两者相加
#
# 对固定 U0 的直接通道：
#     对 nu_j 与 tau_h[j] 分别拟合完整周期 Fourier 基频向量
#     force_per_velocity = dot(tau_1, nu_1) / dot(nu_1, nu_1)
#     equivalent_D = -force_per_velocity
#     报告随频率/振幅的趋势；不把单通道符号当作全六自由度被动性判据
#
# 创建三个 shape=(6,6) 的矩阵；只填 v、w、q、r 对角项，其余保持 NaN
# 输出 CSV、JSON、逐试次、逐频率和 ±50 ms 人工相位扰动诊断
# ```

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=180)


def resolve_data_root() -> tuple[Path, Path]:
    """Support launching Jupyter from the repository root or from environment/pmm."""
    cwd = Path.cwd().resolve()
    repository_candidate = cwd / "environment" / "pmm"
    if (repository_candidate / "pure_sway1").is_dir():
        return cwd, repository_candidate
    if (cwd / "pure_sway1").is_dir():
        return cwd.parent.parent, cwd
    # Keep the unresolved conventional path so a missing input raises at load time.
    return cwd, repository_candidate


REPO_ROOT, DATA_ROOT = resolve_data_root()
OUTPUT_DIR = DATA_ROOT / "pmm_6d_results"

# User-confirmed acquisition definitions: gather records the motor controller at
# 100 Hz; sensor_ records the six-axis force/torque sensor at 500 Hz.
GATHER_HZ = 100.0
SENSOR_HZ = 500.0
SENSOR_BLOCK = 5
MOTION_FIT_START_S = 2.0
MOTION_FIT_END_S = 10.0
LOAD_FIT_START_S = 2.5
LOAD_FIT_END_S = 9.5
POSITION_COUNTS_PER_M = 1_000_000.0
ANGLE_COUNTS_PER_DEG = 18_300.0
YAW_ENCODER_SIGN_TO_H = -1.0
NOMINAL_FREQUENCY_STEP_HZ = 0.1
MOTION_HARMONICS = 3
LOAD_HARMONICS = (1, 3)
SENSOR_TO_MOTION_SHIFT_S = 0.0
SYNCHRONIZATION_STATUS = "user_confirmed_hardware_synchronized"
TIMING_SENSITIVITY_SHIFTS_S = (-0.05, 0.0, 0.05)

# Raw sensor columns are [TX, TY, TZ, FX, FY, FZ] in the user-confirmed bench
# axis-label convention A=(X_forward,Y_left,Z_down).  A is left-handed, so it is
# never used as a computational rigid-body frame.  Components are mapped into
# the right-handed pool frame H=(F,L,U) before any rotation or cross product.
BENCH_COMPONENTS_TO_POOL_FLU = np.diag([1.0, 1.0, -1.0])

# These are channel/reaction polarities in the raw bench convention, separate
# from the geometric axis map above.  The original pure scripts use Y=-FY and
# N=TZ.  Consequently pure force has polarity -1, while moment has no extra
# polarity: its required N=-TZ in FLU comes from the bench z-down conversion.
# The vertical force campaign has the opposite raw polarity; its moment channels
# use the same raw convention as pure.
SENSOR_FORCE_POLARITY_IN_BENCH = {"pure": -1.0, "vertical": 1.0}
SENSOR_MOMENT_POLARITY_IN_BENCH = {"pure": 1.0, "vertical": 1.0}

DOF_LABELS = ("u", "v", "w", "p", "q", "r")
WRENCH_LABELS = ("X", "Y", "Z", "K", "M", "N")
EXCITED_INDICES = (1, 2, 4, 5)
UNEXCITED_INDICES = (0, 3)

print("repository root:", REPO_ROOT)
print("PMM data root:", DATA_ROOT)

# %% [markdown]
# ## 坐标、vertical 安装和质量属性
#
# 原始台架轴标签为 `A=(X前,Y左,Z下)`，输入行向量先按
# `v_H_row=v_A_row @ diag(1,1,-1)` 转成右手池坐标 `H=(F,L,U)`。
# 设正常艇体坐标为 `B=(u,v,w)`。`R_HB` 的列是艇体
# 基向量在池坐标中的表示，所以列向量满足 `v_H=R_HB v_B`，本代码使用行历史时
# 写成 `v_B_row=v_H_row @ R_HB`。
#
# - pure：`R_HB=I`；SolidWorks 空间方向是 `+Z前、+Y左、+X下`；
# - vertical：艇体沿 `+F` 顺时针 90°，在右手 FLU 中就是主动
#   `Rx(+90°)`；此时 SolidWorks 空间方向变为 `+Z前、+Y上、+X左`。
#
# SolidWorks 坐标轴附着于艇体，因此两次装配转回“正常艇体 FLU”时都使用
# `v_B=[S_Z,S_Y,-S_X]`。vertical 的空间轴变化由单独的 `R_HB` 表示，不能把
# 这两个变换重复应用。下面同时检查 `R_HB_vertical @ Q_BS == Q_HS_vertical`。

# %%
def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z_history(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    result = np.zeros((len(angle), 3, 3))
    result[:, 0, 0] = c
    result[:, 0, 1] = -s
    result[:, 1, 0] = s
    result[:, 1, 1] = c
    result[:, 2, 2] = 1.0
    return result


Q_BS = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
Q_HS_PURE = Q_BS.copy()
Q_HS_VERTICAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
R_HB_PURE = np.eye(3)
R_HB_VERTICAL = rotation_x(math.pi / 2.0)


def solidworks_positive_products_to_tensor(values_kg_mm2: list[list[float]]) -> np.ndarray:
    """SolidWorks positive-product notation -> standard inertia tensor, kg mm² -> kg m²."""
    matrix = np.asarray(values_kg_mm2, dtype=float) * 1.0e-6
    matrix[~np.eye(3, dtype=bool)] *= -1.0
    return matrix


@dataclass(frozen=True)
class RigidProperties:
    name: str
    mass_kg: float
    material_volume_m3: float
    surface_area_m2: float
    r_OG_B_m: np.ndarray
    inertia_G_B_kg_m2: np.ndarray
    inertia_O_B_reported_kg_m2: np.ndarray
    parallel_axis_residual_kg_m2: np.ndarray


def make_rigid_properties(
    name: str,
    mass_kg: float,
    volume_mm3: float,
    area_mm2: float,
    com_S_mm: list[float],
    inertia_G_S_positive: list[list[float]],
    inertia_O_S_positive: list[list[float]],
) -> RigidProperties:
    r_OG_B = Q_BS @ (np.asarray(com_S_mm, dtype=float) * 1.0e-3)
    inertia_G_B = Q_BS @ solidworks_positive_products_to_tensor(inertia_G_S_positive) @ Q_BS.T
    inertia_O_B = Q_BS @ solidworks_positive_products_to_tensor(inertia_O_S_positive) @ Q_BS.T
    parallel_axis = mass_kg * (
        float(r_OG_B @ r_OG_B) * np.eye(3) - np.outer(r_OG_B, r_OG_B)
    )
    return RigidProperties(
        name=name,
        mass_kg=mass_kg,
        material_volume_m3=volume_mm3 * 1.0e-9,
        surface_area_m2=area_mm2 * 1.0e-6,
        r_OG_B_m=r_OG_B,
        inertia_G_B_kg_m2=inertia_G_B,
        inertia_O_B_reported_kg_m2=inertia_O_B,
        parallel_axis_residual_kg_m2=inertia_O_B - inertia_G_B - parallel_axis,
    )


PURE_RIGID = make_rigid_properties(
    "pure",
    5.123,
    9_047_323.744,
    877_335.930,
    [-6.066, -0.005, 0.484],
    [
        [42_966.430, -0.005, -17.757],
        [-0.005, 37_325.524, -1.784],
        [-17.757, -1.784, 20_734.482],
    ],
    [
        [42_967.630, 0.150, -32.799],
        [0.150, 37_515.239, -1.797],
        [-32.799, -1.797, 20_922.997],
    ],
)

VERTICAL_RIGID = make_rigid_properties(
    "vertical",
    5.123,
    9_558_845.734,
    955_242.593,
    [-11.031, 2.809, 0.536],
    [
        [43_744.974, 157.177, -27.157],
        [157.177, 40_537.674, -7.710],
        [-27.157, -7.710, 27_409.119],
    ],
    [
        [43_786.864, -1.562, -57.461],
        [-1.562, 41_162.598, 0.006],
        [-57.461, 0.006, 28_072.987],
    ],
)

print("vertical axis-map max error:", np.max(np.abs(R_HB_VERTICAL @ Q_BS - Q_HS_VERTICAL)))
for props in (PURE_RIGID, VERTICAL_RIGID):
    print(f"\n{props.name}: r_OG_B [m] =", props.r_OG_B_m)
    print("I_G_B [kg m²] =\n", props.inertia_G_B_kg_m2)
    print("parallel-axis max residual [kg m²] =", np.max(np.abs(props.parallel_axis_residual_kg_m2)))

# %% [markdown]
# 平行轴残差仅来自报告的小数舍入；这个检查也验证了 SolidWorks “正张量记数法”
# 的非对角项必须取负后才能作为标准惯性张量使用。报告里的“体积”被保留为元数据，
# 但不擅自当成排水体积，因此不从它构造浮力/恢复力。

# %%
@dataclass(frozen=True)
class TrialPlan:
    dof_name: str
    dof_index: int
    family: str
    raw_kind: str
    repeat: int
    file_id: int
    nominal_frequency_hz: float
    gather_path: Path
    sensor_path: Path
    mount: str


DOF_SPECS = {
    "sway": dict(index=1, family="pure_sway", raw_kind="sway", mount="pure"),
    "heave": dict(index=2, family="vertical_sway", raw_kind="sway", mount="vertical"),
    "pitch": dict(index=4, family="vertical_yaw", raw_kind="yaw", mount="vertical"),
    "yaw": dict(index=5, family="pure_yaw", raw_kind="yaw", mount="pure"),
}


def build_manifest() -> list[TrialPlan]:
    suffix = {1: "ang0", 2: "ang60", 3: "ang120"}
    plans: list[TrialPlan] = []
    for dof_name, spec in DOF_SPECS.items():
        ids = range(8, 15) if spec["raw_kind"] == "sway" else range(22, 29)
        base = 7 if spec["raw_kind"] == "sway" else 21
        for repeat in range(1, 4):
            folder = DATA_ROOT / f"{spec['family']}{repeat}"
            for file_id in ids:
                stem = str(file_id) if spec["raw_kind"] == "sway" else f"{file_id}_{suffix[repeat]}"
                plans.append(
                    TrialPlan(
                        dof_name=dof_name,
                        dof_index=spec["index"],
                        family=spec["family"],
                        raw_kind=spec["raw_kind"],
                        repeat=repeat,
                        file_id=file_id,
                        nominal_frequency_hz=round(
                            NOMINAL_FREQUENCY_STEP_HZ * (file_id - base), 10
                        ),
                        gather_path=folder / f"gather_{stem}.csv",
                        sensor_path=folder / f"sensor_{stem}.csv",
                        mount=spec["mount"],
                    )
                )
    return plans


MANIFEST = build_manifest()
print("planned raw trial pairs:", len(MANIFEST))

# %%
@dataclass(frozen=True)
class FourierFit:
    frequency_hz: float
    harmonics: int
    center_s: float
    coefficients: np.ndarray
    r2: float

    def evaluate(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = np.asarray(time_s, dtype=float)
        omega = 2.0 * np.pi * self.frequency_hz
        value = self.coefficients[0] + self.coefficients[1] * (time - self.center_s)
        velocity = np.full_like(time, self.coefficients[1])
        acceleration = np.zeros_like(time)
        offset = 2
        for harmonic in range(1, self.harmonics + 1):
            a_sin, a_cos = self.coefficients[offset : offset + 2]
            phase = harmonic * omega * time
            h_omega = harmonic * omega
            value += a_sin * np.sin(phase) + a_cos * np.cos(phase)
            velocity += h_omega * (a_sin * np.cos(phase) - a_cos * np.sin(phase))
            acceleration -= h_omega**2 * (a_sin * np.sin(phase) + a_cos * np.cos(phase))
            offset += 2
        return value, velocity, acceleration


def fourier_design(time_s: np.ndarray, frequency_hz: float, harmonics: int) -> tuple[np.ndarray, float]:
    center = float(np.mean(time_s))
    columns = [np.ones_like(time_s), time_s - center]
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * np.pi * harmonic * frequency_hz * time_s
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns), center


def fit_fourier(time_s: np.ndarray, values: np.ndarray, frequency_hz: float, harmonics: int) -> FourierFit:
    design, center = fourier_design(time_s, frequency_hz, harmonics)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    prediction = design @ coefficients
    denominator = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - float(np.sum((values - prediction) ** 2)) / denominator
    return FourierFit(frequency_hz, harmonics, center, coefficients, r2)


def estimate_frequency(time_s: np.ndarray, values: np.ndarray, nominal_hz: float) -> float:
    low, high = 0.96 * nominal_hz, 1.02 * nominal_hz
    best = nominal_hz
    for _ in range(4):
        grid = np.linspace(low, high, 61)
        scores = []
        for frequency in grid:
            design, _ = fourier_design(time_s, float(frequency), MOTION_HARMONICS)
            residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
            scores.append(float(residual @ residual))
        index = int(np.argmin(scores))
        best = float(grid[index])
        step = float(grid[1] - grid[0])
        low = max(0.96 * nominal_hz, best - step)
        high = min(1.02 * nominal_hz, best + step)
    return best


def block_average_sensor(sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = len(sensor) // SENSOR_BLOCK
    averaged = sensor[: blocks * SENSOR_BLOCK].reshape(blocks, SENSOR_BLOCK, 6).mean(axis=1)
    centers = np.arange(blocks) * SENSOR_BLOCK + (SENSOR_BLOCK - 1.0) / 2.0
    return centers / SENSOR_HZ, averaged


def rigid_body_wrench(
    mass_kg: float,
    inertia_G_B: np.ndarray,
    linear_velocity_B: np.ndarray,
    linear_derivative_B: np.ndarray,
    angular_velocity_B: np.ndarray,
    angular_acceleration_B: np.ndarray,
) -> np.ndarray:
    force = mass_kg * (linear_derivative_B + np.cross(angular_velocity_B, linear_velocity_B))
    angular_momentum = angular_velocity_B @ inertia_G_B.T
    moment = angular_acceleration_B @ inertia_G_B.T + np.cross(angular_velocity_B, angular_momentum)
    return np.column_stack((force, moment))


@dataclass
class Trial:
    plan: TrialPlan
    frequency_hz: float
    time_s: np.ndarray
    coordinate: np.ndarray
    nu: np.ndarray
    nu_dot: np.ndarray
    hydro_wrench: np.ndarray
    measured_wrench: np.ndarray
    rigid_wrench: np.ndarray
    motion_r2: float
    forward_r2: float
    gather_rows: int
    sensor_rows: int


def build_trial(plan: TrialPlan, sensor_to_motion_shift_s: float = SENSOR_TO_MOTION_SHIFT_S) -> Trial:
    gather = np.loadtxt(plan.gather_path)
    sensor = np.loadtxt(plan.sensor_path, delimiter=",")

    gather_time = np.arange(len(gather), dtype=float) / GATHER_HZ
    motion_mask = (gather_time >= MOTION_FIT_START_S) & (gather_time <= MOTION_FIT_END_S)
    forward_position = gather[:, 1] / POSITION_COUNTS_PER_M
    lateral_position = gather[:, 4] / POSITION_COUNTS_PER_M
    yaw_angle = np.deg2rad(
        YAW_ENCODER_SIGN_TO_H * (gather[:, 10] - gather[0, 10]) / ANGLE_COUNTS_PER_DEG
    )
    frequency_signal = lateral_position if plan.raw_kind == "sway" else yaw_angle
    frequency = estimate_frequency(
        gather_time[motion_mask], frequency_signal[motion_mask], plan.nominal_frequency_hz
    )
    fit_forward = fit_fourier(
        gather_time[motion_mask], forward_position[motion_mask], frequency, MOTION_HARMONICS
    )
    fit_lateral = fit_fourier(
        gather_time[motion_mask], lateral_position[motion_mask], frequency, MOTION_HARMONICS
    )
    fit_angle = (
        fit_fourier(gather_time[motion_mask], yaw_angle[motion_mask], frequency, MOTION_HARMONICS)
        if plan.raw_kind == "yaw"
        else None
    )

    sensor_time, sensor_average = block_average_sensor(sensor)
    load_mask = (sensor_time >= LOAD_FIT_START_S) & (sensor_time <= LOAD_FIT_END_S)
    sensor_time = sensor_time[load_mask]
    sensor_average = sensor_average[load_mask]
    motion_time = sensor_time + sensor_to_motion_shift_s

    _, dx, ddx = fit_forward.evaluate(motion_time)
    lateral_value, dy, ddy = fit_lateral.evaluate(motion_time)
    if fit_angle is None:
        psi = np.zeros_like(motion_time)
        yaw_rate = np.zeros_like(motion_time)
        yaw_acceleration = np.zeros_like(motion_time)
        motion_r2 = fit_lateral.r2
    else:
        psi, yaw_rate, yaw_acceleration = fit_angle.evaluate(motion_time)
        motion_r2 = fit_angle.r2

    # First rotate inertial F/L motion into the instantaneous yaw frame.  The
    # derivative is the derivative of components in that rotating frame.
    c, s = np.cos(psi), np.sin(psi)
    velocity_yaw_frame = np.column_stack((c * dx + s * dy, -s * dx + c * dy, np.zeros_like(dx)))
    derivative_yaw_frame = np.column_stack(
        (
            -s * yaw_rate * dx + c * ddx + c * yaw_rate * dy + s * ddy,
            -c * yaw_rate * dx - s * ddx - s * yaw_rate * dy + c * ddy,
            np.zeros_like(dx),
        )
    )
    omega_yaw_frame = np.column_stack((np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_rate))
    alpha_yaw_frame = np.column_stack(
        (np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_acceleration)
    )

    base_rotation = R_HB_PURE if plan.mount == "pure" else R_HB_VERTICAL
    rigid = PURE_RIGID if plan.mount == "pure" else VERTICAL_RIGID
    velocity_O_B = velocity_yaw_frame @ base_rotation
    derivative_O_B = derivative_yaw_frame @ base_rotation
    omega_B = omega_yaw_frame @ base_rotation
    alpha_B = alpha_yaw_frame @ base_rotation
    velocity_G_B = velocity_O_B + np.cross(omega_B, rigid.r_OG_B_m)
    derivative_G_B = derivative_O_B + np.cross(alpha_B, rigid.r_OG_B_m)
    nu = np.column_stack((velocity_G_B, omega_B))
    nu_dot = np.column_stack((derivative_G_B, alpha_B))

    if plan.raw_kind == "sway":
        coordinate_B = np.column_stack(
            (np.zeros_like(lateral_value), lateral_value, np.zeros_like(lateral_value))
        ) @ base_rotation
        coordinate = coordinate_B[:, plan.dof_index]
    else:
        angle_B = np.column_stack((np.zeros_like(psi), np.zeros_like(psi), psi)) @ base_rotation
        coordinate = angle_B[:, plan.dof_index - 3]

    # Keep the raw bench-axis conversion separate from acquisition/reaction
    # polarity.  In particular, raw +FZ/+TZ point down and therefore become
    # -FZ/-TZ in the right-handed FLU pool frame.
    force_A = SENSOR_FORCE_POLARITY_IN_BENCH[plan.mount] * sensor_average[:, 3:6]
    moment_A_at_O = SENSOR_MOMENT_POLARITY_IN_BENCH[plan.mount] * sensor_average[:, 0:3]
    force_H = force_A @ BENCH_COMPONENTS_TO_POOL_FLU
    moment_H_at_O = moment_A_at_O @ BENCH_COMPONENTS_TO_POOL_FLU
    R_z = rotation_z_history(psi)
    R_HB = np.einsum("nij,jk->nik", R_z, base_rotation)
    force_B = np.einsum("ni,nij->nj", force_H, R_HB)
    moment_B_at_O = np.einsum("ni,nij->nj", moment_H_at_O, R_HB)
    moment_B_at_G = moment_B_at_O - np.cross(rigid.r_OG_B_m, force_B)
    measured = np.column_stack((force_B, moment_B_at_G))
    rigid_load = rigid_body_wrench(
        rigid.mass_kg,
        rigid.inertia_G_B_kg_m2,
        velocity_G_B,
        derivative_G_B,
        omega_B,
        alpha_B,
    )
    # The balance records the body-on-support reaction R=-F_support.  Newton--Euler
    # gives tau_rigid = F_support + tau_h, hence tau_h = R + tau_rigid.  The
    # original numerical scripts subtracted the rigid term and therefore counted
    # the rigid inertia twice in the apparent added mass.
    hydro = measured + rigid_load
    return Trial(
        plan=plan,
        frequency_hz=frequency,
        time_s=sensor_time,
        coordinate=coordinate,
        nu=nu,
        nu_dot=nu_dot,
        hydro_wrench=hydro,
        measured_wrench=measured,
        rigid_wrench=rigid_load,
        motion_r2=motion_r2,
        forward_r2=fit_forward.r2,
        gather_rows=len(gather),
        sensor_rows=len(sensor),
    )


TRIALS = [build_trial(plan) for plan in MANIFEST]
print("built trials:", len(TRIALS))

# %%
def residualize(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    centered = time_s - np.mean(time_s)
    nuisance = np.column_stack((np.ones_like(centered), centered))
    return values - nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]


def sampling_and_sign_audit(trials: list[Trial]) -> tuple[dict[str, float], dict[str, float]]:
    first = trials[0]
    duration_gather = (first.gather_rows - 1) / GATHER_HZ
    duration_sensor = (first.sensor_rows - 1) / SENSOR_HZ
    trial_mean_towing_speeds = np.asarray(
        [np.mean(trial.nu[:, 0]) for trial in trials], dtype=float
    )
    sampling = {
        "gather_duration_s_example": duration_gather,
        "sensor_duration_s_example": duration_sensor,
        "sensor_to_gather_row_ratio_example": first.sensor_rows / first.gather_rows,
        "mean_towing_speed_m_s": float(np.mean(np.concatenate([trial.nu[:, 0] for trial in trials]))),
        "trial_mean_towing_speed_min_m_s": float(np.min(trial_mean_towing_speeds)),
        "trial_mean_towing_speed_max_m_s": float(np.max(trial_mean_towing_speeds)),
        "trial_mean_towing_speed_std_m_s": float(np.std(trial_mean_towing_speeds)),
    }
    slopes: dict[str, float] = {}
    for dof_name in DOF_SPECS:
        values = []
        for trial in trials:
            if trial.plan.dof_name != dof_name:
                continue
            j = trial.plan.dof_index
            acceleration = residualize(trial.time_s, trial.nu_dot[:, j])
            measured = residualize(trial.time_s, trial.measured_wrench[:, j])
            values.append(float((acceleration @ measured) / (acceleration @ acceleration)))
        slopes[dof_name] = float(np.median(values))
    return sampling, slopes


SAMPLING_AUDIT, SIGN_AUDIT = sampling_and_sign_audit(TRIALS)
print("sampling audit:", SAMPLING_AUDIT)
print("median mapped measured/qdot slopes (selected polarity):", SIGN_AUDIT)

yaw_trials = [trial for trial in TRIALS if trial.plan.dof_name == "yaw"]
yaw_purity = float(
    np.median(
        [
            (0.5 * np.ptp(trial.nu[:, 1])) / (0.5 * np.ptp(trial.nu[:, 5]))
            for trial in yaw_trials
        ]
    )
)
print("pure-yaw median sway/rate amplitude ratio [m]:", yaw_purity)

# %% [markdown]
# `median mapped measured/qdot slopes` 在四类试验中应保持同一惯性相位。这里不把
# 系数裁剪为正，也不做正定投影；这个审计只是把实际使用的传感器极性显式暴露出来。
# 用户确认 gather 与 sensor_ 为硬同步，因此主结果的零时差是实验事实。末尾的
# ±50 ms 扫描只作为人工相位扰动诊断，不代表实际触发时差的不确定性。

# %%
def harmonic_projection(trial: Trial) -> tuple[np.ndarray, np.ndarray]:
    j = trial.plan.dof_index
    q = trial.nu[:, j]
    q_dot = trial.nu_dot[:, j]
    design = np.column_stack((-q_dot, -q, -q * np.abs(q), -trial.coordinate))
    design = residualize(trial.time_s, design)
    target = residualize(trial.time_s, trial.hydro_wrench)
    basis_columns = []
    for harmonic in LOAD_HARMONICS:
        phase = 2.0 * np.pi * harmonic * trial.frequency_hz * trial.time_s
        basis_columns.extend((np.sin(phase), np.cos(phase)))
    basis = np.column_stack(basis_columns)
    return (
        np.linalg.lstsq(basis, design, rcond=None)[0],
        np.linalg.lstsq(basis, target, rcond=None)[0],
    )


def huber_fit(X: np.ndarray, y: np.ndarray, tuning: float = 1.5) -> np.ndarray:
    scales = np.linalg.norm(X, axis=0)
    X_scaled = X / scales
    beta_scaled = np.linalg.lstsq(X_scaled, y, rcond=None)[0]
    for _ in range(40):
        residual = y - X_scaled @ beta_scaled
        center = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - center))
        cutoff = tuning * sigma
        weights = np.ones_like(residual)
        large = np.abs(residual) > cutoff
        weights[large] = cutoff / np.abs(residual[large])
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(X_scaled * root[:, None], y * root, rcond=None)[0]
        if np.linalg.norm(updated - beta_scaled) <= 1.0e-10 * (1.0 + np.linalg.norm(beta_scaled)):
            beta_scaled = updated
            break
        beta_scaled = updated
    return beta_scaled / scales


@dataclass
class DofFit:
    dof_name: str
    dof_index: int
    X: np.ndarray
    Y: np.ndarray
    coefficients: np.ndarray
    prediction: np.ndarray
    r2: float
    condition_number: float


def fit_one_dof(dof_name: str, trials: list[Trial]) -> DofFit:
    selected = [trial for trial in trials if trial.plan.dof_name == dof_name]
    compressed = [harmonic_projection(trial) for trial in selected]
    X = np.vstack([item[0] for item in compressed])
    dof_index = DOF_SPECS[dof_name]["index"]
    Y = np.vstack([item[1] for item in compressed])[:, dof_index]
    coefficients = huber_fit(X, Y)
    prediction = X @ coefficients
    denominator = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1.0 - np.sum((Y - prediction) ** 2) / denominator
    standardized = X / np.linalg.norm(X, axis=0)
    return DofFit(
        dof_name=dof_name,
        dof_index=dof_index,
        X=X,
        Y=Y,
        coefficients=coefficients,
        prediction=prediction,
        r2=float(r2),
        condition_number=float(np.linalg.cond(standardized)),
    )


FITS = {name: fit_one_dof(name, TRIALS) for name in DOF_SPECS}
M_ADDED = np.full((6, 6), np.nan)
D_LINEAR = np.full((6, 6), np.nan)
D_QUADRATIC = np.full((6, 6), np.nan)
K_NUISANCE = np.full((6, 6), np.nan)
for fit in FITS.values():
    j = fit.dof_index
    M_ADDED[j, j] = fit.coefficients[0]
    D_LINEAR[j, j] = fit.coefficients[1]
    D_QUADRATIC[j, j] = fit.coefficients[2]
    K_NUISANCE[j, j] = fit.coefficients[3]

for name, fit in FITS.items():
    print(f"\n{name}: standardized condition={fit.condition_number:.3f}")
    print(f"direct-channel R² = {fit.r2:.6f}")

# %% [markdown]
# ## 固定航速 surge 偶次耦合项与基频同相诊断
#
# ITTC 的 underwater-vehicle captive-test 指南明确把平移试验的 surge 项列为
# `X_vv`、`X_ww`，把旋转试验的 surge 项列为 `X_rr`、`X_qq`。这些项在
# 正负半周期都增加阻力，所以先用 Fourier 模型取得完整周期均值，再跨振幅辨识。
# 每个 campaign 单独保留截距；把四套装配/清零截距强迫成同一个值会把零点差异
# 转嫁给斜率。独立的 2 倍频导数用于检查同一个无记忆二次项能否同时解释
# DC 与 2f 响应。
#
# - ITTC 7.5-02-06-07 (2024):
#   https://ittc.info/media/11876/75-02-06-07.pdf
# - Fossen marine craft model（六自由度惯性、科氏和阻力项的统一记号）:
#   https://www.fossen.biz/html/marineCraftModel.html
# - Fossen MSS `forceSurgeDamping`（单一稳态速度只能锚定总 surge 阻力；若采用
#   二次独占闭合，可用 `D_uu=R(U)/U²`）:
#   https://github.com/cybergalactic/MSS/blob/master/LIBRARY/modeling/forceSurgeDamping.m

# %%
SURGE_TERM_BY_DOF = {
    "sway": "X_vv",
    "heave": "X_ww",
    "pitch": "X_qq",
    "yaw": "X_rr",
}
SURGE_TERM_UNITS_BY_DOF = {
    "sway": "N*s^2/m^2",
    "heave": "N*s^2/m^2",
    "pitch": "N*s^2",
    "yaw": "N*s^2",
}


@dataclass
class SurgeEvenFit:
    dof_name: str
    term: str
    units: str
    intercept_force_N: float
    force_derivative: float
    resistance_magnitude: float
    r2: float
    rmse_N: float
    second_harmonic_force_derivative: float
    second_harmonic_r2: float
    leave_one_repeat_force_derivatives: np.ndarray
    cycle_mean_square_velocity: np.ndarray
    cycle_mean_surge_force_N: np.ndarray


def fit_surge_even_coupling(dof_name: str, trials: list[Trial]) -> SurgeEvenFit:
    selected = [trial for trial in trials if trial.plan.dof_name == dof_name]
    j = DOF_SPECS[dof_name]["index"]
    cycle_mean_square_velocity = []
    cycle_mean_surge_force = []
    second_harmonic_square_velocity = []
    second_harmonic_surge_force = []
    for trial in selected:
        velocity_fit = fit_fourier(
            trial.time_s, trial.nu[:, j], trial.frequency_hz, MOTION_HARMONICS
        )
        square_velocity_fit = fit_fourier(
            trial.time_s, trial.nu[:, j] ** 2, trial.frequency_hz, MOTION_HARMONICS
        )
        surge_force_fit = fit_fourier(
            trial.time_s, trial.hydro_wrench[:, 0], trial.frequency_hz, MOTION_HARMONICS
        )
        # Parseval over a complete period.  The fitted linear trend is nuisance
        # and is not extrapolated into the cycle mean.
        cycle_mean_square_velocity.append(
            velocity_fit.coefficients[0] ** 2
            + 0.5 * np.sum(velocity_fit.coefficients[2:] ** 2)
        )
        cycle_mean_surge_force.append(surge_force_fit.coefficients[0])
        # coefficient offsets: [constant, trend, 1sin, 1cos, 2sin, 2cos, ...]
        second_harmonic_square_velocity.extend(square_velocity_fit.coefficients[4:6])
        second_harmonic_surge_force.extend(surge_force_fit.coefficients[4:6])
    cycle_mean_square_velocity = np.asarray(cycle_mean_square_velocity)
    cycle_mean_surge_force = np.asarray(cycle_mean_surge_force)
    design = np.column_stack((np.ones(len(selected)), cycle_mean_square_velocity))
    beta = huber_fit(design, cycle_mean_surge_force)
    prediction = design @ beta
    denominator = np.sum((cycle_mean_surge_force - np.mean(cycle_mean_surge_force)) ** 2)
    r2 = 1.0 - np.sum((cycle_mean_surge_force - prediction) ** 2) / denominator
    harmonic_design = np.asarray(second_harmonic_square_velocity)[:, None]
    harmonic_target = np.asarray(second_harmonic_surge_force)
    harmonic_derivative = float(huber_fit(harmonic_design, harmonic_target)[0])
    harmonic_prediction = harmonic_design[:, 0] * harmonic_derivative
    harmonic_denominator = np.sum((harmonic_target - np.mean(harmonic_target)) ** 2)
    harmonic_r2 = 1.0 - np.sum((harmonic_target - harmonic_prediction) ** 2) / harmonic_denominator
    leave_one_repeat = []
    for omitted_repeat in (1, 2, 3):
        keep = np.array([trial.plan.repeat != omitted_repeat for trial in selected])
        leave_one_repeat.append(huber_fit(design[keep], cycle_mean_surge_force[keep])[1])
    return SurgeEvenFit(
        dof_name=dof_name,
        term=SURGE_TERM_BY_DOF[dof_name],
        units=SURGE_TERM_UNITS_BY_DOF[dof_name],
        intercept_force_N=float(beta[0]),
        force_derivative=float(beta[1]),
        resistance_magnitude=float(-beta[1]),
        r2=float(r2),
        rmse_N=float(np.sqrt(np.mean((cycle_mean_surge_force - prediction) ** 2))),
        second_harmonic_force_derivative=harmonic_derivative,
        second_harmonic_r2=float(harmonic_r2),
        leave_one_repeat_force_derivatives=np.asarray(leave_one_repeat),
        cycle_mean_square_velocity=cycle_mean_square_velocity,
        cycle_mean_surge_force_N=cycle_mean_surge_force,
    )


SURGE_EVEN_FITS = {
    dof_name: fit_surge_even_coupling(dof_name, TRIALS) for dof_name in DOF_SPECS
}


def fixed_u_frequency_response(trials: list[Trial]) -> list[dict[str, object]]:
    """Full-cycle fundamental in-phase derivative for each fixed-U0 trial.

    The dot product of the sine/cosine coefficient pairs is the complete-cycle
    inner product.  Acceleration is in quadrature with velocity, so it cannot
    create the partial-window boundary-energy artefact of -mean(tau*nu).
    """
    rows: list[dict[str, object]] = []
    for trial in trials:
        j = trial.plan.dof_index
        velocity_fit = fit_fourier(
            trial.time_s, trial.nu[:, j], trial.frequency_hz, MOTION_HARMONICS
        )
        wrench_fit = fit_fourier(
            trial.time_s,
            trial.hydro_wrench[:, j],
            trial.frequency_hz,
            MOTION_HARMONICS,
        )
        velocity_fundamental = velocity_fit.coefficients[2:4]
        wrench_fundamental = wrench_fit.coefficients[2:4]
        velocity_norm_squared = float(velocity_fundamental @ velocity_fundamental)
        fluid_force_per_velocity = float(
            (wrench_fundamental @ velocity_fundamental) / velocity_norm_squared
        )
        rows.append(
            {
                "dof": trial.plan.dof_name,
                "repeat": trial.plan.repeat,
                "nominal_frequency_hz": trial.plan.nominal_frequency_hz,
                "estimated_frequency_hz": trial.frequency_hz,
                "mean_forward_speed_m_s": float(np.mean(trial.nu[:, 0])),
                "fundamental_velocity_amplitude": float(np.sqrt(velocity_norm_squared)),
                "fundamental_wrench_amplitude": float(np.linalg.norm(wrench_fundamental)),
                "fluid_wrench_per_velocity_in_phase": fluid_force_per_velocity,
                "equivalent_direct_D_at_trial_amplitude": -fluid_force_per_velocity,
                "coefficient_units": "N*s/m" if j < 3 else "N*m*s",
                "fundamental_mean_fluid_power_W": float(
                    0.5 * (wrench_fundamental @ velocity_fundamental)
                ),
            }
        )
    return rows


FIXED_U_FREQUENCY_RESPONSE = fixed_u_frequency_response(TRIALS)


def summarize_fixed_u_frequency_response(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for dof_name, spec in DOF_SPECS.items():
        values = np.asarray(
            [
                row["equivalent_direct_D_at_trial_amplitude"]
                for row in rows
                if row["dof"] == dof_name
            ],
            dtype=float,
        )
        summary[dof_name] = {
            "trials": len(values),
            "median_equivalent_direct_D": float(np.median(values)),
            "min_equivalent_direct_D": float(np.min(values)),
            "max_equivalent_direct_D": float(np.max(values)),
            "coefficient_units": "N*s/m" if spec["index"] < 3 else "N*m*s",
        }
    return summary


FIXED_U_FREQUENCY_RESPONSE_SUMMARY = summarize_fixed_u_frequency_response(
    FIXED_U_FREQUENCY_RESPONSE
)

# The normal/pure assembly is the deployment orientation.  Its separately
# extrapolated sway/yaw intercepts bracket the unoscillated straight-tow force.
SURGE_REFERENCE_SPEED_M_S = SAMPLING_AUDIT["mean_towing_speed_m_s"]
normal_intercepts = np.array(
    [SURGE_EVEN_FITS["sway"].intercept_force_N, SURGE_EVEN_FITS["yaw"].intercept_force_N]
)
SURGE_REFERENCE_FORCE_ESTIMATE_N = float(np.median(normal_intercepts))
SURGE_REFERENCE_FORCE_RANGE_N = [float(np.min(normal_intercepts)), float(np.max(normal_intercepts))]
SURGE_REFERENCE_RESISTANCE_ESTIMATE_N = -SURGE_REFERENCE_FORCE_ESTIMATE_N
SURGE_REFERENCE_LINEAR_ONLY_CLOSURE = (
    SURGE_REFERENCE_RESISTANCE_ESTIMATE_N / SURGE_REFERENCE_SPEED_M_S
)
SURGE_REFERENCE_QUADRATIC_ONLY_CLOSURE = (
    SURGE_REFERENCE_RESISTANCE_ESTIMATE_N / SURGE_REFERENCE_SPEED_M_S**2
)

print("\nITTC fixed-speed surge even-coupling fits")
print("term       cycle-mean derivative   2f derivative       resistance magnitude       R²    leave-repeat range")
for surge_fit in SURGE_EVEN_FITS.values():
    leave_resistance = -surge_fit.leave_one_repeat_force_derivatives
    print(
        f"{surge_fit.term:<6s} {surge_fit.force_derivative:20.7g} "
        f"{surge_fit.second_harmonic_force_derivative:15.7g} "
        f"{surge_fit.resistance_magnitude:26.7g} {surge_fit.r2:8.4f} "
        f"[{np.min(leave_resistance):.7g}, {np.max(leave_resistance):.7g}] {surge_fit.units}"
    )
print(
    "normal-orientation straight-surge force anchor at "
    f"U={SURGE_REFERENCE_SPEED_M_S:.7g} m/s: "
    f"{SURGE_REFERENCE_FORCE_ESTIMATE_N:.7g} N "
    f"(campaign-intercept span {SURGE_REFERENCE_FORCE_RANGE_N})"
)
print(
    "mutually exclusive surge closures: "
    f"D1_uu={SURGE_REFERENCE_LINEAR_ONLY_CLOSURE:.7g} N*s/m OR "
    f"D2_uu={SURGE_REFERENCE_QUADRATIC_ONLY_CLOSURE:.7g} N*s^2/m^2"
)
print("fixed-U0 fundamental in-phase response:", FIXED_U_FREQUENCY_RESPONSE_SUMMARY)

# %%
M_ADDED_RECOMMENDED = M_ADDED.copy()
D_LINEAR_RECOMMENDED = D_LINEAR.copy()
D_QUADRATIC_RECOMMENDED = D_QUADRATIC.copy()
D_LINEAR_FIXED_U0_SURGE_CLOSURE = D_LINEAR_RECOMMENDED.copy()
D_QUADRATIC_FIXED_U0_SURGE_CLOSURE = D_QUADRATIC_RECOMMENDED.copy()
D_LINEAR_FIXED_U0_SURGE_CLOSURE[0, 0] = 0.0
D_QUADRATIC_FIXED_U0_SURGE_CLOSURE[0, 0] = SURGE_REFERENCE_QUADRATIC_ONLY_CLOSURE
IDENTIFIABILITY_MASK = np.isfinite(M_ADDED)


def matrix_table(matrix: np.ndarray) -> str:
    header = "          " + " ".join(f"{name:>12s}" for name in DOF_LABELS)
    rows = [header]
    for label, row in zip(WRENCH_LABELS, matrix):
        values = " ".join(f"{value:12.6g}" if np.isfinite(value) else f"{'NaN':>12s}" for value in row)
        rows.append(f"{label:>3s}       {values}")
    return "\n".join(rows)


print("\nM_A: direct diagonal PMM identification\n" + matrix_table(M_ADDED_RECOMMENDED))
print("\nD_1: direct diagonal fixed-U0 PMM identification\n" + matrix_table(D_LINEAR_RECOMMENDED))
print("\nD_2: direct diagonal PMM identification\n" + matrix_table(D_QUADRATIC_RECOMMENDED))
print(
    "\nD_2: with quadratic-only straight-surge anchor (fixed-U0 simulation closure)\n"
    + matrix_table(D_QUADRATIC_FIXED_U0_SURGE_CLOSURE)
)

identified_diagonal = np.diag(M_ADDED_RECOMMENDED)[list(EXCITED_INDICES)]
print("\nidentified M_A direct diagonal [v,w,q,r]:", identified_diagonal)

# %% [markdown]
# 三张矩阵只保留直接对角回归。未激励的 `u、p` 对角项和全部非对角项
# 都是 `NaN`。不使用左右对称把未辨识项写成 0，也不使用附加质量互易性补齐他列。

# %%
def fit_frequency_group(
    dof_name: str,
    nominal_frequency_hz: float,
    trials: list[Trial],
    global_fit: DofFit,
) -> dict[str, float | int | str]:
    selected = [
        trial
        for trial in trials
        if trial.plan.dof_name == dof_name
        and math.isclose(trial.plan.nominal_frequency_hz, nominal_frequency_hz)
    ]
    compressed = [harmonic_projection(trial) for trial in selected]
    X = np.vstack([item[0] for item in compressed])
    Y = np.vstack([item[1] for item in compressed])
    index = DOF_SPECS[dof_name]["index"]
    global_restoring = float(global_fit.coefficients[3])
    response = Y[:, index] - X[:, 3] * global_restoring
    beta = huber_fit(X[:, :3], response)
    prediction = X[:, :3] @ beta + X[:, 3] * global_restoring
    full_response = Y[:, index]
    r2 = 1.0 - np.sum((full_response - prediction) ** 2) / np.sum(
        (full_response - np.mean(full_response)) ** 2
    )
    return {
        "dof": dof_name,
        "nominal_frequency_hz": nominal_frequency_hz,
        "mean_estimated_frequency_hz": float(np.mean([trial.frequency_hz for trial in selected])),
        "repeats": len(selected),
        "added_mass_with_global_restoring_removed": float(beta[0]),
        "linear_damping_effective": float(beta[1]),
        "quadratic_damping": float(beta[2]),
        "global_nuisance_restoring": global_restoring,
        "r2": float(r2),
    }


NOMINAL_FREQUENCIES = sorted({plan.nominal_frequency_hz for plan in MANIFEST})
PER_FREQUENCY = [
    fit_frequency_group(dof_name, frequency, TRIALS, FITS[dof_name])
    for dof_name in DOF_SPECS
    for frequency in NOMINAL_FREQUENCIES
]

print("\nDirect-channel frequency audit (global restoring term removed)")
print("dof       f_nom  f_est  repeats       M_A       D_1eff        D_2       R²")
for row in PER_FREQUENCY:
    print(
        f"{row['dof']:<9s} {row['nominal_frequency_hz']:5.1f}  "
        f"{row['mean_estimated_frequency_hz']:6.4f}  {row['repeats']:7d}  "
        f"{row['added_mass_with_global_restoring_removed']:9.4g}  "
        f"{row['linear_damping_effective']:11.4g}  "
        f"{row['quadratic_damping']:9.4g}  {row['r2']:7.4f}"
    )

# %%
def timing_sensitivity() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    trials_by_shift: dict[float, list[Trial]] = {0.0: TRIALS}
    for shift in TIMING_SENSITIVITY_SHIFTS_S:
        if shift != 0.0:
            trials_by_shift[shift] = [build_trial(plan, shift) for plan in MANIFEST]
        shifted_trials = trials_by_shift[shift]
        for dof_name in DOF_SPECS:
            fit = fit_one_dof(dof_name, shifted_trials)
            rows.append(
                {
                    "sensor_to_motion_shift_s": shift,
                    "dof": dof_name,
                    "added_mass": float(fit.coefficients[0]),
                    "linear_damping_effective": float(fit.coefficients[1]),
                    "quadratic_damping": float(fit.coefficients[2]),
                    "nuisance_restoring": float(fit.coefficients[3]),
                    "r2": fit.r2,
                }
            )
    return rows


TIMING_SENSITIVITY = timing_sensitivity()
print("\nTiming sensitivity: direct channels")
for row in TIMING_SENSITIVITY:
    print(
        f"shift={row['sensor_to_motion_shift_s']:+.3f}s {row['dof']:<6s} "
        f"M={row['added_mass']:.6g} D1={row['linear_damping_effective']:.6g} "
        f"D2={row['quadratic_damping']:.6g} K={row['nuisance_restoring']:.6g} "
        f"R²={row['r2']:.5f}"
    )

# %%
def rigid_properties_json(props: RigidProperties) -> dict[str, object]:
    return {
        "mass_kg": props.mass_kg,
        "material_volume_m3_not_used_as_displaced_volume": props.material_volume_m3,
        "surface_area_m2": props.surface_area_m2,
        "com_from_output_origin_body_flu_m": props.r_OG_B_m.tolist(),
        "inertia_at_com_body_flu_kg_m2": props.inertia_G_B_kg_m2.tolist(),
        "parallel_axis_max_residual_kg_m2": float(np.max(np.abs(props.parallel_axis_residual_kg_m2))),
    }


trial_diagnostics = [
    {
        "dof": trial.plan.dof_name,
        "repeat": trial.plan.repeat,
        "file_id": trial.plan.file_id,
        "nominal_frequency_hz": trial.plan.nominal_frequency_hz,
        "estimated_frequency_hz": trial.frequency_hz,
        "motion_r2": trial.motion_r2,
        "forward_r2": trial.forward_r2,
        "paired_samples": len(trial.time_s),
        "gather_rows": trial.gather_rows,
        "sensor_rows": trial.sensor_rows,
        "mean_towing_speed_m_s": float(np.mean(trial.nu[:, 0])),
        "excited_coordinate_fundamental_amplitude": float(
            np.linalg.norm(
                fit_fourier(
                    trial.time_s,
                    trial.coordinate,
                    trial.frequency_hz,
                    MOTION_HARMONICS,
                ).coefficients[2:4]
            )
        ),
        "excited_velocity_fundamental_amplitude": float(
            np.linalg.norm(
                fit_fourier(
                    trial.time_s,
                    trial.nu[:, trial.plan.dof_index],
                    trial.frequency_hz,
                    MOTION_HARMONICS,
                ).coefficients[2:4]
            )
        ),
        "gather_file": str(trial.plan.gather_path.relative_to(DATA_ROOT)),
        "sensor_file": str(trial.plan.sensor_path.relative_to(DATA_ROOT)),
    }
    for trial in TRIALS
]


def strict_json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return strict_json_value(value.tolist())
    if isinstance(value, list):
        return [strict_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [strict_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: strict_json_value(item) for key, item in value.items()}
    return value


RESULT = {
    "role": "historical_experiment_archive_not_runtime_source",
    "definition": "tau_h=-M_A@nu_dot-D_1_eff@nu-D_2@(abs(nu)*nu)-K_nuisance@xi",
    "measurement_equation": {
        "balance_output": "body_on_support_reaction_R_equals_minus_F_support",
        "newton_euler": "tau_rigid=F_support+tau_h",
        "hydrodynamic_wrench_recovery": "tau_h=R+tau_rigid",
        "original_script_difference": (
            "Downloads/jn numerical scripts subtracted the rigid term from the mapped "
            "balance reaction, double-counting rigid inertia in apparent added mass."
        ),
    },
    "frame": "body FLU at center of mass",
    "wrench_order": list(WRENCH_LABELS),
    "velocity_order": list(DOF_LABELS),
    "units_by_block": {
        "added_mass": {
            "force_rows_linear_columns": "kg",
            "force_rows_angular_columns": "kg*m",
            "moment_rows_linear_columns": "kg*m",
            "moment_rows_angular_columns": "kg*m^2",
        },
        "linear_damping": {
            "force_rows_linear_columns": "N*s/m",
            "force_rows_angular_columns": "N*s",
            "moment_rows_linear_columns": "N*s",
            "moment_rows_angular_columns": "N*m*s",
        },
        "quadratic_damping": {
            "force_rows_linear_columns": "N*s^2/m^2",
            "force_rows_angular_columns": "N*s^2",
            "moment_rows_linear_columns": "N*s^2/m",
            "moment_rows_angular_columns": "N*m*s^2",
        },
        "angle_convention": "radian is dimensionless",
    },
    "identification_scope": {
        "published_terms": "direct_diagonal_only",
        "identified_diagonal_dofs": [DOF_LABELS[index] for index in EXCITED_INDICES],
        "unidentified_diagonal_dofs": [DOF_LABELS[index] for index in UNEXCITED_INDICES],
        "off_diagonal_terms": "not_identified_and_left_null",
        "cross_wrench_channels": "not_used_to_publish_hydrodynamic_matrix_coefficients",
        "reciprocity_completion": False,
        "symmetry_zero_fill": False,
    },
    "identifiability_mask": IDENTIFIABILITY_MASK.tolist(),
    "added_mass_direct_diagonal_identified": M_ADDED_RECOMMENDED.tolist(),
    "linear_damping_direct_diagonal_at_mean_towing_speed": D_LINEAR_RECOMMENDED.tolist(),
    "quadratic_damping_direct_diagonal_identified": D_QUADRATIC_RECOMMENDED.tolist(),
    "fixed_U0_quadratic_only_straight_surge_closure": {
        "linear_damping_matrix": D_LINEAR_FIXED_U0_SURGE_CLOSURE.tolist(),
        "quadratic_damping_matrix": D_QUADRATIC_FIXED_U0_SURGE_CLOSURE.tolist(),
        "assumption": (
            "D1_uu=0 and D2_uu=R(U0)/U0^2; this exactly anchors straight-surge "
            "resistance at U0 but does not identify its forward-speed dependence."
        ),
        "remaining_nulls": "p diagonal and every off-diagonal term; none is zero-filled",
    },
    "fixed_speed_surge_even_model": {
        "definition": (
            "Fourier-cycle-mean(X_h)=X0_campaign+X_vv*cycle-mean(v^2)"
            "+X_ww*cycle-mean(w^2)+X_qq*cycle-mean(q^2)+X_rr*cycle-mean(r^2); "
            "each campaign excites one term"
        ),
        "reference_forward_speed_m_s": SURGE_REFERENCE_SPEED_M_S,
        "coefficient_sign_convention": (
            "force_derivative is fluid-on-body X coefficient and is expected negative for added resistance; "
            "resistance_magnitude is its negative"
        ),
        "terms": {
            surge_fit.term: {
                "source_campaign": surge_fit.dof_name,
                "units": surge_fit.units,
                "campaign_zero_oscillation_intercept_force_N": surge_fit.intercept_force_N,
                "force_derivative": surge_fit.force_derivative,
                "resistance_magnitude": surge_fit.resistance_magnitude,
                "r2": surge_fit.r2,
                "rmse_N": surge_fit.rmse_N,
                "second_harmonic_force_derivative_audit": (
                    surge_fit.second_harmonic_force_derivative
                ),
                "second_harmonic_r2_audit": surge_fit.second_harmonic_r2,
                "dc_vs_second_harmonic_warning": (
                    "A memoryless squared-velocity term should give similar cycle-mean and "
                    "second-harmonic derivatives; disagreement indicates unsteady effects, "
                    "facility loads, or insufficient signal-to-noise."
                ),
                "leave_one_repeat_resistance_range": [
                    float(np.min(-surge_fit.leave_one_repeat_force_derivatives)),
                    float(np.max(-surge_fit.leave_one_repeat_force_derivatives)),
                ],
            }
            for surge_fit in SURGE_EVEN_FITS.values()
        },
        "straight_surge_reference_anchor": {
            "method": (
                "median of the separately extrapolated pure-sway and pure-yaw zero-oscillation "
                "intercepts; their span is retained as a campaign systematic range"
            ),
            "fluid_on_body_force_estimate_N": SURGE_REFERENCE_FORCE_ESTIMATE_N,
            "fluid_on_body_force_campaign_span_N": SURGE_REFERENCE_FORCE_RANGE_N,
            "resistance_estimate_N": SURGE_REFERENCE_RESISTANCE_ESTIMATE_N,
            "linear_only_closure_D1_uu_N_s_m": SURGE_REFERENCE_LINEAR_ONLY_CLOSURE,
            "quadratic_only_closure_D2_uu_N_s2_m2": SURGE_REFERENCE_QUADRATIC_ONLY_CLOSURE,
            "closure_warning": (
                "The linear-only and quadratic-only values are mutually exclusive assumptions. "
                "A single forward speed does not identify both coefficients."
            ),
        },
        "current_isaac_D2_compatibility": {
            "compatible": False,
            "odd_D1_D2_sign_and_algebra_match_current_isaac": True,
            "reason": (
                "Isaac currently evaluates D2@(abs(nu)*nu), which is odd in each velocity. "
                "X_vv*v^2, X_ww*w^2, X_qq*q^2 and X_rr*r^2 are even surge-force terms "
                "and require an additional manoeuvring-wrench term."
            ),
            "scope_warning": (
                "The exported odd matrices are empirical manoeuvring derivatives near U0, "
                "not a globally passive damping law for arbitrary forward speed."
            ),
        },
    },
    "fixed_U0_fundamental_in_phase_diagnostic": {
        "method": (
            "Complete-cycle sine/cosine fundamental inner product of direct wrench and "
            "velocity; equivalent_direct_D=-dot(tau_1,nu_1)/dot(nu_1,nu_1)."
        ),
        "interpretation": (
            "A frequency/amplitude diagnostic for local fixed-U0 manoeuvring derivatives, "
            "not a single-channel passivity gate for the coupled six-DOF model."
        ),
        "summary": FIXED_U_FREQUENCY_RESPONSE_SUMMARY,
    },
    "deployment_readiness": {
        "model_scope": "local_fixed_U0_manoeuvring_model",
        "reference_forward_speed_m_s": SURGE_REFERENCE_SPEED_M_S,
        "v_w_q_r_diagonal": "each directly fitted from 21 trials",
        "off_diagonal_terms": "not_published_from_this_PMM_campaign",
        "frequency_amplitude_confounding": (
            "The seven records change forcing frequency and oscillatory velocity amplitude "
            "together. Constant D1/D2 across 0.1--0.7 Hz is therefore a modelling assumption; "
            "per-frequency diagnostics show sign-changing effective in-phase response."
        ),
        "straight_surge_at_reference_speed": "total_resistance_anchored",
        "surge_speed_dependence": "not_identified_from_one_forward_speed",
        "roll_diagonal": "not_identified_not_excited",
        "surge_even_couplings": (
            "identified_as_fixed-U0 empirical terms with independent DC-versus-2f audit"
        ),
        "isaac_requirement": (
            "Keep the odd D1/D2 terms and add a separate even surge wrench for "
            "X_vv*v^2+X_ww*w^2+X_qq*q^2+X_rr*r^2."
        ),
        "coefficient_sign_clipping_or_projection_applied": False,
    },
    "nuisance_restoring_not_part_of_requested_matrices": K_NUISANCE.tolist(),
    "sampling": {
        "gather_stream": "motor_controller",
        "gather_hz": GATHER_HZ,
        "gather_hz_status": "user_confirmed",
        "sensor_stream": "six_axis_force_torque_sensor",
        "sensor_hz": SENSOR_HZ,
        "sensor_hz_status": "user_confirmed",
        "sensor_block_average": SENSOR_BLOCK,
        "motion_fit_window_s": [MOTION_FIT_START_S, MOTION_FIT_END_S],
        "load_fit_window_s": [LOAD_FIT_START_S, LOAD_FIT_END_S],
        "window_origin": {
            "motion_2_to_10_s": "Downloads/jn/gather_smooth_sway_yaw.py",
            "load_2_5_to_9_5_s": [
                "Downloads/jn/numerical_sway.py",
                "Downloads/jn/numerical_yaw.py",
            ],
        },
        "excluded_phases": (
            "approximately 0--2 s start/tow acceleration and 9.5 s onward deceleration, "
            "stop, lateral/angle recentering; F-axis return occurs after the record"
        ),
        "synchronization_status": SYNCHRONIZATION_STATUS,
        "sensor_to_motion_shift_s": SENSOR_TO_MOTION_SHIFT_S,
        **SAMPLING_AUDIT,
    },
    "coordinate_mapping": {
        "raw_bench_axis_labels": ["X_forward", "Y_left", "Z_down"],
        "raw_bench_handedness": "left_handed_axis_labels_not_used_for_cross_products",
        "bench_components_to_pool_FLU": BENCH_COMPONENTS_TO_POOL_FLU.tolist(),
        "pure_body_to_pool": R_HB_PURE.tolist(),
        "vertical_body_to_pool_Rx_plus_90": R_HB_VERTICAL.tolist(),
        "solidworks_body_attached_to_body_flu": Q_BS.tolist(),
        "yaw_encoder_sign_to_pool_FLU": YAW_ENCODER_SIGN_TO_H,
        "sensor_force_polarity_in_raw_bench_axes": SENSOR_FORCE_POLARITY_IN_BENCH,
        "sensor_moment_polarity_in_raw_bench_axes": SENSOR_MOMENT_POLARITY_IN_BENCH,
    },
    "rigid_properties": {
        "pure": rigid_properties_json(PURE_RIGID),
        "vertical": rigid_properties_json(VERTICAL_RIGID),
    },
    "sign_audit_median_mapped_measured_per_qdot": SIGN_AUDIT,
    "pure_yaw_median_sway_to_yaw_rate_amplitude_ratio_m": yaw_purity,
    "fit_diagnostics": {
        name: {
            "standardized_condition_number": fit.condition_number,
            "direct_wrench": WRENCH_LABELS[fit.dof_index],
            "direct_r2": fit.r2,
        }
        for name, fit in FITS.items()
    },
    "notes": [
        "NaN means the term was not published by the direct-diagonal PMM identification; it is not zero. This includes u/p diagonal terms and every off-diagonal term.",
        "D_1_eff includes towing-speed-dependent in-phase effects and unseparated added-mass Coriolis effects.",
        "Only direct v/Y, w/Z, q/M and r/N regressions populate the three matrices; all off-diagonal terms remain null and no reciprocity or symmetry zero-fill is applied.",
        "The SolidWorks material/assembly volume is recorded but is not used as displaced volume.",
        "Coordinate System 1 is assumed to be the PMM motion/load reference origin; replace r_OG if the balance origin differs.",
        "The user confirms that gather and sensor_ are hardware-synchronized; zero timing shift is therefore the experimental setting.",
        "The user confirms the raw PMM bench axis labels are +X forward, +Y left, +Z down; raw vectors are explicitly mapped by diag(1,1,-1) into right-handed FLU before rigid-body operations.",
        "The vertical force polarity is resolved from the campaign-to-campaign inertial phase reversal and should be confirmed against the wiring log.",
        "The +/-50 ms timing sweep is an artificial phase perturbation, not an estimate of actual synchronization uncertainty.",
        "The direct fixed-U0 coefficients are manoeuvring derivatives of a coupled model; an individual diagonal sign is not by itself a six-DOF passivity result.",
        "The old partial-window -mean(tau_j*nu_j) audit was removed because its noninteger-cycle inertial boundary term can be mistaken for damping power.",
        "ITTC-style X_vv/X_ww/X_qq/X_rr are even surge-force couplings and are exported separately; they are not entries of D_2@(abs(nu)*nu).",
        "At the single measured forward speed, total straight-surge resistance can be anchored but D1_uu and D2_uu cannot both be independently estimated.",
        "The seven forcing frequencies also change oscillatory velocity amplitude, so intrinsic frequency dependence and amplitude nonlinearity are not independently controlled; exported D1/D2 are robust empirical sweep coefficients.",
    ],
}


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wrench\\velocity", *DOF_LABELS])
        for label, row in zip(WRENCH_LABELS, matrix):
            writer.writerow([label, *row])


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_matrix(matrix: np.ndarray) -> str:
    lines = [
        "| wrench \\ velocity | " + " | ".join(DOF_LABELS) + " |",
        "|---|" + "---:|" * 6,
    ]
    for label, row in zip(WRENCH_LABELS, matrix):
        values = [f"{value:.7g}" if np.isfinite(value) else "NaN" for value in row]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


SURGE_EVEN_ROWS = [
    {
        "term": surge_fit.term,
        "source_campaign": surge_fit.dof_name,
        "units": surge_fit.units,
        "campaign_intercept_force_N": surge_fit.intercept_force_N,
        "force_derivative": surge_fit.force_derivative,
        "resistance_magnitude": surge_fit.resistance_magnitude,
        "r2": surge_fit.r2,
        "rmse_N": surge_fit.rmse_N,
        "second_harmonic_force_derivative_audit": surge_fit.second_harmonic_force_derivative,
        "second_harmonic_r2_audit": surge_fit.second_harmonic_r2,
        "leave_one_repeat_resistance_min": float(
            np.min(-surge_fit.leave_one_repeat_force_derivatives)
        ),
        "leave_one_repeat_resistance_max": float(
            np.max(-surge_fit.leave_one_repeat_force_derivatives)
        ),
    }
    for surge_fit in SURGE_EVEN_FITS.values()
]


def markdown_surge_even(rows: list[dict[str, object]]) -> str:
    lines = [
        "| term | campaign | cycle-mean derivative | 2f derivative audit | resistance magnitude | units | DC R² | 2f R² | leave-one-repeat resistance range |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['term']} | {row['source_campaign']} | {row['force_derivative']:.7g} | "
            f"{row['second_harmonic_force_derivative_audit']:.7g} | "
            f"{row['resistance_magnitude']:.7g} | {row['units']} | {row['r2']:.4f} | "
            f"{row['second_harmonic_r2_audit']:.4f} | "
            f"[{row['leave_one_repeat_resistance_min']:.7g}, "
            f"{row['leave_one_repeat_resistance_max']:.7g}] |"
        )
    return "\n".join(lines)


def markdown_frequency_response_summary(summary: dict[str, dict[str, object]]) -> str:
    lines = [
        "| DOF | trials | median equivalent direct D | trial range | units |",
        "|---|---:|---:|---:|---|",
    ]
    for dof_name, values in summary.items():
        lines.append(
            f"| {dof_name} | {values['trials']} | "
            f"{values['median_equivalent_direct_D']:.7g} | "
            f"[{values['min_equivalent_direct_D']:.7g}, "
            f"{values['max_equivalent_direct_D']:.7g}] | "
            f"{values['coefficient_units']} |"
        )
    return "\n".join(lines)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for filename, matrix in (
    ("added_mass_identified.csv", M_ADDED),
    ("added_mass_recommended.csv", M_ADDED_RECOMMENDED),
    ("linear_damping_identified_raw.csv", D_LINEAR),
    ("linear_damping_effective.csv", D_LINEAR_RECOMMENDED),
    ("quadratic_damping_identified_raw.csv", D_QUADRATIC),
    ("quadratic_damping.csv", D_QUADRATIC_RECOMMENDED),
    ("linear_damping_fixed_u_surge_closure.csv", D_LINEAR_FIXED_U0_SURGE_CLOSURE),
    ("quadratic_damping_fixed_u_surge_closure.csv", D_QUADRATIC_FIXED_U0_SURGE_CLOSURE),
    ("nuisance_restoring.csv", K_NUISANCE),
):
    write_matrix_csv(OUTPUT_DIR / filename, matrix)

write_dict_rows(OUTPUT_DIR / "trial_diagnostics.csv", trial_diagnostics)
write_dict_rows(OUTPUT_DIR / "per_frequency_diagnostics.csv", PER_FREQUENCY)
write_dict_rows(OUTPUT_DIR / "timing_sensitivity.csv", TIMING_SENSITIVITY)
write_dict_rows(OUTPUT_DIR / "surge_even_coupling.csv", SURGE_EVEN_ROWS)
write_dict_rows(
    OUTPUT_DIR / "fixed_u_frequency_response.csv", FIXED_U_FREQUENCY_RESPONSE
)
(OUTPUT_DIR / "hydrodynamic_matrices.json").write_text(
    json.dumps(strict_json_value(RESULT), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
(OUTPUT_DIR / "REPORT.md").write_text(
    "# PMM 6×6 水动力辨识结果\n\n"
    "本报告只保留为水池模型试验档案，不是模拟器生产系数源；"
    "运行时三张矩阵只由目标尺度 OpenFOAM 工况发布。\n\n"
    "辨识式为 `tau_h=-M_A@nu_dot-D_1_eff@nu-D_2@(abs(nu)*nu)-K_nuisance@xi`；"
    "`K_nuisance` 只用于隔离恢复项。顺序为 "
    "wrench `[X,Y,Z,K,M,N]`、velocity `[u,v,w,p,q,r]`，参考点为质心，坐标为艇体 FLU。\n\n"
    "## 台架到辨识坐标的转换\n\n"
    "用户确认原始 PMM 台架轴标签为 `(+X前,+Y左,+Z下)`。"
    "数据读入后先按 `[x,y,z] -> [x,y,-z]` 转成右手 FLU，"
    "艏向编码器同样取负号；之后才执行 vertical 装配旋转、"
    "力矩平移、Newton--Euler 叉乘和系数回归。"
    "台架左手轴标签不直接用作刚体计算坐标。\n\n"
    "## 附加质量：直接对角项\n\n"
    + markdown_matrix(M_ADDED_RECOMMENDED)
    + "\n\n## 测力计反力符号\n\n"
    + "测力计给出艇体对测力架的反力 `R=-F_support`，因此应使用 "
    "`tau_h=R+tau_rigid`。原脚本的减法会重复计入刚体惯性。\n\n"
    + "## 一次阻力（固定 U0 直接对角项）\n\n"
    + markdown_matrix(D_LINEAR_RECOMMENDED)
    + "\n\n## 二次阻力（固定 U0 直接对角项）\n\n"
    + markdown_matrix(D_QUADRATIC_RECOMMENDED)
    + "\n\n## 固定 U0 的二次独占直航闭合\n\n"
    + "为当前 Isaac 接口提供的可选闭合取 `D1_uu=0`、"
    + f"`D2_uu=R(U0)/U0²={SURGE_REFERENCE_QUADRATIC_ONLY_CLOSURE:.7g} N·s²/m²`。"
    + "它在参考航速精确复现直航总阻力，但不代表已经测得前进速度律；"
    + "`p,p` 和所有非对角项继续保留 NaN。\n\n一次矩阵：\n\n"
    + markdown_matrix(D_LINEAR_FIXED_U0_SURGE_CLOSURE)
    + "\n\n二次矩阵：\n\n"
    + markdown_matrix(D_QUADRATIC_FIXED_U0_SURGE_CLOSURE)
    + "\n\n## ITTC 固定航速 surge 偶次耦合项\n\n"
    + "力导数为 fluid-on-body 符号；阻力幅值是其相反数。`v²/w²/q²/r²` 是偶函数，"
    "不能写入现有 `D_2@(abs(nu)*nu)`。\n\n"
    + markdown_surge_even(SURGE_EVEN_ROWS)
    + (
        "\n\n`cycle-mean derivative` 来自 Fourier 还原后的完整周期 DC 趋势；"
        "`2f derivative audit` 是独立的二倍频检查。理想无记忆平方项应使两者接近；"
        "表中的差异（尤其 heave 二倍频反号）说明这些 surge 耦合值可以作为固定航速"
        "的经验趋势，但尚不是已验证的全动态 Isaac 系数。\n\nnormal/pure 装配在参考航速 "
        f"{SURGE_REFERENCE_SPEED_M_S:.7g} m/s 的直航 surge 力锚点为 "
        f"{SURGE_REFERENCE_FORCE_ESTIMATE_N:.7g} N，两个 campaign 截距范围为 "
        f"[{SURGE_REFERENCE_FORCE_RANGE_N[0]:.7g}, {SURGE_REFERENCE_FORCE_RANGE_N[1]:.7g}] N。"
        "若仿真必须在当前接口中闭合 surge，可互斥选择 "
        f"`D1_uu={SURGE_REFERENCE_LINEAR_ONLY_CLOSURE:.7g} N·s/m` 或 "
        f"`D2_uu={SURGE_REFERENCE_QUADRATIC_ONLY_CLOSURE:.7g} N·s²/m²`；"
        "单一航速不能同时辨识这两个值，因此不能把它们相加。\n\n"
    )
    + "## 固定 U0 基频同相诊断\n\n"
    + markdown_frequency_response_summary(FIXED_U_FREQUENCY_RESPONSE_SUMMARY)
    + "\n\n这里用 Fourier 正弦/余弦基频向量做完整周期内积，避免非整周期时间窗的"
    "惯性边界能量混入。它显示直接通道随频率和振幅的有效同相导数，但不是把每个"
    "通道单独判为被动/非被动的门槛；固定航速操纵模型还包含 surge 偶次项和耦合项。"
    + "\n\n## 结论与限制\n\n"
    "- 用户确认 `gather` 为电机记录（100 Hz），`sensor_` 为六分力记录（500 Hz）；"
    "每 5 个传感器点块平均后对齐到 100 Hz。\n"
    "- 用户确认台架原始方向为前 `+X`、左 `+Y`、下 `+Z`；"
    "六分力和艏向角已在读入端转到项目 FLU。\n"
    "- 用户确认两路采集硬同步；零时差是实验设置，不再视为待估计参数。\n"
    "- 2–10 s 运动拟合及 2.5–9.5 s 载荷硬裁剪均直接沿用 `Downloads/jn` 原脚本。\n"
    "- 已把原脚本的 `mapped_balance-rigid` 修正为反力关系 `R+rigid`，避免重复计入刚体惯性。\n"
    "- 三张 PMM 矩阵只发布 `v/Y、w/Z、q/M、r/N` 直接对角回归；"
    "`u、p` 对角项和所有非对角项都保留 `NaN`。\n"
    "- 没有对负系数取绝对值，没有对阻力矩阵强制对称/正定，也没有 PSD 投影。\n"
    "- 不用左右对称将未辨识项填 0，不用附加质量互易性补齐他列。\n"
    "- `D_1_eff` 含平均拖航速度下的交叉流与未单独分离的附加质量 Coriolis 同相项。\n"
    "- `timing_sensitivity.csv` 的 ±50 ms 是人工相位扰动诊断，不代表真实同步误差。\n"
    "- `v,w,q,r` 阻力对角项是约 0.202 m/s 的局部固定航速导数；"
    "不能外推成任意前进速度下不变的全局阻力矩阵。\n"
    "- 七个工况同时改变强迫频率和振荡速度幅值；逐频率同相导数会变号，因此全局 `D1/D2` 是 0.1–0.7 Hz 扫频上的经验闭合，不能声称已独立分离固有频率效应与幅值非线性。\n"
    "- `X_vv/X_ww/X_qq/X_rr` 已按 ITTC 的偶次 surge 模型辨识并单独导出；"
    "它们不是当前 Isaac 二次阻力矩阵的 X 行元素。\n"
    "- 约 0.202 m/s 的直航 surge 总阻力已有锚点，但单一航速不能同时拆分 `D1_uu` 与 `D2_uu`。\n"
    "- 若要三张无 NaN 且一次/二次项都由试验独立辨识的纯 PMM 全数值矩阵，"
    "必须补做多个直航速度的 surge 试验与 roll 独立强迫振荡。\n",
    encoding="utf-8",
)
print("\noutputs written to:", OUTPUT_DIR)

# %% [markdown]
# ## 如何使用输出
#
# - `added_mass_identified.csv`、`added_mass_recommended.csv`：相同的 PMM 直接对角结果；
# - `linear_damping_effective.csv`、`quadratic_damping.csv`：约 0.202 m/s 的直接对角项；
# - `linear_damping_identified_raw.csv`、`quadratic_damping_identified_raw.csv`：
#   与上述直接对角结果相同，保留旧文件名以便追溯；
# - `linear_damping_fixed_u_surge_closure.csv`、
#   `quadratic_damping_fixed_u_surge_closure.csv`：用二次独占假设在参考航速闭合
#   `D_uu` 的仿真版本；它不是多航速辨识结果；
# - `surge_even_coupling.csv`：ITTC 固定航速模型的 `X_vv/X_ww/X_qq/X_rr`；
# - `fixed_u_frequency_response.csv`：每个试次的完整周期基频同相导数；
# - `nuisance_restoring.csv`：为避免恢复力污染附加质量而联合估计的辅助量；
# - `trial_diagnostics.csv`、`per_frequency_diagnostics.csv` 和
#   `timing_sensitivity.csv`：分别检查单次记录、频率依赖和人工相位扰动敏感性；
# - `hydrodynamic_matrices.json`：包含矩阵、单位/顺序、坐标、质量属性、掩码和假设的机器可读汇总。
#
# 不能直接把 `NaN` 改成 0。现有数据给出了参考航速下的直航 surge 总阻力锚点，
# 但一次/二次 surge 拆分仍需要多个直航速度；roll 仍需独立试验或明确的外部来源。
# `hydrodynamic_matrices.json` 同时给出线性独占与二次独占的互斥 surge 闭合值，
# 下游只能选择其一，不能把两者同时写入矩阵。
#
# `X_vv/X_ww/X_qq/X_rr` 需要额外的 manoeuvring-wrench 计算：当前 Isaac 的
# `D_2@(abs(nu)*nu)` 是奇函数，不能表示这些偶次项。若直接把它们填进 D2，正负
# 半周期会互相抵消，得到与试验平均 surge 阻力相反的结果。
#
# 同样不能仅凭“阻力应为正”去改每个操纵导数的符号。用户已确认
# gather/sensor_ 硬同步，因此 `timing_sensitivity.csv` 的 ±50 ms 只用于展示人工
# 相位扰动影响。主结果用完整周期 Fourier 同相量诊断频率/振幅趋势；单个对角项不能
# 脱离 surge 偶次项及六自由度耦合单独作能量结论。
#

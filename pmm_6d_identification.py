# %% [markdown]
# # PMM 六自由度水动力矩阵辨识（FLU，质心参考点）
#
# 本脚本是 notebook 的可测试源文件。最终模型采用
#
# `tau_h = -M_A @ nu_dot - D_1 @ nu - D_2 @ (abs(nu) * nu)`，
#
# 行顺序为 `[X,Y,Z,K,M,N]`，列顺序为 `[u,v,w,p,q,r]`。

# %% [markdown]
# ## 实验逻辑与伪代码
#
# 1. 建立池坐标 `H=(F,L,U)` 与艇体坐标 `B=(u,v,w)`；`pure` 使用单位旋转，
#    `vertical` 使用 `R_HB=Rx(+90°)`，因此池中的横荡/艏摇分别变成艇体升沉/纵摇。
# 2. 从 gather 的第 2、5、8、11 列（零基索引 1、4、7、10）恢复纵向位置、
#    横向位置与角度；按每个工况的名义频率附近拟合三阶 Fourier 运动，解析求导。
# 3. gather/sensor 的真实采样率是 200/1000 Hz；六分力 5 点块平均到 200 Hz。
#    根据 pure 原脚本与 vertical 符号审计映射传感器，再按瞬时 `Rz(psi) @ R_HB`
#    旋转到艇体坐标，把力矩从运动原点平移到质心。
# 4. 用对应装配体的质量、质心与质心惯量计算刚体 Newton-Euler 惯性载荷；
#    `tau_h = tau_measured - tau_RB`。
# 5. 每个试次去除常量和线性漂移，再投影到实测频率的 1、3 次谐波。
# 6. 对每个已激励列 `j in {v,w,q,r}` 联合全部频率和三次重复，Huber 回归
#    `tau_h = -M_A[:,j] qdot - D1[:,j] q - D2[:,j] q|q|`。
# 7. 组装 6×6 矩阵。`u、p` 没有独立振荡，保留为 NaN；不能把“没测到”写成 0。

# %%
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import json
import math
import subprocess

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=160)

REPO_ROOT = Path.cwd().resolve()
DATA_ROOT = Path("/home/jining_yang/Downloads/jn")
GIT_FALLBACK_PREFIX = "environment/pmm"
GIT_FALLBACK_REVISION = subprocess.run(
    [
        "git",
        "log",
        "-n",
        "1",
        "--format=%H",
        "--",
        f"{GIT_FALLBACK_PREFIX}/vertical_sway1/gather_8.csv",
    ],
    cwd=REPO_ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
OUTPUT_DIR = REPO_ROOT / "pmm_identification_results"

GATHER_HZ = 200.0
SENSOR_HZ = 1000.0
SENSOR_BLOCK = 5
MOTION_FIT_START_S = 1.0
MOTION_FIT_END_S = 5.0
LOAD_FIT_START_S = 1.25
LOAD_FIT_END_S = 4.75
POSITION_COUNTS_PER_M = 1_000_000.0
ANGLE_COUNTS_PER_DEG = 18_300.0
YAW_ENCODER_SIGN_TO_H = -1.0
NOMINAL_FREQUENCY_STEP_HZ = 0.2
MOTION_HARMONICS = 3
LOAD_HARMONICS = (1, 3)
SENSOR_TO_MOTION_SHIFT_S = 0.0

# raw sensor columns are [TX, TY, TZ, FX, FY, FZ].  pure forces and both
# moment sets keep the resolved reaction sign.  The vertical force channels
# require the opposite polarity: it raises direct-heave R² from about 0.72 to
# 0.99 and makes its added mass consistent with the 0.7-scale CFD counterpart.
SENSOR_FORCE_SIGN_TO_H = {"pure": -1.0, "vertical": 1.0}
SENSOR_MOMENT_SIGN_TO_H = {"pure": -1.0, "vertical": -1.0}

CFD_EXPERIMENT_CONFIG = (
    REPO_ROOT / "environment/openfoam/experiment_configs/jn2_port_starboard_symmetric_minimal_level6.json"
)
CFD_TARGET_RESULTS = (
    REPO_ROOT / "environment/openfoam/results_jn2_port_starboard_symmetric_minimal_level6_v1/config_updates.json"
)

DOF_LABELS = ("u", "v", "w", "p", "q", "r")
WRENCH_LABELS = ("X", "Y", "Z", "K", "M", "N")
EXCITED_INDICES = (1, 2, 4, 5)


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


# SolidWorks 默认坐标 S：+Z 前、+Y 左、+X 下。
# v_B = Q_BS @ v_S，其中 B 是艇体 FLU。
Q_BS = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
R_HB_PURE = np.eye(3)
R_HB_VERTICAL = rotation_x(math.pi / 2.0)


def solidworks_positive_products_to_tensor(values_kg_mm2: list[list[float]]) -> np.ndarray:
    """SolidWorks 正张量记数法 -> 标准惯性矩阵，kg mm² -> kg m²。"""
    matrix = np.asarray(values_kg_mm2, dtype=float) * 1.0e-6
    off_diagonal = ~np.eye(3, dtype=bool)
    matrix[off_diagonal] *= -1.0
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
    residual = inertia_O_B - inertia_G_B - parallel_axis
    return RigidProperties(
        name=name,
        mass_kg=mass_kg,
        material_volume_m3=volume_mm3 * 1.0e-9,
        surface_area_m2=area_mm2 * 1.0e-6,
        r_OG_B_m=r_OG_B,
        inertia_G_B_kg_m2=inertia_G_B,
        inertia_O_B_reported_kg_m2=inertia_O_B,
        parallel_axis_residual_kg_m2=residual,
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

print("Q_BS determinant:", np.linalg.det(Q_BS))
for props in (PURE_RIGID, VERTICAL_RIGID):
    print(f"\n{props.name}: r_OG_B [m] =", props.r_OG_B_m)
    print("I_G_B [kg m²] =\n", props.inertia_G_B_kg_m2)
    print("parallel-axis max residual [kg m²] =", np.max(np.abs(props.parallel_axis_residual_kg_m2)))

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
    gather_relative: Path
    sensor_relative: Path
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
            folder = f"{spec['family']}{repeat}"
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
                        nominal_frequency_hz=NOMINAL_FREQUENCY_STEP_HZ * (file_id - base),
                        gather_relative=Path(folder) / f"gather_{stem}.csv",
                        sensor_relative=Path(folder) / f"sensor_{stem}.csv",
                        mount=spec["mount"],
                    )
                )
    return plans


def read_text_with_provenance(relative: Path) -> tuple[str, str]:
    local = DATA_ROOT / relative
    if local.is_file():
        return local.read_text(encoding="utf-8"), "downloads"
    object_name = f"{GIT_FALLBACK_REVISION}:{GIT_FALLBACK_PREFIX}/{relative.as_posix()}"
    completed = subprocess.run(
        ["git", "show", object_name],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout, f"git:{GIT_FALLBACK_REVISION}"


def load_raw(relative: Path, delimiter: str | None) -> tuple[np.ndarray, str]:
    text, provenance = read_text_with_provenance(relative)
    return np.loadtxt(StringIO(text), delimiter=delimiter), provenance


MANIFEST = build_manifest()
provenance_counts: dict[str, int] = {}
for plan in MANIFEST:
    _, source = read_text_with_provenance(plan.gather_relative)
    provenance_counts[source] = provenance_counts.get(source, 0) + 1
print("planned trial pairs:", len(MANIFEST))
print("gather provenance:", provenance_counts)
print("historical vertical-data revision:", GIT_FALLBACK_REVISION)

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
        low, high = max(0.96 * nominal_hz, best - step), min(1.02 * nominal_hz, best + step)
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
    force = mass_kg * (
        linear_derivative_B + np.cross(angular_velocity_B, linear_velocity_B)
    )
    angular_momentum = angular_velocity_B @ inertia_G_B.T
    moment = angular_acceleration_B @ inertia_G_B.T + np.cross(
        angular_velocity_B, angular_momentum
    )
    return np.column_stack((force, moment))


@dataclass
class Trial:
    plan: TrialPlan
    frequency_hz: float
    time_s: np.ndarray
    nu: np.ndarray
    nu_dot: np.ndarray
    hydro_wrench: np.ndarray
    measured_wrench: np.ndarray
    rigid_wrench: np.ndarray
    motion_r2: float
    x_r2: float
    source_gather: str
    source_sensor: str


def build_trial(plan: TrialPlan) -> Trial:
    gather, source_gather = load_raw(plan.gather_relative, None)
    sensor, source_sensor = load_raw(plan.sensor_relative, ",")

    gather_time = np.arange(len(gather), dtype=float) / GATHER_HZ
    motion_mask = (gather_time >= MOTION_FIT_START_S) & (gather_time <= MOTION_FIT_END_S)
    x_position = (gather[:, 1] + gather[:, 7]) / POSITION_COUNTS_PER_M
    lateral_position = gather[:, 4] / POSITION_COUNTS_PER_M
    yaw_angle = np.deg2rad(
        YAW_ENCODER_SIGN_TO_H
        * (gather[:, 10] - gather[0, 10])
        / ANGLE_COUNTS_PER_DEG
    )
    frequency_signal = lateral_position if plan.raw_kind == "sway" else yaw_angle
    frequency = estimate_frequency(
        gather_time[motion_mask],
        frequency_signal[motion_mask],
        plan.nominal_frequency_hz,
    )
    fit_x = fit_fourier(gather_time[motion_mask], x_position[motion_mask], frequency, MOTION_HARMONICS)
    fit_y = fit_fourier(
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
    motion_time = sensor_time + SENSOR_TO_MOTION_SHIFT_S

    _, dx, ddx = fit_x.evaluate(motion_time)
    _, dy, ddy = fit_y.evaluate(motion_time)
    if fit_angle is None:
        psi = np.zeros_like(motion_time)
        yaw_rate = np.zeros_like(motion_time)
        yaw_acceleration = np.zeros_like(motion_time)
        motion_r2 = fit_y.r2
    else:
        psi, yaw_rate, yaw_acceleration = fit_angle.evaluate(motion_time)
        motion_r2 = fit_angle.r2

    c, s = np.cos(psi), np.sin(psi)
    velocity_yaw_frame = np.column_stack(
        (c * dx + s * dy, -s * dx + c * dy, np.zeros_like(dx))
    )
    derivative_yaw_frame = np.column_stack(
        (
            -s * yaw_rate * dx + c * ddx + c * yaw_rate * dy + s * ddy,
            -c * yaw_rate * dx - s * ddx - s * yaw_rate * dy + c * ddy,
            np.zeros_like(dx),
        )
    )
    omega_yaw_frame = np.column_stack(
        (np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_rate)
    )
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

    # raw = [TX,TY,TZ,FX,FY,FZ].  Force polarity differs between the two
    # mounting campaigns; moment polarity does not.
    force_H = SENSOR_FORCE_SIGN_TO_H[plan.mount] * sensor_average[:, 3:6]
    moment_H_at_O = SENSOR_MOMENT_SIGN_TO_H[plan.mount] * sensor_average[:, 0:3]
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
    hydro = measured - rigid_load
    return Trial(
        plan=plan,
        frequency_hz=frequency,
        time_s=sensor_time,
        nu=nu,
        nu_dot=nu_dot,
        hydro_wrench=hydro,
        measured_wrench=measured,
        rigid_wrench=rigid_load,
        motion_r2=motion_r2,
        x_r2=fit_x.r2,
        source_gather=source_gather,
        source_sensor=source_sensor,
    )


TRIALS = [build_trial(plan) for plan in MANIFEST]
print("built trials:", len(TRIALS))

audit_gather, _ = load_raw(MANIFEST[0].gather_relative, None)
audit_sensor, _ = load_raw(MANIFEST[0].sensor_relative, ",")
timestamp_step = float(np.median(np.diff(audit_gather[:, 0])))
velocity_reconstruction_error = np.sqrt(
    np.mean(
        (
            np.diff(audit_gather[:, 1])
            - timestamp_step * audit_gather[:-1, 2]
        )
        ** 2
    )
)
mean_towing_speed = float(np.mean(np.concatenate([trial.nu[:, 0] for trial in TRIALS])))
print("sampling audit:")
print("  gather timestamp-count step =", timestamp_step)
print("  RMS of Δposition - step*recorded_velocity =", velocity_reconstruction_error)
print("  sensor/gather row ratio =", len(audit_sensor) / len(audit_gather))
print("  resolved mean towing speed [m/s] =", mean_towing_speed)

# %%
def residualize(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    centered = time_s - np.mean(time_s)
    nuisance = np.column_stack((np.ones_like(centered), centered))
    return values - nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]


def harmonic_projection(trial: Trial) -> tuple[np.ndarray, np.ndarray]:
    j = trial.plan.dof_index
    q = trial.nu[:, j]
    q_dot = trial.nu_dot[:, j]
    # Coefficients directly follow the positive Fossen M_A, D_1, D_2 convention.
    design = np.column_stack((-q_dot, -q, -q * np.abs(q)))
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
    r2_by_wrench: np.ndarray
    condition_number: float


def fit_one_dof(dof_name: str, trials: list[Trial]) -> DofFit:
    selected = [trial for trial in trials if trial.plan.dof_name == dof_name]
    compressed = [harmonic_projection(trial) for trial in selected]
    X = np.vstack([item[0] for item in compressed])
    Y = np.vstack([item[1] for item in compressed])
    coefficients = np.column_stack([huber_fit(X, Y[:, i]) for i in range(6)])
    prediction = X @ coefficients
    denominator = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum((Y - prediction) ** 2, axis=0) / denominator
    standardized = X / np.linalg.norm(X, axis=0)
    return DofFit(
        dof_name=dof_name,
        dof_index=DOF_SPECS[dof_name]["index"],
        X=X,
        Y=Y,
        coefficients=coefficients,
        prediction=prediction,
        r2_by_wrench=r2,
        condition_number=float(np.linalg.cond(standardized)),
    )


FITS = {name: fit_one_dof(name, TRIALS) for name in DOF_SPECS}
M_ADDED = np.full((6, 6), np.nan)
D_LINEAR = np.full((6, 6), np.nan)
D_QUADRATIC = np.full((6, 6), np.nan)
for fit in FITS.values():
    M_ADDED[:, fit.dof_index] = fit.coefficients[0]
    D_LINEAR[:, fit.dof_index] = fit.coefficients[1]
    D_QUADRATIC[:, fit.dof_index] = fit.coefficients[2]

for name, fit in FITS.items():
    print(f"\n{name}: condition={fit.condition_number:.3f}")
    print("R² [X,Y,Z,K,M,N] =", fit.r2_by_wrench)
print("\nidentified M_A columns [kg / kg m / kg m² as applicable]:\n", M_ADDED)
print("\nidentified D_1 columns (effective at towing condition):\n", D_LINEAR)
print("\nidentified D_2 columns:\n", D_QUADRATIC)

# %%
def reciprocal_added_mass_completion(raw: np.ndarray) -> np.ndarray:
    """Use only added-mass reciprocity; the two unmeasured diagonal blocks remain NaN."""
    completed = raw.copy()
    for missing_column in (0, 3):
        for measured_column in EXCITED_INDICES:
            completed[measured_column, missing_column] = raw[missing_column, measured_column]
    for row in EXCITED_INDICES:
        for column in EXCITED_INDICES:
            completed[row, column] = 0.5 * (raw[row, column] + raw[column, row])
    return completed


def identified_diagonal(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((6, 6))
    for index in range(6):
        result[index, index] = matrix[index, index]
    return result


M_ADDED_RECIPROCAL = reciprocal_added_mass_completion(M_ADDED)
M_ADDED_DIAGONAL = identified_diagonal(M_ADDED)
D_LINEAR_DIAGONAL = identified_diagonal(D_LINEAR)
D_QUADRATIC_DIAGONAL = identified_diagonal(D_QUADRATIC)


def matrix_table(matrix: np.ndarray) -> str:
    header = "          " + " ".join(f"{name:>12s}" for name in DOF_LABELS)
    rows = [header]
    for label, row in zip(WRENCH_LABELS, matrix):
        values = " ".join(f"{value:12.6g}" if np.isfinite(value) else f"{'NaN':>12s}" for value in row)
        rows.append(f"{label:>3s}       {values}")
    return "\n".join(rows)


def markdown_matrix(matrix: np.ndarray) -> str:
    lines = [
        "| wrench \\ velocity | " + " | ".join(DOF_LABELS) + " |",
        "|---|" + "---:|" * 6,
    ]
    for label, row in zip(WRENCH_LABELS, matrix):
        values = [f"{value:.7g}" if np.isfinite(value) else "NaN" for value in row]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


print("\nM_A\n" + matrix_table(M_ADDED))
print("\nM_A with reciprocity-only completion\n" + matrix_table(M_ADDED_RECIPROCAL))
print("\nD_1\n" + matrix_table(D_LINEAR))
print("\nD_2\n" + matrix_table(D_QUADRATIC))

identified_subblock = M_ADDED_RECIPROCAL[np.ix_(EXCITED_INDICES, EXCITED_INDICES)]
print("\nM_A identified symmetric-subblock eigenvalues:", np.linalg.eigvalsh(identified_subblock))


def fit_frequency_group(dof_name: str, nominal_frequency_hz: float) -> dict[str, float | int | str]:
    selected = [
        trial
        for trial in TRIALS
        if trial.plan.dof_name == dof_name
        and math.isclose(trial.plan.nominal_frequency_hz, nominal_frequency_hz)
    ]
    compressed = [harmonic_projection(trial) for trial in selected]
    X = np.vstack([item[0] for item in compressed])
    Y = np.vstack([item[1] for item in compressed])
    index = DOF_SPECS[dof_name]["index"]
    beta = huber_fit(X, Y[:, index])
    prediction = X @ beta
    response = Y[:, index]
    r2 = 1.0 - np.sum((response - prediction) ** 2) / np.sum(
        (response - np.mean(response)) ** 2
    )
    return {
        "dof": dof_name,
        "nominal_frequency_hz": nominal_frequency_hz,
        "mean_estimated_frequency_hz": float(np.mean([trial.frequency_hz for trial in selected])),
        "repeats": len(selected),
        "added_mass": float(beta[0]),
        "linear_damping_effective": float(beta[1]),
        "quadratic_damping": float(beta[2]),
        "r2": float(r2),
    }


PER_FREQUENCY = [
    fit_frequency_group(dof_name, frequency)
    for dof_name in DOF_SPECS
    for frequency in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
]
print("\nDirect-channel frequency audit")
print("dof       f_nom  f_est   repeats      M_A        D_1eff       D_2       R²")
for row in PER_FREQUENCY:
    print(
        f"{row['dof']:<9s} {row['nominal_frequency_hz']:5.1f}  "
        f"{row['mean_estimated_frequency_hz']:6.4f}  {row['repeats']:7d}  "
        f"{row['added_mass']:9.4g}  {row['linear_damping_effective']:11.4g}  "
        f"{row['quadratic_damping']:9.4g}  {row['r2']:7.4f}"
    )

diagnostics = []
for trial in TRIALS:
    diagnostics.append(
        {
            "dof": trial.plan.dof_name,
            "repeat": trial.plan.repeat,
            "file_id": trial.plan.file_id,
            "nominal_frequency_hz": trial.plan.nominal_frequency_hz,
            "estimated_frequency_hz": trial.frequency_hz,
            "motion_r2": trial.motion_r2,
            "x_r2": trial.x_r2,
            "samples": len(trial.time_s),
            "gather_source": trial.source_gather,
            "sensor_source": trial.source_sensor,
        }
    )

result = {
    "regression_definition": "tau_h=-M_A@nu_dot-D_1_eff@nu-D_2@(abs(nu)*nu)",
    "frame": "body FLU at center of mass",
    "wrench_order": list(WRENCH_LABELS),
    "velocity_order": list(DOF_LABELS),
    "experimentally_excited_columns": [DOF_LABELS[index] for index in EXCITED_INDICES],
    "unidentified_columns": ["u", "p"],
    "added_mass_identified_columns": M_ADDED.tolist(),
    "added_mass_reciprocity_only_completion": M_ADDED_RECIPROCAL.tolist(),
    "linear_damping_effective_at_towing_condition": D_LINEAR.tolist(),
    "quadratic_damping": D_QUADRATIC.tolist(),
    "diagonal_only_views": {
        "added_mass": M_ADDED_DIAGONAL.tolist(),
        "linear_damping_effective": D_LINEAR_DIAGONAL.tolist(),
        "quadratic_damping": D_QUADRATIC_DIAGONAL.tolist(),
    },
    "mean_towing_speed_m_s_by_experiment": {
        name: float(np.mean(np.concatenate([trial.nu[:, 0] for trial in TRIALS if trial.plan.dof_name == name])))
        for name in DOF_SPECS
    },
    "sensor_to_motion_shift_s": SENSOR_TO_MOTION_SHIFT_S,
    "vertical_data_revision": GIT_FALLBACK_REVISION,
    "notes": [
        "NaN means the input DOF was not independently excited; it is not zero.",
        "D_1_eff includes towing-speed-dependent in-phase and unseparated added-mass-Coriolis effects.",
        "Do not combine D_1_eff unchanged with a separate full C_A implementation until u/p added mass is supplied and C_A is separated.",
        "Material/assembly volume from SolidWorks is recorded but is not treated as displaced volume.",
        "The negative heave added-mass result is preserved; the data/sign/timing chain must be resolved before deployment.",
    ],
}


def strict_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [strict_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: strict_json_value(item) for key, item in value.items()}
    return value

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for filename, matrix in (
    ("added_mass.csv", M_ADDED),
    ("added_mass_reciprocity_only.csv", M_ADDED_RECIPROCAL),
    ("linear_damping_effective.csv", D_LINEAR),
    ("quadratic_damping.csv", D_QUADRATIC),
    ("added_mass_diagonal_identified.csv", M_ADDED_DIAGONAL),
    ("linear_damping_diagonal_identified.csv", D_LINEAR_DIAGONAL),
    ("quadratic_damping_diagonal_identified.csv", D_QUADRATIC_DIAGONAL),
):
    np.savetxt(
        OUTPUT_DIR / filename,
        matrix,
        delimiter=",",
        header=",".join(DOF_LABELS),
        comments="",
    )
(OUTPUT_DIR / "hydrodynamic_matrices.json").write_text(
    json.dumps(strict_json_value(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
(OUTPUT_DIR / "trial_diagnostics.json").write_text(
    json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(OUTPUT_DIR / "per_frequency_diagnostics.json").write_text(
    json.dumps(PER_FREQUENCY, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
(OUTPUT_DIR / "REPORT.md").write_text(
    "# PMM 6×6 水动力辨识结果\n\n"
    "模型顺序为 wrench `[X,Y,Z,K,M,N]`、velocity `[u,v,w,p,q,r]`，参考点为质心，"
    "坐标为艇体 FLU。回归式为 `tau_h=-M_A@nu_dot-D_1_eff@nu-D_2@(abs(nu)*nu)`。\n\n"
    "## 附加质量：实验列\n\n"
    + markdown_matrix(M_ADDED)
    + "\n\n## 附加质量：只用互易性补齐\n\n"
    + markdown_matrix(M_ADDED_RECIPROCAL)
    + "\n\n## 一次阻力（试验拖航状态下的等效项）\n\n"
    + markdown_matrix(D_LINEAR)
    + "\n\n## 二次阻力\n\n"
    + markdown_matrix(D_QUADRATIC)
    + "\n\n## 必须保留的结论\n\n"
    "- `u` 与 `p` 没有独立振荡，相关输入列为 `NaN`，不是零。\n"
    "- vertical/heave 给出负附加质量，互易补齐后的已测子块也有负特征值；未取绝对值、未裁剪。\n"
    "- `D_1_eff` 吸收了拖航速度相关同相项和未分离的附加质量 Coriolis 项；"
    "在补齐 `u/p` 附加质量并显式分离 `C_A` 前，不应直接与仿真器的完整 `C_A` 重复叠加。\n"
    "- Downloads 中缺少 42 条 vertical 试次；本次运行从 Git 历史对象 `"
    + GIT_FALLBACK_REVISION
    + "` 只读恢复这些原始记录，未恢复或改写用户删除的工作树文件。\n",
    encoding="utf-8",
)
print("\noutputs:", OUTPUT_DIR)

# %% [markdown]
# ## 如何解释最终输出
#
# notebook 同时输出原始的 6×6 实验列、仅对附加质量应用互易性后的矩阵，以及
# 对角视图。`NaN` 是可辨识性结论：现有 PMM 只激励了 `v,w,q,r`，没有纯 `u` 与
# 纯 `p` 振荡。更重要的是，vertical/heave 的附加质量为负，且随频率从接近零降到
# 约 -11 kg；这不是数值求导噪声造成的孤立点，不能用绝对值或 PSD 投影掩盖。
#
# 下一步若要得到可直接部署的三张全数值矩阵，需要补齐两类证据：
#
# 1. 纯 surge 与纯 roll 强迫振荡，补齐 `u,p` 输入列；
# 2. vertical 试验的六分力正负号、传感器是否随装配体滚转、以及 gather/sensor
#    触发时差的原始接线/采集记录。确认后重跑本 notebook，而不是在结果端改符号。

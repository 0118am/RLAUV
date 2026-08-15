"""Artifact tables, metadata, plots, and report rendering for PMM fits."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .pmm_common import *
from .pmm_config import PreflightResult, SixDofConfig
from .pmm_trials import SixDofTrial


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_rows(
    preflight_result: PreflightResult,
    built: Mapping[tuple[str, int, int], SixDofTrial],
    config: SixDofConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in preflight_result.audit:
        plan = audit.plan
        trial = built.get((plan.dof, plan.repeat, plan.file_id))
        row: dict[str, Any] = {
            "dof": plan.dof,
            "repeat": plan.repeat,
            "file_id": plan.file_id,
            "nominal_frequency_hz": f"{plan.nominal_frequency_hz:.10g}",
            "estimated_frequency_hz": "" if trial is None else f"{trial.frequency_hz:.10g}",
            "timing_group": plan.timing_group,
            "gather_file": str(plan.gather_path),
            "sensor_file": str(plan.sensor_path),
            "status": audit.status,
            "exclusion_reason": audit.reason,
            "gather_rows": "" if audit.gather_rows is None else audit.gather_rows,
            "sensor_rows": "" if audit.sensor_rows is None else audit.sensor_rows,
            "required_sensor_rows": audit.required_sensor_rows,
            "paired_rows": "",
            "motion_fit_r2": "",
            "u_mean_m_s": "",
            "surge_force_mean_n": "",
            "surge_force_oscillatory_rms_n": "",
            "q_amplitude": "",
            "condition_raw": "",
            "sensor_time_shift_ms": f"{config.shift_ms(plan.timing_group):.10g}",
            "quality_flag": audit.status if audit.status != "included" else "",
        }
        if trial is not None:
            d = trial.diagnostics
            for key in (
                "paired_rows", "motion_fit_r2", "u_mean_m_s", "surge_force_mean_n",
                "surge_force_oscillatory_rms_n", "q_amplitude", "condition_raw",
            ):
                row[key] = f"{d[key]:.10g}"
            flags: list[str] = []
            if d["motion_fit_r2"] < float(config.quality["minimum_motion_fit_r2"]):
                flags.append("motion_fit_low")
            ratio = d["frequency_ratio_to_nominal"]
            lo = float(config.frequency_estimation["search_lower_fraction"])
            hi = float(config.frequency_estimation["search_upper_fraction"])
            if math.isclose(ratio, lo, rel_tol=0.0, abs_tol=1e-6) or math.isclose(ratio, hi, rel_tol=0.0, abs_tol=1e-6):
                flags.append("frequency_search_boundary")
            row["quality_flag"] = ";".join(flags) if flags else "ok"
        rows.append(row)
    return rows


def _make_plot(path: Path, panels: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3 if len(panels) > 4 else 2
    rows = int(math.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (label, result) in zip(axes_array, panels):
        y = np.asarray(result["y"])
        prediction = np.asarray(result["prediction"])
        stride = max(1, len(y) // 5000)
        axis.scatter(y[::stride], prediction[::stride], s=5, alpha=0.22, rasterized=True)
        lo = float(min(np.min(y), np.min(prediction)))
        hi = float(max(np.max(y), np.max(prediction)))
        axis.plot([lo, hi], [lo, hi], color="#b91c1c", lw=1.4)
        axis.set_title(f"{label}: R2={result['full_r2']:.3f}")
        axis.set_xlabel("detrended hydrodynamic load")
        axis.set_ylabel("prediction")
        axis.grid(alpha=0.25)
    for axis in axes_array[len(panels):]:
        axis.set_visible(False)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _metadata(
    config: SixDofConfig,
    preflight_result: PreflightResult,
    trials_by_dof: Mapping[str, Sequence[SixDofTrial]],
    results: Mapping[tuple[str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_scope": "six_dof_diagonal_per_frequency_hybrid",
        "result_scale": {
            "wet_length_m": float(config.model["wet_length_m"]),
            "geometry_scale_of_real_robot": float(config.model["geometry_scale_of_real_robot"]),
            "PMM_and_CFD_length_ratio": 1.0,
            "coefficient_scale_factor_applied": 1.0,
            "full_scale_conversion_performed": False,
        },
        "raw_data_policy": "read_only",
        "planned_trials": len(preflight_result.planned),
        "included_trials": len(preflight_result.included),
        "excluded_trials": sum(row.status == "excluded" for row in preflight_result.audit),
        "included_trials_by_dof": {dof: len(trials_by_dof[dof]) for dof in DOFS},
        "config": config.resolved_dict(),
        "coordinate_transform": {
            "apparatus_frame": "H FLU: x forward, y left, z up",
            "body_frame": "B FLU: x forward, y left, z up",
            "sensor_frame": "upright raw balance frame S; unchanged between horizontal and vertical tests",
            "sensor_mount_status": config.apparatus["sensor_mount_status"],
            "horizontal_R_HB": config.apparatus["body_to_apparatus_rotation_horizontal"],
            "vertical_R_HB": config.apparatus["body_to_apparatus_rotation_vertical"],
            "observed_vertical_encoder_initial_offset_deg": config.apparatus["observed_vertical_encoder_initial_offset_deg"],
            "observed_vertical_encoder_offset_interpretation": config.apparatus["observed_vertical_encoder_offset_interpretation"],
            "assumption": "vertical model is rolled +90 degrees about body FLU +x only; the encoder initial offset is not a body yaw rotation",
            "status": config.apparatus["vertical_mount_status"],
            "derivation": "twists and wrenches are explicitly rotated H to B; changing +90 to -90 reverses each diagonal generalized motion and conjugate load together",
            "scope": "diagonal terms only; no cross-coupling inference",
        },
        "pitch_restoring": {
            "included_in_fit": False,
            "reason": "the rolled-model pitch axis is parallel to gravity; normal-attitude hydrostatic restoring is separate",
        },
        "wrench_and_rigid_body": {
            "raw_sensor_order": list(SENSOR_COLUMNS),
            "sensor_mount": config.apparatus["sensor_mount"],
            "sensor_axes_frame": config.apparatus["sensor_axes_frame"],
            "sensor_to_H_wrench_matrix_horizontal": config.apparatus["sensor_to_H_wrench_matrix_horizontal"],
            "sensor_to_H_wrench_matrix_vertical": config.apparatus["sensor_to_H_wrench_matrix_vertical"],
            "wrench_mapping_scope": "unchanged upright balance: negate the complete raw reaction wrench so [X,Y,Z,K,M,N]=-[FX,FY,FZ,TX,TY,TZ], then apply the model-roll transform",
            "raw_wrench_reference": config.apparatus["wrench_reference"],
            "com_from_motion_origin_flu_m": config.model["com_from_motion_origin_flu_m"],
            "translation_formula": "M_COM=M_origin-r_origin_to_COM cross F using confirmed force channels",
            "rigid_force_formula": "m*(v_dot+omega cross v) at COM",
            "rigid_moment_formula": "I_COM*omega_dot+omega cross (I_COM*omega)",
            "hydrodynamic_target": "measured_wrench_at_COM-rigid_body_wrench_at_COM",
        },
        "time_alignment": {
            group: {
                "sensor_time_shift_ms": config.shift_ms(group),
                "status": config.timing[f"{group}_status"],
                "definition": "sensor(t) is paired with motion(t+shift)",
            }
            for group in ("horizontal", "vertical")
        },
        "frequency_estimation": {
            "method": "per-trial nonlinear grid refinement of constant+trend+three-harmonic motion fit",
            "estimated_frequency_hz_by_trial": {
                f"{trial.plan.dof}/r{trial.plan.repeat}/id{trial.plan.file_id}": trial.frequency_hz
                for dof in DOFS
                for trial in trials_by_dof[dof]
            },
        },
        "load_fit": {
            "method": config.fit["method"],
            "harmonics": config.fit["load_harmonics"],
            "regression": "one Huber fit over repeated trials at each nominal frequency; no cross-frequency pooling",
            "huber_k": config.fit["huber_k"],
            "frequency_group_results": {
                f"{dof}/{frequency:g}Hz": {
                    "included_repeats": int(result["included_repeats"]),
                    "mean_estimated_frequency_hz": float(result["mean_estimated_frequency_hz"]),
                    "estimated_frequency_std_hz": float(result["estimated_frequency_std_hz"]),
                    "full_r2": float(result["full_r2"]),
                    "fit_quality_flag": (
                        "low_load_fit_r2"
                        if float(result["full_r2"])
                        < float(config.quality["minimum_load_fit_r2_for_flag"])
                        else "ok"
                    ),
                    "standardized_condition_number": float(result["condition"]),
                }
                for (dof, frequency), result in sorted(
                    results.items(), key=lambda item: (DOFS.index(item[0][0]), item[0][1])
                )
            },
        },
    }


def _report(
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> str:
    lines = [
        "# PMM 单频率六自由度对角水动力结果",
        "",
        "## 范围",
        "",
        "结果保持在实物 PMM 模型的 `0.562 m` 尺度；PMM 与 CFD 都是真实机器人的 0.7 模型，因此系数缩放因子严格为 `1.0`，没有执行 0.7→1.0 换算。",
        "",
        "每个 DOF 的 0.1–0.7 Hz 七个名义频率分别拟合。每组只合并同一名义频率的重复试次，不跨频率联合回归。矩阵顺序为 `[u,v,w,p,q,r]`；`Y←v`、`Z←w`、`M←q`、`N←r` 来自 PMM，未激励的 surge/roll 使用既有先验补齐。",
        "",
        "六维传感器保持直立，原始值表示模型施加给传感器的反力/反力矩，因此完整取反为 `[X,Y,Z,K,M,N]=-[FX,FY,FZ,TX,TY,TZ]` 后才得到作用在模型上的FLU广义力。vertical仅把模型绕前向F轴旋转 `+90°`，即 `R_HB=Rx(+90°)`；传感器本身不随模型旋转。",
        "",
        f"计划 {metadata['planned_trials']} 条，纳入 {metadata['included_trials']} 条，显式排除 {metadata['excluded_trials']} 条。",
        "",
        "## 湿态质量假设",
        "",
        "材料已确认为PLA，且CAD材料密度低估15%。因此先将CAD质量 `6.4163 kg` 修正为干态 `7.378745 kg`，再采用FFF PLA长期浸水试验的平均增重 `2.5%`，得到湿态刚体质量 `7.563213625 kg`。其中密度修正增加 `0.962445 kg`，材料吸水增加 `0.184468625 kg`。",
        "",
        "约9 kg的体感重量仍比上述PLA湿态估计高约 `1.437 kg`，这部分按自由水/滞留水处理，不纳入刚体扣除；只有能够证明与模型机械锁定、无晃动和交换的水才应计入刚体质量。密度与吸水均暂按均匀分布，故惯量按原CAD张量乘 `1.15×1.025=1.17875`。资料：[PLA/PETG浸水研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC8036839/)、[PLA 28天浸水研究](https://pubs.rsc.org/en/content/articlehtml/2022/ma/d2ma00707j)、[ISO 62对多孔材料的适用性限制](https://www.iso.org/standard/41672.html)。",
        "",
        "## 系数",
        "",
        "| DOF | 名义频率/Hz | 实测均值/Hz | 重复数 | 项 | 系数 | 单位 | R² |",
        "|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in rows:
        term = str(row["term"]).replace("|", "\\|")
        lines.append(
            f"| {row['dof']} | {row['nominal_frequency_hz']} | "
            f"{row['mean_estimated_frequency_hz']} | {row['included_repeats']} | "
            f"{term} | {row['coefficient']} | {row['unit']} | {row['full_r2']} |"
        )
    excluded = [row for row in diagnostics if row["status"] == "excluded"]
    lines.extend(["", "## 数据质量", ""])
    if excluded:
        for row in excluded:
            lines.append(f"- 排除 `{row['sensor_file']}`：{row['exclusion_reason']}")
    else:
        lines.append("- 无排除记录。")
    low_fit_groups = sorted(
        {
            (str(row["dof"]), float(row["nominal_frequency_hz"]), float(row["full_r2"]))
            for row in rows
            if row["fit_quality_flag"] == "low_load_fit_r2"
        },
        key=lambda item: (DOFS.index(item[0]), item[1]),
    )
    if low_fit_groups:
        formatted = "、".join(
            f"{dof} {frequency:g} Hz (R²={score:.3f})"
            for dof, frequency, score in low_fit_groups
        )
        lines.append(f"- 低于载荷拟合提示阈值 R²=0.8：{formatted}。这些组保留但应降低权重。")
    else:
        lines.append("- 所有单频载荷拟合均达到 R²=0.8 提示阈值。")
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- horizontal和vertical都先将六维传感器完整反力取反；vertical仅把模型绕艇体前向F轴旋转+90°。",
            "- 每个试次都保留前进surge速度与X向力：u进入u·q项，完整FX/FY/FZ参与质心力矩平移。没有独立的纯surge振荡加速度，因此surge附加质量仍不能由这些记录单独识别；roll没有试验。",
            "- `u/p` 是文献先验，`v/w/q/r` 是 PMM 试验识别；完整 6×6 是混合来源矩阵。",
            "- 载荷先逐试次投影到各自实测频率的 1–3 次谐波，再按名义频率分组进行 Huber 拟合；不同频率之间没有共享系数。",
            "- PMM刚体扣除使用PLA密度+15%并叠加2.5%吸水后的湿质量7.563213625 kg，惯量为原CAD值的1.17875倍；约9 kg体感值中剩余的自由水不作刚体扣除。",
            "- PMM与CFD属于同一0.562 m、0.7几何模型，本目录所有结果均可直接与CFD模型尺度结果比较。",
            "- Fossen 输出保留原始拟合符号，不会把负阻尼静默取绝对值。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

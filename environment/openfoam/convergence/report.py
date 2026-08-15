"""JSON and Markdown rendering for convergence comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


VARIANTS = ("coarse", "nominal", "fine", "dt", "domain")


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.7g}"


def _markdown_header(report: Mapping[str, Any]) -> list[str]:
    case = report["case"]
    return [
        f"# Convergence report: {case['case_name']}",
        "",
        (
            f"Excitation `{case['dof']}` / main load `{case['main_wrench']}`; "
            f"amplitude `{case['amplitude_si']:.7g}` SI, frequency "
            f"`{case['frequency_hz']:.7g} Hz`. Force sign is fluid-on-body."
        ),
        "",
        "All variants use identical motion definitions and complete sampled cycles.",
        "",
        "| Variant | MA | DL | DQ | D_eff at v_peak | Load amplitude | Load phase (deg) | Odd-fit residual RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]


def _markdown_variant_rows(report: Mapping[str, Any]) -> list[str]:
    variants = report["variants"]
    lines: list[str] = []
    for name in VARIANTS:
        item = variants[name]
        coefficient = item["coefficients"]
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    _format_number(coefficient["added_mass"]),
                    _format_number(coefficient["linear_damping"]),
                    _format_number(coefficient["quadratic_damping"]),
                    _format_number(coefficient["effective_damping_at_peak_speed"]),
                    _format_number(item["main_load"]["amplitude"]),
                    _format_number(item["main_load"]["phase_deg_relative_to_displacement_sine"]),
                    _format_number(item["fit"]["odd_model_residual_rms"]),
                )
            )
            + " |"
        )
    return lines


def _markdown_grid_rows(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## Grid convergence",
        "",
        "| Metric | coarse vs nominal (%) | nominal vs fine (%) | GCI fine (%) | observed order | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    grid_metrics = report["comparisons"]["grid"]["metrics"]
    for name in (
        "added_mass",
        "effective_damping_at_peak_speed",
        "main_load_fundamental_amplitude",
        "odd_model_residual_rms",
    ):
        item = grid_metrics[name]
        gci = item["gci"]
        lines.append(
            f"| {name} | "
            f"{_format_number(item['coarse_vs_nominal']['absolute_relative_difference_percent'])} | "
            f"{_format_number(item['nominal_vs_fine']['absolute_relative_difference_percent'])} | "
            f"{_format_number(gci.get('fine_grid_gci_percent'))} | "
            f"{_format_number(gci.get('observed_order'))} | {gci['status']} |"
        )
    phase = grid_metrics["main_load_phase"]
    phase_gci = phase["gci_degrees"]
    lines.append(
        "| main_load_phase (absolute deg) | "
        f"{_format_number(phase['coarse_vs_nominal']['absolute_difference_deg'])} | "
        f"{_format_number(phase['nominal_vs_fine']['absolute_difference_deg'])} | "
        f"{_format_number(phase_gci.get('fine_grid_gci_absolute'))} | "
        f"{_format_number(phase_gci.get('observed_order'))} | {phase_gci['status']} |"
    )
    return lines


def _markdown_cross_check_rows(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## Time-step and domain checks",
        "",
        "Percentages use the refined-time-step or expanded-domain result as reference; phase is an absolute circular difference.",
        "",
        "| Metric | nominal vs refined dt | nominal vs expanded domain |",
        "|---|---:|---:|",
    ]
    timestep = report["comparisons"]["time_step"]["metrics"]
    domain = report["comparisons"]["domain"]["metrics"]
    for name in (
        "added_mass",
        "effective_damping_at_peak_speed",
        "main_load_fundamental_amplitude",
        "odd_model_residual_rms",
    ):
        lines.append(
            f"| {name} (%) | "
            f"{_format_number(timestep[name]['absolute_relative_difference_percent'])} | "
            f"{_format_number(domain[name]['absolute_relative_difference_percent'])} |"
        )
    lines.extend(
        (
            "| main_load_phase (deg) | "
            f"{_format_number(timestep['main_load_phase']['absolute_difference_deg'])} | "
            f"{_format_number(domain['main_load_phase']['absolute_difference_deg'])} |",
            "",
        )
    )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = _markdown_header(report)
    lines.extend(_markdown_variant_rows(report))
    lines.extend(_markdown_grid_rows(report))
    lines.extend(_markdown_cross_check_rows(report))
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "convergence_report.json"
    markdown_path = destination / "convergence_report.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}

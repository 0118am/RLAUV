"""Fit tether dynamics updates from validated tension and drag CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.identification.fitters import fit_tether_drag_coefficient, fit_tether_spring_damper  # noqa: E402
from environment.profiles.pool_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolProfileAuditOptions,
    pool_profile_calibration_log_schemas,
    validate_pool_calibration_log_directory,
)


TETHER_LOG_FILENAMES = ("tether_tension_samples.csv", "tether_drag_samples.csv")


@dataclass(frozen=True)
class TetherCalibrationPipelineResult:
    cfg_updates: dict[str, Any]
    diagnostics: dict[str, Any]
    source_files: tuple[str, ...]

    def update_payload(self) -> dict[str, Any]:
        return {"cfg_updates": self.cfg_updates, "domain_randomization_updates": {}}

    def report_dict(self) -> dict[str, Any]:
        return {
            "source_files": list(self.source_files),
            "cfg_updates": self.cfg_updates,
            "diagnostics": self.diagnostics,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit tether profile updates from calibration CSV logs.")
    parser.add_argument("log_dir", type=Path, help="Directory containing tether calibration CSV logs.")
    parser.add_argument("--anchor-pos-w", type=float, nargs=3, default=(0.0, 0.0, 8.0))
    parser.add_argument("--attach-offset-b", type=float, nargs=3, default=(-0.2, 0.0, 0.0))
    parser.add_argument("--num-segments", type=int, default=1)
    parser.add_argument("--segment-diameter", type=float, default=0.004)
    parser.add_argument("--segment-density", type=float, default=1100.0)
    parser.add_argument("--segment-buoyancy-density", type=float, default=997.0)
    parser.add_argument("--enable-winch", action="store_true", help="Enable rate-limited tether slack payout/retrieval.")
    parser.add_argument("--winch-target-length", type=float, help="Measured or commanded winch target slack length in m.")
    parser.add_argument("--winch-reel-speed", type=float, default=0.0, help="Maximum winch reel speed in m/s.")
    parser.add_argument("--winch-min-length", type=float, default=0.0, help="Minimum allowed winch slack length in m.")
    parser.add_argument("--winch-max-length", type=float, default=20.0, help="Maximum allowed winch slack length in m.")
    parser.add_argument(
        "--slack-candidates",
        type=float,
        nargs="+",
        help="Optional explicit slack-length candidates in meters.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output builder-compatible updates JSON path.")
    parser.add_argument("--report", type=Path, help="Optional detailed fit diagnostics JSON path.")
    return parser


def fit_tether_calibration_logs(
    log_dir: Path,
    *,
    anchor_pos_w: Sequence[float] = (0.0, 0.0, 8.0),
    attach_offset_b: Sequence[float] = (-0.2, 0.0, 0.0),
    num_segments: int = 1,
    segment_diameter: float = 0.004,
    segment_density: float = 1100.0,
    segment_buoyancy_density: float = 997.0,
    slack_length_candidates: Sequence[float] | None = None,
    winch_enabled: bool = False,
    winch_target_length: float | None = None,
    winch_reel_speed: float = 0.0,
    winch_min_length: float = 0.0,
    winch_max_length: float = 20.0,
) -> TetherCalibrationPipelineResult:
    if len(anchor_pos_w) != 3 or len(attach_offset_b) != 3:
        raise ValueError("anchor_pos_w and attach_offset_b must have length 3.")
    if int(num_segments) != num_segments or int(num_segments) < 1:
        raise ValueError("num_segments must be a positive integer.")
    if float(segment_diameter) <= 0.0:
        raise ValueError("segment_diameter must be positive.")
    if float(segment_density) < 0.0 or float(segment_buoyancy_density) < 0.0:
        raise ValueError("Segment densities must be non-negative.")
    if float(winch_reel_speed) < 0.0:
        raise ValueError("winch_reel_speed must be non-negative.")
    if float(winch_min_length) < 0.0 or float(winch_max_length) < float(winch_min_length):
        raise ValueError("Winch length bounds must satisfy 0 <= min <= max.")
    if winch_target_length is not None and float(winch_target_length) < 0.0:
        raise ValueError("winch_target_length must be non-negative.")

    schemas = pool_profile_calibration_log_schemas(
        NOMINAL_POOL_DYNAMICS_PROFILE,
        PoolProfileAuditOptions(tether_expected=True, domain_randomization_expected=False),
    )
    schema_by_filename = {schema.filename: schema for schema in schemas}
    source_files = tuple(filename for filename in TETHER_LOG_FILENAMES if (log_dir / filename).is_file())
    if not source_files:
        raise ValueError(f"No supported tether calibration logs found in {log_dir}.")
    validation = validate_pool_calibration_log_directory(
        log_dir,
        tuple(schema_by_filename[filename] for filename in source_files),
    )
    if not validation.is_valid:
        messages = "; ".join(
            f"{issue.filename}:{issue.row_number or '-'}:{issue.column or '-'} {issue.message}"
            for issue in validation.issues
            if issue.severity == "error"
        )
        raise ValueError(f"Tether calibration log validation failed: {messages}")

    cfg_updates: dict[str, Any] = {
        "tether_enabled": True,
        "tether_anchor_pos_w": [float(value) for value in anchor_pos_w],
        "tether_attach_offset_b": [float(value) for value in attach_offset_b],
        "tether_num_segments": int(num_segments),
        "tether_segment_diameter": float(segment_diameter),
        "tether_segment_density": float(segment_density),
        "tether_segment_buoyancy_density": float(segment_buoyancy_density),
        "tether_winch_enabled": bool(winch_enabled),
        "tether_winch_target_length": float(winch_target_length if winch_target_length is not None else 2.0),
        "tether_winch_reel_speed": float(winch_reel_speed),
        "tether_winch_min_length": float(winch_min_length),
        "tether_winch_max_length": float(winch_max_length),
    }
    diagnostics: dict[str, Any] = {"validation": validation.to_dict()}

    tension_path = log_dir / "tether_tension_samples.csv"
    if tension_path.is_file():
        rows = _read_csv_rows(tension_path)
        fit = fit_tether_spring_damper(
            _float_column(rows, "length_m"),
            _float_column(rows, "tension_n"),
            _float_column(rows, "velocity_along_tether_mps"),
            slack_length_candidates=slack_length_candidates,
        )
        cfg_updates.update(fit.to_cfg_updates())
        if winch_target_length is None:
            cfg_updates["tether_winch_target_length"] = float(fit.slack_length)
        diagnostics["spring_damper"] = {
            "slack_length_m": fit.slack_length,
            "stiffness_n_m": fit.stiffness,
            "damping_n_s_m": fit.damping,
            "residual_rms_n": fit.residual_rms,
            "sample_count": fit.sample_count,
        }

    drag_path = log_dir / "tether_drag_samples.csv"
    if drag_path.is_file():
        rows = _read_csv_rows(drag_path)
        fit = fit_tether_drag_coefficient(
            _vector_columns(
                rows,
                ("relative_velocity_x_mps", "relative_velocity_y_mps", "relative_velocity_z_mps"),
            ),
            _vector_columns(rows, ("drag_force_x_n", "drag_force_y_n", "drag_force_z_n")),
        )
        cfg_updates.update(fit.to_cfg_updates())
        diagnostics["drag"] = {
            "drag_coeff": fit.drag_coeff,
            "residual_rms_n": fit.residual_rms,
            "sample_count": fit.sample_count,
        }

    return TetherCalibrationPipelineResult(cfg_updates, diagnostics, source_files)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return rows


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _vector_columns(rows: list[dict[str, str]], names: tuple[str, ...]) -> list[list[float]]:
    return [[float(row[name]) for name in names] for row in rows]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = fit_tether_calibration_logs(
            args.log_dir,
            anchor_pos_w=args.anchor_pos_w,
            attach_offset_b=args.attach_offset_b,
            num_segments=args.num_segments,
            segment_diameter=args.segment_diameter,
            segment_density=args.segment_density,
            segment_buoyancy_density=args.segment_buoyancy_density,
            slack_length_candidates=args.slack_candidates,
            winch_enabled=args.enable_winch,
            winch_target_length=args.winch_target_length,
            winch_reel_speed=args.winch_reel_speed,
            winch_min_length=args.winch_min_length,
            winch_max_length=args.winch_max_length,
        )
        _write_json(args.output, result.update_payload())
        if args.report is not None:
            _write_json(args.report, result.report_dict())
    except Exception as exc:
        print(f"Failed to fit tether calibration logs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

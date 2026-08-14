"""Audit a measured pool dynamics profile JSON before training or experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.profiles.pool_profile import (  # noqa: E402
    PoolCalibrationLogSchema,
    PoolCalibrationLogValidationReport,
    PoolCalibrationTask,
    PoolProfileAuditOptions,
    PoolProfileAuditReport,
    audit_pool_dynamics_profile,
    load_pool_dynamics_profile_json,
    pool_profile_calibration_log_schemas,
    pool_profile_calibration_update_template,
    pool_profile_calibration_tasks,
    validate_pool_calibration_log_directory,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a PoolDynamicsProfile JSON for high-fidelity pool simulation readiness.",
    )
    parser.add_argument("profile_json", type=Path, help="Path to a PoolDynamicsProfile JSON file.")
    parser.add_argument("--near-boundaries", action="store_true", help="Expect trajectories near pool walls/floor.")
    parser.add_argument("--near-surface", action="store_true", help="Expect trajectories near the free surface.")
    parser.add_argument("--tether", action="store_true", help="Expect a physical tether/safety cable.")
    parser.add_argument("--spatial-current", action="store_true", help="Expect spatially varying water current.")
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="Do not warn when domain_randomization is absent.",
    )
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON.")
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="Print experiment-oriented calibration tasks instead of the audit summary.",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Print a JSON skeleton for missing calibration cfg/domain update values.",
    )
    parser.add_argument(
        "--log-schemas",
        action="store_true",
        help="Print JSON schemas for experiment log CSV files implied by the calibration tasks.",
    )
    parser.add_argument(
        "--write-log-templates",
        type=Path,
        help="Write empty CSV header templates and schemas.json to this directory.",
    )
    parser.add_argument(
        "--validate-log-dir",
        type=Path,
        help="Validate experiment CSV files in this directory against the generated schemas.",
    )
    parser.add_argument(
        "--max-log-issues",
        type=int,
        default=50,
        help="Maximum validation errors reported per CSV file.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with code 2 when warnings or critical findings are present.",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 2 when critical findings are present.",
    )
    return parser


def audit_options_from_args(args: argparse.Namespace) -> PoolProfileAuditOptions:
    return PoolProfileAuditOptions(
        near_boundaries_expected=bool(args.near_boundaries),
        near_surface_expected=bool(args.near_surface),
        tether_expected=bool(args.tether),
        spatial_current_expected=bool(args.spatial_current),
        domain_randomization_expected=not bool(args.no_domain_randomization),
    )


def load_and_audit_profile(profile_path: Path, options: PoolProfileAuditOptions) -> PoolProfileAuditReport:
    profile = load_pool_dynamics_profile_json(profile_path)
    return audit_pool_dynamics_profile(profile, options)


def load_calibration_tasks(profile_path: Path, options: PoolProfileAuditOptions) -> tuple[PoolCalibrationTask, ...]:
    profile = load_pool_dynamics_profile_json(profile_path)
    return pool_profile_calibration_tasks(profile, options)


def load_calibration_update_template(profile_path: Path, options: PoolProfileAuditOptions) -> dict:
    profile = load_pool_dynamics_profile_json(profile_path)
    return pool_profile_calibration_update_template(profile, options)


def load_calibration_log_schemas(
    profile_path: Path,
    options: PoolProfileAuditOptions,
) -> tuple[PoolCalibrationLogSchema, ...]:
    profile = load_pool_dynamics_profile_json(profile_path)
    return pool_profile_calibration_log_schemas(profile, options)


def write_calibration_log_templates(directory: Path, schemas: tuple[PoolCalibrationLogSchema, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for schema in schemas:
        target = directory / schema.filename
        target.write_text(",".join(schema.csv_header) + "\n", encoding="utf-8")
    (directory / "schemas.json").write_text(
        json.dumps([schema.to_dict() for schema in schemas], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_calibration_log_validation_report(report: PoolCalibrationLogValidationReport) -> str:
    lines = [
        f"Log directory: {report.directory}",
        f"Valid: {'yes' if report.is_valid else 'no'}",
        f"Files: {len(report.expected_files)} expected",
        f"Issues: {report.error_count} errors, {report.warning_count} warnings",
    ]
    for issue in report.issues:
        location = issue.filename
        if issue.row_number is not None:
            location += f":{issue.row_number}"
        if issue.column is not None:
            location += f" [{issue.column}]"
        lines.append(f"[{issue.severity.upper()}] {location}: {issue.message}")
    return "\n".join(lines)


def exit_code_for_log_validation(report: PoolCalibrationLogValidationReport) -> int:
    return 0 if report.is_valid else 2


def format_audit_report(report: PoolProfileAuditReport) -> str:
    lines = [
        f"Profile: {report.profile_name}",
        f"Readiness score: {report.readiness_score:.2f}",
        "Findings: "
        f"{report.counts_by_severity.get('critical', 0)} critical, "
        f"{report.counts_by_severity.get('warning', 0)} warning, "
        f"{report.counts_by_severity.get('info', 0)} info",
    ]
    for finding in report.findings:
        lines.append("")
        lines.append(f"[{finding.severity.upper()}] {finding.section}")
        lines.append(f"  {finding.message}")
        lines.append(f"  Recommendation: {finding.recommendation}")
    return "\n".join(lines)


def format_calibration_tasks(profile_name: str, tasks: tuple[PoolCalibrationTask, ...]) -> str:
    lines = [
        f"Profile: {profile_name}",
        f"Calibration tasks: {len(tasks)}",
    ]
    for index, task in enumerate(tasks, start=1):
        lines.append("")
        lines.append(f"{index}. [{task.priority}] {task.section} ({task.severity})")
        lines.append(f"   {task.title}")
        lines.append(f"   Reason: {task.reason}")
        lines.append(f"   Experiment: {task.experiment}")
        if task.calibration_functions:
            lines.append(f"   Functions: {', '.join(task.calibration_functions)}")
        if task.update_keys:
            lines.append(f"   Update keys: {', '.join(task.update_keys)}")
    return "\n".join(lines)


def exit_code_for_report(
    report: PoolProfileAuditReport,
    *,
    fail_on_warning: bool,
    fail_on_critical: bool,
) -> int:
    counts = report.counts_by_severity
    if fail_on_warning and (counts.get("warning", 0) > 0 or counts.get("critical", 0) > 0):
        return 2
    if fail_on_critical and counts.get("critical", 0) > 0:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        options = audit_options_from_args(args)
        if args.validate_log_dir is not None:
            log_schemas = load_calibration_log_schemas(args.profile_json, options)
            log_validation = validate_pool_calibration_log_directory(
                args.validate_log_dir,
                log_schemas,
                max_issues_per_file=args.max_log_issues,
            )
        elif args.log_schemas or args.write_log_templates is not None:
            log_schemas = load_calibration_log_schemas(args.profile_json, options)
        elif args.template:
            template = load_calibration_update_template(args.profile_json, options)
        elif args.checklist:
            profile = load_pool_dynamics_profile_json(args.profile_json)
            tasks = pool_profile_calibration_tasks(profile, options)
        else:
            report = load_and_audit_profile(args.profile_json, options)
    except Exception as exc:
        print(f"Failed to audit profile: {exc}", file=sys.stderr)
        return 1

    if args.validate_log_dir is not None:
        if args.json:
            print(json.dumps(log_validation.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_calibration_log_validation_report(log_validation))
        return exit_code_for_log_validation(log_validation)

    if args.log_schemas or args.write_log_templates is not None:
        if args.write_log_templates is not None:
            write_calibration_log_templates(args.write_log_templates, log_schemas)
        if args.log_schemas or args.write_log_templates is None:
            print(json.dumps([schema.to_dict() for schema in log_schemas], indent=2, sort_keys=True))
        return 0

    if args.template:
        print(json.dumps(template, indent=2, sort_keys=True))
        return 0

    if args.checklist:
        if args.json:
            print(json.dumps([task.to_dict() for task in tasks], indent=2, sort_keys=True))
        else:
            print(format_calibration_tasks(profile.name, tasks))
        return 0

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_audit_report(report))
    return exit_code_for_report(
        report,
        fail_on_warning=bool(args.fail_on_warning),
        fail_on_critical=bool(args.fail_on_critical),
    )


if __name__ == "__main__":
    raise SystemExit(main())

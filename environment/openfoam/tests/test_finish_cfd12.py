from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from environment.openfoam import finish_cfd12


class FinishCfd12Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = self.root / "cases"
        self.cases.mkdir()
        self.config = self.root / "campaign.json"
        self.config_payload = {
            "schema_version": 1,
            "openfoam_version": "v2512",
            "solver": "pimpleFoam",
            "translation_amplitudes_m": [0.01, 0.025],
            "rotation_amplitudes_deg": [2.0, 5.0],
            "frequencies_hz": [1.5],
            "settle_cycles": 2,
            "sample_cycles": 4,
        }
        self.config.write_text(json.dumps(self.config_payload), encoding="utf-8")
        self.case_names_by_dof: dict[str, list[str]] = {}
        for dof_index, dof in enumerate(("u", "v", "w", "p", "q", "r")):
            rotation = dof_index >= 3
            amplitudes = [2.0, 5.0] if rotation else [0.01, 0.025]
            names: list[str] = []
            for amplitude in amplitudes:
                suffix = f"{amplitude:.3f}" if not rotation else f"{amplitude:.1f}"
                name = f"{dof}_amp{suffix}_f1p50hz"
                names.append(name)
                case = self.cases / name
                case.mkdir()
                axis = [0, 0, 0]
                axis[dof_index if dof_index < 3 else dof_index - 3] = 1
                metadata = {
                    "schema_version": 1,
                    "openfoam_version": "v2512",
                    "solver": "pimpleFoam",
                    "case_name": name,
                    "dof": dof,
                    "dof_index": dof_index,
                    "kind": "rotation" if rotation else "translation",
                    "axis": axis,
                    "amplitude_m": None if rotation else amplitude,
                    "amplitude_deg": amplitude if rotation else None,
                    "amplitude_rad": math.radians(amplitude) if rotation else None,
                    "frequency_hz": 1.5,
                    "omega_rad_s": 3.0 * math.pi,
                    "settle_cycles": 2,
                    "sample_cycles": 4,
                    "purpose": "identification",
                    "include_in_fit": True,
                }
                (case / "motion.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.case_names_by_dof[dof] = names
        mesh = self.cases / "mesh_case"
        mesh.mkdir()
        (mesh / "motion.json").write_text(
            json.dumps({"purpose": "shared_mesh", "dof": None, "include_in_fit": False}),
            encoding="utf-8",
        )

    def _campaign(self) -> finish_cfd12.Campaign:
        return finish_cfd12._load_campaign(self.cases, self.config)

    def _write_markers(self, schema_version: int = 2) -> None:
        for case in self._campaign().case_dirs:
            (case / ".completed").write_text(
                json.dumps({"schema_version": schema_version}), encoding="utf-8"
            )

    def _write_valid_fit_outputs(
        self,
        destination: Path,
        *,
        bootstrap_samples: int = 200,
        passivity_samples: int = 10000,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        zero = np.zeros((6, 6)).tolist()
        identity = np.eye(6).tolist()
        cases = []
        fits = {}
        convergence = []
        for dof in ("u", "v", "w", "p", "q", "r"):
            names = self.case_names_by_dof[dof]
            fits[dof] = {
                "rank": 3,
                "case_names": names,
                "complete_cycles_by_case": {name: 4 for name in names},
            }
            for name in names:
                cases.append({"case_name": name, "dof": dof, "sample_cycles": 4})
                convergence.append(
                    {
                        "case_name": name,
                        "cycles": [
                            {"cycle_id": cycle, "fit": {"rank": 3}}
                            for cycle in range(4)
                        ],
                        "last_two_cycle_comparison": {"available": True},
                    }
                )
        updates = {
            "added_mass_diag": identity,
            "linear_damping": identity,
            "quadratic_damping": identity,
        }
        report = {
            "schema_version": 1,
            "matrices": {
                "added_mass_raw": identity,
                "added_mass": identity,
                "linear_damping": identity,
                "quadratic_damping": identity,
            },
            "config_updates": updates,
            "diagnostics": {
                "fit_by_excited_dof": fits,
                "cycle_convergence_by_case": convergence,
                "added_mass_projection": {"enabled": True},
                "passivity": {
                    "observed_negative_fraction": 0.0,
                    "random_negative_fraction": 0.0,
                    "random_sample_count": passivity_samples,
                },
            },
            "confidence_intervals": (
                {"samples": bootstrap_samples} if bootstrap_samples else {}
            ),
            "cases": cases,
            "options": {
                "project_added_mass_psd": True,
                "bootstrap_samples": bootstrap_samples,
                "passivity_samples": passivity_samples,
            },
        }
        (destination / "hydrodynamic_fit.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (destination / "config_updates.json").write_text(
            json.dumps(updates), encoding="utf-8"
        )
        for name in (
            "added_mass.csv",
            "added_mass_raw.csv",
            "linear_damping.csv",
            "quadratic_damping.csv",
        ):
            (destination / name).write_text("matrix\n", encoding="utf-8")

    def test_campaign_is_exactly_two_amplitudes_for_every_dof_at_1p5_hz(self) -> None:
        campaign = self._campaign()

        self.assertEqual(len(campaign.case_dirs), 12)
        self.assertEqual(campaign.solver, "pimpleFoam")
        self.assertEqual(set(campaign.case_names), set(sum(self.case_names_by_dof.values(), [])))

        extra = self.cases / "u_extra"
        extra.mkdir()
        payload = json.loads((campaign.case_dirs[0] / "motion.json").read_text(encoding="utf-8"))
        payload["case_name"] = extra.name
        (extra / "motion.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(finish_cfd12.FinishFailure, "exactly 12"):
            self._campaign()

    def test_completion_requires_schema_v2_and_revalidates_runner_evidence(self) -> None:
        campaign = self._campaign()
        self._write_markers(schema_version=1)
        with mock.patch.object(
            finish_cfd12.run_cases, "_validated_completion", return_value=(True, "ok")
        ) as validator:
            snapshot = finish_cfd12._completion_snapshot(campaign)
        self.assertFalse(snapshot.complete)
        self.assertEqual(validator.call_count, 0)

        self._write_markers(schema_version=2)
        with mock.patch.object(
            finish_cfd12.run_cases, "_validated_completion", return_value=(True, "ok")
        ) as validator:
            snapshot = finish_cfd12._completion_snapshot(campaign)
        self.assertTrue(snapshot.complete)
        self.assertEqual(validator.call_count, 12)

    def test_runner_log_watcher_reads_existing_and_appended_failures(self) -> None:
        path = self.root / "runner.log"
        path.write_text("[start] u\n", encoding="utf-8")
        watcher = finish_cfd12.RunnerLogWatcher(path)
        self.assertIsNone(watcher.failure())
        with path.open("a", encoding="utf-8") as stream:
            stream.write("[fail] u: pimpleFoam failed; see log\n")
        self.assertIn("[fail]", watcher.failure() or "")

        path.write_text("Traceback (most recent call last):\n", encoding="utf-8")
        self.assertIn("Traceback", watcher.failure() or "")

    def test_process_probe_matches_only_runner_for_exact_cases_dir(self) -> None:
        proc = self.root / "proc"
        proc.mkdir()
        for pid, target in ((101, self.cases), (102, self.root / "other")):
            process = proc / str(pid)
            process.mkdir()
            (process / "cmdline").write_bytes(
                b"python3\0environment/openfoam/run_cases.py\0--cases-dir\0"
                + str(target).encode()
                + b"\0"
            )
            os.symlink(self.root, process / "cwd")
        self.assertEqual(
            finish_cfd12._find_runner_processes(self.cases, proc_root=proc), {101}
        )

    def test_monitor_fails_when_observed_runners_disappear_incomplete(self) -> None:
        campaign = self._campaign()
        self._write_markers(schema_version=2)
        probes = iter(({321}, set()))
        with mock.patch.object(
            finish_cfd12.run_cases,
            "_validated_completion",
            return_value=(False, "force.dat truncated"),
        ):
            with self.assertRaisesRegex(
                finish_cfd12.FinishFailure, "exited before the campaign completed"
            ) as caught:
                finish_cfd12._wait_for_completion(
                    campaign,
                    [],
                    0.01,
                    process_probe=lambda _: next(probes),
                    sleep=lambda _: None,
                )
        self.assertEqual(caught.exception.details["observed_runner_pids"], [321])

    def test_fit_acceptance_rejects_nonzero_passivity_fraction(self) -> None:
        staging = self.root / "staging"
        self._write_valid_fit_outputs(staging)
        acceptance = finish_cfd12._validate_fit_outputs(
            staging, self._campaign(), 200, 10000
        )
        self.assertTrue(acceptance["added_mass_psd"])

        report_path = staging / "hydrodynamic_fit.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["diagnostics"]["passivity"]["random_negative_fraction"] = 0.001
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(finish_cfd12.FinishFailure, "must be exactly zero"):
            finish_cfd12._validate_fit_outputs(staging, self._campaign(), 200, 10000)

    def test_fit_runs_in_staging_and_publishes_a_complete_directory(self) -> None:
        campaign = self._campaign()
        output = self.root / "published"
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> SimpleNamespace:
            commands.append(command)
            staging = Path(command[command.index("--output-dir") + 1])
            self._write_valid_fit_outputs(staging)
            return SimpleNamespace(returncode=0, stdout="fit complete\n")

        complete = finish_cfd12.CompletionSnapshot(campaign.case_dirs, {})
        with mock.patch.object(finish_cfd12.subprocess, "run", side_effect=fake_run), mock.patch.object(
            finish_cfd12, "_completion_snapshot", return_value=complete
        ):
            acceptance = finish_cfd12._fit_and_publish(campaign, output, 200, 10000)

        self.assertTrue(acceptance["config_updates_consistent"])
        self.assertTrue((output / "finish_status.json").is_file())
        self.assertTrue((output / "hydrodynamic_fit.json").is_file())
        self.assertIn("--project-added-mass-psd", commands[0])
        self.assertFalse(list(self.root.glob(".published.staging-*")))

        # A restart is idempotent only for the same accepted publication.
        with mock.patch.object(finish_cfd12.subprocess, "run") as runner:
            finish_cfd12._fit_and_publish(campaign, output, 200, 10000)
        runner.assert_not_called()

    def test_main_writes_atomic_failure_record_without_partial_output(self) -> None:
        bad_config = self.root / "bad.json"
        bad_config.write_text("{}\n", encoding="utf-8")
        output = self.root / "result"
        status = finish_cfd12.main(
            [
                "--cases-dir",
                str(self.cases),
                "--config",
                str(bad_config),
                "--output-dir",
                str(output),
                "--wait-seconds",
                "0.01",
            ]
        )

        self.assertEqual(status, 1)
        self.assertFalse(output.exists())
        failure = output.with_name("result.failure.json")
        payload = json.loads(failure.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "failed")
        self.assertIn("v2512 pimpleFoam", payload["reason"])


if __name__ == "__main__":
    unittest.main()

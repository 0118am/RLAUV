from __future__ import annotations

import contextlib
import concurrent.futures
import io
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from environment.openfoam import run_cases


class RunCaseIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = self.root / "u_amp0p025m_f1p50hz"
        (self.case / "constant" / "polyMesh").mkdir(parents=True)
        (self.case / "constant" / "polyMesh" / "boundary").write_text("boundary\n", encoding="utf-8")
        self.motion = {
            "schema_version": 1,
            "openfoam_version": "v2512",
            "solver": "pimpleFoam",
            "case_name": self.case.name,
            "purpose": "identification",
            "settle_end_s": 1.0,
            "end_time_s": 2.0,
        }
        (self.case / "motion.json").write_text(json.dumps(self.motion), encoding="utf-8")
        self._write_log()
        self._write_split_output()

    def _write_log(self, text: str | None = None) -> None:
        if text is None:
            text = (
                "Create time\n\n"
                "Time = 1\n"
                "Courant Number mean: 0.1 max: 0.2\n"
                "Time = 2\n"
                "ExecutionTime = 10 s  ClockTime = 11 s\n\n"
                "End\n"
            )
        (self.case / "log.pimpleFoam").write_text(text, encoding="utf-8")

    @staticmethod
    def _vector_file(times: list[float], *, value: str = "1") -> str:
        lines = [
            "# Time total_x total_y total_z pressure_x pressure_y pressure_z "
            "viscous_x viscous_y viscous_z\n"
        ]
        for time_s in times:
            lines.append(f"{time_s:.12g} {value} 2 3 {value} 2 3 0 0 0\n")
        return "".join(lines)

    def _write_split_output(
        self,
        force_times: list[float] | None = None,
        moment_times: list[float] | None = None,
    ) -> Path:
        output = self.case / "postProcessing" / "forces" / "1"
        output.mkdir(parents=True, exist_ok=True)
        force = [1.0, 1.5, 2.0] if force_times is None else force_times
        moment = force if moment_times is None else moment_times
        (output / "force.dat").write_text(self._vector_file(force), encoding="utf-8")
        (output / "moment.dat").write_text(self._vector_file(moment), encoding="utf-8")
        return output

    def _write_valid_marker(self) -> None:
        validation = run_cases._validate_case_outputs(self.case, "pimpleFoam", self.motion)
        marker = {
            "schema_version": run_cases._MARKER_SCHEMA_VERSION,
            "status": "completed",
            "case": self.case.name,
            "solver": "pimpleFoam",
            "foam_api": "2512",
            "mpi_ranks": 1,
            "motion": self.motion,
            "validation": validation,
            "elapsed_s": 12.5,
        }
        (self.case / ".completed").write_text(json.dumps(marker), encoding="utf-8")

    def test_valid_v2512_outputs_may_start_at_settle_end(self) -> None:
        evidence = run_cases._validate_case_outputs(self.case, "pimpleFoam")

        self.assertEqual(evidence["solver_end_time_s"], 2.0)
        self.assertEqual(evidence["force_end_time_s"], 2.0)
        self.assertEqual(evidence["moment_end_time_s"], 2.0)
        self.assertEqual(evidence["force_files"], ["postProcessing/forces/1/force.dat"])

    def test_suffixed_restart_pair_completes_an_old_partial_segment(self) -> None:
        output = self.case / "postProcessing" / "forces" / "1"
        self._write_split_output(force_times=[1.0, 1.5], moment_times=[1.0, 1.5])
        with self.assertRaisesRegex(RuntimeError, "force.dat ends at"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

        (output / "force_0.dat").write_text(
            self._vector_file([1.5, 2.0]), encoding="utf-8"
        )
        (output / "moment_0.dat").write_text(
            self._vector_file([1.5, 2.0]), encoding="utf-8"
        )
        evidence = run_cases._validate_case_outputs(self.case, "pimpleFoam")
        self.assertEqual(evidence["force_end_time_s"], 2.0)
        self.assertEqual(
            evidence["force_files"],
            [
                "postProcessing/forces/1/force.dat",
                "postProcessing/forces/1/force_0.dat",
            ],
        )

        # A later restart is authoritative even if the older file happened
        # to cover endTime; truncation of the newest suffix must fail closed.
        self._write_split_output()
        (output / "force_0.dat").write_text(
            self._vector_file([1.5, 1.9]), encoding="utf-8"
        )
        (output / "moment_0.dat").write_text(
            self._vector_file([1.5, 1.9]), encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "force.dat ends at"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

        self._write_split_output(force_times=[1.0, 1.5], moment_times=[1.0, 1.5])
        (output / "force_0.dat").unlink()
        (output / "moment_0.dat").unlink()
        with self.assertRaisesRegex(RuntimeError, "force.dat ends at"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

    def test_solver_log_rejects_abnormal_end_and_obvious_failure_tokens(self) -> None:
        bad_logs = {
            "missing End": "Time = 2\nExecutionTime = 10 s\n",
            "fatal": "Time = 2\nFOAM FATAL ERROR: bad mesh\nEnd\n",
            "negativeVolumeCells": "Time = 2\nnegativeVolumeCells: 4\nEnd\n",
            "negative minimum": "Time = 2\nminimum volume = -1e-12\nEnd\n",
            "NaN": "Time = 2\nCourant Number mean: nan max: nan\nEnd\n",
        }
        for label, content in bad_logs.items():
            with self.subTest(label=label):
                self._write_log(content)
                with self.assertRaises(RuntimeError):
                    run_cases._validate_case_outputs(self.case, "pimpleFoam")

    def test_solver_time_not_wall_time_must_cover_motion_end(self) -> None:
        self._write_log("Time = 1.9\nExecutionTime = 500 s  ClockTime = 500 s\nEnd\n")

        with self.assertRaisesRegex(RuntimeError, "solver log ends at"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

    def test_benign_negative_volume_wording_does_not_false_fail(self) -> None:
        for wording in (
            "No cells with negative volume",
            "0 negativeVolumeCells",
            "minimum volume remains non-negative",
        ):
            with self.subTest(wording=wording):
                self._write_log(f"Time = 2\n{wording}\nExecutionTime = 1 s\nEnd\n")
                run_cases._validate_case_outputs(self.case, "pimpleFoam")

    def test_split_outputs_must_be_paired_finite_aligned_and_reach_end(self) -> None:
        output = self.case / "postProcessing" / "forces" / "1"

        (output / "moment.dat").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing OpenCFD v2512"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

        self._write_split_output(moment_times=[1.0, 1.5, 1.9])
        with self.assertRaisesRegex(RuntimeError, "times do not align"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

        self._write_split_output(moment_times=[1.0, 1.5, 2.000001])
        with self.assertRaisesRegex(RuntimeError, "times do not align"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

        self._write_split_output(force_times=[1.0, 2.0], moment_times=[1.0, 1.5])
        crossed = self.case / "postProcessing" / "forces" / "1.5"
        crossed.mkdir(parents=True)
        (crossed / "force.dat").write_text(self._vector_file([1.5]), encoding="utf-8")
        (crossed / "moment.dat").write_text(self._vector_file([1.5, 2.0]), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "times do not align"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")
        shutil.rmtree(crossed)

        self._write_split_output()
        (output / "force.dat").write_text(self._vector_file([1.0, 2.0], value="nan"), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "non-finite force output"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

        (output / "force.dat").write_text("2 1 2 3 4\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "malformed v2512"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

    def test_legacy_combined_forces_file_does_not_satisfy_v2512_gate(self) -> None:
        output = self.case / "postProcessing" / "forces" / "1"
        (output / "force.dat").unlink()
        (output / "moment.dat").unlink()
        (output / "forces.dat").write_text(
            "2 ((1 2 3) (0 0 0)) ((4 5 6) (0 0 0))\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "missing OpenCFD v2512"):
            run_cases._validate_case_outputs(self.case, "pimpleFoam")

    def test_resume_skips_only_when_marker_and_current_outputs_validate(self) -> None:
        self._write_valid_marker()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_cases._discover(self.root, [], True, "pimpleFoam"), [])

        (self.case / "postProcessing" / "forces" / "1" / "moment.dat").unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_cases._discover(self.root, [], True, "pimpleFoam"), [self.case])

    def test_resume_reruns_malformed_or_stale_marker(self) -> None:
        marker = self.case / ".completed"
        for label, content in {
            "malformed": "not json\n",
            "old schema": json.dumps({"case": self.case.name, "solver": "pimpleFoam"}),
        }.items():
            with self.subTest(label=label):
                marker.write_text(content, encoding="utf-8")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run_cases._discover(self.root, [], True, "pimpleFoam"), [self.case])

        self._write_valid_marker()
        changed = dict(self.motion, end_time_s=2.5)
        (self.case / "motion.json").write_text(json.dumps(changed), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_cases._discover(self.root, [], True, "pimpleFoam"), [self.case])

    def test_run_one_removes_stale_marker_and_writes_only_after_validation(self) -> None:
        marker = self.case / ".completed"
        marker.write_text("stale\n", encoding="utf-8")

        with mock.patch.object(
            run_cases.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                run_cases._run_one(self.case, "pimpleFoam", 1, False, False)
        self.assertFalse(marker.exists())

        def successful_process(command: list[str], **kwargs: object) -> SimpleNamespace:
            stream = kwargs["stdout"]
            stream.write("Time = 2\nExecutionTime = 1 s\nEnd\n")
            return SimpleNamespace(returncode=0)

        with mock.patch.object(run_cases.subprocess, "run", side_effect=successful_process):
            run_cases._run_one(self.case, "pimpleFoam", 1, False, False)
        payload = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["foam_api"], "2512")
        self.assertFalse(payload["bind_to_core"])
        self.assertEqual(payload["validation"]["force_end_time_s"], 2.0)

        (self.case / "postProcessing" / "forces" / "1" / "moment.dat").unlink()
        with mock.patch.object(run_cases.subprocess, "run", side_effect=successful_process):
            with self.assertRaisesRegex(RuntimeError, "missing OpenCFD v2512"):
                run_cases._run_one(self.case, "pimpleFoam", 1, False, False)
        self.assertFalse(marker.exists())

    def test_main_requires_exact_foam_api_for_real_runs(self) -> None:
        argv = ["run_cases.py", "--cases-dir", str(self.root), "--resume"]
        for api in (None, "2506", "2512.0"):
            environment = {} if api is None else {"FOAM_API": api}
            with self.subTest(api=api), mock.patch.object(sys, "argv", argv), mock.patch.dict(
                os.environ, environment, clear=True
            ):
                with self.assertRaisesRegex(SystemExit, "requires OpenCFD API 2512"):
                    run_cases.main()

        self._write_valid_marker()
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {"FOAM_API": "2512"}, clear=True
        ), mock.patch.object(run_cases.shutil, "which", return_value="/fake/command"), contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(run_cases.main(), 0)

    def test_dry_run_needs_no_runtime_and_touches_no_marker_or_log(self) -> None:
        marker = self.case / ".completed"
        marker.write_text("sentinel marker\n", encoding="utf-8")
        log = self.case / "log.pimpleFoam"
        before_log = log.read_text(encoding="utf-8")
        stdout = io.StringIO()
        argv = ["run_cases.py", "--cases-dir", str(self.root), "--np", "1", "--dry-run"]

        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {"FOAM_API": "wrong"}, clear=True
        ), mock.patch.object(run_cases.shutil, "which", return_value=None), contextlib.redirect_stdout(stdout):
            self.assertEqual(run_cases.main(), 0)

        self.assertIn('"commands": [["pimpleFoam", "-case"', stdout.getvalue())
        self.assertEqual(marker.read_text(encoding="utf-8"), "sentinel marker\n")
        self.assertEqual(log.read_text(encoding="utf-8"), before_log)

    def test_core_binding_is_explicit_in_mpi_command_and_completion_marker(self) -> None:
        commands: list[list[str]] = []

        def successful_process(command: list[str], **kwargs: object) -> SimpleNamespace:
            commands.append(command)
            if command[0] == "mpirun":
                kwargs["stdout"].write("Time = 2\nExecutionTime = 1 s\nEnd\n")
            return SimpleNamespace(returncode=0)

        with mock.patch.object(run_cases.subprocess, "run", side_effect=successful_process):
            run_cases._run_one(
                self.case,
                "pimpleFoam",
                4,
                False,
                False,
                bind_to_core=True,
            )

        mpi = next(command for command in commands if command[0] == "mpirun")
        self.assertEqual(
            mpi[:8],
            [
                "mpirun",
                "-np",
                "4",
                "--map-by",
                "core",
                "--bind-to",
                "core",
                "pimpleFoam",
            ],
        )
        marker = json.loads((self.case / ".completed").read_text(encoding="utf-8"))
        self.assertTrue(marker["bind_to_core"])
        self.assertIsNone(marker["cpu_set"])

    def test_cpu_set_wraps_only_mpi_command_and_is_recorded(self) -> None:
        commands: list[list[str]] = []

        def successful_process(command: list[str], **kwargs: object) -> SimpleNamespace:
            commands.append(command)
            if command[0] == "taskset":
                kwargs["stdout"].write("Time = 2\nExecutionTime = 1 s\nEnd\n")
            return SimpleNamespace(returncode=0)

        with mock.patch.object(run_cases.subprocess, "run", side_effect=successful_process):
            run_cases._run_one(
                self.case,
                "pimpleFoam",
                4,
                False,
                False,
                bind_to_core=True,
                cpu_set="0-3",
            )

        self.assertEqual(commands[0][0], "foamDictionary")
        self.assertEqual(commands[1][0], "decomposePar")
        self.assertEqual(
            commands[2][:11],
            [
                "taskset",
                "-c",
                "0-3",
                "mpirun",
                "-np",
                "4",
                "--map-by",
                "core",
                "--bind-to",
                "core",
                "pimpleFoam",
            ],
        )
        marker = json.loads((self.case / ".completed").read_text(encoding="utf-8"))
        self.assertEqual(marker["cpu_set"], "0-3")

    def test_cpu_set_syntax_count_and_disjointness(self) -> None:
        self.assertEqual(
            run_cases._parse_cpu_set("0-3,8,10-11"),
            ((0, 3), (8, 8), (10, 11)),
        )
        run_cases._validate_cpu_sets(["0-7", "8-15"], 2)

        invalid = {
            "": "must not be empty",
            "0-": "comma-separated",
            "3-1": "ends before",
            "0-3,3-7": "duplicate or overlapping",
            "0,0": "duplicate or overlapping",
            "0 1": "comma-separated",
        }
        for value, message in invalid.items():
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                run_cases._parse_cpu_set(value)

        with self.assertRaisesRegex(ValueError, "received 1 .* --jobs is 2"):
            run_cases._validate_cpu_sets(["0-7"], 2)
        with self.assertRaisesRegex(ValueError, "mutually disjoint"):
            run_cases._validate_cpu_sets(["0-7", "7-15"], 2)

    def test_cpu_set_cli_requires_mpi_binding_and_one_set_per_job(self) -> None:
        common = ["run_cases.py", "--cases-dir", str(self.root), "--dry-run"]
        invalid_argv = [
            (common + ["--np", "1", "--cpu-set", "0"], "requires --np greater than 1"),
            (common + ["--np", "4", "--cpu-set", "0-3"], "requires --bind-to-core"),
            (
                common
                + ["--np", "4", "--jobs", "2", "--bind-to-core", "--cpu-set", "0-3"],
                "received 1 .* --jobs is 2",
            ),
            (
                common
                + [
                    "--np",
                    "4",
                    "--jobs",
                    "2",
                    "--bind-to-core",
                    "--cpu-set",
                    "0-3",
                    "--cpu-set",
                    "3-6",
                ],
                "mutually disjoint",
            ),
        ]
        for argv, message in invalid_argv:
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, message):
                    run_cases.main()

    def test_concurrent_cases_lease_unique_cpu_set_slots(self) -> None:
        cpu_slots: queue.Queue[str] = queue.Queue()
        cpu_slots.put("0-3")
        cpu_slots.put("4-7")
        active: set[str] = set()
        observed: dict[str, str] = {}
        lock = threading.Lock()
        rendezvous = threading.Barrier(2)

        def fake_run_one(
            case: Path,
            solver: str,
            ranks: int,
            reconstruct: bool,
            dry_run: bool,
            bind_to_core: bool,
            cpu_set: str,
        ) -> tuple[str, float]:
            del solver, ranks, reconstruct, dry_run, bind_to_core
            with lock:
                self.assertNotIn(cpu_set, active)
                active.add(cpu_set)
                observed[case.name] = cpu_set
            rendezvous.wait(timeout=2.0)
            with lock:
                active.remove(cpu_set)
            return case.name, 0.0

        cases = [self.root / f"case_{index}" for index in range(4)]
        with mock.patch.object(run_cases, "_run_one", side_effect=fake_run_one):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        run_cases._run_one_with_cpu_slot,
                        cpu_slots,
                        case,
                        "pimpleFoam",
                        4,
                        False,
                        True,
                        True,
                    )
                    for case in cases
                ]
                for future in futures:
                    future.result(timeout=3.0)

        self.assertEqual(set(observed), {case.name for case in cases})
        self.assertEqual(set(observed.values()), {"0-3", "4-7"})
        self.assertEqual(cpu_slots.qsize(), 2)

    def test_core_binding_rejects_serial_execution(self) -> None:
        argv = [
            "run_cases.py",
            "--cases-dir",
            str(self.root),
            "--np",
            "1",
            "--bind-to-core",
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "requires --np greater than 1"):
                run_cases.main()


if __name__ == "__main__":
    unittest.main()

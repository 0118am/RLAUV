from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from environment.openfoam.analysis.fit import (
    FitOptions,
    _OddCase,
    _cycle_convergence_diagnostic,
    _odd_project,
    _scaled_lstsq,
    added_mass_coriolis_product,
    analyze_cases,
)
from environment.openfoam.analysis.forces import ForceSeries, load_case_forces, parse_forces_file
from environment.openfoam.analysis.motion import (
    CaseData,
    MotionSpec,
    axis_angle_rotation,
    transform_wrench_to_body,
)


def _vector_text(vector: np.ndarray) -> str:
    return "(" + " ".join(f"{float(value):.17g}" for value in vector) + ")"


class ForcesParserTests(unittest.TestCase):
    def test_openfoam_components_are_summed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forces.dat"
            path.write_text(
                "# Time forces(pressure viscous porous) moment(pressure viscous porous)\n"
                "0.25 ((1 2 3) (0.1 0.2 0.3) (0.01 0.02 0.03)) "
                "((4 5 6) (0.4 0.5 0.6) (0.04 0.05 0.06))\n",
                encoding="utf-8",
            )
            result = parse_forces_file(path)
            np.testing.assert_allclose(result.force_global[0], (1.11, 2.22, 3.33))
            np.testing.assert_allclose(result.moment_global[0], (4.44, 5.55, 6.66))

    def test_restart_duplicate_prefers_later_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            first = case / "postProcessing" / "auvForces" / "0"
            second = case / "postProcessing" / "auvForces" / "1"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "forces.dat").write_text(
                "0 ((1 0 0) (0 0 0)) ((0 0 0) (0 0 0))\n"
                "1 ((2 0 0) (0 0 0)) ((0 0 0) (0 0 0))\n",
                encoding="utf-8",
            )
            (second / "forces.dat").write_text(
                "1 ((3 0 0) (0 0 0)) ((0 0 0) (0 0 0))\n"
                "2 ((4 0 0) (0 0 0)) ((0 0 0) (0 0 0))\n",
                encoding="utf-8",
            )
            result = load_case_forces(case)
            np.testing.assert_allclose(result.time_s, (0, 1, 2))
            np.testing.assert_allclose(result.force_global[:, 0], (1, 3, 4))

    def test_v2512_split_files_align_and_deduplicate_restarts(self) -> None:
        def split_file(rows: list[tuple[float, tuple[float, float, float]]]) -> str:
            lines = [
                "# Time total_x total_y total_z pressure_x pressure_y pressure_z "
                "viscous_x viscous_y viscous_z\n"
            ]
            for time_s, total in rows:
                # Deliberately unrelated components prove the parser consumes
                # total_* directly instead of summing all nine values.
                lines.append(
                    f"{time_s:.12g} {total[0]} {total[1]} {total[2]} "
                    "100 200 300 10 20 30\n"
                )
            return "".join(lines)

        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            first = case / "postProcessing" / "forces" / "0"
            second = case / "postProcessing" / "forces" / "1"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "force.dat").write_text(
                split_file([(0.0, (1, 2, 3)), (1.0, (2, 3, 4))]), encoding="utf-8"
            )
            (first / "moment.dat").write_text(
                split_file([(5.0e-10, (10, 20, 30)), (1.0 + 5.0e-10, (20, 30, 40))]),
                encoding="utf-8",
            )
            (second / "force.dat").write_text(
                split_file([(1.0, (3, 4, 5)), (2.0, (4, 5, 6))]), encoding="utf-8"
            )
            (second / "moment.dat").write_text(
                split_file([(1.0 + 4.0e-10, (30, 40, 50)), (2.0 + 4.0e-10, (40, 50, 60))]),
                encoding="utf-8",
            )

            result = load_case_forces(case)
            np.testing.assert_allclose(result.time_s, (0, 1, 2), atol=1.0e-9)
            np.testing.assert_allclose(result.force_global, ((1, 2, 3), (3, 4, 5), (4, 5, 6)))
            np.testing.assert_allclose(
                result.moment_global,
                ((10, 20, 30), (30, 40, 50), (40, 50, 60)),
            )
            self.assertEqual(len(result.source_files), 4)

    def test_v2512_numeric_filename_suffix_has_later_restart_priority(self) -> None:
        def split_file(rows: list[tuple[float, float]]) -> str:
            header = "# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z\n"
            return header + "".join(
                f"{time_s:.12g} {value} 0 0 {value} 0 0 0 0 0\n"
                for time_s, value in rows
            )

        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            output = case / "postProcessing" / "forces" / "0"
            output.mkdir(parents=True)
            (output / "force.dat").write_text(
                split_file([(0.0, 1.0), (1.000001, 2.0)]), encoding="utf-8"
            )
            (output / "moment.dat").write_text(
                split_file([(0.0, 10.0), (1.000001, 20.0)]), encoding="utf-8"
            )
            (output / "force_0.dat").write_text(
                split_file([(1.0, 3.0), (2.0, 4.0)]), encoding="utf-8"
            )
            (output / "moment_0.dat").write_text(
                split_file([(1.0, 30.0), (2.0, 40.0)]), encoding="utf-8"
            )

            result = load_case_forces(case)
            np.testing.assert_allclose(result.time_s, (0.0, 1.0, 2.0))
            np.testing.assert_allclose(result.force_global[:, 0], (1.0, 3.0, 4.0))
            np.testing.assert_allclose(result.moment_global[:, 0], (10.0, 30.0, 40.0))
            self.assertEqual(len(result.source_files), 4)


class FrameTransformTests(unittest.TestCase):
    def test_translation_shifts_fixed_cofr_moment_to_com(self) -> None:
        motion = MotionSpec(
            case_name="surge",
            dof="u",
            dof_index=0,
            motion_kind="translation",
            axis=np.array((1.0, 0.0, 0.0)),
            amplitude_si=1.0,
            omega_rad_s=2.0 * math.pi,
        )
        # At t=0.25 the COM is (1,0,0). A y-force of 2 adds +2 z-moment
        # about the fixed origin, so moment 5 about origin is moment 3 at COM.
        series = ForceSeries(
            np.array((0.25,)),
            np.array(((0.0, 2.0, 0.0),)),
            np.array(((0.0, 0.0, 5.0),)),
        )
        _, _, _, wrench = transform_wrench_to_body(series, motion)
        np.testing.assert_allclose(wrench[0], (0, 2, 0, 0, 0, 3), atol=1.0e-12)

    def test_rotation_maps_global_wrench_back_to_flu(self) -> None:
        motion = MotionSpec.from_mapping(
            {
                "case_name": "yaw",
                "dof": "r",
                "kind": "rotation",
                "axis": [0, 0, 1],
                "amplitude_m": None,
                "amplitude_deg": 90,
                "amplitude_rad": math.pi / 2.0,
                "frequency_hz": 1,
                "centre_of_rotation_m": [0, 0, 0],
            }
        )
        series = ForceSeries(
            np.array((0.25,)),
            np.array(((0.0, 1.0, 0.0),)),
            np.array(((-2.0, 0.0, 0.0),)),
        )
        _, _, _, wrench = transform_wrench_to_body(series, motion)
        np.testing.assert_allclose(wrench[0], (1, 0, 0, 0, 2, 0), atol=1.0e-12)


class OddProjectionSamplingTests(unittest.TestCase):
    @staticmethod
    def _synthetic_case(time_s: np.ndarray, name: str) -> CaseData:
        motion = MotionSpec(
            case_name=name,
            dof="u",
            dof_index=0,
            motion_kind="translation",
            axis=np.array((1.0, 0.0, 0.0)),
            amplitude_si=0.2,
            omega_rad_s=2.0 * math.pi,
            settle_cycles=0.0,
            sample_cycles=1.0,
        )
        eta_scalar, nu_scalar, nudot_scalar = motion.kinematics(time_s)
        eta = np.zeros((time_s.size, 6))
        nu = np.zeros_like(eta)
        nudot = np.zeros_like(eta)
        eta[:, 0] = eta_scalar
        nu[:, 0] = nu_scalar
        nudot[:, 0] = nudot_scalar
        design = np.column_stack(
            (-nudot_scalar, -nu_scalar, -np.abs(nu_scalar) * nu_scalar)
        )
        coefficients = np.array((2.3, 1.7, 0.8))
        # sin(3*phase) is odd across a half-period and orthogonal to all three
        # regression columns on a uniform phase grid.  A raw, phase-clustered
        # least-squares fit would incorrectly turn it into coefficient bias.
        discrepancy = 0.35 * np.sin(6.0 * math.pi * time_s)
        wrench = np.zeros((time_s.size, 6))
        wrench[:, 0] = design @ coefficients + discrepancy + 0.4
        zero = np.zeros((time_s.size, 3))
        series = ForceSeries(time_s, zero, zero)
        return CaseData(name, motion, time_s, eta, nu, nudot, wrench, series)

    def test_nonuniform_adaptive_sampling_does_not_bias_uniform_phase_fit(self) -> None:
        unit = np.linspace(0.0, 1.0, 4097)
        uniform = self._synthetic_case(unit, "uniform")
        # Strictly increasing phase warp creates a roughly 3:1 raw density
        # contrast while preserving identical cycle endpoints and physics.
        warped_time = unit + 0.45 / (2.0 * math.pi) * np.sin(2.0 * math.pi * unit)
        nonuniform = self._synthetic_case(warped_time, "adaptive")
        uniform_odd = _odd_project(uniform, phase_samples_per_cycle=256)
        adaptive_odd = _odd_project(nonuniform, phase_samples_per_cycle=256)
        uniform_fit, _ = _scaled_lstsq(uniform_odd.design, uniform_odd.target)
        adaptive_fit, _ = _scaled_lstsq(adaptive_odd.design, adaptive_odd.target)
        np.testing.assert_allclose(uniform_fit[:, 0], (2.3, 1.7, 0.8), atol=2.0e-6)
        np.testing.assert_allclose(adaptive_fit[:, 0], (2.3, 1.7, 0.8), atol=2.0e-5)
        np.testing.assert_allclose(adaptive_fit[:, 0], uniform_fit[:, 0], atol=2.0e-5)
        self.assertEqual(uniform_odd.design.shape[0], 128)
        self.assertEqual(adaptive_odd.design.shape[0], 128)

    def test_requested_partial_cycle_is_rejected_instead_of_underweighted(self) -> None:
        time_s = np.linspace(0.0, 0.9, 1000)
        case = self._synthetic_case(time_s, "truncated")
        with self.assertRaisesRegex(ValueError, "sample cycle.*incomplete"):
            _odd_project(case, phase_samples_per_cycle=256)

    def test_middle_restart_gap_is_not_bridged_by_interpolation(self) -> None:
        time_s = np.linspace(0.0, 1.0, 1001)
        time_s = time_s[(time_s <= 0.42) | (time_s >= 0.58)]
        case = self._synthetic_case(time_s, "restart_gap")
        with self.assertRaisesRegex(ValueError, "large raw time gap"):
            _odd_project(case, phase_samples_per_cycle=256)


class CycleConvergenceDiagnosticTests(unittest.TestCase):
    def test_reports_cycle_coefficients_and_robust_last_cycle_changes(self) -> None:
        phase = np.arange(64, dtype=float) * math.pi / 64.0
        one_cycle_design = np.column_stack(
            (
                np.sin(phase),
                -np.cos(phase),
                -np.abs(np.cos(phase)) * np.cos(phase),
            )
        )
        coefficient_cycles = []
        target_cycles = []
        for cycle, added_main in enumerate((2.0, 2.2, 2.31)):
            coefficients = np.zeros((3, 6))
            coefficients[0] = np.linspace(added_main, added_main + 0.5, 6)
            coefficients[1] = np.linspace(0.4, 0.9, 6)
            coefficients[2] = np.linspace(0.7, 1.2, 6)
            # Exercise the near-zero relative-denominator path for the main
            # linear coefficient while the main odd waveform remains finite.
            if cycle >= 1:
                coefficients[1, 0] = 1.0e-16 * (cycle - 1)
            # A truly zero wrench component must produce an explicit
            # unavailable normalization, not NaN, infinity, or silent pass.
            coefficients[:, 5] = 0.0
            coefficient_cycles.append(coefficients)
            target_cycles.append(one_cycle_design @ coefficients)

        item = _OddCase(
            "synthetic_cycle_drift",
            0,
            np.tile(one_cycle_design, (3, 1)),
            np.concatenate(target_cycles, axis=0),
            np.zeros((3 * phase.size, 6)),
            np.repeat(np.arange(3), phase.size),
        )
        report = _cycle_convergence_diagnostic(item)

        self.assertEqual(report["automatic_acceptance"], "not_evaluated_no_threshold_configured")
        self.assertEqual(len(report["cycles"]), 3)
        self.assertEqual(report["cycles"][0]["odd_phase_row_count"], 64)
        np.testing.assert_allclose(
            report["cycles"][2]["coefficients_by_wrench"]["added_mass_raw"],
            coefficient_cycles[2][0],
            atol=1.0e-14,
        )

        comparison = report["last_two_cycle_comparison"]
        self.assertTrue(comparison["available"])
        self.assertEqual(comparison["previous_cycle_id"], 1)
        self.assertEqual(comparison["latest_cycle_id"], 2)
        self.assertEqual(comparison["main_response_wrench"], "X")
        added_change = comparison["main_response_coefficient_changes"]["added_mass_raw"]
        self.assertAlmostEqual(added_change["previous"], 2.2, places=13)
        self.assertAlmostEqual(added_change["latest"], 2.31, places=13)
        self.assertAlmostEqual(added_change["absolute_relative_change_percent"], 5.0, places=11)
        self.assertEqual(added_change["relative_change_status"], "reported")

        linear_change = comparison["main_response_coefficient_changes"]["linear_damping"]
        self.assertIsNone(linear_change["absolute_relative_change_percent"])
        self.assertEqual(
            linear_change["relative_change_status"],
            "near_zero_previous_use_absolute_and_load_normalized_change",
        )
        self.assertIsNotNone(linear_change["absolute_change_normalized_by_odd_load_percent"])

        waveform_changes = comparison["odd_load_waveform_changes_by_wrench"]
        self.assertEqual(set(waveform_changes), {"X", "Y", "Z", "K", "M", "N"})
        self.assertGreater(waveform_changes["X"]["rms_of_phase_aligned_pointwise_change"], 0.0)
        self.assertIsNone(
            waveform_changes["N"]["phase_aligned_change_normalized_by_pooled_rms_percent"]
        )
        self.assertEqual(
            waveform_changes["N"]["normalization_status"],
            "unavailable_both_waveforms_near_zero",
        )

    def test_one_cycle_explains_why_last_two_comparison_is_unavailable(self) -> None:
        phase = np.arange(8, dtype=float) * math.pi / 8.0
        design = np.column_stack(
            (np.sin(phase), np.cos(phase), np.abs(np.cos(phase)) * np.cos(phase))
        )
        coefficients = np.ones((3, 6))
        report = _cycle_convergence_diagnostic(
            _OddCase(
                "one_cycle",
                2,
                design,
                design @ coefficients,
                np.zeros((phase.size, 6)),
                np.zeros(phase.size, dtype=int),
            )
        )
        self.assertEqual(
            report["last_two_cycle_comparison"],
            {"available": False, "reason": "fewer_than_two_complete_cycles"},
        )


class FullMatrixRecoveryTests(unittest.TestCase):
    @staticmethod
    def _write_synthetic_case(
        root: Path,
        dof_index: int,
        variant: int,
        added_mass: np.ndarray,
        linear: np.ndarray,
        quadratic: np.ndarray,
        bias: np.ndarray,
    ) -> Path:
        names = ("u", "v", "w", "p", "q", "r")
        dof = names[dof_index]
        kind = "translation" if dof_index < 3 else "rotation"
        axis_index = dof_index if dof_index < 3 else dof_index - 3
        axis = np.eye(3)[axis_index]
        frequency_hz = 0.75 + 0.2 * variant + 0.03 * dof_index
        omega = 2.0 * math.pi * frequency_hz
        amplitude = (0.035 + 0.018 * variant) if dof_index < 3 else math.radians(2.5 + 2.0 * variant)
        settle_cycles = 1
        sample_cycles = 3
        case = root / f"osc_{dof}_{variant}"
        output = case / "postProcessing" / "forces" / "0"
        output.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "case_name": case.name,
            "dof": dof,
            "dof_index": dof_index,
            "kind": kind,
            "axis": axis.tolist(),
            "amplitude_m": amplitude if dof_index < 3 else None,
            "amplitude_deg": math.degrees(amplitude) if dof_index >= 3 else None,
            "amplitude_rad": amplitude if dof_index >= 3 else None,
            "frequency_hz": frequency_hz,
            "omega_rad_s": omega,
            "phase_rad": 0.0,
            "settle_cycles": settle_cycles,
            "sample_cycles": sample_cycles,
            "centre_of_rotation_m": [0.0, 0.0, 0.0],
            "com_initial_global_m": [0.07, -0.03, 0.02],
            "include_in_fit": True,
        }
        (case / "motion.json").write_text(json.dumps(metadata), encoding="utf-8")

        samples_per_cycle = 128
        period = 1.0 / frequency_hz
        time = np.arange((settle_cycles + sample_cycles) * samples_per_cycle + 1) * period / samples_per_cycle
        argument = omega * time
        scalar_eta = amplitude * np.sin(argument)
        scalar_nu = amplitude * omega * np.cos(argument)
        scalar_nudot = -amplitude * omega**2 * np.sin(argument)
        nu = np.zeros((time.size, 6))
        nudot = np.zeros_like(nu)
        nu[:, dof_index] = scalar_nu
        nudot[:, dof_index] = scalar_nudot
        tau_body = (
            bias
            - nudot @ added_mass.T
            - added_mass_coriolis_product(nu, added_mass)
            - nu @ linear.T
            - (np.abs(nu) * nu) @ quadratic.T
        )

        com_initial = np.asarray(metadata["com_initial_global_m"])
        if kind == "translation":
            rotations = np.broadcast_to(np.eye(3), (time.size, 3, 3))
            com = com_initial + scalar_eta[:, None] * axis
        else:
            rotations = axis_angle_rotation(axis, scalar_eta)
            cofr = np.asarray(metadata["centre_of_rotation_m"])
            com = cofr + np.einsum("nij,j->ni", rotations, com_initial - cofr)
        force_global = np.einsum("nij,nj->ni", rotations, tau_body[:, :3])
        moment_com_global = np.einsum("nij,nj->ni", rotations, tau_body[:, 3:])
        cofr = np.asarray(metadata["centre_of_rotation_m"])
        moment_origin_global = moment_com_global + np.cross(com - cofr, force_global)

        lines = ["# Time forces(pressure viscous porous) moment(pressure viscous porous)\n"]
        zero = np.zeros(3)
        for sample_time, force, moment in zip(time, force_global, moment_origin_global):
            lines.append(
                f"{sample_time:.17g} "
                f"({_vector_text(force)} {_vector_text(zero)} {_vector_text(zero)}) "
                f"({_vector_text(moment)} {_vector_text(zero)} {_vector_text(zero)})\n"
            )
        (output / "forces.dat").write_text("".join(lines), encoding="utf-8")
        return case

    def test_recovers_every_entry_with_even_coriolis_and_moving_frames(self) -> None:
        rng = np.random.default_rng(42)
        added_mass = rng.normal(0.0, 0.3, (6, 6)) + np.diag(np.linspace(2.0, 4.5, 6))
        linear = rng.normal(0.0, 0.15, (6, 6)) + np.diag(np.linspace(1.0, 2.0, 6))
        quadratic = rng.normal(0.0, 0.1, (6, 6)) + np.diag(np.linspace(0.5, 1.0, 6))
        bias = np.array((0.4, -0.3, 0.2, -0.1, 0.35, -0.25))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                self._write_synthetic_case(root, dof, variant, added_mass, linear, quadratic, bias)
                for dof in range(6)
                for variant in range(2)
            ]
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "motion.json").write_text(
                json.dumps(
                    {
                        "case_name": "baseline",
                        "kind": "baseline",
                        "purpose": "stationary_tare",
                        "dof": None,
                        "dof_index": None,
                        "axis": [0, 0, 0],
                        "amplitude_m": None,
                        "amplitude_rad": None,
                        "frequency_hz": None,
                        "include_in_fit": False,
                    }
                ),
                encoding="utf-8",
            )
            cases.append(baseline)
            result = analyze_cases(
                cases,
                output_dir=root / "results",
                config={
                    "analysis": {
                        "project_added_mass_psd": False,
                        "bootstrap_samples": 0,
                        "passivity_samples": 0,
                        "phase_samples_per_cycle": 128,
                    }
                },
            )
            np.testing.assert_allclose(result.added_mass_raw, added_mass, rtol=2.0e-9, atol=2.0e-9)
            np.testing.assert_allclose(result.added_mass, added_mass, rtol=2.0e-9, atol=2.0e-9)
            np.testing.assert_allclose(result.linear_damping, linear, rtol=2.0e-9, atol=2.0e-9)
            np.testing.assert_allclose(result.quadratic_damping, quadratic, rtol=2.0e-9, atol=2.0e-9)

            cycle_reports = result.diagnostics["cycle_convergence_by_case"]
            self.assertEqual(len(cycle_reports), 12)
            self.assertTrue(all(len(item["cycles"]) == 3 for item in cycle_reports))
            self.assertTrue(
                all(item["last_two_cycle_comparison"]["available"] for item in cycle_reports)
            )

            updates = json.loads((root / "results" / "config_updates.json").read_text(encoding="utf-8"))
            self.assertEqual(set(updates), {"added_mass_diag", "linear_damping", "quadratic_damping"})
            self.assertEqual(np.asarray(updates["added_mass_diag"]).shape, (6, 6))
            written_report = json.loads(
                (root / "results" / "hydrodynamic_fit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                len(written_report["diagnostics"]["cycle_convergence_by_case"]), 12
            )
            for filename in (
                "hydrodynamic_fit.json",
                "added_mass.csv",
                "added_mass_raw.csv",
                "linear_damping.csv",
                "quadratic_damping.csv",
            ):
                self.assertTrue((root / "results" / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from environment.openfoam.convergence.compare import (
    DEFAULT_GRID_CONFIGS,
    DEFAULT_VARIANT_CONFIGS,
    GRID_VARIANTS,
    VARIANTS,
    _acceptance_assessment,
    _mesh_characteristic_size,
    compare_cases,
    main,
)
from environment.openfoam import run_cases


def _vector(value: tuple[float, float, float]) -> str:
    return "(" + " ".join(f"{item:.17g}" for item in value) + ")"


class ConvergenceComparisonTests(unittest.TestCase):
    def test_nonmonotonic_grid_uses_provisional_nominal_fine_gate(self) -> None:
        grid = {
            metric: {
                "gci": {
                    "available": False,
                    "status": "GCI unavailable: sequence is not monotonic",
                },
                "nominal_vs_fine": {"absolute_relative_difference_percent": 0.5},
            }
            for metric in (
                "added_mass",
                "effective_damping_at_peak_speed",
                "main_load_fundamental_amplitude",
            )
        }
        two_variant = {
            metric: {"absolute_relative_difference_percent": 0.5}
            for metric in (
                "added_mass",
                "effective_damping_at_peak_speed",
                "main_load_fundamental_amplitude",
            )
        }
        two_variant["main_load_phase"] = {"absolute_difference_deg": 0.25}
        assessment = _acceptance_assessment(grid, two_variant, two_variant)
        self.assertFalse(assessment["overall_pass"])
        self.assertTrue(assessment["all_numeric_limits_pass"])
        self.assertEqual(assessment["status"], "provisional_pass")
        self.assertTrue(assessment["checks"]["grid_added_mass"]["provisional"])
        self.assertEqual(
            assessment["checks"]["grid_added_mass"]["source"],
            "nominal_vs_fine_absolute_relative_difference_percent",
        )

    @staticmethod
    def _write_case(
        root: Path,
        variant: str,
        *,
        added_mass: float,
        linear_damping: float,
        quadratic_damping: float,
        stop_cycles: float = 8.0,
        amplitude: float = 0.08,
    ) -> Path:
        case = root / variant / "v_amp0p08m_f1p00hz"
        output = case / "postProcessing" / "forces" / "0"
        output.mkdir(parents=True)
        config_variant = (
            "dt" if variant.startswith("dt") else
            "domain" if variant.startswith("domain") else
            variant
        )
        config_path = DEFAULT_VARIANT_CONFIGS[config_variant]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        omega = 2.0 * math.pi
        metadata = {
            "schema_version": 1,
            "openfoam_version": config["openfoam_version"],
            "solver": config["solver"],
            "case_name": "v_amp0p08m_f1p00hz",
            "dof": "v",
            "dof_index": 1,
            "kind": "translation",
            "axis": [0.0, 1.0, 0.0],
            "amplitude_m": amplitude,
            "frequency_hz": 1.0,
            "omega_rad_s": omega,
            "phase_rad": 0.0,
            "settle_cycles": config["settle_cycles"],
            "sample_cycles": config["sample_cycles"],
            "settle_end_s": float(config["settle_cycles"]),
            "end_time_s": float(config["settle_cycles"] + config["sample_cycles"]),
            "delta_t_s": 1.0 / config["steps_per_cycle"],
            "initial_delta_t_s": (
                config["initial_delta_t_fraction"] / config["steps_per_cycle"]
            ),
            "max_co": config["max_co"],
            "rho_kg_m3": config["rho_kg_m3"],
            "nu_m2_s": config["nu_m2_s"],
            "force_patch": config["force_patch"],
            "centre_of_rotation_m": config["centre_of_rotation_m"],
        }
        (case / "motion.json").write_text(json.dumps(metadata), encoding="utf-8")

        steps_per_cycle = config["steps_per_cycle"]
        time = np.arange(int(stop_cycles * steps_per_cycle) + 1, dtype=float) / steps_per_cycle
        phase = omega * time
        velocity = amplitude * omega * np.cos(phase)
        acceleration = -amplitude * omega**2 * np.sin(phase)
        load = (
            -added_mass * acceleration
            - linear_damping * velocity
            - quadratic_damping * np.abs(velocity) * velocity
            + 0.2
        )
        force_lines = [
            "# Time total_x total_y total_z pressure_x pressure_y pressure_z "
            "viscous_x viscous_y viscous_z\n"
        ]
        moment_lines = list(force_lines)
        for sample_time, sample_load in zip(time, load):
            force_lines.append(
                f"{sample_time:.17g} 0 {sample_load:.17g} 0 "
                f"0 {sample_load:.17g} 0 0 0 0\n"
            )
            moment_lines.append(f"{sample_time:.17g} 0 0 0 0 0 0 0 0 0\n")
        (output / "force.dat").write_text("".join(force_lines), encoding="utf-8")
        (output / "moment.dat").write_text("".join(moment_lines), encoding="utf-8")

        shared_geometry = root / "geometry" / config["geometry_filename"]
        shared_geometry.parent.mkdir(parents=True, exist_ok=True)
        shared_geometry.write_bytes(b"synthetic convergence geometry\n")
        tri_surface = case / "constant" / "triSurface"
        tri_surface.mkdir(parents=True)
        (tri_surface / config["geometry_filename"]).symlink_to(shared_geometry.resolve())

        mesh_owner = "nominal" if config_variant == "dt" else variant
        mesh_case = root / mesh_owner / "mesh_case"
        poly_mesh = mesh_case / "constant" / "polyMesh"
        poly_mesh.mkdir(parents=True, exist_ok=True)
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            (poly_mesh / name).touch()
        block = config["block_mesh"]
        lower = block["domain_min"]
        upper = block["domain_max"]
        vertices = (
            (lower[0], lower[1], lower[2]),
            (upper[0], lower[1], lower[2]),
            (upper[0], upper[1], lower[2]),
            (lower[0], upper[1], lower[2]),
            (lower[0], lower[1], upper[2]),
            (upper[0], lower[1], upper[2]),
            (upper[0], upper[1], upper[2]),
            (lower[0], upper[1], upper[2]),
        )
        system = mesh_case / "system"
        system.mkdir(parents=True, exist_ok=True)
        vertex_text = "\n".join(f"    {_vector(vertex)}" for vertex in vertices)
        cells = " ".join(str(value) for value in block["base_cells"])
        (system / "blockMeshDict").write_text(
            f"vertices\n(\n{vertex_text}\n);\nblocks\n(\n"
            f"hex (0 1 2 3 4 5 6 7) ({cells}) simpleGrading (1 1 1)\n);\n",
            encoding="utf-8",
        )
        logs = mesh_case / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        cell_count = math.prod(block["base_cells"])
        (logs / "checkMesh.log").write_text(
            f"Mesh stats\n    cells: {cell_count}\nMesh OK.\nEnd\n", encoding="utf-8"
        )
        (logs / "mesh_volume_validation.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "expected_displaced_volume_m3": 0.0113,
                    "excluded_volume_m3": 0.0113,
                    "relative_error": 0.0,
                    "relative_tolerance": 0.03,
                }
            ),
            encoding="utf-8",
        )
        constant = case / "constant"
        (constant / "polyMesh").symlink_to(poly_mesh.resolve(), target_is_directory=True)

        case_system = case / "system"
        case_system.mkdir(parents=True)
        cofr = _vector(tuple(config["centre_of_rotation_m"]))
        case_system.joinpath("controlDict").write_text(
            f"application {config['solver']};\n"
            f"endTime {metadata['end_time_s']:.17g};\n"
            f"deltaT {metadata['initial_delta_t_s']:.17g};\n"
            f"maxDeltaT {metadata['delta_t_s']:.17g};\n"
            f"maxCo {config['max_co']:.17g};\n"
            f"purgeWrite {config['purge_write']};\n"
            "functions\n{\nforces\n{\n"
            "executeControl timeStep;\nexecuteInterval 1;\n"
            "writeControl timeStep;\nwriteInterval 1;\n"
            f"timeStart {metadata['settle_end_s']:.17g};\n"
            f"patches ({config['force_patch']});\n"
            f"rhoInf {config['rho_kg_m3']:.17g};\nCofR {cofr};\n"
            "}\n}\n",
            encoding="utf-8",
        )
        constant.joinpath("transportProperties").write_text(
            f"nu {config['nu_m2_s']:.17g};\n", encoding="utf-8"
        )
        initial = case / "0"
        initial.mkdir()
        amplitude_vector = _vector(tuple(amplitude * np.asarray(metadata["axis"])))
        initial.joinpath("pointDisplacement").write_text(
            "boundaryField\n{\n"
            f"{config['force_patch']}\n{{\n"
            "type oscillatingDisplacement;\n"
            f"amplitude {amplitude_vector};\n"
            f"omega {omega:.17g};\n"
            "}\n}\n",
            encoding="utf-8",
        )

        (case / "log.pimpleFoam").write_text(
            f"Time = {stop_cycles:.17g}\nEnd\n", encoding="utf-8"
        )
        try:
            validation = run_cases._validate_case_outputs(case, "pimpleFoam", metadata)
        except RuntimeError:
            # Deliberately truncated test cases retain an invalid marker so
            # the production completion gate rejects them.
            (case / ".completed").write_text("{}", encoding="utf-8")
        else:
            marker = {
                "schema_version": run_cases._MARKER_SCHEMA_VERSION,
                "status": "completed",
                "case": case.name,
                "solver": "pimpleFoam",
                "foam_api": "2512",
                "mpi_ranks": 1,
                "motion": metadata,
                "validation": validation,
                "elapsed_s": 1.0,
            }
            (case / ".completed").write_text(json.dumps(marker), encoding="utf-8")
        return case

    def _make_campaign(self, root: Path) -> dict[str, Path]:
        sizes = {
            name: _mesh_characteristic_size(DEFAULT_GRID_CONFIGS[name])
            for name in GRID_VARIANTS
        }
        paths: dict[str, Path] = {}
        for variant in GRID_VARIANTS:
            h2 = sizes[variant] ** 2
            paths[variant] = self._write_case(
                root,
                variant,
                added_mass=10.0 + 30.0 * h2,
                linear_damping=5.0 + 30.0 * h2,
                quadratic_damping=2.0 + 12.0 * h2,
            )
        paths["dt"] = self._write_case(
            root,
            "dt",
            added_mass=10.42,
            linear_damping=5.14,
            quadratic_damping=2.055,
        )
        paths["domain"] = self._write_case(
            root,
            "domain",
            added_mass=10.38,
            linear_damping=5.12,
            quadratic_damping=2.048,
        )
        return paths

    def test_single_column_fit_comparisons_and_gci(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_campaign(Path(directory))
            report = compare_cases(paths)

        self.assertEqual(report["case"]["dof"], "v")
        fine_h = _mesh_characteristic_size(DEFAULT_GRID_CONFIGS["fine"])
        expected_fine = 10.0 + 30.0 * fine_h**2
        self.assertAlmostEqual(
            report["variants"]["fine"]["coefficients"]["added_mass"],
            expected_fine,
            places=3,
        )
        peak_speed = 0.08 * 2.0 * math.pi
        expected_effective = (5.0 + 30.0 * fine_h**2) + (
            2.0 + 12.0 * fine_h**2
        ) * peak_speed
        self.assertAlmostEqual(
            report["variants"]["fine"]["coefficients"][
                "effective_damping_at_peak_speed"
            ],
            expected_effective,
            places=3,
        )
        gci = report["comparisons"]["grid"]["metrics"]["added_mass"]["gci"]
        self.assertTrue(gci["monotonic"])
        self.assertTrue(gci["available"])
        self.assertAlmostEqual(gci["observed_order"], 2.0, places=9)
        self.assertAlmostEqual(gci["richardson_extrapolated"], 10.0, places=3)
        self.assertLess(report["variants"]["nominal"]["fit"]["odd_model_residual_rms"], 1.0e-3)
        self.assertIn("absolute_relative_difference_percent", report["comparisons"]["time_step"]["metrics"]["added_mass"])
        self.assertIn("absolute_relative_difference_percent", report["comparisons"]["domain"]["metrics"]["added_mass"])
        self.assertIn("overall_pass", report["acceptance"])
        self.assertTrue(report["acceptance"]["checks"]["grid_added_mass"]["pass"])
        self.assertFalse(report["acceptance"]["overall_pass"])
        self.assertEqual(report["acceptance"]["status"], "fail")
        self.assertEqual(report["acceptance"]["limits"]["grid"]["added_mass_percent"], 2.0)
        self.assertEqual(
            report["acceptance"]["limits"]["time_step"]["effective_damping_percent"],
            3.0,
        )
        self.assertEqual(report["acceptance"]["limits"]["domain"]["added_mass_percent"], 1.0)
        self.assertEqual(len(report["acceptance"]["checks"]), 11)
        for check in report["acceptance"]["checks"].values():
            self.assertIn("pass", check)
            self.assertIn("status", check)
            self.assertTrue(check["reason"])

    def test_rejects_motion_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_campaign(Path(directory))
            paths["domain"] = self._write_case(
                Path(directory),
                "domain_mismatch",
                added_mass=10.38,
                linear_damping=5.12,
                quadratic_damping=2.048,
                amplitude=0.081,
            )
            with self.assertRaisesRegex(ValueError, "motion mismatch for amplitude_si"):
                compare_cases(paths)

    def test_rejects_incomplete_requested_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._make_campaign(root)
            broken = self._write_case(
                root,
                "dt_broken",
                added_mass=10.42,
                linear_damping=5.14,
                quadratic_damping=2.055,
                stop_cycles=7.8,
            )
            paths["dt"] = broken
            with self.assertRaisesRegex(ValueError, "case completion evidence is invalid"):
                compare_cases(paths)

    def test_rejects_restart_gap_crossing_cycle_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_campaign(Path(directory))
            for filename in ("force.dat", "moment.dat"):
                path = paths["dt"] / "postProcessing" / "forces" / "0" / filename
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                kept = [lines[0]]
                for line in lines[1:]:
                    time_s = float(line.split(maxsplit=1)[0])
                    if not 4.8 < time_s < 5.2:
                        kept.append(line)
                path.write_text("".join(kept), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw time gap.*exceeds configured"):
                compare_cases(paths)

    def test_rejects_uniform_force_history_undersampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_campaign(Path(directory))
            for filename in ("force.dat", "moment.dat"):
                path = paths["dt"] / "postProcessing" / "forces" / "0" / filename
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                kept = [lines[0], *lines[1:-1:100], lines[-1]]
                path.write_text("".join(kept), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw time gap.*exceeds configured"):
                compare_cases(paths)

    def test_rejects_reusing_nominal_as_every_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_campaign(Path(directory))
            duplicated = {name: paths["nominal"] for name in VARIANTS}
            with self.assertRaisesRegex(ValueError, "five distinct case directories"):
                compare_cases(duplicated)

    def test_rejects_solver_input_that_disagrees_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_campaign(Path(directory))
            control = paths["fine"] / "system" / "controlDict"
            text = control.read_text(encoding="utf-8").replace("maxCo 0.5;", "maxCo 0.4;")
            control.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "controlDict maxCo"):
                compare_cases(paths)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._make_campaign(root)
            output = root / "report"
            arguments: list[str] = []
            for variant in VARIANTS:
                arguments.extend((f"--{variant}", str(paths[variant])))
            arguments.extend(("--output-dir", str(output)))
            self.assertEqual(main(arguments), 0)
            parsed = json.loads((output / "convergence_report.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["case"]["case_name"], "v_amp0p08m_f1p00hz")
            markdown = (output / "convergence_report.md").read_text(encoding="utf-8")
            self.assertIn("Grid convergence", markdown)
            self.assertIn("Time-step and domain checks", markdown)


if __name__ == "__main__":
    unittest.main()

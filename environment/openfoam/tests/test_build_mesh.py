from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from environment.openfoam import build_mesh
from environment.openfoam import generate_cases
from environment.openfoam.build_mesh import (
    _check_mesh_audit,
    _generator_command,
    _mesh_volume_validation,
    _parser,
    _prepare_command,
    _snappy_mesh_audit,
    _surface_check_failures,
    _verify_prepared_input,
)
from environment.openfoam.run_cases import _command_plan, _discover


_ZERO_FACE_ERROR_BLOCK = """Checking faces in error :
    non-orthogonality > 70 degrees                         : 0
    faces with face pyramid volume < 1e-15                 : 0
    faces with face-decomposition tet quality < 1e-30      : 0
    faces with concavity > 80 degrees                      : 0
    faces with skewness > 4 (internal) or 20 (boundary)    : 0
    faces with interpolation weights (0..1) < 0.02         : 0
    faces with volume ratio of neighbour cells < 0.01      : 0
    faces with face twist < 0.02                           : 0
    faces on cells with determinant < 0.001                : 0
"""

_PASSING_SNAPPY_OUTPUT = f"""Checking final mesh ...
{_ZERO_FACE_ERROR_BLOCK}
Finished meshing without any errors
End
"""

_CURRENT_STRICT_CHECK_OUTPUT = f"""Checking topology...
    Boundary definition OK.
    Cell to face addressing OK.
    Point usage OK.
    Upper triangular ordering OK.
    Face vertices OK.
    Topological cell zip-up check OK.
  <<Number of duplicate (not baffle) faces found: 2. This might indicate a problem.
  <<Number of faces with non-consecutive shared points: 34. This might indicate a problem.
    Number of regions: 1 (OK).
Checking geometry...
    Min volume = 2.854692023e-12. Max volume = 0.0005950534511.  Total volume = 127.9881059.  Cell volumes OK.
 ***Max skewness = 16.46337122, 12 highly skew faces detected which may impair the quality of the results
   *There are 22608 faces with concave angles between consecutive edges.
 ***Cells with small determinant (< 0.001) found, number of cells: 3371
 ***Concave cells (using face planes) found, number of cells: 203483
 ***Faces with small interpolation weight (< 0.05) found, number of faces: 2279
{_ZERO_FACE_ERROR_BLOCK}
Failed 4 mesh checks.
End
"""


def _hard_failures(output: str) -> list[str]:
    return list(_check_mesh_audit(output)["hard_failures"])


def _locked_rotor_report() -> dict:
    assemblies = []
    for index in range(1, 9):
        axis = [0.6, 0.8, 0.0] if index == 1 else [0.0, 0.0, 1.0]
        centre = [100.0 + index, 200.0, 300.0]
        shaft_end = [
            value + 27.397463135207 * direction
            for value, direction in zip(centre, axis, strict=True)
        ]
        shaft_start = [
            value - 42.897463135207 * direction
            for value, direction in zip(shaft_end, axis, strict=True)
        ]
        motor_end = [
            value + 35.102536865 * direction
            for value, direction in zip(shaft_end, axis, strict=True)
        ]
        assemblies.append(
            {
                "label": f"T{index}",
                "representation": "single_axisymmetric_smooth_motor_envelope",
                "connector_radius_mm": 2.05,
                "propeller_centre_body_mm": centre,
                "connector_axis_start_body_mm": shaft_start,
                "connector_axis_end_body_mm": shaft_end,
                "motor_profile_axis_end_body_mm": motor_end,
                "axis_direction_body": axis,
                "source_volume_relative_error": 0.00281112,
                "maximum_source_volume_relative_error": 0.01,
                "common_volume_mm3": {
                    "mount": 376.2,
                    "hub": 41.4,
                    "propeller": 234.9,
                },
                "minimum_common_volume_mm3": {
                    "mount": 1.0,
                    "hub": 1.0,
                    "propeller": 1.0,
                },
            }
        )
    return {
        "output_frame": "body_flu_com",
        "locked_rotor_condition": "fully_assembled_static_locked",
        "locked_rotor_assemblies": assemblies,
    }


class SurfaceCheckGateTests(unittest.TestCase):
    def test_accepts_one_closed_consistent_non_intersecting_surface(self) -> None:
        output = """
Surface has no illegal triangles.
Surface is closed. All edges connected to two faces.
Number of unconnected parts : 1
Number of zones (connected area with consistent normal) : 1
Surface is not self-intersecting
"""
        self.assertEqual(_surface_check_failures(output), [])

    def test_rejects_dirty_multi_part_surface(self) -> None:
        output = """
Surface has 265 illegal triangles.
Surface is not closed.
Number of unconnected parts : 220
Number of zones (connected area with consistent normal) : 904
Surface has 144814 self-intersections
"""
        self.assertEqual(
            set(_surface_check_failures(output)),
            {
                "illegal triangles",
                "closed two-face edges",
                "self intersections",
                "exactly one connected part",
                "one consistently oriented normal zone",
            },
        )


class MeshQualityGateTests(unittest.TestCase):
    def test_accepts_current_strict_check_as_pass_with_recorded_warnings(self) -> None:
        audit = _check_mesh_audit(_CURRENT_STRICT_CHECK_OUTPUT)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["hard_failures"], [])
        self.assertEqual(audit["extended_checks_reported_failed"], 4)
        self.assertEqual(audit["connected_regions"], 1)
        self.assertGreater(len(audit["warnings"]), 4)
        self.assertNotIn("Mesh OK.", _CURRENT_STRICT_CHECK_OUTPUT)
        self.assertEqual(_hard_failures(_CURRENT_STRICT_CHECK_OUTPUT), [])

    def test_accepts_snappy_only_when_final_configured_counts_are_zero(self) -> None:
        audit = _snappy_mesh_audit(_PASSING_SNAPPY_OUTPUT)
        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["final_configured_face_errors"]), 9)

    def test_rejects_nonzero_configured_count_in_snappy_and_check_mesh(self) -> None:
        bad_block = _ZERO_FACE_ERROR_BLOCK.replace(
            "faces with skewness > 4 (internal) or 20 (boundary)    : 0",
            "faces with skewness > 4 (internal) or 20 (boundary)    : 1",
        )
        snappy = _PASSING_SNAPPY_OUTPUT.replace(_ZERO_FACE_ERROR_BLOCK, bad_block)
        check_mesh = _CURRENT_STRICT_CHECK_OUTPUT.replace(
            _ZERO_FACE_ERROR_BLOCK, bad_block
        )

        self.assertFalse(_snappy_mesh_audit(snappy)["passed"])
        failures = _hard_failures(check_mesh)
        self.assertTrue(
            any("configured mesh-quality limits exceeded" in item for item in failures)
        )

    def test_rejects_truncation_fatal_diagnostics_and_incomplete_evidence(self) -> None:
        truncated = _CURRENT_STRICT_CHECK_OUTPUT.rsplit("End", 1)[0]
        fatal = _CURRENT_STRICT_CHECK_OUTPUT.replace(
            "End\n", "FOAM FATAL ERROR: corrupted owner list\nEnd\n"
        )
        incomplete = "Checking geometry...\nMesh OK.\nEnd\n"

        self.assertTrue(
            any("terminal End" in item for item in _hard_failures(truncated))
        )
        self.assertTrue(any("fatal diagnostic" in item for item in _hard_failures(fatal)))
        self.assertGreater(len(_hard_failures(incomplete)), 3)

    def test_rejects_negative_volume_multiple_regions_and_missing_core_topology(self) -> None:
        negative = _CURRENT_STRICT_CHECK_OUTPUT.replace(
            "Min volume = 2.854692023e-12", "Min volume = -2.0e-12"
        )
        multiple = _CURRENT_STRICT_CHECK_OUTPUT.replace(
            "Number of regions: 1 (OK).", "Number of regions: 2 (OK)."
        )
        missing_core = _CURRENT_STRICT_CHECK_OUTPUT.replace(
            "    Face vertices OK.\n", ""
        )

        self.assertTrue(any("minimum cell volume" in item for item in _hard_failures(negative)))
        self.assertTrue(any("exactly one" in item for item in _hard_failures(multiple)))
        self.assertTrue(any("Face vertices OK" in item for item in _hard_failures(missing_core)))


class GeneratedControlDictTests(unittest.TestCase):
    def test_formal_config_keeps_only_four_field_times(self) -> None:
        cfg = generate_cases.load_config(
            Path(generate_cases.__file__).resolve().parent / "config.json"
        )
        rendered = generate_cases.render_control_dict(
            generate_cases.motion_specs(cfg)[0], cfg
        )
        self.assertIn("purgeWrite          4;", rendered)

    def test_rejects_disabled_field_purging(self) -> None:
        config_path = Path(generate_cases.__file__).resolve().parent / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["purge_write"] = 0
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "config.json"
            candidate.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "purge_write"):
                generate_cases.load_config(candidate)


class ParameterizedMeshDictionaryTests(unittest.TestCase):
    _CONFIG = Path(generate_cases.__file__).resolve().parent / "config.json"

    def test_formal_config_renders_all_parameterized_dictionary_values(self) -> None:
        cfg = generate_cases.load_config(self._CONFIG)
        block = generate_cases.render_block_mesh_dict(cfg)
        control = generate_cases.render_control_dict(generate_cases.motion_specs(cfg)[0], cfg)
        without_rotors = json.loads(json.dumps(cfg))
        without_rotors["locked_rotor_mesh"]["enabled"] = False
        snappy = generate_cases.render_snappy_hex_mesh_dict(without_rotors, None)

        self.assertIn("    (-3 -2 -2)", block)
        self.assertIn("    (5 2 2)", block)
        self.assertIn("(96 48 48) simpleGrading", block)
        self.assertIn("maxLocalCells       16000000;", snappy)
        self.assertIn("maxGlobalCells      16000000;", snappy)
        self.assertIn("addLayers       true;", snappy)
        self.assertIn("auv { nSurfaceLayers 4; }", snappy)
        self.assertIn("nFeatureSnapIter    8;", snappy)
        self.assertIn("explicitFeatureSnap false;", snappy)
        self.assertIn("maxCo               0.5;", control)
        self.assertIn("type            yPlus;", control)
        self.assertIn("libs            (fieldFunctionObjects);", control)
        formal_spec = generate_cases.motion_specs(cfg)[0]
        formal_time = generate_cases.timeline(formal_spec, cfg)
        self.assertAlmostEqual(
            formal_time["initial_delta_t_s"], 0.05 * formal_time["delta_t_s"]
        )
        self.assertIn(
            f"deltaT              {generate_cases.fmt(formal_time['initial_delta_t_s'])};",
            control,
        )
        self.assertIn(
            f"maxDeltaT           {generate_cases.fmt(formal_time['delta_t_s'])};",
            control,
        )
        self.assertNotIn("__BLOCK_MESH_", block)
        self.assertNotIn("__SNAPPY_CELL_LIMITS_", snappy)

    def test_no_layer_switch_and_smooth_wall_functions_are_rendered(self) -> None:
        cfg = generate_cases.load_config(self._CONFIG)
        no_layers = json.loads(json.dumps(cfg))
        no_layers["snappy"]["add_layers"] = False
        no_layers["locked_rotor_mesh"]["enabled"] = False

        snappy = generate_cases.render_snappy_hex_mesh_dict(no_layers, None)
        nut = generate_cases.render_wall_function_field("nut", no_layers)
        omega = generate_cases.render_wall_function_field("omega", no_layers)

        self.assertIn("addLayers       false;", snappy)
        self.assertNotIn("__SNAPPY_ADD_LAYERS_", snappy)
        self.assertNotIn("__SNAPPY_LAYER_CONTROLS_", snappy)
        self.assertIn("blending        exponential;", nut)
        self.assertIn("blending        exponential;", omega)

    def test_solver_performance_controls_default_and_variant_are_rendered(self) -> None:
        baseline = generate_cases.load_config(
            Path(generate_cases.__file__).resolve().parent
            / "experiment_configs"
            / "cfd12_no_layers_level6.json"
        )
        performance = generate_cases.load_config(
            Path(generate_cases.__file__).resolve().parent
            / "experiment_configs"
            / "cfd12_no_layers_level6_performance.json"
        )

        baseline_solution = generate_cases.render_fv_solution(baseline)
        performance_solution = generate_cases.render_fv_solution(performance)

        self.assertTrue(baseline["move_mesh_outer_correctors"])
        self.assertEqual(baseline["gamg_update_interval"], 1)
        self.assertIn("moveMeshOuterCorrectors yes;", baseline_solution)
        self.assertEqual(baseline_solution.count("updateInterval  1;"), 2)

        self.assertFalse(performance["move_mesh_outer_correctors"])
        self.assertEqual(performance["gamg_update_interval"], 10)
        self.assertIn("moveMeshOuterCorrectors no;", performance_solution)
        self.assertEqual(performance_solution.count("updateInterval  10;"), 2)
        self.assertNotIn("__GAMG_UPDATE_INTERVAL__", performance_solution)
        self.assertNotIn("__MOVE_MESH_OUTER_CORRECTORS__", performance_solution)

        rendered_metadata = generate_cases.metadata(
            generate_cases.motion_specs(performance)[0], performance
        )
        self.assertFalse(rendered_metadata["move_mesh_outer_correctors"])
        self.assertEqual(rendered_metadata["gamg_update_interval"], 10)

    def test_rejects_invalid_layer_and_wall_function_settings(self) -> None:
        base = json.loads(self._CONFIG.read_text(encoding="utf-8"))
        invalid: list[tuple[dict, str]] = []
        bad_bool = json.loads(json.dumps(base))
        bad_bool["snappy"]["add_layers"] = 1
        invalid.append((bad_bool, "add_layers"))
        bad_count = json.loads(json.dumps(base))
        bad_count["snappy"]["n_surface_layers"] = 0
        invalid.append((bad_count, "n_surface_layers"))
        bad_blending = json.loads(json.dumps(base))
        bad_blending["wall_function_blending"] = "implicit"
        invalid.append((bad_blending, "wall_function_blending"))
        bad_motion_switch = json.loads(json.dumps(base))
        bad_motion_switch["move_mesh_outer_correctors"] = 0
        invalid.append((bad_motion_switch, "move_mesh_outer_correctors"))
        bad_gamg_interval = json.loads(json.dumps(base))
        bad_gamg_interval["gamg_update_interval"] = 0
        invalid.append((bad_gamg_interval, "gamg_update_interval"))

        for cfg, message in invalid:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "config.json"
                candidate.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    generate_cases.load_config(candidate)

    def test_rejects_invalid_block_mesh_settings(self) -> None:
        base = json.loads(self._CONFIG.read_text(encoding="utf-8"))
        invalid = (
            ("short domain", ("block_mesh", "domain_min"), [-3.0, -2.0]),
            ("nonfinite domain", ("block_mesh", "domain_max"), [5.0, 2.0, float("nan")]),
            ("reversed domain", ("block_mesh", "domain_min"), [5.0, -2.0, -2.0]),
            ("boolean cell", ("block_mesh", "base_cells"), [True, 48, 48]),
            ("float cell", ("block_mesh", "base_cells"), [96.0, 48, 48]),
            ("zero cell", ("block_mesh", "base_cells"), [96, 0, 48]),
        )
        for label, path, value in invalid:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                cfg = json.loads(json.dumps(base))
                cfg[path[0]][path[1]] = value
                candidate = Path(directory) / "config.json"
                candidate.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "block_mesh"):
                    generate_cases.load_config(candidate)

    def test_rejects_invalid_snappy_limits_and_max_co(self) -> None:
        base = json.loads(self._CONFIG.read_text(encoding="utf-8"))
        invalid = (
            ("boolean local", ("snappy", "max_local_cells"), True, "snappy"),
            ("zero global", ("snappy", "max_global_cells"), 0, "snappy"),
            ("global below local", ("snappy", "max_global_cells"), 100, "snappy"),
            ("boolean maxCo", ("max_co",), True, "max_co"),
            ("nonfinite maxCo", ("max_co",), float("nan"), "max_co"),
            ("zero maxCo", ("max_co",), 0.0, "max_co"),
        )
        for label, path, value, message in invalid:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                cfg = json.loads(json.dumps(base))
                if len(path) == 1:
                    cfg[path[0]] = value
                else:
                    cfg[path[0]][path[1]] = value
                candidate = Path(directory) / "config.json"
                candidate.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    generate_cases.load_config(candidate)

    def test_rejects_missing_or_unknown_parameter_keys(self) -> None:
        base = json.loads(self._CONFIG.read_text(encoding="utf-8"))
        variants = []
        missing_block = json.loads(json.dumps(base))
        del missing_block["block_mesh"]["base_cells"]
        variants.append(("missing block", missing_block, "Missing block_mesh"))
        unknown_block = json.loads(json.dumps(base))
        unknown_block["block_mesh"]["cells"] = [96, 48, 48]
        variants.append(("unknown block", unknown_block, "Unknown block_mesh"))
        missing_snappy = json.loads(json.dumps(base))
        del missing_snappy["snappy"]["max_global_cells"]
        variants.append(("missing snappy", missing_snappy, "Missing snappy"))
        unknown_snappy = json.loads(json.dumps(base))
        unknown_snappy["snappy"]["max_cells"] = 16000000
        variants.append(("unknown snappy", unknown_snappy, "Unknown snappy"))

        for label, cfg, message in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "config.json"
                candidate.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    generate_cases.load_config(candidate)


class ConvergenceConfigTests(unittest.TestCase):
    _OPENFOAM = Path(generate_cases.__file__).resolve().parent
    _CONFIGS = _OPENFOAM / "convergence" / "configs"

    def _load(self, name: str) -> dict:
        return generate_cases.load_config(self._CONFIGS / f"{name}.json")

    def test_variants_preserve_formal_design_outside_reviewed_overrides(self) -> None:
        formal = generate_cases.load_config(self._OPENFOAM / "config.json")
        allowed = {
            "mesh_coarse": {"block_mesh"},
            "mesh_nominal": set(),
            "mesh_fine": {"block_mesh", "snappy"},
            "dt800": {"steps_per_cycle", "max_co"},
            "domain_expanded": {"block_mesh"},
        }
        for name, permitted_keys in allowed.items():
            with self.subTest(variant=name):
                variant = self._load(name)
                self.assertEqual(set(variant), set(formal))
                for key in formal:
                    if key not in permitted_keys:
                        self.assertEqual(variant[key], formal[key], key)
                self.assertEqual(len(generate_cases.motion_specs(variant)), 24)
                self.assertEqual(variant["geometry_path"], formal["geometry_path"])
                self.assertEqual(variant["purge_write"], 4)
                self.assertEqual(variant["settle_cycles"], 3)
                self.assertEqual(variant["sample_cycles"], 5)

    def test_mesh_and_outer_domain_spacings_match_reviewed_values(self) -> None:
        expected_level7_mm = {
            "mesh_coarse": 0.8680555555555556,
            "mesh_nominal": 0.6510416666666666,
            "mesh_fine": 0.48828125,
        }
        base_spacing: dict[str, float] = {}
        for name, expected_mm in expected_level7_mm.items():
            cfg = self._load(name)
            block = cfg["block_mesh"]
            spacing = [
                (upper - lower) / cells
                for lower, upper, cells in zip(
                    block["domain_min"],
                    block["domain_max"],
                    block["base_cells"],
                    strict=True,
                )
            ]
            self.assertAlmostEqual(max(spacing), min(spacing), places=14)
            base_spacing[name] = spacing[0]
            self.assertAlmostEqual(spacing[0] / (2**7) * 1000.0, expected_mm)

        self.assertAlmostEqual(
            base_spacing["mesh_coarse"] / base_spacing["mesh_nominal"], 4.0 / 3.0
        )
        self.assertAlmostEqual(
            base_spacing["mesh_nominal"] / base_spacing["mesh_fine"], 4.0 / 3.0
        )

        expanded_cfg = self._load("domain_expanded")
        expanded = expanded_cfg["block_mesh"]
        expanded_spacing = [
            (upper - lower) / cells
            for lower, upper, cells in zip(
                expanded["domain_min"],
                expanded["domain_max"],
                expanded["base_cells"],
                strict=True,
            )
        ]
        self.assertEqual(expanded["domain_min"], [-4.0, -2.5, -2.5])
        self.assertEqual(expanded["domain_max"], [6.0, 2.5, 2.5])
        self.assertEqual(expanded["base_cells"], [120, 60, 60])
        for spacing in expanded_spacing:
            self.assertAlmostEqual(spacing, base_spacing["mesh_nominal"])
        expanded_dict = generate_cases.render_block_mesh_dict(expanded_cfg)
        self.assertIn("    (-4 -2.5 -2.5)", expanded_dict)
        self.assertIn("    (6 2.5 2.5)", expanded_dict)
        self.assertIn("(120 60 60) simpleGrading", expanded_dict)

    def test_fine_cap_and_dt800_are_effective(self) -> None:
        formal = generate_cases.load_config(self._OPENFOAM / "config.json")
        fine = self._load("mesh_fine")
        dt800 = self._load("dt800")
        self.assertGreaterEqual(fine["snappy"]["max_local_cells"], 24000000)
        self.assertGreaterEqual(fine["snappy"]["max_global_cells"], 24000000)
        self.assertEqual(dt800["steps_per_cycle"], 800)
        self.assertEqual(dt800["max_co"], 0.25)

        formal_spec = generate_cases.motion_specs(formal)[1]
        dt800_spec = generate_cases.motion_specs(dt800)[1]
        formal_time = generate_cases.timeline(formal_spec, formal)
        refined_time = generate_cases.timeline(dt800_spec, dt800)
        self.assertAlmostEqual(refined_time["delta_t_s"], 0.5 * formal_time["delta_t_s"])
        self.assertEqual(refined_time["end_time_s"], formal_time["end_time_s"])
        self.assertEqual(refined_time["write_interval_s"], formal_time["write_interval_s"])

        fine_without_rotors = json.loads(json.dumps(fine))
        fine_without_rotors["locked_rotor_mesh"]["enabled"] = False
        fine_block = generate_cases.render_block_mesh_dict(fine)
        fine_snappy = generate_cases.render_snappy_hex_mesh_dict(
            fine_without_rotors, None
        )
        dt800_control = generate_cases.render_control_dict(dt800_spec, dt800)
        self.assertIn("(128 64 64) simpleGrading", fine_block)
        self.assertIn("maxLocalCells       24000000;", fine_snappy)
        self.assertIn("maxGlobalCells      24000000;", fine_snappy)
        self.assertIn("maxCo               0.25;", dt800_control)


class FirstMatrixExperimentConfigTests(unittest.TestCase):
    _OPENFOAM = Path(generate_cases.__file__).resolve().parent

    def test_two_amplitude_single_frequency_design_has_twelve_cases(self) -> None:
        for level in (6, 7):
            with self.subTest(level=level):
                cfg = generate_cases.load_config(
                    self._OPENFOAM
                    / "experiment_configs"
                    / f"cfd12_no_layers_level{level}.json"
                )
                specs = generate_cases.motion_specs(cfg)
                self.assertEqual(len(specs), 12)
                self.assertEqual({spec.frequency_hz for spec in specs}, {1.5})
                self.assertEqual(cfg["settle_cycles"], 2)
                self.assertEqual(cfg["sample_cycles"], 4)
                self.assertFalse(cfg["snappy"]["add_layers"])
                self.assertEqual(cfg["locked_rotor_mesh"]["rotor_level"], level)
                self.assertEqual(cfg["wall_function_blending"], "exponential")

    def test_performance_variant_changes_only_solver_performance_controls(self) -> None:
        baseline = generate_cases.load_config(
            self._OPENFOAM / "experiment_configs" / "cfd12_no_layers_level6.json"
        )
        performance = generate_cases.load_config(
            self._OPENFOAM
            / "experiment_configs"
            / "cfd12_no_layers_level6_performance.json"
        )
        self.assertEqual(
            {
                key
                for key in baseline
                if baseline[key] != performance[key]
            },
            {"move_mesh_outer_correctors", "gamg_update_interval"},
        )


class MeshVolumeGateTests(unittest.TestCase):
    _BLOCK = "Mesh Information\n  boundingBox: (-3 -2 -2) (5 2 2)\nEnd\n"

    def test_accepts_snappy_exclusion_matching_measured_displacement(self) -> None:
        expected = 0.011304505834
        result = _mesh_volume_validation(
            self._BLOCK,
            f"Total volume = {128.0 - expected:.12g}.\nMesh OK.\n",
            expected,
            0.03,
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["domain_volume_m3"], 128.0)
        self.assertAlmostEqual(result["excluded_volume_m3"], expected, places=9)

    def test_rejects_the_old_three_litre_mesh(self) -> None:
        result = _mesh_volume_validation(
            self._BLOCK,
            "Total volume = 127.996562.\nMesh OK.\n",
            0.011304505834,
            0.03,
        )
        self.assertFalse(result["passed"])
        self.assertGreater(result["relative_error"], 0.6)

    def test_fails_closed_on_missing_volume_records(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "boundingBox"):
            _mesh_volume_validation("End\n", "Total volume = 1\n", 1.0, 0.03)
        with self.assertRaisesRegex(RuntimeError, "total volume"):
            _mesh_volume_validation(self._BLOCK, "Mesh OK.\n", 1.0, 0.03)


class GeometryCommandTests(unittest.TestCase):
    def test_body_frame_affine_options_are_forwarded_to_geometry_preparation(self) -> None:
        args = _parser().parse_args(
            [
                "/tmp/dealt.STL",
                "--expected-sha256",
                "a" * 64,
                "--expected-displaced-volume-m3",
                "0.011304505834",
                "--axis-map",
                "z,x,y",
                "--translate-after-map",
                "-307",
                "-201.269",
                "-77.523",
                "--backend",
                "vtk",
            ]
        )

        command = _prepare_command(
            args,
            Path("/tmp/auv.stl"),
            Path("/tmp/geometry_provenance.json"),
        )

        axis_index = command.index("--axis-map")
        self.assertEqual(command[axis_index : axis_index + 2], ["--axis-map", "z,x,y"])
        translate_index = command.index("--translate-after-map")
        self.assertEqual(
            command[translate_index : translate_index + 4],
            ["--translate-after-map", "-307.0", "-201.269", "-77.523"],
        )

    def test_repair_report_is_forwarded_to_mesh_case_generator(self) -> None:
        args = _parser().parse_args(
            [
                "/tmp/body.stl",
                "--expected-displaced-volume-m3",
                "0.011304505834",
                "--repair-report",
                "/tmp/selection_report.json",
            ]
        )
        command = _generator_command(args, "--mesh-case-only")
        self.assertEqual(command.count("--output"), 1)
        output_index = command.index("--output")
        self.assertEqual(command[output_index + 1], str(args.cases_dir.resolve()))
        report_index = command.index("--repair-report")
        self.assertEqual(
            command[report_index + 1], str(Path("/tmp/selection_report.json").resolve())
        )


class LockedRotorRefinementTests(unittest.TestCase):
    def test_canted_refinement_and_isotropic_near_fields_come_from_report(self) -> None:
        cfg = generate_cases.load_config(
            Path(generate_cases.__file__).resolve().parent / "config.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "repair.json"
            report.write_text(json.dumps(_locked_rotor_report()), encoding="utf-8")
            locked = generate_cases.load_locked_rotor_report(report, cfg)
            rendered = generate_cases.render_snappy_hex_mesh_dict(cfg, locked)

        self.assertNotIn("__LOCKED_ROTOR_", rendered)
        self.assertEqual(rendered.count("type    searchableCylinder;"), 24)
        self.assertEqual(rendered.count("type    searchableSphere;"), 8)
        self.assertIn("rotorT1", rendered)
        self.assertIn("shaftT8", rendered)
        self.assertIn("motorT8", rendered)
        self.assertIn("nearFieldT8", rendered)
        self.assertEqual(rendered.count("levels ((1e15 7));"), 24)
        self.assertEqual(rendered.count("levels ((1e15 5));"), 8)
        # T1 centre=(.101,.2,.3)m and its report-derived axis=(.6,.8,0).
        # A 30 mm rotor cylinder therefore uses +/-15 mm along that canted axis.
        self.assertIn("point1  (0.092 0.188 0.3);", rendered)
        self.assertIn("point2  (0.11 0.212 0.3);", rendered)
        self.assertIn("centre  (0.101 0.2 0.3);", rendered)
        self.assertIn("radius  0.045;", rendered)
        self.assertIn("radius  0.006;", rendered)
        self.assertIn("radius  0.02;", rendered)
        self.assertIn("point2  (0.1385 0.25 0.3);", rendered)

    def test_mesh_case_generation_requires_and_records_all_eight_axes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = root / "body.stl"
            geometry.write_bytes(b"fixture")
            report = root / "repair.json"
            report.write_text(json.dumps(_locked_rotor_report()), encoding="utf-8")
            output = root / "cases"
            status = generate_cases.main(
                [
                    "--mesh-case-only",
                    "--output",
                    str(output),
                    "--geometry",
                    str(geometry),
                    "--geometry-mode",
                    "copy",
                    "--repair-report",
                    str(report),
                ]
            )
            rendered = (output / "mesh_case" / "system" / "snappyHexMeshDict").read_text(
                encoding="utf-8"
            )
            block = (output / "mesh_case" / "system" / "blockMeshDict").read_text(
                encoding="utf-8"
            )
            control = (output / "mesh_case" / "system" / "controlDict").read_text(
                encoding="utf-8"
            )

        self.assertEqual(status, 0)
        self.assertEqual(rendered.count("type    searchableCylinder;"), 24)
        self.assertEqual(rendered.count("type    searchableSphere;"), 8)
        self.assertIn("(96 48 48) simpleGrading", block)
        self.assertIn("maxLocalCells       16000000;", rendered)
        self.assertIn("maxGlobalCells      16000000;", rendered)
        self.assertIn("maxCo               0.5;", control)

    def test_report_with_failed_connector_overlap_is_rejected(self) -> None:
        cfg = generate_cases.load_config(
            Path(generate_cases.__file__).resolve().parent / "config.json"
        )
        payload = _locked_rotor_report()
        payload["locked_rotor_assemblies"][3]["common_volume_mm3"]["propeller"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "repair.json"
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "common-volume gates"):
                generate_cases.load_locked_rotor_report(report, cfg)


class PreparedInputTests(unittest.TestCase):
    _CLEAN_SURFACE_OUTPUT = """
Surface has no illegal triangles.
Surface is closed. All edges connected to two faces.
Number of unconnected parts : 1
Number of zones (connected area with consistent normal) : 1
Surface is not self-intersecting
"""

    def test_verifies_exact_sha256_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "prepared.STL"
            geometry.write_bytes(b"binary stl fixture")
            expected = hashlib.sha256(geometry.read_bytes()).hexdigest().upper()

            _verify_prepared_input(geometry, expected)

    def test_records_sha256_without_user_supplied_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "prepared.stl"
            geometry.write_bytes(b"binary stl fixture")
            actual = hashlib.sha256(geometry.read_bytes()).hexdigest()

            self.assertEqual(_verify_prepared_input(geometry, None), actual)

    def test_rejects_sha256_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "prepared.stl"
            geometry.write_bytes(b"binary stl fixture")

            with self.assertRaisesRegex(RuntimeError, "prepared input SHA-256 mismatch"):
                _verify_prepared_input(geometry, "0" * 64)

    def test_rejects_missing_file_and_non_stl_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "prepared input does not exist"):
                _verify_prepared_input(root / "missing.stl", "0" * 64)

            wrong_suffix = root / "prepared.obj"
            wrong_suffix.write_bytes(b"fixture")
            with self.assertRaisesRegex(RuntimeError, "must use the .stl suffix"):
                _verify_prepared_input(wrong_suffix, "0" * 64)

    def test_prepared_input_skips_preparation_but_runs_surface_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = root / "prepared.stl"
            geometry.write_bytes(b"binary stl fixture")
            expected = hashlib.sha256(geometry.read_bytes()).hexdigest()
            provenance = root / "reports" / "geometry.json"

            def fake_run(command, *, log=None, dry_run=False):
                if command[0] == "surfaceCheck":
                    return self._CLEAN_SURFACE_OUTPUT
                if command[0] == "blockMesh":
                    return "boundingBox: (-3 -2 -2) (5 2 2)\n"
                if command[0] == "snappyHexMesh":
                    return _PASSING_SNAPPY_OUTPUT
                if command[0] == "checkMesh":
                    return _CURRENT_STRICT_CHECK_OUTPUT.replace(
                        "Total volume = 127.9881059",
                        "Total volume = 127",
                    )
                return ""

            stdout = io.StringIO()
            with (
                mock.patch.object(build_mesh, "_require_commands"),
                mock.patch.object(build_mesh, "_prepare_command") as prepare_command,
                mock.patch.object(build_mesh, "_run", side_effect=fake_run) as run,
                contextlib.redirect_stdout(stdout),
            ):
                status = build_mesh.main(
                    [
                        str(geometry),
                        "--expected-displaced-volume-m3",
                        "1",
                        "--prepared-input",
                        "--provenance",
                        str(provenance),
                        "--cases-dir",
                        str(root / "cases"),
                        "--mesh-only",
                    ]
                )

            self.assertEqual(status, 0)
            prepare_command.assert_not_called()
            surface_calls = [
                call
                for call in run.call_args_list
                if call.args[0] and call.args[0][0] == "surfaceCheck"
            ]
            self.assertEqual(len(surface_calls), 1)
            self.assertEqual(
                surface_calls[0].args[0],
                ["surfaceCheck", "-checkSelfIntersection", str(geometry.resolve())],
            )
            self.assertEqual(
                surface_calls[0].kwargs["log"],
                provenance.resolve().with_name("surfaceCheck.log"),
            )

    def test_prepared_input_rejects_nondefault_preparation_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = root / "prepared.stl"
            geometry.write_bytes(b"binary stl fixture")
            expected = hashlib.sha256(geometry.read_bytes()).hexdigest()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = build_mesh.main(
                    [
                        str(geometry),
                        "--expected-displaced-volume-m3",
                        "1",
                        "--prepared-input",
                        "--axis-map",
                        "z,x,y",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("cannot be combined", stderr.getvalue())
            self.assertIn("--axis-map", stderr.getvalue())

    def test_prepared_dry_run_still_verifies_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "prepared.stl"
            geometry.write_bytes(b"binary stl fixture")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = build_mesh.main(
                    [
                        str(geometry),
                        "--expected-sha256",
                        "0" * 64,
                        "--expected-displaced-volume-m3",
                        "1",
                        "--prepared-input",
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("SHA-256 mismatch", stderr.getvalue())

    def test_bad_prepared_sha_stops_before_surface_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / "prepared.stl"
            geometry.write_bytes(b"binary stl fixture")
            stderr = io.StringIO()
            with (
                mock.patch.object(build_mesh, "_require_commands"),
                mock.patch.object(build_mesh, "_run") as run,
                contextlib.redirect_stderr(stderr),
            ):
                status = build_mesh.main(
                    [
                        str(geometry),
                        "--expected-sha256",
                        "0" * 64,
                        "--expected-displaced-volume-m3",
                        "1",
                        "--prepared-input",
                    ]
                )

            self.assertEqual(status, 1)
            run.assert_not_called()
            self.assertIn("prepared input SHA-256 mismatch", stderr.getvalue())

    def test_input_and_displaced_volume_remain_required_in_prepared_mode(self) -> None:
        parser = _parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing_input:
                parser.parse_args(
                    [
                        "--prepared-input",
                        "--expected-displaced-volume-m3",
                        "1",
                    ]
                )
            with self.assertRaises(SystemExit) as missing_volume:
                parser.parse_args(
                    [
                        "/tmp/prepared.stl",
                        "--prepared-input",
                    ]
                )
            without_sha = parser.parse_args(
                [
                    "/tmp/prepared.stl",
                    "--prepared-input",
                    "--expected-displaced-volume-m3",
                    "1",
                ]
            )

        self.assertEqual(missing_input.exception.code, 2)
        self.assertEqual(missing_volume.exception.code, 2)
        self.assertIsNone(without_sha.expected_sha256)


class CaseDiscoveryTests(unittest.TestCase):
    def test_shared_mesh_is_not_scheduled_but_baseline_is(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, purpose in (("mesh_case", "shared_mesh"), ("baseline", "stationary_tare")):
                case = root / name
                (case / "constant" / "polyMesh").mkdir(parents=True)
                (case / "constant" / "polyMesh" / "boundary").write_text("", encoding="utf-8")
                (case / "motion.json").write_text(
                    json.dumps({"case_name": name, "purpose": purpose}), encoding="utf-8"
                )
            self.assertEqual([case.name for case in _discover(root, [], False)], ["baseline"])

    def test_parallel_plan_sets_requested_decomposition_size(self) -> None:
        case = Path("/tmp/example_case")
        commands = [command for command, _ in _command_plan(case, "pimpleFoam", 4, True)]
        self.assertEqual(commands[0][-2:], ["-set", "4"])
        self.assertEqual(commands[1][:2], ["decomposePar", "-force"])
        self.assertIn("-np", commands[2])
        self.assertEqual(commands[-1][0], "reconstructPar")

        bound_commands = [
            command
            for command, _ in _command_plan(
                case, "pimpleFoam", 4, False, bind_to_core=True
            )
        ]
        self.assertEqual(
            bound_commands[2][:7],
            ["mpirun", "-np", "4", "--map-by", "core", "--bind-to", "core"],
        )


if __name__ == "__main__":
    unittest.main()

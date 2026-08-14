from __future__ import annotations

import json
import copy
import math
from pathlib import Path
import tempfile
import unittest

from environment.openfoam.tools.repair_step import (
    EXPECTED_BODY_TRANSLATION_MM,
    EXPECTED_SOURCE_COM_BODY_MM,
    _body_direction_from_step,
    _body_point_from_step,
    _preflight_paths,
    _validate_config,
)


class RepairStepSafetyTests(unittest.TestCase):
    def test_body_axis_mapping_is_right_handed_cyclic_permutation(self) -> None:
        self.assertEqual(
            _body_point_from_step([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
            [3.0, 1.0, 2.0],
        )

    def test_reported_com_translation_is_applied_after_axis_mapping(self) -> None:
        self.assertEqual(EXPECTED_SOURCE_COM_BODY_MM, (-1.306, 0.061, 2.385))
        self.assertEqual(EXPECTED_BODY_TRANSLATION_MM, (1.306, -0.061, -2.385))
        transformed = _body_point_from_step(
            [1.0, 2.0, 3.0], EXPECTED_BODY_TRANSLATION_MM
        )
        for actual, expected in zip(
            transformed, [4.306, 0.939, -0.385], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [
                com + shift
                for com, shift in zip(
                    EXPECTED_SOURCE_COM_BODY_MM,
                    EXPECTED_BODY_TRANSLATION_MM,
                    strict=True,
                )
            ],
            [0.0, 0.0, 0.0],
        )

    def test_motor_axis_uses_rotation_without_com_translation(self) -> None:
        self.assertEqual(_body_direction_from_step([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
        transformed = _body_direction_from_step([-0.3420201433, 0.0, -0.9396926208])
        for actual, expected in zip(
            transformed, (-0.9396926208, -0.3420201433, 0.0), strict=True
        ):
            self.assertAlmostEqual(actual, expected)

    def test_preflight_rejects_aliases_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.step"
            config = root / "config.json"
            source.write_bytes(b"step")
            config.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must all be distinct"):
                _preflight_paths(source, config, source, root / "report.json", True)
            with self.assertRaisesRegex(ValueError, "must all be distinct"):
                _preflight_paths(source, config, root / "out.stl", config, True)

    def test_preflight_checks_both_outputs_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.step"
            config = root / "config.json"
            output = root / "out.stl"
            report = root / "report.json"
            source.write_bytes(b"step")
            config.write_text("{}", encoding="utf-8")
            report.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "report.json"):
                _preflight_paths(source, config, output, report, False)
            self.assertFalse(output.exists())

            resolved = _preflight_paths(source, config, output, report, True)
            self.assertEqual(resolved, tuple(path.resolve() for path in (source, config, output, report)))

    def test_reviewed_selection_config_is_self_consistent(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "geometry"
            / "verification_assembly_repair.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        groups = config["groups"]
        shell_count = config["source"]["expected_shell_count"]

        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["source"]["units"], "mm")
        self.assertEqual(config["selection"]["default_closed_shell_action"], "keep")
        self.assertEqual(config["selection"]["open_shell_action"], "remove")
        self.assertEqual(
            config["output_frame"]["source_com_body_mm"],
            list(EXPECTED_SOURCE_COM_BODY_MM),
        )
        self.assertEqual(
            config["output_frame"]["translation_mm"],
            list(EXPECTED_BODY_TRANSLATION_MM),
        )
        self.assertTrue(config["output_frame"]["reference_assumption"])
        self.assertEqual(
            config["selection"]["replace_groups"],
            {"thruster_motor_with_cable": "smooth_axisymmetric_envelope"},
        )

        for name, indices in groups.items():
            self.assertEqual(len(indices), len(set(indices)), name)
            self.assertTrue(
                all(type(index) is int and 1 <= index <= shell_count for index in indices),
                name,
            )

        remove_names = config["selection"]["remove_groups"]
        preserve_names = config["selection"]["preserve_groups"]
        self.assertTrue(set(remove_names + preserve_names) <= set(groups))
        removed = {index for name in remove_names for index in groups[name]}
        preserved = {index for name in preserve_names for index in groups[name]}
        self.assertFalse(removed & preserved)
        self.assertEqual(groups["main_pressure_hull"], [30])
        self.assertIn("main_pressure_hull", preserve_names)
        self.assertNotIn(30, removed)
        self.assertEqual(groups["pressure_hull_endcaps"], [42, 43])
        self.assertIn("pressure_hull_endcaps", preserve_names)
        self.assertTrue({42, 43} <= preserved)
        self.assertEqual(groups["main_closed_cell_buoyancy_material"], [257])
        self.assertIn("main_closed_cell_buoyancy_material", preserve_names)
        self.assertIn(257, preserved)
        propellers = set(groups["propeller_3blade"] + groups["propeller_4blade"])
        hubs = set(groups["propeller_hub_or_nut"])
        self.assertTrue(propellers <= preserved)
        self.assertTrue(hubs <= preserved)
        self.assertFalse((propellers | hubs) & removed)

        volume_validation = config["volume_validation"]
        self.assertEqual(
            volume_validation["target_displaced_volume_mm3"], 11_304_505.834
        )
        self.assertEqual(
            volume_validation["wrapped_surface_relative_tolerance"], 0.02
        )
        self.assertEqual(
            volume_validation["snappy_excluded_volume_relative_tolerance"], 0.055
        )

        motors = set(groups["thruster_motor_with_cable"])
        replacement = config["motor_replacement"]
        self.assertTrue(motors <= removed)
        self.assertEqual(motors, {int(index) for index in replacement["mount_side_by_shell"]})
        self.assertEqual(motors, {int(index) for index in replacement["mount_shells_by_motor"]})
        self.assertTrue(
            all(
                set(mounts) <= preserved
                for mounts in replacement["mount_shells_by_motor"].values()
            )
        )
        landmarks = config["output_frame"]["validation"]["propeller_landmarks"]
        step_origin_nominal = {
            "T1": [142.0, -160.0, -64.5],
            "T2": [-142.0, -160.0, -64.5],
            "T3": [142.0, 160.0, -64.5],
            "T4": [-142.0, 160.0, -64.5],
            "T5": [-140.88, 104.15, 42.5],
            "T6": [-140.88, -104.15, 42.5],
            "T7": [140.88, 104.15, 42.5],
            "T8": [140.88, -104.15, 42.5],
        }
        for item in landmarks:
            expected = [
                    value + shift
                    for value, shift in zip(
                        step_origin_nominal[item["label"]],
                        EXPECTED_BODY_TRANSLATION_MM,
                        strict=True,
                    )
                ]
            for actual, expected_value in zip(
                item["expected_body_mm"], expected, strict=True
            ):
                self.assertAlmostEqual(actual, expected_value)
        self.assertTrue(math.isfinite(replacement["mount_extension_mm"]))
        self.assertGreater(replacement["mount_extension_mm"], 0.0)
        self.assertTrue(math.isfinite(replacement["minimum_common_volume_mm3"]))
        self.assertGreater(replacement["minimum_common_volume_mm3"], 0.0)

        locked = config["locked_propeller"]
        self.assertEqual(locked["condition"], "fully_assembled_static_locked")
        self.assertEqual(locked["nominal_motor_shaft_radius_mm"], 2.0)
        self.assertEqual(locked["connector_radius_mm"], 2.05)
        self.assertEqual(locked["shaft_tip_extension_mm"], 0.5)
        self.assertEqual(locked["shaft_refinement_end_mm"], 0.0)
        self.assertEqual(locked["maximum_source_volume_relative_error"], 0.01)
        profile = locked["axisymmetric_profile_mm"]
        self.assertEqual(len(profile), 19)
        self.assertEqual(profile[0], [-42.897463135207, 0.0])
        self.assertEqual(profile[-1], [35.102536865, 0.0])
        self.assertIn([-1.370861642017, 16.292991320832], profile)
        assemblies = locked["assemblies_by_motor"]
        self.assertEqual({int(index) for index in assemblies}, motors)
        self.assertEqual(
            {item["propeller_shell"] for item in assemblies.values()}, propellers
        )
        self.assertEqual({item["hub_shell"] for item in assemblies.values()}, hubs)
        self.assertEqual(
            {item["label"] for item in assemblies.values()},
            {f"T{index}" for index in range(1, 9)},
        )
        self.assertEqual(
            set(locked["minimum_common_volume_mm3"]),
            {"mount", "hub", "propeller"},
        )

        sealed = config["sealed_pressure_boundary"]
        self.assertEqual(sealed["condition"], "waterproof_assembled_vehicle")
        self.assertEqual(
            sealed["representation"], "two_tube_opening_sealing_disks"
        )
        self.assertEqual(sealed["hull_shell"], 30)
        self.assertEqual(set(sealed["endcap_shells"]), {42, 43})
        self.assertEqual(sealed["endcap_by_hull_end"], {"start": 43, "end": 42})
        self.assertEqual(sealed["expected_inner_radius_mm"], 60.0)
        self.assertEqual(sealed["expected_outer_radius_mm"], 65.0)
        self.assertEqual(sealed["expected_hull_length_mm"], 300.0)
        self.assertEqual(sealed["radial_wall_overlap_mm"], 0.5)
        self.assertEqual(sealed["disk_half_thickness_mm"], 0.5)
        self.assertEqual(
            set(sealed["minimum_common_volume_mm3"]),
            {
                "start_disk_hull",
                "start_disk_endcap",
                "end_disk_hull",
                "end_disk_endcap",
            },
        )

        buoyancy = config["buoyancy_material_validation"]
        self.assertEqual(buoyancy["shell"], 257)
        self.assertEqual(
            buoyancy["role"], "waterproof closed-cell main buoyancy material"
        )
        self.assertEqual(buoyancy["condition"], "waterproof_closed_cell")
        self.assertEqual(
            buoyancy["identification_status"],
            "high-confidence geometry/placement inference",
        )
        self.assertAlmostEqual(
            buoyancy["expected_closed_solid_volume_mm3"], 2_674_187.521821612
        )
        self.assertEqual(len(buoyancy["expected_bbox_step_mm"]), 6)
        self.assertIn("geometry_union", buoyancy["hydrodynamic_accounting"])
        self.assertIn("never add", buoyancy["hydrodynamic_accounting"])

        landmark_indices = [item["shell_index"] for item in landmarks]
        labels = [item["label"] for item in landmarks]
        self.assertEqual(set(landmark_indices), propellers)
        self.assertEqual(len(landmark_indices), len(set(landmark_indices)))
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(
            all(
                len(item["expected_body_mm"]) == 3
                and all(math.isfinite(float(value)) for value in item["expected_body_mm"])
                for item in landmarks
            )
        )

        triangulation = config["triangulation"]
        self.assertGreater(triangulation["linear_deflection_mm"], 0.0)
        self.assertGreater(triangulation["angular_deflection_rad"], 0.0)
        self.assertEqual(
            config["triangulation_reference"],
            {
                "null_triangulation_face_indices": [10856],
                "binary_triangle_count": 1_205_609,
                "repeated_vertex_triangle_count": 239,
                "zero_area_triangle_count": 241,
                "note": config["triangulation_reference"]["note"],
            },
        )

        _validate_config(config, Path(config["source"]["basename"]))

        provenance_only_sha = copy.deepcopy(config)
        provenance_only_sha["source"]["sha256"] = "0" * 64
        _validate_config(provenance_only_sha, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["selection"]["remove_groups"][0] = "misspelled_group"
        with self.assertRaisesRegex(ValueError, "Unknown reviewed shell group"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["motor_replacement"]["mount_side_by_shell"]["29"] = "unknown"
        with self.assertRaisesRegex(ValueError, "vmin or vmax"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["selection"]["preserve_groups"].remove("main_pressure_hull")
        with self.assertRaisesRegex(ValueError, "pressure hull shell 30"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["selection"]["preserve_groups"].remove("pressure_hull_endcaps")
        with self.assertRaisesRegex(ValueError, "endcap shells 42/43"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["selection"]["preserve_groups"].remove(
            "main_closed_cell_buoyancy_material"
        )
        with self.assertRaisesRegex(ValueError, "buoyancy-material shell 257"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["buoyancy_material_validation"]["identification_status"] = "proven"
        with self.assertRaisesRegex(ValueError, "identification confidence"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["selection"]["preserve_groups"].remove("propeller_3blade")
        with self.assertRaisesRegex(ValueError, "propeller and hub"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["locked_propeller"]["connector_radius_mm"] = 2.5
        with self.assertRaisesRegex(ValueError, "reviewed bore overlap"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["locked_propeller"]["axisymmetric_profile_mm"][5][0] = -100.0
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["locked_propeller"]["maximum_source_volume_relative_error"] = 0.1
        with self.assertRaisesRegex(ValueError, "source-volume tolerance"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["sealed_pressure_boundary"]["radial_wall_overlap_mm"] = 6.0
        with self.assertRaisesRegex(ValueError, "radial overlap"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["sealed_pressure_boundary"]["endcap_by_hull_end"] = {
            "start": 42,
            "end": 42,
        }
        with self.assertRaisesRegex(ValueError, "one reviewed end fitting"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["output_frame"]["translation_mm"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "move the reviewed source COM"):
            _validate_config(malformed, Path(config["source"]["basename"]))

        malformed = copy.deepcopy(config)
        malformed["volume_validation"]["wrapped_surface_relative_tolerance"] = 1.0
        with self.assertRaisesRegex(ValueError, "wrapped surface relative tolerance"):
            _validate_config(malformed, Path(config["source"]["basename"]))


if __name__ == "__main__":
    unittest.main()

"""Small synthetic tests for the voxel exterior wrapper."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import voxel_wrap  # noqa: E402


BACKEND = voxel_wrap.backend_status()


class VolumeValidationTests(unittest.TestCase):
    def test_cli_requires_expected_volume(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing_volume:
                voxel_wrap._build_parser().parse_args(
                    ["source.stl", "output.stl", "--voxel-size", "0.5"]
                )
        self.assertEqual(missing_volume.exception.code, 2)

    def test_accepts_volume_within_tolerance(self) -> None:
        result = voxel_wrap._volume_validation(1.019, 1.0, 0.02)
        self.assertTrue(result["enabled"])
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["relative_error"], 0.019)

    def test_rejects_pre_fix_pressure_hull_wrap(self) -> None:
        result = voxel_wrap._volume_validation(
            7_686_508.750,
            11_304_505.834,
            0.02,
        )
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["relative_error"], 0.3200491, places=6)

    def test_rejects_invalid_contract_values(self) -> None:
        with self.assertRaisesRegex(voxel_wrap.VoxelWrapError, "expected volume"):
            voxel_wrap._volume_validation(1.0, 0.0, 0.02)
        with self.assertRaisesRegex(voxel_wrap.VoxelWrapError, "relative tolerance"):
            voxel_wrap._volume_validation(1.0, 1.0, 1.0)


@unittest.skipUnless(BACKEND.get("ready"), BACKEND.get("reason", "voxel backend unavailable"))
class VoxelWrapTests(unittest.TestCase):
    @staticmethod
    def _write_cubes(path: Path, centers: tuple[tuple[float, float, float], ...]) -> None:
        import vtk

        append = vtk.vtkAppendPolyData()
        sources = []
        for center in centers:
            cube = vtk.vtkCubeSource()
            cube.SetCenter(*center)
            cube.SetXLength(1.0)
            cube.SetYLength(1.0)
            cube.SetZLength(1.0)
            append.AddInputConnection(cube.GetOutputPort())
            sources.append(cube)
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(append.GetOutputPort())
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(triangles.GetOutputPort())
        writer.SetFileTypeToBinary()
        if int(writer.Write()) != 1:
            raise RuntimeError("could not write synthetic STL")

    @classmethod
    def _write_overlapping_cubes(cls, path: Path) -> None:
        cls._write_cubes(path, ((0.0, 0.0, 0.0), (0.6, 0.0, 0.0)))

    def test_overlapping_shells_become_one_watertight_reported_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "overlapping.stl"
            output = root / "wrapped.stl"
            report_path = root / "wrapped.json"
            self._write_overlapping_cubes(source)

            report = voxel_wrap.wrap_surface(
                source,
                output,
                voxel_size=0.1,
                report_path=report_path,
                closing_iterations=1,
                pad_voxels=4,
                repair_iterations=8,
                repair_radius=1,
                max_voxels=1_000_000,
            )

            self.assertTrue(output.is_file())
            self.assertTrue(report_path.is_file())
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["output"]["sha256"], report["output"]["sha256"])
            self.assertGreaterEqual(report["input"]["topology"]["connected_regions"], 2)
            topology = report["output"]["topology"]
            self.assertEqual(topology["boundary_edges"], 0)
            self.assertEqual(topology["non_manifold_edges"], 0)
            self.assertEqual(topology["connected_regions"], 1)
            self.assertTrue(topology["watertight_manifold"])
            self.assertFalse(report["parameters"]["surface_smoothing"])
            self.assertEqual(report["parameters"]["outside_connectivity"], 6)
            self.assertTrue(report["parameters"]["require_single_component"])
            self.assertIn("input_points_to_output_surface", report["distance"])
            self.assertIn("output_points_to_input_surface", report["distance"])
            self.assertGreater(report["output"]["triangles"], 0)
            self.assertFalse(report["volume_validation"]["enabled"])

            with self.assertRaisesRegex(voxel_wrap.VoxelWrapError, "refusing to overwrite"):
                voxel_wrap.wrap_surface(
                    source,
                    output,
                    voxel_size=0.1,
                    report_path=report_path,
                    max_voxels=1_000_000,
                )

    def test_multiple_components_require_an_explicit_cli_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "separated.stl"
            strict_output = root / "strict.stl"
            strict_report = root / "strict.json"
            allowed_output = root / "allowed.stl"
            allowed_report = root / "allowed.json"
            self._write_cubes(source, ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

            with self.assertRaisesRegex(voxel_wrap.VoxelWrapError, "components=2"):
                voxel_wrap.wrap_surface(
                    source,
                    strict_output,
                    voxel_size=0.1,
                    report_path=strict_report,
                    max_voxels=1_000_000,
                )
            self.assertFalse(strict_output.exists())
            self.assertFalse(strict_report.exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = voxel_wrap.main(
                    [
                        str(source),
                        str(allowed_output),
                        "--voxel-size",
                        "0.1",
                        "--json",
                        str(allowed_report),
                        "--allow-multiple",
                        "--max-voxels",
                        "1000000",
                        "--expected-volume",
                        "2",
                        "--volume-relative-tolerance",
                        "0.5",
                    ]
                )

            self.assertEqual(status, 0, stdout.getvalue())
            report = json.loads(allowed_report.read_text(encoding="utf-8"))
            self.assertFalse(report["parameters"]["require_single_component"])
            topology = report["output"]["topology"]
            self.assertEqual(topology["connected_regions"], 2)
            self.assertFalse(topology["single_component"])
            self.assertTrue(topology["watertight_manifold"])

    def test_volume_gate_fails_before_publishing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cube.stl"
            output = root / "wrapped.stl"
            report_path = root / "wrapped.json"
            self._write_cubes(source, ((0.0, 0.0, 0.0),))

            with self.assertRaisesRegex(
                voxel_wrap.VoxelWrapError, "displaced-volume gate failed"
            ):
                voxel_wrap.wrap_surface(
                    source,
                    output,
                    voxel_size=0.1,
                    report_path=report_path,
                    max_voxels=1_000_000,
                    expected_volume=10.0,
                    volume_relative_tolerance=0.02,
                )
            self.assertFalse(output.exists())
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()

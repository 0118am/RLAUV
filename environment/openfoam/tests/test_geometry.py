"""Focused tests for the standalone OpenFOAM geometry utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_environment  # noqa: E402
import inspect_stl  # noqa: E402
import prepare_geometry  # noqa: E402


def _normal(triangle: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    a, b, c = triangle
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    magnitude = math.sqrt(sum(value * value for value in cross))
    return tuple(value / magnitude for value in cross)


def _write_binary_stl(
    path: Path,
    triangles: list[tuple[tuple[float, float, float], ...]],
    *,
    header: bytes = b"synthetic unittest STL",
) -> None:
    padded_header = header[:80].ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(padded_header)
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            values = (*_normal(triangle), *triangle[0], *triangle[1], *triangle[2], 0)
            stream.write(struct.pack("<12fH", *values))


def _tetrahedron() -> list[tuple[tuple[float, float, float], ...]]:
    p0 = (0.0, 0.0, 0.0)
    p1 = (1.0, 0.0, 0.0)
    p2 = (0.0, 1.0, 0.0)
    p3 = (0.0, 0.0, 1.0)
    # Every edge occurs exactly twice; winding is outward.
    return [
        (p0, p2, p1),
        (p0, p1, p3),
        (p0, p3, p2),
        (p1, p2, p3),
    ]


class InspectSTLTests(unittest.TestCase):
    def test_binary_metadata_sha_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tetra.stl"
            _write_binary_stl(source, _tetrahedron(), header=b"closed tetra")

            report = inspect_stl.inspect_stl(source, topology=False)

            self.assertEqual(report["format"], "binary")
            self.assertEqual(report["triangle_count"], 4)
            self.assertEqual(report["size_bytes"], 84 + 4 * 50)
            self.assertEqual(report["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(report["bbox"]["min"], [0.0, 0.0, 0.0])
            self.assertEqual(report["bbox"]["max"], [1.0, 1.0, 1.0])
            self.assertEqual(report["bbox"]["size"], [1.0, 1.0, 1.0])
            self.assertEqual(report["binary"]["header_ascii"], "closed tetra")
            self.assertEqual(report["degenerate_triangles"], 0)

    def test_strict_fails_when_topology_is_not_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "one-triangle.stl"
            output = Path(directory) / "report.json"
            _write_binary_stl(source, [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))])

            status = inspect_stl.main(
                [str(source), "--strict", "--no-topology", "--json", str(output)]
            )

            self.assertEqual(status, inspect_stl.EXIT_STRICT_FAILURE)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(report["topology"]["watertight"])

    @unittest.skipUnless(importlib.util.find_spec("vtk"), "VTK is optional")
    def test_vtk_confirms_closed_tetrahedron_and_rejects_open_triangle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed = root / "closed.stl"
            open_surface = root / "open.stl"
            _write_binary_stl(closed, _tetrahedron())
            _write_binary_stl(
                open_surface,
                [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
            )

            closed_report = inspect_stl.inspect_stl(closed)
            open_report = inspect_stl.inspect_stl(open_surface)

            self.assertTrue(inspect_stl.is_confirmed_watertight(closed_report))
            self.assertEqual(closed_report["topology"]["boundary_edges"], 0)
            self.assertFalse(inspect_stl.is_confirmed_watertight(open_report))
            self.assertEqual(open_report["topology"]["boundary_edges"], 3)


class PrepareGeometryTests(unittest.TestCase):
    def test_axis_map_requires_a_signed_permutation(self) -> None:
        self.assertEqual(prepare_geometry._parse_axis_map("z,-x,+y"), ("z", "-x", "y"))
        with self.assertRaisesRegex(
            prepare_geometry.GeometryPreparationError, "use each input axis exactly once"
        ):
            prepare_geometry._parse_axis_map("x,x,z")

    def test_hash_mismatch_refuses_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.stl"
            output = root / "output.stl"
            _write_binary_stl(source, _tetrahedron())

            with self.assertRaisesRegex(prepare_geometry.GeometryPreparationError, "SHA-256 mismatch"):
                prepare_geometry.prepare_geometry(
                    source,
                    output,
                    expected_sha256="0" * 64,
                    allow_dirty=True,
                )

            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".provenance.json").exists())

    def test_dirty_or_unaudited_input_is_refused_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "open.stl"
            output = root / "scaled.stl"
            _write_binary_stl(
                source,
                [((1.0, 2.0, 3.0), (2.0, 2.0, 3.0), (1.0, 4.0, 3.0))],
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            with self.assertRaisesRegex(
                prepare_geometry.GeometryPreparationError, "not confirmed watertight"
            ):
                prepare_geometry.prepare_geometry(source, output, expected_sha256=digest)

            self.assertFalse(output.exists())

    @unittest.skipUnless(importlib.util.find_spec("vtk"), "VTK scaling backend is optional")
    def test_allow_dirty_scales_about_origin_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "off-origin.stl"
            output = root / "scaled.stl"
            _write_binary_stl(
                source,
                [((10.0, 20.0, 30.0), (11.0, 20.0, 30.0), (10.0, 22.0, 30.0))],
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            provenance = prepare_geometry.prepare_geometry(
                source,
                output,
                expected_sha256=digest,
                scale=0.001,
                backend="vtk",
                allow_dirty=True,
            )

            output_report = inspect_stl.inspect_stl(output, topology=False)
            for actual, expected in zip(output_report["bbox"]["min"], [0.01, 0.02, 0.03]):
                self.assertAlmostEqual(actual, expected, places=8)
            for actual, expected in zip(output_report["bbox"]["max"], [0.011, 0.022, 0.03]):
                self.assertAlmostEqual(actual, expected, places=8)
            self.assertEqual(provenance["transform"]["origin"], [0.0, 0.0, 0.0])
            self.assertEqual(provenance["transform"]["translation"], [0.0, 0.0, 0.0])
            self.assertFalse(provenance["transform"]["auto_center"])
            self.assertTrue(provenance["validation"]["dirty_override_used"])
            sidecar = Path(str(output) + ".provenance.json")
            self.assertTrue(sidecar.is_file())
            persisted = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(persisted["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    @unittest.skipUnless(importlib.util.find_spec("vtk"), "VTK transform backend is optional")
    def test_axis_map_then_translation_is_applied_and_affine_bounds_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cad-frame.stl"
            output = root / "body-frame.stl"
            _write_binary_stl(
                source,
                [((10.0, 20.0, 30.0), (14.0, 20.0, 30.0), (10.0, 26.0, 32.0))],
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            provenance = prepare_geometry.prepare_geometry(
                source,
                output,
                expected_sha256=digest,
                scale=0.1,
                axis_map="z,-x,y",
                translate_after_map=(-30.0, 14.0, -20.0),
                backend="vtk",
                allow_dirty=True,
            )

            output_report = inspect_stl.inspect_stl(output, topology=False)
            for actual, expected in zip(output_report["bbox"]["min"], [0.0, 0.0, 0.0]):
                self.assertAlmostEqual(actual, expected, places=7)
            for actual, expected in zip(output_report["bbox"]["max"], [0.2, 0.4, 0.6]):
                self.assertAlmostEqual(actual, expected, places=7)
            transform = provenance["transform"]
            self.assertEqual(transform["axis_map"], ["z", "-x", "y"])
            self.assertEqual(transform["translate_after_map"], [-30.0, 14.0, -20.0])
            self.assertEqual(
                transform["axis_map_matrix"],
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
            self.assertEqual(transform["affine_matrix"], provenance["validation"]["bounds"]["affine_matrix"])
            self.assertTrue(provenance["validation"]["bounds"]["affine_transform_match"])
            self.assertFalse(provenance["validation"]["bounds"]["uniform_scale_about_origin"])
            self.assertTrue(provenance["backend"]["winding_reversed_for_reflection"])


class CheckEnvironmentTests(unittest.TestCase):
    def test_fake_loaded_opencfd_environment_needs_no_real_openfoam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_bin = Path(directory)
            fake_command = fake_bin / "blockMesh"
            fake_command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_command.chmod(0o755)
            environment = {
                "PATH": str(fake_bin),
                "WM_PROJECT": "OpenFOAM",
                "WM_PROJECT_VERSION": "v2512",
                "WM_PROJECT_DIR": str(fake_bin / "OpenFOAM-v2512"),
                "FOAM_API": "2512",
            }

            report = check_environment.inspect_environment(
                required_commands=("blockMesh",),
                environ=environment,
                minimum_api=2506,
            )

            self.assertTrue(report["ready"])
            self.assertEqual(report["openfoam"]["distribution"], "OpenCFD")
            self.assertEqual(report["openfoam"]["api"], 2512)
            self.assertEqual(report["missing_required_commands"], [])


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from environment.openfoam.tools.scale_binary_stl import scale_binary_stl


class ScaleBinaryStlTests(unittest.TestCase):
    def test_scales_vertices_preserves_layout_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.stl"
            output = root / "output.stl"
            report = root / "output.json"
            header = b"test".ljust(80, b"\0")
            triangle = struct.pack(
                "<12fH",
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1000.0,
                0.0,
                0.0,
                0.0,
                2000.0,
                0.0,
                7,
            )
            source.write_bytes(header + struct.pack("<I", 1) + triangle)
            payload = scale_binary_stl(
                source,
                output,
                report,
                scale=0.001,
                chunk_triangles=1,
                force=False,
            )
            self.assertEqual(output.stat().st_size, 134)
            self.assertEqual(payload["output"]["triangle_count"], 1)
            self.assertEqual(payload["output"]["bbox"]["max"], [1.0, 2.0, 0.0])
            self.assertIsNone(payload["source"]["provided_expected_sha256"])
            self.assertEqual(payload["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(json.loads(report.read_text())["transform"]["origin"], [0.0, 0.0, 0.0])

            dtype = np.dtype(
                [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
            )
            data = np.memmap(output, dtype=dtype, mode="r", offset=84, shape=(1,))
            np.testing.assert_allclose(
                data[0]["vertices"],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            )
            np.testing.assert_allclose(data[0]["normal"], [0.0, 0.0, 1.0])
            self.assertEqual(int(data[0]["attribute"]), 7)
            del data

            with self.assertRaises(FileExistsError):
                scale_binary_stl(
                    source,
                    output,
                    report,
                    scale=0.001,
                    chunk_triangles=1,
                    force=False,
                )


if __name__ == "__main__":
    unittest.main()

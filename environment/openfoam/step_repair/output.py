"""Body-frame checks and STL serialization for STEP repair."""

from __future__ import annotations

import math
from pathlib import Path
import struct

from environment.openfoam.step_repair.core import _bbox
from environment.openfoam.step_repair.pressure_boundary import _body_point_from_step

def _validate_body_frame(shells, validation: dict, translation_body_mm) -> dict:
    """Fingerprint the COM-centred body mapping with labelled thrusters."""

    from OCP.TopoDS import TopoDS

    landmarks = validation["propeller_landmarks"]
    if not landmarks:
        raise RuntimeError("Frame validation requires labelled propeller landmarks")
    if any(len(item["expected_body_mm"]) != 3 for item in landmarks):
        raise RuntimeError("Every expected thruster centre must contain three coordinates")

    matches = []
    for item in landmarks:
        index = int(item["shell_index"])
        bounds = _bbox(TopoDS.Shell_s(shells.FindKey(index)))
        centre_step = [
            (bounds[axis] + bounds[axis + 3]) / 2.0 for axis in range(3)
        ]
        measured = _body_point_from_step(centre_step, translation_body_mm)
        expected = [float(value) for value in item["expected_body_mm"]]
        matches.append(
            {
                "source_shell_index": index,
                "measured_body_mm": measured,
                "expected_label": item["label"],
                "expected_body_mm": expected,
                "error_mm": math.dist(measured, expected),
            }
        )

    maximum_error = max(item["error_mm"] for item in matches)
    tolerance = float(validation["max_nearest_error_mm"])
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise RuntimeError("Frame-validation tolerance must be finite and positive")
    if maximum_error > tolerance:
        raise RuntimeError(
            "STEP-to-body frame validation failed: "
            f"maximum thruster-centre error {maximum_error:.6g} mm exceeds "
            f"{tolerance:.6g} mm"
        )

    return {
        "method": "fixed labelled propeller-centre landmarks",
        "translation_body_mm": [float(value) for value in translation_body_mm],
        "tolerance_mm": tolerance,
        "maximum_error_mm": maximum_error,
        "matches": matches,
    }


def _write_stl(shape, path: Path, deflection: float, angular_deflection: float) -> dict:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS

    mesher = BRepMesh_IncrementalMesh(
        shape, deflection, False, angular_deflection, True
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade triangulation did not complete")
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_ShapeEnum.TopAbs_FACE, faces)
    null_faces = []
    face_triangle_count = 0
    for index in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(index))
        triangulation = BRep_Tool.Triangulation_s(face, TopLoc_Location())
        if triangulation is None:
            null_faces.append(index)
        else:
            face_triangle_count += triangulation.NbTriangles()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    if not writer.Write(shape, str(path)):
        raise RuntimeError(f"Failed to write STL: {path}")
    return {
        "open_cascade_face_count": faces.Extent(),
        "null_triangulation_face_indices": null_faces,
        "open_cascade_face_triangle_count": face_triangle_count,
    }


def _audit_binary_stl(path: Path) -> dict:
    """Report finite/degenerate records without building a global edge graph."""

    import numpy as np

    record_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
        align=False,
    )
    with path.open("rb") as stream:
        header = stream.read(80)
        raw_count = stream.read(4)
    if len(header) != 80 or len(raw_count) != 4:
        raise RuntimeError("OpenCascade wrote a truncated STL header")
    triangle_count = struct.unpack("<I", raw_count)[0]
    expected_size = 84 + triangle_count * record_dtype.itemsize
    if record_dtype.itemsize != 50 or path.stat().st_size != expected_size:
        raise RuntimeError("OpenCascade output is not an exact binary STL")
    records = np.memmap(path, dtype=record_dtype, mode="r", offset=84)
    vertices = np.asarray(records["vertices"])
    normals = np.asarray(records["normal"])
    finite = bool(np.isfinite(vertices).all() and np.isfinite(normals).all())
    repeated = (
        np.all(vertices[:, 0] == vertices[:, 1], axis=1)
        | np.all(vertices[:, 1] == vertices[:, 2], axis=1)
        | np.all(vertices[:, 2] == vertices[:, 0], axis=1)
    )
    cross = np.cross(
        vertices[:, 1].astype(np.float64) - vertices[:, 0],
        vertices[:, 2].astype(np.float64) - vertices[:, 0],
    )
    zero_area = np.einsum("ij,ij->i", cross, cross) <= 1e-24
    report = {
        "binary_layout_valid": True,
        "binary_triangle_count": triangle_count,
        "all_values_finite": finite,
        "repeated_vertex_triangle_count": int(np.count_nonzero(repeated)),
        "zero_area_triangle_count": int(np.count_nonzero(zero_area)),
    }
    del records
    if not finite:
        raise RuntimeError("Intermediate STL contains non-finite coordinates or normals")
    return report


def _preflight_paths(
    source: Path, config: Path, output: Path, report: Path, force: bool
) -> tuple[Path, Path, Path, Path]:
    resolved = tuple(path.resolve() for path in (source, config, output, report))
    if len(set(resolved)) != len(resolved):
        raise ValueError("input, config, output and report paths must all be distinct")
    source, config, output, report = resolved
    if not source.is_file():
        raise FileNotFoundError(source)
    if not config.is_file():
        raise FileNotFoundError(config)
    existing = [path for path in (output, report) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite " + ", ".join(map(str, existing)) + "; pass --force"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return source, config, output, report

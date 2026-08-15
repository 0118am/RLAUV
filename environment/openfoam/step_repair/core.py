"""OCP loading and shared solid operations for STEP repair."""

from __future__ import annotations

import math
from pathlib import Path
import sys

OPENFOAM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = OPENFOAM_ROOT / "geometry" / "verification_assembly_repair.json"
DEFAULT_OCP_SITE = OPENFOAM_ROOT / ".runtime" / "cadquery-ocp" / "site-packages"
EXPECTED_OCP_VERSION = "7.9.3.1"

def _load_ocp() -> str:
    if not DEFAULT_OCP_SITE.is_dir():
        raise SystemExit(
            "Pinned OCP runtime is unavailable. Run environment/openfoam/install_cad_tools.sh first."
        )
    sys.path.insert(0, str(DEFAULT_OCP_SITE))
    try:
        import OCP
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OCP is unavailable. Install the pinned CAD runtime with "
            "environment/openfoam/install_cad_tools.sh first."
        ) from exc
    actual_version = getattr(OCP, "__version__", "unknown")
    module_path = Path(OCP.__file__).resolve()
    if actual_version != EXPECTED_OCP_VERSION or not module_path.is_relative_to(
        DEFAULT_OCP_SITE.resolve()
    ):
        raise SystemExit(
            "Wrong OCP runtime loaded: "
            f"version={actual_version}, path={module_path}; expected "
            f"{EXPECTED_OCP_VERSION} below {DEFAULT_OCP_SITE.resolve()}"
        )
    return actual_version


def _count_subshapes(shape, kind) -> int:
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    mapped = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, mapped)
    return mapped.Extent()


def _shell_map(shape):
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    mapped = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_ShapeEnum.TopAbs_SHELL, mapped)
    return mapped


def _read_step(path: Path):
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP reader failed with status {status}")
    transferred = reader.TransferRoots()
    if transferred < 1:
        raise RuntimeError("STEP contains no transferable roots")
    return reader.OneShape()


def _closed_solid(shell):
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Solid

    if not BRep_Tool.IsClosed_s(shell):
        return None
    solid = ShapeFix_Solid().SolidFromShell(shell)
    if solid.IsNull() or not BRepCheck_Analyzer(solid, True).IsValid():
        raise RuntimeError("A retained closed shell could not be made into a valid solid")
    return solid


def _dominant_motor_cylinder(shell, mount_extension_mm: float, mount_side: str):
    """Return a smooth cylinder matching the largest cylindrical motor face."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    candidates = []
    explorer = TopExp_Explorer(shell, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face, True)
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            candidates.append(
                {
                    "radius": cylinder.Radius(),
                    "axis": cylinder.Axis(),
                    "vmin": surface.FirstVParameter(),
                    "vmax": surface.LastVParameter(),
                }
            )
        explorer.Next()
    if not candidates:
        raise RuntimeError("Reviewed motor shell has no cylindrical face")

    radius = max(item["radius"] for item in candidates)
    dominant = [item for item in candidates if abs(item["radius"] - radius) < 1e-6]
    reference = dominant[0]
    axis = reference["axis"]
    origin = axis.Location()
    direction = axis.Direction()
    aligned = []
    for item in dominant:
        other = item["axis"]
        dot = direction.Dot(other.Direction())
        if abs(abs(dot) - 1.0) > 1e-7:
            continue
        delta = other.Location().XYZ() - origin.XYZ()
        if delta.Crossed(direction.XYZ()).Modulus() > 1e-5:
            continue
        offset = delta.Dot(direction.XYZ())
        if dot > 0:
            aligned.extend([offset + item["vmin"], offset + item["vmax"]])
        else:
            aligned.extend([offset - item["vmax"], offset - item["vmin"]])
    if not aligned:
        raise RuntimeError("Unable to consolidate dominant motor-cylinder faces")
    vmin, vmax = min(aligned), max(aligned)
    measured_height = vmax - vmin
    if not (15.0 <= radius <= 18.0 and 20.0 <= measured_height <= 40.0):
        raise RuntimeError(
            "Motor signature changed unexpectedly: "
            f"radius={radius:g}, height={measured_height:g}"
        )
    if not (0.0 <= mount_extension_mm <= 10.0):
        raise RuntimeError("Motor mount extension must be between 0 and 10 mm")
    if mount_side not in {"vmin", "vmax"}:
        raise RuntimeError(f"Unknown reviewed motor mount side: {mount_side!r}")
    height = measured_height + mount_extension_mm
    start_parameter = vmin - mount_extension_mm if mount_side == "vmin" else vmin
    start = origin.XYZ() + direction.XYZ() * start_parameter
    axis2 = gp_Ax2(
        gp_Pnt(start.X(), start.Y(), start.Z()),
        gp_Dir(direction.X(), direction.Y(), direction.Z()),
    )
    solid = BRepPrimAPI_MakeCylinder(axis2, radius, height).Solid()
    return solid, {
        "radius_mm": radius,
        "measured_main_casing_height_mm": measured_height,
        "height_mm": height,
        "mount_extension_mm": mount_extension_mm,
        "reviewed_mount_side": mount_side,
        "start_step_mm": [start.X(), start.Y(), start.Z()],
        "direction_step": [direction.X(), direction.Y(), direction.Z()],
    }


def _make_compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def _common_volume(first, second) -> float:
    """Return the solid intersection volume, failing closed on Boolean errors."""

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    operation = BRepAlgoAPI_Common(first, second)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError("OpenCascade failed to validate motor/mount intersection")
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(operation.Shape(), properties, True, False, False)
    return abs(float(properties.Mass()))


def _solid_volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, True, False, False)
    return abs(float(properties.Mass()))


EXPECTED_SOURCE_COM_BODY_MM = (-1.306, 0.061, 2.385)
EXPECTED_BODY_TRANSLATION_MM = tuple(-value for value in EXPECTED_SOURCE_COM_BODY_MM)


def _to_body_flu(shape, translation_body_mm):
    """Apply the STEP-axis permutation, then translate the measured COM to zero."""

    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    tx, ty, tz = (float(value) for value in translation_body_mm)
    transform = gp_Trsf()
    transform.SetValues(
        0.0, 0.0, 1.0, tx,
        1.0, 0.0, 0.0, ty,
        0.0, 1.0, 0.0, tz,
    )
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _bbox(shape) -> list[float]:
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return list(box.Get())


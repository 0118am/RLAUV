"""CAD-validated physical parameters for the AUV verification vehicle.

All runtime values use the COM-centred body frame: x-forward, y-left, z-up.
SolidWorks coordinate system 1 has those axis directions, but its origin is
offset from the measured COM; STEP-derived CFD geometry is translated by the
recorded COM offset before use.  This module is the single source of truth for
AUV geometry, rigid-body properties, and measured T1--T8 force curves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AUVModel:
    """SolidWorks mass properties and measured geometry of the AUV rig.

    ``center_of_mass_offset_m`` is zero because the Isaac/PhysX body frame is
    explicitly configured as COM-centred.  Coordinate system 1 is x-forward,
    y-left, z-up, but its reported origin is not the COM.  Force arms below are
    runtime COM-relative values; STEP geometry has its own explicit axis map
    and COM translation in the OpenFOAM preparation workflow.
    """

    mass_kg: float = 11.301
    water_density_kg_m3: float = 1000.0
    # SolidWorks reports material volume, not the closed external volume that
    # displaces water.  Keep the independently identified displacement used by
    # the nearly neutrally buoyant assembled vehicle.
    solid_material_volume_m3: float = 0.008690716111
    displaced_volume_m3: float = 0.011304505834
    surface_area_m2: float = 2.514359189
    visual_bounds_size_m: tuple[float, float, float] = (0.561500000, 0.401999756, 0.190621773)

    center_of_mass_from_coordinate_system_1_m: tuple[float, float, float] = (-0.001306, 0.000061, 0.002385)
    center_of_mass_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The replacement mass-property report does not include a new CoB.  This
    # remains the independently identified assembled-vehicle value.
    center_of_buoyancy_from_com_m: tuple[float, float, float] = (0.003498, -0.000060, 0.018494)

    # SolidWorks reports positive products Lxy/Lxz/Lyz.  Standard mechanics,
    # URDF and PhysX inertia matrices use their negatives off diagonal.
    inertia_tensor_body_kg_m2: tuple[tuple[float, float, float], ...] = (
        (0.115628684, -0.000010883, -0.001001989),
        (-0.000010883, 0.201210129, -0.000004539),
        (-0.001001989, -0.000004539, 0.259427119),
    )

    # Canonical action/measurement order. The centers are measured relative
    # to the COM in the FLU body frame and converted from millimetres to metres.
    # T1--T4 are vertical (FR, RR, FL, RL); T5--T8 are horizontal
    # (RL, RR, FL, FR).  Keeping these physical IDs preserves the policy action
    # channels while matching the mirror pairs in the measured force table.
    thruster_labels: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
    thruster_positions_body_m: tuple[tuple[float, float, float], ...] = (
        (0.13400, -0.16000, -0.17098),   # T1 vertical/front-right
        (-0.15000, -0.16000, -0.17098),  # T2 vertical/rear-right
        (0.13400, 0.16000, -0.17098),    # T3 vertical/front-left
        (-0.15000, 0.16000, -0.17098),   # T4 vertical/rear-left (T2 mirror)
        (-0.15039, 0.10360, -0.06312),   # T5 horizontal/rear-left (T6 mirror)
        (-0.15039, -0.10360, -0.06312),  # T6 horizontal/rear-right
        (0.13439, 0.10360, -0.06312),    # T7 horizontal/front-left (T8 mirror)
        (0.13439, -0.10360, -0.06312),   # T8 horizontal/front-right
    )

    # Measured body-frame T60 vector-force model.  Each thruster stores the
    # four FLU vectors (a_negative, b_negative, a_positive, b_positive), with
    # vector components ordered (Fx, Fy, Fz).  Quadratic coefficients are in
    # N/us^2 and linear coefficients in N/us.  Curve signs and off-axis forces
    # are already encoded; there is no separate scalar thrust, polarity, or
    # fixed-axis conversion in the runtime chain.
    thruster_pwm_center_us: float = 1500.0
    thruster_pwm_half_range_us: float = 200.0
    thruster_pwm_deadband_us: float = 25.0
    thruster_force_curve_coefficients: tuple[tuple[tuple[float, float, float], ...], ...] = (
        (
            (1.04794e-5, -2.24161e-5, -4.84864e-5),
            (2.90491e-4, 9.17132e-5, -3.62652e-2),
            (8.44745e-6, -2.31153e-5, 9.42738e-5),
            (3.96402e-3, 4.95147e-3, 5.61775e-3),
        ),
        (
            (-1.23746e-5, -3.31253e-5, -9.25861e-5),
            (1.92894e-3, 2.86340e-3, -1.983995e-2),
            (-3.41477e-5, -1.53342e-5, -6.65934e-6),
            (8.59225e-3, 7.88179e-3, 4.28832e-2),
        ),
        (
            (1.04794e-5, 2.24161e-5, -4.84864e-5),
            (2.90491e-4, -9.17132e-5, -3.62652e-2),
            (8.44745e-6, 2.31153e-5, 9.42738e-5),
            (3.96402e-3, -4.95147e-3, 5.61775e-3),
        ),
        (
            (-1.23746e-5, 3.31253e-5, -9.25861e-5),
            (1.92894e-3, -2.86340e-3, -1.983995e-2),
            (-3.41477e-5, 1.53342e-5, -6.65934e-6),
            (8.59225e-3, -7.88179e-3, 4.28832e-2),
        ),
        (
            (-2.87273e-5, -5.99843e-7, 2.54886e-7),
            (-2.10643e-2, -1.06914e-2, -2.08868e-3),
            (-1.85457e-5, 1.74530e-5, -1.77918e-6),
            (2.07221e-2, -6.78815e-3, -8.46072e-3),
        ),
        (
            (-2.87273e-5, 5.99843e-7, 2.54886e-7),
            (-2.10643e-2, 1.06914e-2, -2.08868e-3),
            (-1.85457e-5, -1.74530e-5, -1.77918e-6),
            (2.07221e-2, 6.78815e-3, -8.46072e-3),
        ),
        (
            (3.02662e-5, 4.69924e-7, 2.31237e-5),
            (2.83503e-2, -1.21290e-2, -1.42489e-2),
            (-6.96610e-5, 1.68788e-5, 2.93961e-5),
            (4.26677e-3, -7.72990e-3, -2.07979e-2),
        ),
        (
            (3.02662e-5, -4.69924e-7, 2.31237e-5),
            (2.83503e-2, 1.21290e-2, -1.42489e-2),
            (-6.96610e-5, -1.68788e-5, 2.93961e-5),
            (4.26677e-3, 7.72990e-3, -2.07979e-2),
        ),
    )


AUV = AUVModel()

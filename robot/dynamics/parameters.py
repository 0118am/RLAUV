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
    # SolidWorks reports material volume, not the closed external volume that
    # displaces water. Pool measurement shows buoyancy exceeds weight by a
    # 0.24 kg-equivalent at rho=1000 kg/m^3: V=(11.301+0.24)/1000.
    solid_material_volume_m3: float = 0.008690716111
    displaced_volume_m3: float = 0.011541
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

    # Canonical action/measurement order.  The propeller centres are the
    # verified COM-relative installation coordinates supplied in millimetres
    # and converted here to metres.  T1--T4 are vertical (FR, RR, FL, RL);
    # T5--T8 are horizontal (RL, RR, FL, FR).  The hash-pinned STEP record
    # below remains the source of the fixed shaft axes, not of these centres.
    thruster_installation_source_step_sha256: str = (
        "9777be1c028d8ebb18f61118466d17671aee1f5860ea8144717c50bc65d6ba07"
    )
    thruster_installation_frame: str = "body_flu_com"
    thruster_labels: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
    thruster_positions_body_m: tuple[tuple[float, float, float], ...] = (
        (0.13400, -0.16000, -0.17098),   # T1 vertical/front-right
        (-0.15000, -0.16000, -0.17098),  # T2 vertical/rear-right
        (0.13400, 0.16000, -0.17098),    # T3 vertical/front-left
        (-0.15000, 0.16000, -0.17098),   # T4 vertical/rear-left
        (-0.15039, 0.10360, -0.06312),   # T5 horizontal/rear-left
        (-0.15039, -0.10360, -0.06312),  # T6 horizontal/rear-right
        (0.13439, 0.10360, -0.06312),    # T7 horizontal/front-left
        (0.13439, -0.10360, -0.06312),   # T8 horizontal/front-right
    )
    thruster_axes_body: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.939692620786, 0.342020143326, 0.0),
        (0.939692620786, -0.342020143326, 0.0),
        (-0.939692620786, 0.342020143326, 0.0),
        (-0.939692620786, -0.342020143326, 0.0),
    )

    # Measured body-frame T60 vector-force model. Each thruster stores the four
    # physical-PWM FLU vectors (a_positive, b_positive, a_negative, b_negative),
    # with vector components ordered (Fx, Fy, Fz). Quadratic coefficients are
    # in N/us^2 and linear coefficients in N/us. Curve signs, including the
    # installed left/right PWM orientation, and off-axis forces are already
    # encoded. Never negate the PWM offset per thruster at evaluation time.
    thruster_pwm_center_us: float = 1500.0
    thruster_pwm_half_range_us: float = 200.0
    thruster_pwm_deadband_us: float = 25.0
    thruster_force_curve_coefficients: tuple[tuple[tuple[float, float, float], ...], ...] = (
        (
            (1.0479446747411169e-05, -2.2416124822308705e-05, -4.8486422690078336e-05),
            (0.00029049139889470002, 9.1713158089046663e-05, -0.036265199090186197),
            (8.4474520804025243e-06, -2.3115342868926689e-05, 9.4273754131380441e-05),
            (0.0039640197202361, 0.0049514676540881996, 0.0056177491461797003),
        ),
        (
            (-1.2374551071730197e-05, -3.3125295417693135e-05, -9.2586109270650197e-05),
            (0.0019289405138891, 0.0028634040004971001, -0.019839952204023199),
            (-3.4147700627779172e-05, -1.5334235466764995e-05, -6.6593389013093715e-06),
            (0.0085922479354993003, 0.0078817943418689994, 0.042883242917725398),
        ),
        (
            (1.0479446747411169e-05, 2.2416124822308705e-05, -4.8486422690078336e-05),
            (0.00029049139889470002, -9.1713158089046663e-05, -0.036265199090186197),
            (8.4474520804025243e-06, 2.3115342868926689e-05, 9.4273754131380441e-05),
            (0.0039640197202361, -0.0049514676540881996, 0.0056177491461797003),
        ),
        (
            (-1.2374551071730197e-05, 3.3125295417693135e-05, -9.2586109270650197e-05),
            (0.0019289405138891, -0.0028634040004971001, -0.019839952204023199),
            (-3.4147700627779172e-05, 1.5334235466764995e-05, -6.6593389013093715e-06),
            (0.0085922479354993003, -0.0078817943418689994, 0.042883242917725398),
        ),
        (
            (-2.8727300664564519e-05, -5.9984305658458156e-07, 2.5488577481102081e-07),
            (-0.021064261061346699, -0.0106913668168963, -0.0020886827450356999),
            (-1.8545677155981932e-05, 1.7453036222353297e-05, -1.7791792921878693e-06),
            (0.0207220916590331, -0.0067881453100422998, -0.0084607241962699007),
        ),
        (
            (-2.8727300664564519e-05, 5.9984305658458156e-07, 2.5488577481102081e-07),
            (-0.021064261061346699, 0.0106913668168963, -0.0020886827450356999),
            (-1.8545677155981932e-05, -1.7453036222353297e-05, -1.7791792921878693e-06),
            (0.0207220916590331, 0.0067881453100422998, -0.0084607241962699007),
        ),
        (
            (3.0266152915549253e-05, 4.6992370553306008e-07, 2.3123665856172451e-05),
            (0.028350257707977399, -0.0121290379475416, -0.014248856239900101),
            (-6.9661024358955455e-05, 1.6878772341498014e-05, 2.9396098190587793e-05),
            (0.0042667706930280998, -0.00772990265755, -0.020797881665092301),
        ),
        (
            (3.0266152915549253e-05, -4.6992370553306008e-07, 2.3123665856172451e-05),
            (0.028350257707977399, 0.0121290379475416, -0.014248856239900101),
            (-6.9661024358955455e-05, -1.6878772341498014e-05, 2.9396098190587793e-05),
            (0.0042667706930280998, 0.00772990265755, -0.020797881665092301),
        ),
    )


AUV = AUVModel()

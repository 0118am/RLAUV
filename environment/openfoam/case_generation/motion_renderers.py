"""Render motion, fields, solver controls, and schema-5 case metadata."""

from __future__ import annotations

import math
from typing import Any, Mapping

from environment.openfoam.case_generation.config import (
    CaseSpec,
    DEFAULT_TEMPLATE,
    ramped_sinusoid_peak_factors,
)
from environment.openfoam.case_generation.formatting import foam_header, fmt


def ambient_turbulence_state(cfg: Mapping[str, Any]) -> dict[str, float]:
    """Convert the campaign-wide turbulence reference into SST k and omega."""

    reference = cfg["ambient_turbulence_reference"]
    speed = float(reference["reference_speed_m_s"])
    intensity_percent = float(reference["turbulence_intensity_percent"])
    intensity = intensity_percent / 100.0
    length_scale = float(reference["turbulence_length_scale_m"])
    kinetic_energy = 1.5 * (intensity * speed) ** 2
    omega = math.sqrt(kinetic_energy) / (0.09 ** 0.25 * length_scale)
    return {
        "reference_speed_m_s": speed,
        "intensity_percent": intensity_percent,
        "length_scale_m": length_scale,
        "k_m2_s2": kinetic_energy,
        "omega_s_inv": omega,
    }


def render_turbulence_field(name: str, cfg: Mapping[str, Any]) -> str:
    """Render the three fields required by fully turbulent kOmegaSST."""

    state = ambient_turbulence_state(cfg)
    blending = str(cfg["wall_function_blending"])
    if name == "k":
        value = fmt(state["k_m2_s2"])
        dimensions = "[0 2 -2 0 0 0 0]"
        wall = f"type kqRWallFunction;\n        value uniform {value};"
        far = (
            f"type freestream;\n        freestreamValue uniform {value};"
            f"\n        value uniform {value};"
        )
    elif name == "omega":
        value = fmt(state["omega_s_inv"])
        dimensions = "[0 0 -1 0 0 0 0]"
        wall = (
            "type omegaWallFunction;\n"
            f"        blending {blending};\n        value uniform {value};"
        )
        far = (
            f"type freestream;\n        freestreamValue uniform {value};"
            f"\n        value uniform {value};"
        )
    elif name == "nut":
        value = "0"
        dimensions = "[0 2 -1 0 0 0 0]"
        wall = (
            "type nutkWallFunction;\n"
            f"        blending {blending};\n        value uniform 0;"
        )
        far = "type calculated;\n        value uniform 0;"
    else:
        raise ValueError(f"Unsupported kOmegaSST field: {name}")
    return foam_header(name, "volScalarField") + f"""
dimensions      {dimensions};
internalField   uniform {value};

boundaryField
{{
    auv
    {{
        {wall}
    }}
    farField
    {{
        {far}
    }}
}}
"""


def render_velocity_field(spec: CaseSpec, cfg: Mapping[str, Any]) -> str:
    """Render a steady equivalent tow or a quiescent oscillatory far field."""

    del cfg
    water_velocity = tuple(-value for value in spec.body_velocity_b_m_s)
    velocity = " ".join(fmt(value) for value in water_velocity)
    wall_type = "fixedValue" if spec.kind == "steady_translation" else "movingWallVelocity"
    if spec.kind == "steady_translation":
        far_field = (
            "type            freestreamVelocity;\n"
            f"        freestreamValue uniform ({velocity});\n"
            f"        value           uniform ({velocity});"
        )
    else:
        # freestreamVelocity is singular when U=(0,0,0).
        far_field = (
            "type            pressureInletOutletVelocity;\n"
            "        tangentialVelocity uniform (0 0 0);\n"
            "        value           uniform (0 0 0);"
        )
    return foam_header("U", "volVectorField") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({velocity});

boundaryField
{{
    auv
    {{
        type            {wall_type};
        value           uniform (0 0 0);
    }}
    farField
    {{
        {far_field}
    }}
}}
"""


def render_pressure_field(spec: CaseSpec, cfg: Mapping[str, Any]) -> str:
    del cfg
    if spec.kind == "steady_translation":
        far_field = (
            "type            freestreamPressure;\n"
            "        freestreamValue uniform 0;\n"
            "        value           uniform 0;"
        )
    else:
        far_field = "type fixedValue;\n        value uniform 0;"
    return foam_header("p", "volScalarField") + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    auv
    {{
        type            zeroGradient;
    }}
    farField
    {{
        {far_field}
    }}
}}
"""


def render_fv_solution(spec: CaseSpec, cfg: Mapping[str, Any]) -> str:
    template = (DEFAULT_TEMPLATE / "system" / "fvSolution").read_text(encoding="utf-8")
    gamg_marker = "        // __GAMG_UPDATE_INTERVAL__"
    motion_marker = "    // __MOVE_MESH_OUTER_CORRECTORS__"
    outer_marker = "__PIMPLE_OUTER_CORRECTORS__"
    if template.count(gamg_marker) != 2:
        raise RuntimeError("fvSolution GAMG update markers changed")
    if template.count(motion_marker) != 1 or template.count(outer_marker) != 1:
        raise RuntimeError("fvSolution PIMPLE markers changed")
    return (
        template.replace(
            gamg_marker,
            f"        updateInterval  {int(cfg['gamg_update_interval'])};",
        )
        .replace(
            motion_marker,
            "    moveMeshOuterCorrectors "
            f"{'yes' if spec.is_oscillatory and cfg['move_mesh_outer_correctors'] else 'no'};",
        )
        .replace(outer_marker, str(int(cfg["pimple_outer_correctors"])))
    )


def render_dynamic_mesh_dict(spec: CaseSpec, cfg: Mapping[str, Any]) -> str:
    del cfg
    if not spec.is_oscillatory:
        return foam_header("dynamicMeshDict") + """
dynamicFvMesh   staticFvMesh;
"""
    return foam_header("dynamicMeshDict") + """
dynamicFvMesh   dynamicMotionSolverFvMesh;
motionSolverLibs (fvMotionSolvers);
motionSolver    displacementLaplacian;
diffusivity     quadratic inverseDistance 1(auv);
"""


def timeline(spec: CaseSpec, cfg: dict[str, Any]) -> dict[str, float]:
    if spec.kind == "steady_translation":
        damping = cfg["damping_identification"]
        speed = math.sqrt(sum(value * value for value in spec.body_velocity_b_m_s))
        convection_time = float(cfg["reference_length_m"]) / speed
        settle_end = float(damping["steady_settle_body_lengths"]) * convection_time
        sample_duration = float(damping["steady_sample_body_lengths"]) * convection_time
        max_delta_t = min(
            float(damping["steady_max_delta_t_s"]),
            convection_time / int(damping["steady_steps_per_body_length"]),
        )
        return {
            "period_s": 0.0,
            "omega_rad_s": 0.0,
            "ramp_end_s": 0.0,
            "settle_end_s": settle_end,
            "end_time_s": settle_end + sample_duration,
            "delta_t_s": max_delta_t,
            "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
            "write_interval_s": sample_duration / int(cfg["writes_per_cycle"]),
        }
    if spec.frequency_hz is None:
        maximum_frequency = max(
            float(cfg["added_mass_identification"]["frequency_hz"]),
            float(cfg["damping_identification"]["rotation_frequency_hz"]),
        )
        max_delta_t = 1.0 / (maximum_frequency * int(cfg["steps_per_cycle"]))
        return {
            "period_s": 0.0,
            "omega_rad_s": 0.0,
            "ramp_end_s": 0.0,
            "settle_end_s": 0.0,
            "end_time_s": 1.0 / maximum_frequency,
            "delta_t_s": max_delta_t,
            "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
            "write_interval_s": 1.0 / maximum_frequency,
        }
    period = 1.0 / float(spec.frequency_hz)
    max_delta_t = period / int(cfg["steps_per_cycle"])
    ramp_end = float(spec.ramp_cycles) * period
    settle_end = ramp_end + float(spec.settle_cycles_after_ramp) * period
    end_time = settle_end + float(spec.sample_cycles) * period
    return {
        "period_s": period,
        "omega_rad_s": 2.0 * math.pi * float(spec.frequency_hz),
        "ramp_end_s": ramp_end,
        "settle_end_s": settle_end,
        "end_time_s": end_time,
        "delta_t_s": max_delta_t,
        "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
        "write_interval_s": period / int(cfg["writes_per_cycle"]),
    }


def render_point_displacement(spec: CaseSpec, cfg: dict[str, Any]) -> str:
    time = timeline(spec, cfg)
    if spec.kind == "translation":
        body = f"""        type            rampedRigidBodyDisplacement;
        motionKind      translation;
        axis            ({' '.join(str(value) for value in spec.axis)});
        origin          (0 0 0);
        amplitude       {fmt(spec.amplitude_m or 0.0)};
        omega           {fmt(time['omega_rad_s'])};
        phase           0;
        rampDuration    {fmt(time['ramp_end_s'])};
        value           uniform (0 0 0);"""
    elif spec.kind == "rotation":
        origin = " ".join(fmt(float(value)) for value in cfg["centre_of_rotation_m"])
        body = f"""        type            rampedRigidBodyDisplacement;
        motionKind      rotation;
        axis            ({' '.join(str(value) for value in spec.axis)});
        origin          ({origin});
        amplitude       {fmt(spec.amplitude_rad or 0.0)};
        omega           {fmt(time['omega_rad_s'])};
        phase           0;
        rampDuration    {fmt(time['ramp_end_s'])};
        value           uniform (0 0 0);"""
    else:
        body = """        type            fixedValue;
        value           uniform (0 0 0);"""
    return foam_header("pointDisplacement", "pointVectorField") + f"""
dimensions      [0 1 0 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    auv
    {{
{body}
    }}
    farField
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
}}
"""


def render_control_dict(spec: CaseSpec, cfg: dict[str, Any]) -> str:
    time = timeline(spec, cfg)
    origin = " ".join(fmt(float(value)) for value in cfg["centre_of_rotation_m"])
    motion_library = (
        'libs                ("librampedAuvMotion.so");\n'
        if spec.is_oscillatory
        else ""
    )
    return foam_header("controlDict") + f"""
application         pimpleFoam;
{motion_library}startFrom           startTime;
startTime           0;
stopAt              endTime;
endTime             {fmt(time['end_time_s'])};
deltaT              {fmt(time['initial_delta_t_s'])};
adjustTimeStep      yes;
maxCo               {fmt(float(cfg['max_co']))};
maxDeltaT           {fmt(time['delta_t_s'])};
writeControl        adjustable;
writeInterval       {fmt(time['write_interval_s'])};
purgeWrite          {int(cfg['purge_write'])};
writeFormat         binary;
writePrecision      10;
writeCompression    off;
timeFormat          general;
timePrecision       10;
runTimeModifiable   true;

functions
{{
    forces
    {{
        type            forces;
        libs            (forces);
        executeControl  timeStep;
        executeInterval {int(cfg['force_execute_interval'])};
        writeControl    timeStep;
        writeInterval   {int(cfg['force_execute_interval'])};
        timeStart       0;
        log             true;
        patches         (auv);
        rho             rhoInf;
        rhoInf          {fmt(float(cfg['rho_kg_m3']))};
        CofR            ({origin});
    }}
    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        executeControl  writeTime;
        writeControl    writeTime;
        writeFields     false;
    }}
}}
"""


def render_transport_properties(cfg: dict[str, Any]) -> str:
    return foam_header("transportProperties") + f"""
transportModel  Newtonian;
nu              {fmt(float(cfg['nu_m2_s']))};
"""


def metadata(
    spec: CaseSpec,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    time = timeline(spec, cfg)
    if spec.frequency_hz:
        ramp_cycles = time["ramp_end_s"] / time["period_s"]
        settle_cycles = time["settle_end_s"] / time["period_s"]
        sample_cycles = (time["end_time_s"] - time["settle_end_s"]) / time["period_s"]
    else:
        ramp_cycles = settle_cycles = sample_cycles = 0.0
    result: dict[str, Any] = {
        "schema_version": 5,
        "openfoam_version": cfg["openfoam_version"],
        "solver": cfg["solver"],
        "flow_model": cfg["flow_model"],
        "case_name": spec.name,
        "case_family": spec.family,
        "dof": spec.dof,
        "dof_index": spec.dof_index,
        "kind": spec.kind,
        "axis": list(spec.axis),
        "amplitude_m": spec.amplitude_m,
        "amplitude_deg": spec.amplitude_deg,
        "amplitude_rad": spec.amplitude_rad,
        "velocity_amplitude_m_s": spec.velocity_amplitude_m_s,
        "rate_amplitude_rad_s": spec.rate_amplitude_rad_s,
        "body_velocity_b_m_s": list(spec.body_velocity_b_m_s),
        "frequency_hz": spec.frequency_hz,
        "omega_rad_s": time["omega_rad_s"],
        "phase_rad": 0.0,
        "period_s": time["period_s"],
        "ramp_cycles": ramp_cycles,
        "ramp_end_s": time["ramp_end_s"],
        "settle_cycles_after_ramp": spec.settle_cycles_after_ramp,
        "settle_cycles": settle_cycles,
        "sample_cycles": sample_cycles,
        "settle_end_s": time["settle_end_s"],
        "end_time_s": time["end_time_s"],
        "delta_t_s": time["delta_t_s"],
        "initial_delta_t_s": time["initial_delta_t_s"],
        "max_co": float(cfg["max_co"]),
        "rho_kg_m3": float(cfg["rho_kg_m3"]),
        "nu_m2_s": float(cfg["nu_m2_s"]),
        "campaign_target_envelope": {
            "translation_speed_limit_m_s": float(
                cfg["target_translation_speed_limit_m_s"]
            ),
            "rotation_rate_limit_rad_s": float(cfg["target_rotation_rate_limit_rad_s"]),
        },
        "wall_function_blending": cfg["wall_function_blending"],
        "ambient_turbulence_boundary": ambient_turbulence_state(cfg),
        "move_mesh_outer_correctors": bool(cfg["move_mesh_outer_correctors"]),
        "gamg_update_interval": int(cfg["gamg_update_interval"]),
        "pimple_outer_correctors": int(cfg["pimple_outer_correctors"]),
        "force_execute_interval": int(cfg["force_execute_interval"]),
        "centre_of_rotation_m": [float(value) for value in cfg["centre_of_rotation_m"]],
        "com_initial_global_m": [float(value) for value in cfg["centre_of_rotation_m"]],
        "force_patches": ["auv"],
        "purpose": spec.purpose,
        "include_in_fit": spec.purpose == "identification",
        "matrix_structure": "full_response_port_starboard_reflection_symmetric",
        "motion_boundary_condition": (
            "rampedRigidBodyDisplacement" if spec.is_oscillatory else "fixedValue"
        ),
    }
    if spec.is_oscillatory:
        velocity_factor, acceleration_factor, jerk_factor = (
            ramped_sinusoid_peak_factors(float(spec.ramp_cycles))
        )
        peak = float(
            spec.velocity_amplitude_m_s
            if spec.kind == "translation"
            else spec.rate_amplitude_rad_s
        )
        result["prescribed_motion_envelope"] = {
            "nominal_velocity_or_rate_si": peak,
            "maximum_velocity_or_rate_si_including_ramp": peak * velocity_factor,
            "nominal_acceleration_si": peak * time["omega_rad_s"],
            "maximum_acceleration_si_including_ramp": (
                peak * time["omega_rad_s"] * acceleration_factor
            ),
            "nominal_jerk_si": peak * time["omega_rad_s"] ** 2,
            "maximum_jerk_si_including_ramp": (
                peak * time["omega_rad_s"] ** 2 * jerk_factor
            ),
        }
    return result

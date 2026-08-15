"""Motion, solver, field, and metadata rendering."""

from __future__ import annotations

import math
from typing import Any, Mapping

from environment.openfoam.case_generation.config import CaseSpec, DEFAULT_TEMPLATE
from environment.openfoam.case_generation.formatting import _finite_vector, fmt, foam_header

def render_wall_function_field(name: str, cfg: Mapping[str, Any]) -> str:
    """Render a turbulence field with an explicit low/high-Re blender."""

    if name not in {"nut", "omega"}:
        raise ValueError(f"Unsupported wall-function field: {name}")
    template = (DEFAULT_TEMPLATE / "0" / name).read_text(encoding="utf-8")
    marker = "        // __WALL_FUNCTION_BLENDING__"
    if template.count(marker) != 1:
        raise RuntimeError(f"{name} wall-function blending marker changed")
    return template.replace(
        marker,
        f"        blending        {cfg['wall_function_blending']};",
    )


def render_velocity_field(cfg: Mapping[str, Any]) -> str:
    """Render a uniform towing-stream velocity without rotating the fixed far field."""

    velocity = _finite_vector(
        cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0)),
        "background_velocity_m_s",
    )
    foam_velocity = _foam_vector(velocity)
    return foam_header("U", "volVectorField") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform {foam_velocity};

boundaryField
{{
    auv
    {{
        type            movingWallVelocity;
        value           uniform (0 0 0);
    }}
    farField
    {{
        type            freestream;
        freestreamValue uniform {foam_velocity};
        value           uniform {foam_velocity};
    }}
}}
"""


def render_fv_solution(cfg: Mapping[str, Any]) -> str:
    """Render dynamic-mesh and GAMG performance controls explicitly."""

    template = (DEFAULT_TEMPLATE / "system" / "fvSolution").read_text(
        encoding="utf-8"
    )
    gamg_marker = "        // __GAMG_UPDATE_INTERVAL__"
    motion_marker = "    // __MOVE_MESH_OUTER_CORRECTORS__"
    outer_marker = "__PIMPLE_OUTER_CORRECTORS__"
    if template.count(gamg_marker) != 2:
        raise RuntimeError("fvSolution GAMG update markers changed")
    if template.count(motion_marker) != 1:
        raise RuntimeError("fvSolution dynamic-mesh marker changed")
    if template.count(outer_marker) != 1:
        raise RuntimeError("fvSolution PIMPLE outer-corrector marker changed")
    return template.replace(
        gamg_marker,
        f"        updateInterval  {int(cfg['gamg_update_interval'])};",
    ).replace(
        motion_marker,
        "    moveMeshOuterCorrectors "
        f"{'yes' if cfg['move_mesh_outer_correctors'] else 'no'};",
    ).replace(
        outer_marker,
        str(int(cfg["pimple_outer_correctors"])),
    )


def timeline(spec: CaseSpec, cfg: dict[str, Any]) -> dict[str, float]:
    if spec.frequency_hz is None:
        max_frequency = max(float(v) for v in cfg["frequencies_hz"])
        max_delta_t = 1.0 / (max_frequency * int(cfg["steps_per_cycle"]))
        return {
            "period_s": 0.0,
            "omega_rad_s": 0.0,
            "settle_end_s": float(cfg.get("baseline_settle_s", 0.5)),
            "end_time_s": float(cfg.get("baseline_duration_s", 2.0)),
            "delta_t_s": max_delta_t,
            "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
            "write_interval_s": 1.0 / max_frequency / int(cfg.get("writes_per_cycle", 4)),
        }
    period = 1.0 / spec.frequency_hz
    max_delta_t = period / int(cfg["steps_per_cycle"])
    fixed_start = cfg.get("fixed_analysis_start_s")
    fixed_end = cfg.get("fixed_end_time_s")
    if fixed_start is not None and fixed_end is not None:
        settle_end = float(fixed_start)
        end_time = float(fixed_end)
    else:
        settle_end = float(cfg["settle_cycles"]) * period
        convective_lengths = float(cfg.get("minimum_settle_convective_lengths", 0.0))
        if convective_lengths > 0.0:
            towing_speed = abs(float(cfg["background_velocity_m_s"][0]))
            convective_time = (
                convective_lengths * float(cfg["characteristic_length_m"]) / towing_speed
            )
            settle_end = max(settle_end, convective_time)
        end_time = settle_end + float(cfg["sample_cycles"]) * period
    return {
        "period_s": period,
        "omega_rad_s": 2.0 * math.pi * spec.frequency_hz,
        "settle_end_s": settle_end,
        "end_time_s": end_time,
        "delta_t_s": max_delta_t,
        "initial_delta_t_s": float(cfg["initial_delta_t_fraction"]) * max_delta_t,
        "write_interval_s": period / int(cfg.get("writes_per_cycle", 4)),
    }


def render_point_displacement(spec: CaseSpec, cfg: dict[str, Any]) -> str:
    time = timeline(spec, cfg)
    if spec.kind == "translation":
        vector = tuple(spec.amplitude_m * component for component in spec.axis)  # type: ignore[operator]
        body = f"""        type            oscillatingDisplacement;
        amplitude       ({' '.join(fmt(v) for v in vector)});
        omega           {fmt(time['omega_rad_s'])};
        value           uniform (0 0 0);"""
    elif spec.kind == "rotation":
        origin = " ".join(fmt(float(v)) for v in cfg["centre_of_rotation_m"])
        body = f"""        type            angularOscillatingDisplacement;
        axis            ({' '.join(str(v) for v in spec.axis)});
        origin          ({origin});
        angle0          0;
        amplitude       {fmt(spec.amplitude_rad or 0.0)};
        omega           {fmt(time['omega_rad_s'])};
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
    origin = " ".join(fmt(float(v)) for v in cfg["centre_of_rotation_m"])
    purge_write = int(cfg.get("purge_write", 4))
    max_co = float(cfg["max_co"])
    return foam_header("controlDict") + f"""
application         pimpleFoam;
startFrom           startTime;
startTime           0;
stopAt              endTime;
endTime             {fmt(time['end_time_s'])};
deltaT              {fmt(time['initial_delta_t_s'])};
adjustTimeStep      yes;
maxCo               {fmt(max_co)};
maxDeltaT           {fmt(time['delta_t_s'])};
writeControl        adjustable;
writeInterval       {fmt(time['write_interval_s'])};
purgeWrite          {purge_write};
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
        executeInterval {int(cfg.get('force_execute_interval', 1))};
        writeControl    timeStep;
        writeInterval   {int(cfg.get('force_execute_interval', 1))};
        // Record through the settling interval as well.  The fitter uses the
        // samples bracketing settle_end_s to interpolate an exact full-cycle
        // phase grid; starting output at settle_end_s would lose that bracket
        // under adaptive time stepping.
        timeStart       0;
        log             true;
        patches         ({cfg.get('force_patch', 'auv')});
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
    }}
}}
"""


def render_transport_properties(cfg: dict[str, Any]) -> str:
    return foam_header("transportProperties") + f"""
transportModel  Newtonian;
nu              {fmt(float(cfg['nu_m2_s']))};
"""


def metadata(spec: CaseSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    time = timeline(spec, cfg)
    if spec.frequency_hz:
        settle_cycles = time["settle_end_s"] / time["period_s"]
        sample_cycles = (time["end_time_s"] - time["settle_end_s"]) / time["period_s"]
    else:
        settle_cycles = 0.0
        sample_cycles = 0.0
    return {
        "schema_version": 1,
        "openfoam_version": cfg["openfoam_version"],
        "solver": cfg["solver"],
        "case_name": spec.name,
        "dof": spec.dof,
        "dof_index": spec.dof_index,
        "kind": spec.kind,
        "axis": list(spec.axis),
        "amplitude_m": spec.amplitude_m,
        "amplitude_deg": spec.amplitude_deg,
        "amplitude_rad": spec.amplitude_rad,
        "frequency_hz": spec.frequency_hz,
        "omega_rad_s": time["omega_rad_s"],
        "period_s": time["period_s"],
        "settle_cycles": settle_cycles,
        "sample_cycles": sample_cycles,
        "settle_end_s": time["settle_end_s"],
        "end_time_s": time["end_time_s"],
        "delta_t_s": time["delta_t_s"],
        "initial_delta_t_s": time["initial_delta_t_s"],
        "max_co": float(cfg["max_co"]),
        "rho_kg_m3": float(cfg["rho_kg_m3"]),
        "nu_m2_s": float(cfg["nu_m2_s"]),
        "wall_function_blending": cfg["wall_function_blending"],
        "move_mesh_outer_correctors": bool(cfg["move_mesh_outer_correctors"]),
        "gamg_update_interval": int(cfg["gamg_update_interval"]),
        "pimple_outer_correctors": int(cfg.get("pimple_outer_correctors", 2)),
        "force_execute_interval": int(cfg.get("force_execute_interval", 1)),
        "centre_of_rotation_m": [float(v) for v in cfg["centre_of_rotation_m"]],
        "force_patch": cfg.get("force_patch", "auv"),
        "background_velocity_m_s": [
            float(value) for value in cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0))
        ],
        "background_fluid_velocity_body_m_s": [
            float(value) for value in cfg.get("background_velocity_m_s", (0.0, 0.0, 0.0))
        ],
        "purpose": spec.purpose,
        "include_in_fit": spec.purpose == "identification",
    }

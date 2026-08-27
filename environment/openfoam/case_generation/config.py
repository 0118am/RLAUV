"""Configuration and construction of the preliminary 24-case CFD campaign."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

OPENFOAM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = OPENFOAM_ROOT / "config.json"
DEFAULT_TEMPLATE = OPENFOAM_ROOT / "case_template"
DEFAULT_OUTPUT = OPENFOAM_ROOT / "cases"

DOFS = {
    "u": (0, "translation", (1, 0, 0)),
    "v": (1, "translation", (0, 1, 0)),
    "w": (2, "translation", (0, 0, 1)),
    "p": (3, "rotation", (1, 0, 0)),
    "q": (4, "rotation", (0, 1, 0)),
    "r": (5, "rotation", (0, 0, 1)),
}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_values(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    result = tuple(_positive(item, name) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


@lru_cache(maxsize=None)
def ramped_sinusoid_peak_factors(ramp_cycles: float) -> tuple[float, float, float]:
    """Peak velocity, acceleration, and jerk factors of the quintic ramp."""

    cycles = _positive(ramp_cycles, "ramp_cycles")
    phase_span = 2.0 * math.pi * cycles
    peaks = [1.0, 1.0, 1.0]
    for index in range(20001):
        x = index / 20000.0
        ramp = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
        ramp_x = 30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4
        ramp_xx = 60.0 * x - 180.0 * x**2 + 120.0 * x**3
        ramp_xxx = 60.0 - 360.0 * x + 360.0 * x**2
        phase = phase_span * x
        sine = math.sin(phase)
        cosine = math.cos(phase)
        values = (
            ramp * cosine + ramp_x * sine / phase_span,
            -ramp * sine
            + 2.0 * ramp_x * cosine / phase_span
            + ramp_xx * sine / phase_span**2,
            -ramp * cosine
            - 3.0 * ramp_x * sine / phase_span
            + 3.0 * ramp_xx * cosine / phase_span**2
            + ramp_xxx * sine / phase_span**3,
        )
        peaks = [max(previous, abs(value)) for previous, value in zip(peaks, values)]
    return tuple(peaks)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: str
    dof: str | None
    dof_index: int | None
    kind: str
    axis: tuple[int, int, int]
    amplitude_m: float | None = None
    amplitude_deg: float | None = None
    amplitude_rad: float | None = None
    frequency_hz: float | None = None
    velocity_amplitude_m_s: float | None = None
    rate_amplitude_rad_s: float | None = None
    body_velocity_b_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ramp_cycles: float = 0.0
    settle_cycles_after_ramp: float = 0.0
    sample_cycles: float = 0.0
    purpose: str = "identification"

    @property
    def is_oscillatory(self) -> bool:
        return self.kind in {"translation", "rotation"}


def _validate_config(cfg: dict[str, Any]) -> None:
    required = {
        "schema_version", "design", "design_notes", "target_translation_speed_limit_m_s",
        "target_rotation_rate_limit_rad_s",
        "reference_length_m", "openfoam_version", "solver", "flow_model",
        "fluid_reference", "geometry_filename", "geometry_path", "force_patches",
        "rho_kg_m3", "nu_m2_s", "ambient_turbulence_reference",
        "wall_function_blending", "move_mesh_outer_correctors", "gamg_update_interval",
        "pimple_outer_correctors", "force_execute_interval", "centre_of_rotation_m",
        "geometry_audit", "damping_identification",
        "added_mass_identification", "steps_per_cycle", "initial_delta_t_fraction",
        "max_co", "writes_per_cycle", "purge_write",
        "block_mesh", "snappy", "locked_rotor_mesh", "analysis",
    }
    missing = sorted(required - set(cfg))
    extra = sorted(set(cfg) - required)
    if missing or extra:
        raise ValueError(f"Config keys differ from schema 5; missing={missing}, extra={extra}")
    if cfg["schema_version"] != 5 or cfg["design"] != "full_response_24_case":
        raise ValueError("Only schema 5 full_response_24_case is supported")
    if cfg["openfoam_version"] != "v2512" or cfg["solver"] != "pimpleFoam":
        raise ValueError("The campaign targets OpenCFD v2512 pimpleFoam")
    if cfg["flow_model"] != "single_phase_kOmegaSST_wall_function":
        raise ValueError("The campaign uses fully turbulent kOmegaSST wall functions")
    if cfg["force_patches"] != ["auv"]:
        raise ValueError("The no-layer mesh must use the single wall patch ['auv']")
    if Path(str(cfg["geometry_filename"])).suffix.lower() != ".obj":
        raise ValueError("geometry_filename must be an OBJ")
    if Path(str(cfg["geometry_path"])).suffix.lower() != ".obj":
        raise ValueError("geometry_path must be an OBJ")
    if any(abs(value) > 1.0e-12 for value in _vector(
        cfg["centre_of_rotation_m"], 3, "centre_of_rotation_m"
    )):
        raise ValueError("centre_of_rotation_m must be the body-FLU COM origin")

    translation_limit = _positive(
        cfg["target_translation_speed_limit_m_s"],
        "target_translation_speed_limit_m_s",
    )
    rotation_limit = _positive(
        cfg["target_rotation_rate_limit_rad_s"], "target_rotation_rate_limit_rad_s"
    )
    for name in (
        "rho_kg_m3", "nu_m2_s", "reference_length_m", "max_co",
    ):
        _positive(cfg[name], name)

    turbulence = cfg["ambient_turbulence_reference"]
    if not isinstance(turbulence, dict) or set(turbulence) != {
        "reference_speed_m_s", "turbulence_intensity_percent",
        "turbulence_length_scale_m",
    }:
        raise ValueError("ambient_turbulence_reference has unexpected keys")
    for name, value in turbulence.items():
        _positive(value, f"ambient_turbulence_reference.{name}")

    damping = cfg["damping_identification"]
    speeds = _positive_values(
        damping["translation_speeds_m_s"],
        "damping_identification.translation_speeds_m_s",
    )
    rates = _positive_values(
        damping["rotation_rate_amplitudes_rad_s"],
        "damping_identification.rotation_rate_amplitudes_rad_s",
    )
    if len(speeds) != 2 or len(set(speeds)) != 2:
        raise ValueError("Exactly two distinct steady speeds are required for DL and DQ")
    if len(rates) != 2 or len(set(rates)) != 2:
        raise ValueError("Exactly two distinct rotation-rate amplitudes are required")
    if max(speeds) > translation_limit or max(rates) > rotation_limit:
        raise ValueError("Damping cases exceed the target RL velocity envelope")
    rotation_frequency = _positive(
        damping["rotation_frequency_hz"],
        "damping_identification.rotation_frequency_hz",
    )
    maximum_angle = max(rates) / (2.0 * math.pi * rotation_frequency)
    if math.degrees(maximum_angle) > _positive(
        damping["maximum_rotation_angle_deg"],
        "damping_identification.maximum_rotation_angle_deg",
    ):
        raise ValueError("Rotational damping motion exceeds maximum_rotation_angle_deg")
    for name in (
        "steady_settle_body_lengths", "steady_sample_body_lengths",
        "steady_max_delta_t_s", "ramp_cycles", "settle_cycles_after_ramp",
        "sample_cycles",
    ):
        _positive(damping[name], f"damping_identification.{name}")
    if int(damping["steady_steps_per_body_length"]) < 1:
        raise ValueError("steady_steps_per_body_length must be positive")

    added = cfg["added_mass_identification"]
    frequency = _positive(added["frequency_hz"], "added_mass_identification.frequency_hz")
    translation_peak = _positive(
        added["translation_velocity_amplitude_m_s"],
        "added_mass_identification.translation_velocity_amplitude_m_s",
    )
    rotation_peak = _positive(
        added["rotation_rate_amplitude_rad_s"],
        "added_mass_identification.rotation_rate_amplitude_rad_s",
    )
    if translation_peak >= translation_limit or rotation_peak >= rotation_limit:
        raise ValueError("Added-mass excitation must remain below the RL velocity envelope")
    if translation_peak / (2.0 * math.pi * frequency) > _positive(
        added["max_translation_displacement_m"],
        "added_mass_identification.max_translation_displacement_m",
    ):
        raise ValueError("Added-mass translation exceeds its displacement limit")
    if math.degrees(rotation_peak / (2.0 * math.pi * frequency)) > _positive(
        added["max_rotation_angle_deg"],
        "added_mass_identification.max_rotation_angle_deg",
    ):
        raise ValueError("Added-mass rotation exceeds its angle limit")
    for name in ("ramp_cycles", "settle_cycles_after_ramp", "sample_cycles"):
        _positive(added[name], f"added_mass_identification.{name}")

    for name in (
        "steps_per_cycle", "writes_per_cycle", "purge_write", "gamg_update_interval",
        "pimple_outer_correctors", "force_execute_interval",
    ):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0.0 < float(cfg["initial_delta_t_fraction"]) <= 1.0:
        raise ValueError("initial_delta_t_fraction must lie in (0, 1]")
    if cfg["wall_function_blending"] not in {"stepwise", "exponential"}:
        raise ValueError("wall_function_blending must be stepwise or exponential")

    block = cfg["block_mesh"]
    lower = _vector(block["domain_min"], 3, "block_mesh.domain_min")
    upper = _vector(block["domain_max"], 3, "block_mesh.domain_max")
    if any(first >= second for first, second in zip(lower, upper)):
        raise ValueError("block_mesh bounds must be ordered")
    cells = block["base_cells"]
    if not isinstance(cells, list) or len(cells) != 3 or any(
        type(value) is not int or value < 1 for value in cells
    ):
        raise ValueError("block_mesh.base_cells must contain three positive integers")
    snappy = cfg["snappy"]
    if not isinstance(snappy, dict) or set(snappy) != {
        "max_local_cells", "max_global_cells", "surface_level", "near_body_level"
    }:
        raise ValueError("snappy has unexpected keys")
    if any(type(value) is not int or value < 1 for value in snappy.values()):
        raise ValueError("snappy values must be positive integers")

    rotor = cfg["locked_rotor_mesh"]
    if rotor.get("enabled") is not True:
        raise ValueError("locked_rotor_mesh.enabled must be true")
    for name in ("rotor_level", "shaft_level", "motor_level", "near_field_level"):
        if type(rotor[name]) is not int or rotor[name] < 1:
            raise ValueError(f"locked_rotor_mesh.{name} must be a positive integer")
    for name in (
        "rotor_radius_m", "rotor_axial_length_m", "shaft_radius_m",
        "motor_radius_m", "motor_upstream_overlap_m", "near_field_radius_m",
    ):
        _positive(rotor[name], f"locked_rotor_mesh.{name}")

    analysis = cfg["analysis"]
    if not isinstance(analysis, dict) or set(analysis) != {"phase_samples_per_cycle"}:
        raise ValueError("analysis has unexpected keys")
    phase_samples = analysis["phase_samples_per_cycle"]
    if type(phase_samples) is not int or phase_samples < 8 or phase_samples % 2:
        raise ValueError("analysis.phase_samples_per_cycle must be an even integer >= 8")


def load_config_with_sources(path: Path) -> tuple[dict[str, Any], tuple[Path, ...]]:
    source = path.resolve()
    with source.open(encoding="utf-8") as stream:
        cfg = json.load(stream, object_pairs_hook=_unique_json_object)
    if not isinstance(cfg, dict):
        raise ValueError(f"{source} must contain a JSON object")
    _validate_config(cfg)
    return cfg, (source,)


def load_config(path: Path) -> dict[str, Any]:
    return load_config_with_sources(path)[0]


def number_token(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def _oscillation_spec(
    *, dof: str, family: str, peak: float, frequency: float,
    ramp_cycles: float, settle_cycles: float, sample_cycles: float,
) -> CaseSpec:
    index, kind, axis = DOFS[dof]
    amplitude = peak / (2.0 * math.pi * frequency)
    unit = "mps" if kind == "translation" else "radps"
    label = "vel" if kind == "translation" else "rate"
    common = dict(
        name=f"{family}_{dof}_{label}{number_token(peak, 3)}{unit}_f{number_token(frequency, 3)}hz",
        family=family,
        dof=dof,
        dof_index=index,
        kind=kind,
        axis=axis,
        frequency_hz=frequency,
        ramp_cycles=ramp_cycles,
        settle_cycles_after_ramp=settle_cycles,
        sample_cycles=sample_cycles,
    )
    if kind == "translation":
        return CaseSpec(
            **common, amplitude_m=amplitude, velocity_amplitude_m_s=peak
        )
    return CaseSpec(
        **common,
        amplitude_deg=math.degrees(amplitude),
        amplitude_rad=amplitude,
        rate_amplitude_rad_s=peak,
    )


def campaign_specs(cfg: dict[str, Any]) -> list[CaseSpec]:
    """Return 12 steady, 6 rotational-damping, and 6 added-mass cases."""

    specs: list[CaseSpec] = []
    damping = cfg["damping_identification"]
    for dof in ("u", "v", "w"):
        index, _, axis = DOFS[dof]
        for speed in damping["translation_speeds_m_s"]:
            for sign, label in ((1.0, "pos"), (-1.0, "neg")):
                value = sign * float(speed)
                specs.append(
                    CaseSpec(
                        name=f"steady_damping_{dof}_{label}_{number_token(float(speed), 3)}mps",
                        family="steady_damping",
                        dof=dof,
                        dof_index=index,
                        kind="steady_translation",
                        axis=axis,
                        body_velocity_b_m_s=tuple(value * component for component in axis),
                    )
                )

    for dof in ("p", "q", "r"):
        for rate in damping["rotation_rate_amplitudes_rad_s"]:
            specs.append(
                _oscillation_spec(
                    dof=dof,
                    family="oscillatory_damping",
                    peak=float(rate),
                    frequency=float(damping["rotation_frequency_hz"]),
                    ramp_cycles=float(damping["ramp_cycles"]),
                    settle_cycles=float(damping["settle_cycles_after_ramp"]),
                    sample_cycles=float(damping["sample_cycles"]),
                )
            )

    added = cfg["added_mass_identification"]
    frequency = float(added["frequency_hz"])
    for dof, (_, kind, _) in DOFS.items():
        peak = float(
            added["translation_velocity_amplitude_m_s"]
            if kind == "translation"
            else added["rotation_rate_amplitude_rad_s"]
        )
        specs.append(
            _oscillation_spec(
                dof=dof,
                family="added_mass",
                peak=peak,
                frequency=frequency,
                ramp_cycles=float(added["ramp_cycles"]),
                settle_cycles=float(added["settle_cycles_after_ramp"]),
                sample_cycles=float(added["sample_cycles"]),
            )
        )
    if len(specs) != 24 or len({spec.name for spec in specs}) != 24:
        raise RuntimeError("The schema-5 design must produce exactly 24 unique cases")
    return specs


def stationary_spec(name: str, purpose: str) -> CaseSpec:
    return CaseSpec(
        name, "shared_mesh", None, None, "stationary", (0, 0, 0), purpose=purpose
    )

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from environment.openfoam.analysis.fit import fit_case_data
from environment.openfoam.analysis.forces import ForceSeries, parse_total_vector_file
from environment.openfoam.analysis.identification import _cross_axis_fraction
from environment.openfoam.analysis.motion import (
    CaseData,
    DOF_NAMES,
    MotionSpec,
    SteadyCaseData,
)
from environment.openfoam.case_execution.planning import _command_plan
from environment.openfoam.case_execution.runner import _remove_parallel_partitions
from environment.openfoam.case_generation.config import (
    campaign_specs,
    load_config,
    ramped_sinusoid_peak_factors,
)
from environment.openfoam.case_generation.mesh_renderers import (
    load_locked_rotor_report,
    render_block_mesh_dict,
    render_snappy_hex_mesh_dict,
)
from environment.openfoam.case_generation.motion_renderers import (
    ambient_turbulence_state,
    metadata,
    render_turbulence_field,
    render_velocity_field,
    timeline,
)
from environment.openfoam.publish_results import _matrix


ROOT = Path(__file__).resolve().parents[2]
OPENFOAM = ROOT / "environment/openfoam"
CONFIG = OPENFOAM / "config.json"
REPAIR = OPENFOAM / "geometry/validated_locked_rotor_v1/selection_report.json"


def test_schema5_design_has_exactly_24_cases() -> None:
    config = load_config(CONFIG)
    specs = campaign_specs(config)
    assert config["schema_version"] == 5
    assert config["design"] == "full_response_24_case"
    assert len(specs) == len({item.name for item in specs}) == 24
    assert sum(item.family == "steady_damping" for item in specs) == 12
    assert sum(item.family == "oscillatory_damping" for item in specs) == 6
    assert sum(item.family == "added_mass" for item in specs) == 6
    assert not any("holdout" in item.family for item in specs)
    assert {
        metadata(item, config)["matrix_structure"] for item in specs
    } == {"full_response_port_starboard_reflection_symmetric"}


def test_config_rejects_duplicate_keys(tmp_path: Path) -> None:
    duplicate = CONFIG.read_text(encoding="utf-8").replace(
        '"schema_version": 5,',
        '"schema_version": 5,\n  "schema_version": 5,',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key: schema_version"):
        load_config(path)


def test_case_speeds_and_timelines_cover_the_rl_envelope() -> None:
    config = load_config(CONFIG)
    specs = campaign_specs(config)
    steady = [item for item in specs if item.family == "steady_damping"]
    assert sorted({abs(item.body_velocity_b_m_s[item.dof_index]) for item in steady}) == [0.08, 0.4]
    low = next(item for item in steady if item.name == "steady_damping_u_pos_0p080mps")
    low_time = timeline(low, config)
    damping = config["damping_identification"]
    assert low_time["settle_end_s"] == pytest.approx(
        damping["steady_settle_body_lengths"]
        * config["reference_length_m"]
        / 0.08
    )
    assert low_time["end_time_s"] == pytest.approx(
        (
            damping["steady_settle_body_lengths"]
            + damping["steady_sample_body_lengths"]
        )
        * config["reference_length_m"]
        / 0.08
    )
    added = next(item for item in specs if item.family == "added_mass")
    added_time = timeline(added, config)
    assert added.frequency_hz == pytest.approx(1.0)
    added_identification = config["added_mass_identification"]
    assert added_time["end_time_s"] == pytest.approx(
        (
            added_identification["ramp_cycles"]
            + added_identification["settle_cycles_after_ramp"]
            + added_identification["sample_cycles"]
        )
        / added.frequency_hz
    )
    assert added_time["delta_t_s"] == pytest.approx(
        1 / (added.frequency_hz * config["steps_per_cycle"])
    )
    assert {
        item.frequency_hz
        for item in specs
        if item.family == "oscillatory_damping"
    } == {1.0}


def test_quintic_ramp_factors_are_finite() -> None:
    velocity, acceleration, jerk = ramped_sinusoid_peak_factors(1.0)
    assert velocity >= 1.0
    assert acceleration >= 1.0
    assert jerk >= 1.0
    assert all(math.isfinite(value) for value in (velocity, acceleration, jerk))


def test_no_layer_single_wall_mesh_is_rendered() -> None:
    config = load_config(CONFIG)
    locked_rotors = load_locked_rotor_report(REPAIR)
    block = render_block_mesh_dict(config)
    snappy = render_snappy_hex_mesh_dict(config, locked_rotors)
    assert "(-3 -1.5 -1.5)" in block
    assert "(90 45 45)" in block
    assert "addLayers       false" in snappy
    assert "level (4 4)" in snappy
    assert "levels ((1e15 5))" in snappy
    assert "auvThinFeatures" not in snappy
    assert "nSurfaceLayers" not in snappy
    quality = (OPENFOAM / "case_template/system/meshQualityDict").read_text(
        encoding="utf-8"
    )
    assert "maxInternalSkewness    3.9" in quality


def test_komega_sst_fields_have_only_one_wall() -> None:
    config = load_config(CONFIG)
    state = ambient_turbulence_state(config)
    assert state["k_m2_s2"] == pytest.approx(2.4e-5)
    for name in ("k", "omega", "nut"):
        field = render_turbulence_field(name, config)
        assert "auv" in field
        assert "auvThinFeatures" not in field
        assert "gammaInt" not in field
        assert "ReThetat" not in field


def test_steady_velocity_uses_opposite_water_velocity() -> None:
    config = load_config(CONFIG)
    spec = next(
        item
        for item in campaign_specs(config)
        if item.name == "steady_damping_u_pos_0p400mps"
    )
    field = render_velocity_field(spec, config)
    assert "freestreamValue uniform (-0.4 0 0)" in field


def test_case_command_plan_has_no_per_case_checkmesh(tmp_path: Path) -> None:
    case = tmp_path / "steady"
    case.mkdir()
    (case / "case.json").write_text(
        json.dumps({"schema_version": 5, "case_family": "steady_damping"}),
        encoding="utf-8",
    )
    plan = _command_plan(case, "pimpleFoam", 1, False)
    commands = [command[0] for command, _ in plan]
    assert commands == ["potentialFoam", "pimpleFoam"]


def test_completed_parallel_partitions_are_discarded(tmp_path: Path) -> None:
    (tmp_path / "processor0").mkdir()
    (tmp_path / "processor7").mkdir()
    (tmp_path / "postProcessing").mkdir()
    (tmp_path / "log.pimpleFoam").write_text("evidence", encoding="utf-8")
    _remove_parallel_partitions(tmp_path)
    assert not (tmp_path / "processor0").exists()
    assert not (tmp_path / "processor7").exists()
    assert (tmp_path / "postProcessing").is_dir()
    assert (tmp_path / "log.pimpleFoam").is_file()


def test_v2512_force_parser(tmp_path: Path) -> None:
    path = tmp_path / "force.dat"
    path.write_text(
        "# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z\n"
        "0.1 (1 2 3) (0 0 0) (1 2 3)\n",
        encoding="utf-8",
    )
    time, values = parse_total_vector_file(path)
    np.testing.assert_allclose(time, [0.1])
    np.testing.assert_allclose(values, [[1, 2, 3]])


def _steady_cases(linear: np.ndarray, quadratic: np.ndarray) -> list[SteadyCaseData]:
    result: list[SteadyCaseData] = []
    for dof in range(3):
        for speed in (0.08, 0.4):
            for sign in (1.0, -1.0):
                velocity = sign * speed
                vector = np.zeros(3)
                vector[dof] = velocity
                wrench_vector = (
                    -linear[:, dof] * velocity
                    - quadratic[:, dof] * abs(velocity) * velocity
                )
                wrench = np.repeat(wrench_vector.reshape(1, 6), 7, axis=0)
                time = np.linspace(0.0, 3.0, 7)
                forces = ForceSeries(time, wrench[:, :3], wrench[:, 3:])
                result.append(
                    SteadyCaseData(
                        f"steady-{dof}-{velocity}",
                        f"steady-{dof}-{velocity}",
                        "steady_damping",
                        DOF_NAMES[dof],
                        dof,
                        vector,
                        1.0,
                        3.0,
                        time,
                        wrench,
                        forces,
                    )
                )
    return result


def _oscillatory_case(
    dof: int,
    family: str,
    frequency: float,
    peak: float,
    added_mass: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> CaseData:
    omega = 2.0 * math.pi * frequency
    period = 1.0 / frequency
    axis = np.eye(3)[dof if dof < 3 else dof - 3]
    motion = MotionSpec(
        case_name=f"{family}-{dof}-{peak}",
        case_family=family,
        dof=DOF_NAMES[dof],
        dof_index=dof,
        motion_kind="translation" if dof < 3 else "rotation",
        axis=axis,
        amplitude_si=peak / omega,
        omega_rad_s=omega,
        ramp_duration_s=period,
        settle_cycles=2.0,
        sample_cycles=3.0,
    )
    time = np.linspace(0.0, 5.0 * period, 2001)
    scalar_eta, scalar_nu, scalar_nudot = motion.kinematics(time)
    eta = np.zeros((time.size, 6))
    nu = np.zeros_like(eta)
    nudot = np.zeros_like(eta)
    eta[:, dof] = scalar_eta
    nu[:, dof] = scalar_nu
    nudot[:, dof] = scalar_nudot
    wrench = (
        -scalar_nudot[:, None] * added_mass[:, dof]
        - scalar_nu[:, None] * linear[:, dof]
        - (np.abs(scalar_nu) * scalar_nu)[:, None] * quadratic[:, dof]
    )
    forces = ForceSeries(time, wrench[:, :3], wrench[:, 3:])
    return CaseData(
        motion.case_name, motion, time, eta, nu, nudot, wrench, forces
    )


def test_synthetic_24_case_fit_recovers_full_reflection_symmetric_matrices() -> None:
    config = load_config(CONFIG)
    added = np.diag([8.0, 12.0, 15.0, 0.12, 0.20, 0.18])
    added[0, 2] = added[2, 0] = 0.4
    added[0, 4] = added[4, 0] = -0.08
    added[1, 3] = added[3, 1] = 0.06
    linear = np.diag([15.0, 20.0, 30.0, 0.10, 0.20, 0.15])
    linear[0, 4], linear[4, 0] = -0.7, 0.15
    linear[1, 3], linear[3, 1] = 0.5, -0.02
    quadratic = np.diag([40.0, 50.0, 60.0, 0.05, 0.08, 0.06])
    quadratic[2, 4], quadratic[4, 2] = 0.9, -0.12
    quadratic[1, 5], quadratic[5, 1] = -0.4, 0.03
    oscillatory = [
        _oscillatory_case(
            dof,
            "added_mass",
            0.35,
            0.04 if dof < 3 else 0.08,
            added,
            linear,
            quadratic,
        )
        for dof in range(6)
    ]
    oscillatory.extend(
        _oscillatory_case(dof, "oscillatory_damping", 0.5, peak, added, linear, quadratic)
        for dof in range(3, 6)
        for peak in (0.2, 0.8)
    )
    result = fit_case_data(_steady_cases(linear, quadratic), oscillatory, config)
    # Odd projection interpolates adaptive-style samples onto a fixed phase
    # grid, so this synthetic recovery includes the same small interpolation
    # error as real force histories.
    np.testing.assert_allclose(result.added_mass, added, rtol=3e-5, atol=1e-9)
    np.testing.assert_allclose(result.linear_damping, linear, rtol=3e-5, atol=1e-4)
    np.testing.assert_allclose(result.quadratic_damping, quadratic, rtol=2e-4, atol=1e-4)
    assert np.count_nonzero(result.added_mass - np.diag(np.diag(result.added_mass))) > 0


def test_cross_axis_load_fraction_converts_moment_with_reference_length() -> None:
    loads = np.zeros((4, 6))
    loads[:, 0] = 1.0
    loads[:, 4] = 0.125
    assert _cross_axis_fraction(loads, 0, 0.5) == pytest.approx(0.25)

    loads[:, 0] = 0.25
    loads[:, 4] = 0.5
    assert _cross_axis_fraction(loads, 4, 0.5) == pytest.approx(0.25)


def test_publisher_accepts_full_finite_matrices() -> None:
    report = {"matrices": {"added_mass": np.eye(6).tolist()}}
    np.testing.assert_allclose(_matrix(report, "added_mass"), np.eye(6))
    report["matrices"]["added_mass"][0][1] = 0.1
    assert _matrix(report, "added_mass")[0, 1] == pytest.approx(0.1)
    report["matrices"]["added_mass"][0][0] = -0.1
    assert _matrix(report, "added_mass")[0, 0] == pytest.approx(-0.1)
    report["matrices"]["added_mass"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite 6x6"):
        _matrix(report, "added_mass")

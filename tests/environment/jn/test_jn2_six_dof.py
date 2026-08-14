from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import jn2.six_dof_identification as six_dof


PROJECT_ROOT = Path(__file__).resolve().parents[3]
JN2_ROOT = PROJECT_ROOT / "jn2"
CONFIG_PATH = JN2_ROOT / "six_dof_config.json"

DOF_RESPONSE_VARIABLE = {
    "sway": ("Y", "v"),
    "heave": ("Z", "w"),
    "pitch": ("M", "q"),
    "yaw": ("N", "r"),
}
DOF_MATRIX_INDEX = {"sway": 1, "heave": 2, "pitch": 4, "yaw": 5}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "as_dict"):
        result = value.as_dict()
        if isinstance(result, Mapping):
            return result
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"cannot convert {type(value).__name__} to a mapping")


def _field(value: Any, *names: str) -> Any:
    mapping = _as_mapping(value)
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AssertionError(f"none of {names!r} found in {sorted(mapping)}")


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in _as_mapping(value).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping) or is_dataclass(item) or hasattr(item, "as_dict"):
            result.update(_flatten_mapping(item, path))
        else:
            result[path.lower()] = item
    return result


def _config_inertia(config: Any, axis: str) -> float:
    index = {"x": 0, "y": 1, "z": 2}[axis]
    return float(config.model_inertia_flu[index, index])


def _replace_model_inertia(raw: dict[str, Any], axis: str, replacement: float) -> bool:
    index = {"x": 0, "y": 1, "z": 2}[axis]
    raw["model"]["inertia_at_com_flu_kg_m2"][index][index] = replacement
    return True


def _metadata_parts(metadata: Any) -> tuple[list[str], list[str], np.ndarray]:
    if isinstance(metadata, tuple) and len(metadata) >= 3:
        names, units, divisors = metadata[:3]
    else:
        names = _field(metadata, "names", "terms", "term_names")
        units = _field(metadata, "units", "term_units")
        divisors = _field(metadata, "divisors", "nondimensional_divisors")
    return list(names), list(units), np.asarray(divisors, dtype=float)


def _preflight_parts(result: Any) -> tuple[list[Any], list[Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        included, audit = result
    else:
        included = _field(result, "included", "included_plans", "trials")
        audit = _field(result, "audit", "audit_rows", "rows")
    return list(included), list(audit)


def _audit_is_included(row: Any) -> bool:
    mapping = _as_mapping(row)
    for key in ("included", "include", "usable"):
        if key in mapping:
            return bool(mapping[key])
    status = str(_field(row, "status", "decision")).lower()
    return status in {"included", "include", "usable", "pass", "passed"}


def _row_text(row: Any) -> str:
    return " ".join(str(value) for value in _as_mapping(row).values())


def _term_exponent(dof: str, response: str, term: str) -> float:
    function = getattr(six_dof, "term_length_exponent", None)
    if function is not None:
        parameter_count = len(inspect.signature(function).parameters)
        if parameter_count == 2:
            return float(function(dof, term))
        return float(function(dof, response, term))

    table = getattr(six_dof, "TERM_LENGTH_EXPONENTS")
    for key in ((dof, response, term), (dof, term), term):
        if key in table:
            return float(table[key])
    raise AssertionError(f"no scaling exponent found for {(dof, response, term)!r}")


def _diagonal(result: Any, kind: str) -> list[Any]:
    aliases = {
        "added_mass": ("added_mass", "added_mass_6d", "M_A"),
        "linear": (
            "linear_damping_effective_at_froude_reference_speed",
            "linear_damping",
            "linear_damping_6d",
            "D_L",
        ),
        "quadratic": ("quadratic_damping", "quadratic_damping_6d", "D_Q"),
    }
    mapping = _as_mapping(result)
    if "matrices" in mapping:
        mapping = _as_mapping(mapping["matrices"])
    for name in aliases[kind]:
        if name in mapping:
            value = list(mapping[name])
            if value and isinstance(value[0], (list, tuple)):
                return [value[index][index] for index in range(6)]
            return value
    raise AssertionError(f"no {kind} diagonal found in {sorted(mapping)}")


def _scaled_rows() -> tuple[list[dict[str, Any]], dict[str, tuple[float, float, float]]]:
    rows: list[dict[str, Any]] = []
    expected: dict[str, tuple[float, float, float]] = {}
    for ordinal, (dof, (response, variable)) in enumerate(DOF_RESPONSE_VARIABLE.items(), start=1):
        added_mass = 100.0 + ordinal
        linear_damping = 200.0 + ordinal
        quadratic_damping = 300.0 + ordinal
        expected[dof] = (added_mass, linear_damping, quadratic_damping)
        common = {"dof": dof, "experiment": dof, "response": response}
        values = {
            f"{response}_u{variable}": -10.0 - ordinal,
            f"{response}_{variable}|{variable}|": -quadratic_damping,
            f"{response}_{variable}dot": -added_mass,
            f"{response}_{variable}@Uref": -linear_damping,
        }
        for term, value in values.items():
            rows.append(
                {
                    **common,
                    "term": term,
                    "coefficient": value,
                    "target_coefficient": value,
                }
            )
    return rows, expected


def test_config_records_distinct_model_pitch_and_yaw_inertias() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    expected_iyy = 60709478.52e-9
    expected_izz = 73524135.11e-9

    assert _config_inertia(config, "y") == pytest.approx(expected_iyy)
    assert _config_inertia(config, "z") == pytest.approx(expected_izz)
    assert _config_inertia(config, "y") != pytest.approx(_config_inertia(config, "z"))


def test_configured_huber_constant_controls_robust_fit() -> None:
    design = np.ones((5, 1))
    load = np.asarray([0.0, 0.1, -0.1, 0.2, 100.0])

    tight = six_dof.robust_fit(design, load, huber_k=0.5)
    loose = six_dof.robust_fit(design, load, huber_k=100.0)

    assert tight[0] < loose[0]


def test_config_uses_explicit_flu_roll_without_yaw_rotation() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    assert "user_confirmed" in config.apparatus["vertical_mount_status"]
    assert config.apparatus["sensor_mount"] == "upright_not_rolled_same_apparatus_orientation"
    assert config.apparatus["sensor_mount_status"] == (
        "upright_not_rolled_and_not_yaw_rotated_user_confirmed"
    )
    assert str(config.apparatus["sensor_axes_frame"]).startswith("raw balance frame S")
    actual = np.asarray(config.apparatus["body_to_apparatus_rotation_vertical"])
    expected = six_dof.rotation_x(np.pi / 2.0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)


def test_flu_plus_90_without_yaw_offset_maps_sway_yaw_to_heave_pitch() -> None:
    rotation = six_dof.rotation_x(np.pi / 2.0)
    linear_B, angular_B = six_dof.rotate_twist_H_to_B(
        np.asarray([[0.0, 2.0, 0.0]]),
        np.asarray([[0.0, 0.0, 3.0]]),
        rotation,
    )
    wrench_B = six_dof.rotate_wrench_H_to_B(
        np.asarray([[0.0, 5.0, 0.0, 0.0, 0.0, 7.0]]), rotation
    )

    np.testing.assert_allclose(linear_B, [[0.0, 0.0, -2.0]], atol=1e-12)
    np.testing.assert_allclose(angular_B, [[0.0, 3.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(wrench_B, [[0.0, 0.0, -5.0, 0.0, 7.0, 0.0]], atol=1e-12)
    assert float(linear_B[0] @ wrench_B[0, :3]) == pytest.approx(10.0)
    assert float(angular_B[0] @ wrench_B[0, 3:]) == pytest.approx(21.0)


def test_upright_sensor_keeps_the_same_fy_polarity_before_model_roll() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    # The balance stays in H.  Raw order is [TX, TY, TZ, FX, FY, FZ].
    raw_sensor = np.asarray([[0.0, 0.0, 7.0, 0.0, -5.0, 0.0]])
    horizontal = np.asarray(config.apparatus["sensor_to_H_wrench_matrix_horizontal"])
    vertical = np.asarray(config.apparatus["sensor_to_H_wrench_matrix_vertical"])
    wrench_H_horizontal = raw_sensor @ horizontal.T
    wrench_H = raw_sensor @ vertical.T
    np.testing.assert_allclose(wrench_H_horizontal, [[0.0, 5.0, 0.0, 0.0, 0.0, 7.0]])
    np.testing.assert_allclose(wrench_H, [[0.0, 5.0, 0.0, 0.0, 0.0, 7.0]])

    rotation = np.asarray(config.apparatus["body_to_apparatus_rotation_vertical"])
    wrench_B = six_dof.rotate_wrench_H_to_B(wrench_H, rotation)
    expected_force = wrench_H[:, :3] @ rotation
    np.testing.assert_allclose(wrench_B[:, :3], expected_force)
    # The apparatus-z moment becomes body-y after the 90-degree roll.
    np.testing.assert_allclose(wrench_B[:, 3:], [[0.0, 7.0, 0.0]], atol=1e-12)


def test_config_rejects_sensor_that_is_not_explicitly_fixed_in_H(tmp_path: Path) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["apparatus"]["sensor_mount"] = "rolled_with_vehicle"
    path = tmp_path / "bad_sensor_mount.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unchanged upright"):
        six_dof.load_config(path)


def test_config_rejects_unconfirmed_fixed_sensor_axis_mapping(tmp_path: Path) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["apparatus"]["sensor_to_H_wrench_matrix_horizontal"][0][3] = 1.0
    path = tmp_path / "invented_fx_mapping.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="sensor-to-H horizontal"):
        six_dof.load_config(path)


def test_wrench_translation_and_rigid_body_com_formulas() -> None:
    wrench_origin = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    r_origin_to_com = np.asarray([0.1, -0.2, 0.3])
    translated = six_dof.translate_wrench_origin_to_com(wrench_origin, r_origin_to_com)
    expected_moment = wrench_origin[0, 3:] - np.cross(r_origin_to_com, wrench_origin[0, :3])
    np.testing.assert_allclose(translated[0, 3:], expected_moment)

    velocity_com, derivative_com = six_dof.motion_origin_to_com_kinematics(
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 2.0, 0.0]]),
        np.asarray([[0.0, 0.0, 3.0]]),
        np.asarray([[0.0, 0.0, 2.0]]),
        np.asarray([0.1, -0.2, 0.3]),
    )
    np.testing.assert_allclose(velocity_com, [[1.6, 0.3, 0.0]])
    np.testing.assert_allclose(derivative_com, [[0.4, 2.2, 0.0]])

    rigid = six_dof.rigid_body_wrench_at_com(
        4.0,
        np.diag([0.2, 0.3, 0.4]),
        np.asarray([[0.5, 0.0, 0.0]]),
        np.asarray([[0.0, 2.0, 0.0]]),
        np.asarray([[0.0, 0.0, 3.0]]),
        np.asarray([[0.0, 0.0, 2.0]]),
    )
    np.testing.assert_allclose(rigid[0, :3], [0.0, 14.0, 0.0])
    np.testing.assert_allclose(rigid[0, 3:], [0.0, 0.0, 0.8])


@pytest.mark.parametrize("axis", ["y", "z"])
def test_config_rejects_nonpositive_model_inertia(tmp_path: Path, axis: str) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert _replace_model_inertia(raw, axis, 0.0), f"model I{axis}{axis} is missing from config"
    path = tmp_path / f"invalid_i{axis}{axis}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"(?i)(i{axis}{axis}|inertia).*(positive|finite)"):
        six_dof.load_config(path)


def test_expected_manifest_contains_all_84_unique_trials() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    plans = list(six_dof.expected_trials(JN2_ROOT, config))

    assert len(plans) == 4 * 7 * 3
    dofs = [_field(plan, "dof", "experiment") for plan in plans]
    assert Counter(dofs) == {dof: 21 for dof in DOF_RESPONSE_VARIABLE}

    frequencies_by_dof = {
        dof: sorted(
            float(_field(plan, "nominal_frequency_hz", "frequency_hz", "frequency"))
            for plan in plans
            if _field(plan, "dof", "experiment") == dof
        )
        for dof in DOF_RESPONSE_VARIABLE
    }
    expected_frequencies = [frequency for frequency in np.arange(0.1, 0.71, 0.1) for _ in range(3)]
    for frequencies in frequencies_by_dof.values():
        np.testing.assert_allclose(frequencies, expected_frequencies, rtol=0.0, atol=1.0e-12)

    path_pairs = {
        (
            Path(_field(plan, "gather_path", "motion_path")),
            Path(_field(plan, "sensor_path", "wrench_path")),
        )
        for plan in plans
    }
    assert len(path_pairs) == 84
    assert all(gather.is_file() and sensor.is_file() for gather, sensor in path_pairs)


def test_preflight_excludes_only_the_two_short_vertical_sway_records() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    included, audit = _preflight_parts(six_dof.preflight(JN2_ROOT, config))

    assert len(audit) == 84
    assert len(included) == 82
    excluded = [row for row in audit if not _audit_is_included(row)]
    assert len(excluded) == 2
    excluded_text = [_row_text(row) for row in excluded]
    assert any("vertical_sway2" in text and "sensor_10.csv" in text for text in excluded_text)
    assert any("vertical_sway2" in text and "sensor_14.csv" in text for text in excluded_text)
    assert all(
        any(word in text.lower() for word in ("short", "coverage", "active", "duration"))
        for text in excluded_text
    )


@pytest.mark.parametrize(
    ("dof", "expected_names", "expected_units"),
    [
        ("sway", ["Y_uv", "Y_v|v|", "Y_vdot"], ["kg/m", "kg/m", "kg"]),
        ("heave", ["Z_uw", "Z_w|w|", "Z_wdot"], ["kg/m", "kg/m", "kg"]),
        ("pitch", ["M_uq", "M_q|q|", "M_qdot"], ["kg*m", "kg*m^2", "kg*m^2"]),
        ("yaw", ["N_ur", "N_r|r|", "N_rdot"], ["kg*m", "kg*m^2", "kg*m^2"]),
    ],
)
def test_coefficient_metadata_uses_fossen_names_and_dimensional_units(
    dof: str,
    expected_names: list[str],
    expected_units: list[str],
) -> None:
    config = six_dof.load_config(CONFIG_PATH)
    names, units, divisors = _metadata_parts(six_dof.coefficient_metadata(dof, config))

    assert names == expected_names
    assert units == expected_units
    assert divisors.shape == (3,)
    assert np.isfinite(divisors).all() and np.all(divisors > 0.0)


@pytest.mark.parametrize(
    ("dof", "expected_exponents"),
    [
        ("sway", (2.0, 2.0, 3.0, 2.5)),
        ("heave", (2.0, 2.0, 3.0, 2.5)),
        ("pitch", (4.0, 5.0, 5.0, 4.5)),
        ("yaw", (4.0, 5.0, 5.0, 4.5)),
    ],
)
def test_target_scaling_uses_force_and_moment_length_exponents(
    dof: str,
    expected_exponents: tuple[float, float, float, float],
) -> None:
    response, variable = DOF_RESPONSE_VARIABLE[dof]
    terms = (
        f"{response}_u{variable}",
        f"{response}_{variable}|{variable}|",
        f"{response}_{variable}dot",
        f"{response}_{variable}@Uref",
    )
    actual = tuple(_term_exponent(dof, response, term) for term in terms)
    assert actual == expected_exponents


def test_six_dof_diagonal_uses_null_for_every_unidentified_6d_entry() -> None:
    rows, expected = _scaled_rows()
    result = six_dof.build_six_dof_diagonal(rows)

    diagonals = {
        "added_mass": _diagonal(result, "added_mass"),
        "linear": _diagonal(result, "linear"),
        "quadratic": _diagonal(result, "quadratic"),
    }
    expected_component = {"added_mass": 0, "linear": 1, "quadratic": 2}
    for kind, diagonal in diagonals.items():
        assert len(diagonal) == 6
        for index, value in enumerate(diagonal):
            dof = next((name for name, dof_index in DOF_MATRIX_INDEX.items() if dof_index == index), None)
            if dof is None:
                assert value is None
            else:
                assert float(value) == pytest.approx(expected[dof][expected_component[kind]])

    mask = list(_field(result, "experimentally_identified_mask"))
    assert mask == [False, True, True, False, True, True]


def test_explicit_flu_vertical_entries_are_selected_directly() -> None:
    rows, _ = _scaled_rows()
    result = six_dof.build_six_dof_diagonal(rows)
    assert _field(result, "vertical_mapping") == "explicit_FLU_R_HB_Rx_plus_90_sensor_upright_same_Y_H_minus_FY_no_yaw_rotation"
    assert list(_field(result, "experimentally_identified_mask")) == [False, True, True, False, True, True]
    assert _diagonal(result, "added_mass")[2] is not None
    assert _diagonal(result, "added_mass")[4] is not None


def test_full_6x6_completion_uses_prior_only_for_unexcited_surged_and_roll() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    rows, expected = _scaled_rows()
    result = six_dof.build_six_dof_diagonal(rows, completion=config.full_6d_completion)

    assert _field(result, "complete_diagonal_mask") == [True] * 6
    assert _field(result, "diagonal_source_by_dof") == [
        "external_literature_prior",
        "jn2_experiment",
        "jn2_experiment",
        "external_literature_prior",
        "jn2_experiment",
        "jn2_experiment",
    ]
    for kind in ("added_mass", "linear", "quadratic"):
        matrix = _field(result, {
            "added_mass": "added_mass",
            "linear": "linear_damping_effective_at_froude_reference_speed",
            "quadratic": "quadratic_damping",
        }[kind])
        assert len(matrix) == 6 and all(len(row) == 6 for row in matrix)
        assert all(matrix[row][column] == 0.0 for row in range(6) for column in range(6) if row != column)

    added = _diagonal(result, "added_mass")
    linear = _diagonal(result, "linear")
    quadratic = _diagonal(result, "quadratic")
    assert added[0] == pytest.approx(10.77)
    assert linear[0] == pytest.approx(38.95)
    assert quadratic[0] == pytest.approx(31.01)
    assert added[3] == pytest.approx(0.103)
    assert linear[3] == pytest.approx(0.0)
    assert quadratic[3] == pytest.approx(2.08)
    for dof, index in DOF_MATRIX_INDEX.items():
        assert added[index] == pytest.approx(expected[dof][0])


def test_simultaneous_mount_sign_flip_preserves_diagonal_coefficients() -> None:
    u = np.array([0.35, 0.38, 0.41, 0.44, 0.47, 0.50])
    q = np.array([-0.8, -0.25, 0.15, 0.55, -0.45, 0.7])
    qdot = np.array([0.3, -0.6, 0.8, -0.25, 0.5, -0.7])
    design = np.column_stack((u * q, q * np.abs(q), qdot))
    expected = np.array([-2.25, -1.4, -0.65])
    load = design @ expected

    helper = getattr(six_dof, "mount_sign_invariant_coefficients", None)
    if helper is not None:
        positive = helper(design, load, 1.0)
        negative = helper(design, load, -1.0)
    else:
        # A sign convention change reverses both the diagonal motion and its
        # conjugate load, so it cannot change the fitted diagonal derivative.
        positive = six_dof.robust_fit(design, load)
        negative = six_dof.robust_fit(-design, -load)

    np.testing.assert_allclose(positive, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(negative, expected, rtol=1.0e-12, atol=1.0e-12)


def test_synthetic_fit_reports_one_robust_diagonal_fit() -> None:
    config = six_dof.load_config(CONFIG_PATH)
    truth = np.array([-2.25, -1.4, -0.65])
    noise_rng = np.random.default_rng(25)
    trials: list[Any] = []

    for frequency_index, frequency in enumerate(np.arange(0.2, 1.41, 0.2)):
        for repeat in range(1, 4):
            time = np.linspace(0.0, 4.8, 80)
            phase = 2.0 * np.pi * frequency * time + repeat * 0.37
            u = 0.4 + 0.025 * np.sin(0.31 * time + frequency_index * 0.1)
            q = np.sin(phase) + 0.18 * np.sin(2.0 * phase + 0.2)
            qdot = (
                2.0 * np.pi * frequency * np.cos(phase)
                + 0.36 * 2.0 * np.pi * frequency * np.cos(2.0 * phase + 0.2)
            )
            design = np.column_stack((u * q, q * np.abs(q), qdot))
            target = design @ truth + 1.2 + 0.08 * time + noise_rng.normal(0.0, 0.008, len(time))
            target[(frequency_index * 3 + repeat) * 7 % len(time)] += 2.5
            plan = six_dof.TrialPlan(
                "sway",
                repeat,
                8 + frequency_index,
                float(frequency),
                Path("unused_gather.csv"),
                Path("unused_sensor.csv"),
                "horizontal",
            )
            trials.append(
                six_dof.SixDofTrial(
                    plan,
                    float(frequency),
                    time,
                    design,
                    target,
                    target.copy(),
                    u,
                    q,
                    qdot,
                    {},
                )
            )

    result = six_dof.fit_dof(trials, config)
    np.testing.assert_allclose(result["beta"], truth, rtol=0.0, atol=3.5e-2)
    assert result["full_r2"] > 0.99
    assert "bootstrap" not in result
    assert "ci_low" not in result

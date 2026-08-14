from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import environment.pmm.six_dof_identification as pmm_fit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PMM_ROOT = PROJECT_ROOT / "environment/pmm"


def _trial(dof: str, nominal_frequency_hz: float, repeat: int) -> pmm_fit.SixDofTrial:
    plan = pmm_fit.TrialPlan(
        dof=dof,
        repeat=repeat,
        file_id=int(round(10.0 * nominal_frequency_hz)),
        nominal_frequency_hz=nominal_frequency_hz,
        gather_path=Path("unused_gather.csv"),
        sensor_path=Path("unused_sensor.csv"),
        timing_group=pmm_fit.DOF_TIMING_GROUP[dof],
    )
    time = np.linspace(0.0, 1.0, 8)
    return pmm_fit.SixDofTrial(
        plan=plan,
        frequency_hz=nominal_frequency_hz + repeat * 1.0e-5,
        time=time,
        X=np.ones((len(time), 3)),
        target=np.ones(len(time)),
        measured=np.ones(len(time)),
        u=np.full(len(time), 0.2),
        q=np.ones(len(time)),
        qdot=np.ones(len(time)),
        diagnostics={},
    )


def test_config_keeps_pmm_and_cfd_at_the_same_physical_scale() -> None:
    config = pmm_fit.load_config(PMM_ROOT / "six_dof_config.json")

    assert float(config.model["wet_length_m"]) == pytest.approx(0.562)
    assert float(config.model["geometry_scale_of_real_robot"]) == pytest.approx(0.7)
    assert float(config.raw["result_scale"]["PMM_and_CFD_length_ratio"]) == pytest.approx(1.0)
    assert float(config.raw["result_scale"]["coefficient_scale_factor_applied"]) == pytest.approx(1.0)
    assert config.raw["result_scale"]["full_scale_conversion_performed"] is False


def test_provisional_wet_rigid_body_is_explicit() -> None:
    config = pmm_fit.load_config(PMM_ROOT / "six_dof_config.json")
    wet = config.model["wet_mass_assumption"]

    assert float(config.model["mass_kg"]) == pytest.approx(7.563213625)
    assert wet["material"] == "PLA"
    assert float(wet["CAD_dry_mass_before_density_correction_kg"]) == pytest.approx(6.4163)
    assert float(wet["density_corrected_dry_mass_kg"]) == pytest.approx(7.378745)
    assert float(wet["material_sorption_mass_gain_kg"]) == pytest.approx(0.184468625)
    assert "excluded" in str(wet["free_or_retained_water_policy"])
    assert "pending" in str(wet["status"])


def test_complete_sensor_reaction_wrench_is_negated_before_vertical_roll() -> None:
    config = pmm_fit.load_config(PMM_ROOT / "six_dof_config.json")
    raw = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    sensor_to_h = np.asarray(config.apparatus["sensor_to_H_wrench_matrix_vertical"])
    wrench_h = raw @ sensor_to_h.T

    np.testing.assert_allclose(wrench_h, [[-4.0, -5.0, -6.0, -1.0, -2.0, -3.0]])
    rotation = np.asarray(config.apparatus["body_to_apparatus_rotation_vertical"])
    wrench_b = pmm_fit.rotate_wrench_H_to_B(wrench_h, rotation)
    np.testing.assert_allclose(wrench_b, [[-4.0, -6.0, 5.0, -1.0, -3.0, 2.0]])


def test_fit_by_frequency_never_pools_distinct_nominal_frequencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = pmm_fit.load_config(PMM_ROOT / "six_dof_config.json")
    trials_by_dof = {
        dof: [
            _trial(dof, frequency, repeat)
            for frequency in (0.1, 0.2)
            for repeat in (1, 2, 3)
        ]
        for dof in pmm_fit.DOFS
    }
    observed_groups: list[tuple[str, float, int]] = []

    def fake_fit(
        trials: list[pmm_fit.SixDofTrial],
        unused_config: pmm_fit.SixDofConfig,
    ) -> dict[str, object]:
        frequencies = {round(trial.plan.nominal_frequency_hz, 10) for trial in trials}
        assert len(frequencies) == 1
        observed_groups.append((trials[0].plan.dof, frequencies.pop(), len(trials)))
        values = np.ones((6, 3))
        target = np.ones(6)
        return {
            "beta": np.zeros(3),
            "full_r2": 1.0,
            "fit_domain": "test",
            "harmonics": [1, 2, 3],
            "condition": 1.0,
            "X": values,
            "y": target,
            "prediction": target,
        }

    monkeypatch.setattr(pmm_fit, "fit_dof", fake_fit)
    results = pmm_fit.fit_by_frequency(trials_by_dof, config)

    assert len(results) == len(pmm_fit.DOFS) * 2
    assert len(observed_groups) == len(results)
    assert all(repeats == 3 for _, _, repeats in observed_groups)
    assert {frequency for _, frequency, _ in observed_groups} == {0.1, 0.2}

"""Regression tests for shared evaluation labels and request validation."""

from __future__ import annotations

import pytest

from simulation.isaac.workflows.common.evaluation_cases import (
    build_evaluation_case_label,
    validate_evaluation_parameters,
)


def test_fixed_disturbances_have_stable_non_nominal_labels() -> None:
    assert build_evaluation_case_label(current_w=(0.0, 0.0, 0.0)) == "cur_0_0_0"
    assert build_evaluation_case_label(damping_scale=1.2) == "damp1p2"
    assert build_evaluation_case_label(thruster_scale=0.8) == "thr0p8"


def test_sampled_dr_and_manual_overrides_share_one_composite_label() -> None:
    label = build_evaluation_case_label(
        sample_domain_randomization=True,
        domain_randomization_name="pool recipe",
        seed=7,
        current_w=(0.1, 0.0, -0.2),
        smooth_current=True,
        current_variation_std=0.03,
    )
    assert label == "dr_pool_recipe_seed7_cur_0p1_0_m0p2_smooth0p03"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"duration_s": 0.0},
        {"duration_s": float("nan")},
        {"duration_s": 1.0, "current_w": (0.0, float("nan"), 0.0)},
        {"duration_s": 1.0, "current_variation_std": -1.0},
        {"duration_s": 1.0, "current_tau": 0.0},
        {"duration_s": 1.0, "thruster_scale": -1.0},
        {"duration_s": 1.0, "num_envs": 0},
        {"duration_s": 1.0, "random_curve_count": 0},
    ),
)
def test_invalid_evaluation_parameters_fail_before_launch(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_evaluation_parameters(**kwargs)

"""Versioned domain-randomization recipe regression tests."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment.profiles.domain_randomization import (
    DomainRandomizationSpec,
    apply_domain_randomization_spec,
    complete_domain_randomization_profile,
    domain_randomization_parameters_requiring_sources,
    domain_randomization_spec_from_dict,
    load_domain_randomization_spec_json,
    write_domain_randomization_spec_json,
)
from environment.profiles.pool_profile import (
    NOMINAL_POOL_DYNAMICS_PROFILE,
    DomainRandomizationProfile,
    PoolDynamicsProfile,
)


ROOT = Path(__file__).resolve().parents[3]


def _complete(
    parameters: DomainRandomizationProfile,
    base_profile: PoolDynamicsProfile = NOMINAL_POOL_DYNAMICS_PROFILE,
) -> DomainRandomizationProfile:
    return complete_domain_randomization_profile(parameters, base_profile)


def _test_sources(parameters: DomainRandomizationProfile) -> dict[str, str]:
    return {
        name: "synthetic regression-test source"
        for name in domain_randomization_parameters_requiring_sources(parameters)
    }


def test_checked_in_auv_recipe_is_valid() -> None:
    spec = load_domain_randomization_spec_json(
        ROOT / "environment/profiles/configs/auv_pool_openfoam_hydrodynamics_dr_v1.json"
    )

    assert spec.name == "auv-pool-openfoam-hydrodynamics-dr-v1"
    assert spec.parameters.use_custom_randomization is True
    assert tuple(spec.parameters.enabled_features) == (
        "rigid_body",
        "current",
        "hydrodynamics",
        "actuators",
        "battery",
    )
    assert spec.parameters.water_current_max_by_stage == [0.0, 0.05, 0.1, 0.15, 0.2]
    assert spec.parameters.added_mass_log_std_by_stage == [0.0, 0.0, 0.05, 0.08, 0.12]


def test_spec_round_trip_preserves_recipe(tmp_path: Path) -> None:
    base_profile = PoolDynamicsProfile(name="measured-pool")
    parameters = _complete(
        DomainRandomizationProfile(
            use_custom_randomization=True,
            mass_range=[10.0, 10.2],
            water_current_max_by_stage=[0.0, 0.1],
            water_current_vertical_max_by_stage=[0.0, 0.02],
            water_current_variation_std_by_stage=[0.0, 0.01],
            damping_scale_by_stage=[0.0, 0.1],
            added_mass_log_std_by_stage=[0.0, 0.08],
            thruster_scale_by_stage=[0.0, 0.1],
            thruster_tau_scale_by_stage=[0.0, 0.2],
            disturbance_curriculum=True,
            disturbance_curriculum_stage_steps=[100],
        ),
        base_profile,
    )
    spec = DomainRandomizationSpec(
        name="round-trip",
        description="test",
        base_profile_name="measured-pool",
        parameters=parameters,
        parameter_sources=_test_sources(parameters),
        metadata={"campaign": "synthetic"},
    )
    path = tmp_path / "recipe.json"

    write_domain_randomization_spec_json(spec, path)
    loaded = load_domain_randomization_spec_json(path)

    assert loaded == spec


def test_apply_spec_records_identity_and_updates_nested_cfg() -> None:
    parameters = _complete(
        DomainRandomizationProfile(
            use_custom_randomization=True,
            mass_range=[9.9, 10.3],
        )
    )
    spec = DomainRandomizationSpec(
        name="apply-test",
        parameters=parameters,
        parameter_sources=_test_sources(parameters),
    )
    cfg = SimpleNamespace(domain_randomization=SimpleNamespace())

    returned = apply_domain_randomization_spec(cfg, spec)

    assert returned is cfg
    assert cfg.domain_randomization.mass_range == [9.9, 10.3]
    assert cfg.domain_randomization_spec_name == "apply-test"


def test_apply_spec_rejects_wrong_measured_profile() -> None:
    measured = PoolDynamicsProfile(name="measured-a")
    expected = PoolDynamicsProfile(name="measured-b")
    spec = DomainRandomizationSpec(
        name="bound-recipe",
        base_profile_name="measured-b",
        parameters=_complete(DomainRandomizationProfile(), expected),
    )

    with pytest.raises(ValueError, match="expects base profile"):
        apply_domain_randomization_spec(
            SimpleNamespace(domain_randomization=SimpleNamespace()),
            spec,
            base_profile=measured,
        )


def test_spec_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown domain-randomization parameter"):
        domain_randomization_spec_from_dict(
            {
                "schema_version": 1,
                "name": "bad",
                "parameters": {"mass_rng": [1.0, 2.0]},
            }
        )


def test_spec_rejects_coerced_schema_types() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        domain_randomization_spec_from_dict(
            {
                "schema_version": 1,
                "name": 123,
                "parameters": _complete(DomainRandomizationProfile()).to_cfg_updates(),
            }
        )


def test_spec_rejects_mismatched_curriculum_arrays() -> None:
    with pytest.raises(ValueError, match="disturbance by-stage arrays must have matching lengths"):
        DomainRandomizationSpec(
            name="bad-stages",
            parameters=DomainRandomizationProfile(
                water_current_max_by_stage=[0.0, 0.1],
                damping_scale_by_stage=[0.0, 0.1, 0.2],
            ),
        ).validate()


def test_spec_rejects_partial_parameter_overlay() -> None:
    with pytest.raises(ValueError, match="parameters must be complete"):
        DomainRandomizationSpec(
            name="partial",
            parameters=DomainRandomizationProfile(mass_range=[10.0, 10.1]),
        ).validate()


def test_spec_rejects_missing_source_for_active_range() -> None:
    parameters = _complete(
        DomainRandomizationProfile(
            use_custom_randomization=True,
            mass_range=[10.0, 10.1],
        )
    )
    with pytest.raises(ValueError, match="require parameter_sources: mass_range"):
        DomainRandomizationSpec(name="missing-source", parameters=parameters).validate()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (DomainRandomizationProfile(mass_range=[-1.0, 1.0]), "mass_range must be positive"),
        (
            DomainRandomizationProfile(thruster_command_dropout_probability_range=[0.0, 1.1]),
            "thruster_command_dropout_probability_range must be in",
        ),
        (
            DomainRandomizationProfile(thruster_command_delay_steps_range=[-1, 0]),
            "thruster_command_delay_steps_range must be non-negative",
        ),
        (DomainRandomizationProfile(volume_range=[1.0, float("nan")]), "must be finite"),
        (
            DomainRandomizationProfile(use_custom_randomization="true"),
            "use_custom_randomization must be boolean",
        ),
        (
            DomainRandomizationProfile(thruster_scale_by_stage=[1.1]),
            "thruster_scale_by_stage must not exceed 1.0",
        ),
    ],
)
def test_randomization_profile_rejects_unphysical_ranges(
    parameters: DomainRandomizationProfile,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parameters.validate()


def test_payload_ensemble_validates_correlated_physical_samples() -> None:
    parameters = DomainRandomizationProfile(
        payload_samples=[
            {
                "name": "camera-rig",
                "weight": 2.0,
                "mass": 11.8,
                "volume": 0.0117,
                "inertia": [0.22, 0.31, 0.34],
                "center_of_mass_offset": [0.01, 0.0, -0.02],
                "com_to_cob_offset": [0.0, 0.0, 0.015],
                "linear_damping_scale": [1.0, 1.0, 1.05, 1.0, 1.0, 1.0],
                "quadratic_damping_scale": 1.1,
                "added_mass_scale": 1.02,
            }
        ]
    )

    parameters.validate()

    with pytest.raises(ValueError, match=r"payload_samples\[0\].mass must be positive"):
        DomainRandomizationProfile(
            payload_samples=[
                {
                    "mass": 0.0,
                    "volume": 0.0117,
                    "inertia": [0.22, 0.31, 0.34],
                    "center_of_mass_offset": [0.0, 0.0, 0.0],
                    "com_to_cob_offset": [0.0, 0.0, 0.0],
                }
            ]
        ).validate()

    with pytest.raises(ValueError, match="inertia triangle inequalities"):
        DomainRandomizationProfile(
            payload_samples=[
                {
                    "mass": 11.8,
                    "volume": 0.0117,
                    "inertia": [1.0, 0.1, 0.1],
                    "center_of_mass_offset": [0.0, 0.0, 0.0],
                    "com_to_cob_offset": [0.0, 0.0, 0.0],
                }
            ]
        ).validate()


def test_environment_does_not_override_configured_torch_seed() -> None:
    source = (ROOT / "simulation/isaac/env.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    manual_seed_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "manual_seed"
    ]

    assert manual_seed_calls == []


def test_spec_json_does_not_contain_non_finite_values() -> None:
    parameters = _complete(DomainRandomizationProfile())
    with pytest.raises(ValueError, match="metadata must contain finite JSON-compatible values"):
        DomainRandomizationSpec(
            name="non-finite",
            parameters=parameters,
            metadata={"invalid": float("nan")},
        ).validate()

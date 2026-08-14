"""Unit tests for explicit runtime DR feature selection."""

from types import SimpleNamespace

import pytest

from environment.profiles.features import (
    DOMAIN_RANDOMIZATION_FEATURES,
    domain_randomization_feature_enabled,
    normalize_domain_randomization_features,
)
from environment.profiles.pool_profile import DomainRandomizationProfile


class _Env:
    def __init__(self, features, *, enabled: bool = True):
        self.cfg = SimpleNamespace(domain_randomization=SimpleNamespace(enabled_features=features))
        self._enabled = enabled

    def _domain_randomization_enabled(self) -> bool:
        return self._enabled


def test_legacy_missing_feature_list_enables_all_groups() -> None:
    assert normalize_domain_randomization_features(None) == DOMAIN_RANDOMIZATION_FEATURES
    env = _Env(None)
    assert all(domain_randomization_feature_enabled(env, name) for name in DOMAIN_RANDOMIZATION_FEATURES)


def test_explicit_feature_subset_is_independent() -> None:
    env = _Env(["actuators", "battery"])
    assert domain_randomization_feature_enabled(env, "actuators")
    assert domain_randomization_feature_enabled(env, "battery")
    assert not domain_randomization_feature_enabled(env, "current")
    assert not domain_randomization_feature_enabled(_Env(["actuators"], enabled=False), "actuators")


def test_unknown_or_duplicate_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown domain-randomization feature"):
        normalize_domain_randomization_features(["not-a-feature"])
    with pytest.raises(ValueError, match="Duplicate domain-randomization feature"):
        DomainRandomizationProfile(enabled_features=["battery", "battery"]).validate()

"""Names and selection rules for independently managed DR feature groups."""

from __future__ import annotations

from collections.abc import Iterable


DOMAIN_RANDOMIZATION_FEATURES = (
    "rigid_body",
    "current",
    "hydrodynamics",
    "actuators",
    "battery",
)
ALL_DOMAIN_RANDOMIZATION_FEATURES = frozenset(DOMAIN_RANDOMIZATION_FEATURES)


def normalize_domain_randomization_features(features: Iterable[str]) -> tuple[str, ...]:
    """Validate and canonicalize explicitly selected feature names.

    An empty collection is valid and deliberately means that the run samples
    no feature group.
    """

    if features is None:
        raise ValueError("domain_randomization.enabled_features must be explicitly configured.")
    if isinstance(features, str):
        raise ValueError(
            "domain_randomization.enabled_features must be a sequence of feature names, not a string."
        )
    try:
        names = tuple(str(name) for name in features)
    except TypeError as exc:
        raise ValueError("domain_randomization.enabled_features must be a sequence.") from exc
    unknown = sorted(set(names) - ALL_DOMAIN_RANDOMIZATION_FEATURES)
    if unknown:
        raise ValueError(
            "Unknown domain-randomization feature(s): " + ", ".join(unknown) + ". "
            "Supported features: " + ", ".join(DOMAIN_RANDOMIZATION_FEATURES) + "."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate domain-randomization feature(s): " + ", ".join(duplicates) + ".")
    return names


def domain_randomization_feature_enabled(env, name: str) -> bool:
    """Return whether one feature is active for this reset/step.

    The environment's global train/eval gate remains authoritative.
    """

    if name not in ALL_DOMAIN_RANDOMIZATION_FEATURES:
        raise ValueError(f"Unknown domain-randomization feature {name!r}.")
    if not env._domain_randomization_enabled():
        return False
    selected = env.cfg.domain_randomization.enabled_features
    return name in normalize_domain_randomization_features(selected)

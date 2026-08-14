"""Composable runtime domain-randomization features.

The recipe format lives in :mod:`environment.profiles.domain_randomization`; this
package owns the independently selectable runtime feature groups used by the
environment.  Keeping the two layers separate makes a recipe an auditable
parameter snapshot while a training run can explicitly select a subset of its
uncertainties.
"""

from environment.profiles.features import (
    ALL_DOMAIN_RANDOMIZATION_FEATURES,
    DOMAIN_RANDOMIZATION_FEATURES,
    domain_randomization_feature_enabled,
    normalize_domain_randomization_features,
)

__all__ = [
    "ALL_DOMAIN_RANDOMIZATION_FEATURES",
    "DOMAIN_RANDOMIZATION_FEATURES",
    "domain_randomization_feature_enabled",
    "normalize_domain_randomization_features",
]

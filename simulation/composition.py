"""Resolve environment and robot sources for a simulator runtime."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation.domain_randomization import (
    DomainRandomizationProfile,
    DomainRandomizationSpec,
    resolve_domain_randomization_spec,
)
from environment.profile import (
    EnvironmentProfile,
    resolve_environment_profile,
)
from robot.runtime import RobotRuntimeProfile, T60_RUNTIME


@dataclass(frozen=True)
class RuntimeComposition:
    """Resolved environment and robot sources for one simulator runtime."""

    environment: EnvironmentProfile
    robot: RobotRuntimeProfile = T60_RUNTIME
    randomization: DomainRandomizationSpec | None = None

    def apply(self, cfg: Any) -> Any:
        self.robot.validate()
        physics_dt_s = float(cfg.sim.dt)
        for key, value in self.environment.to_cfg_updates().items():
            setattr(cfg, key, copy.deepcopy(value))
        for key, value in self.robot.to_runtime_cfg_updates(physics_dt_s).items():
            setattr(cfg, key, copy.deepcopy(value))

        profile = (
            DomainRandomizationProfile(
                use_custom_randomization=False,
                enabled_features=(),
            )
            if self.randomization is None
            else self.randomization.parameters
        )
        for key, value in profile.model_dump(mode="python").items():
            setattr(cfg.domain_randomization, key, copy.deepcopy(value))
        if self.randomization is None:
            cfg.domain_randomization_spec_name = None
        else:
            cfg.domain_randomization_spec_name = self.randomization.name
        cfg.environment_profile_name = self.environment.name
        return cfg


def resolve_runtime_composition(
    environment_profile: EnvironmentProfile | str | Path,
    domain_randomization_spec: DomainRandomizationSpec | str | Path | None = None,
    *,
    robot: RobotRuntimeProfile = T60_RUNTIME,
) -> RuntimeComposition:
    """Resolve, validate, and return one explicit runtime composition."""

    environment = resolve_environment_profile(environment_profile)
    randomization = (
        None
        if domain_randomization_spec in (None, "")
        else resolve_domain_randomization_spec(domain_randomization_spec)
    )
    return RuntimeComposition(environment=environment, robot=robot, randomization=randomization)

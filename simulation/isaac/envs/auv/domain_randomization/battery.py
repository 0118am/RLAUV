"""Battery initial-voltage and voltage-sag DR feature."""

from __future__ import annotations

from environment.profiles.features import domain_randomization_feature_enabled
from environment.profiles.random_sampling import sample_bounded_normal


def reset_battery(env, env_ids, stage: int) -> None:
    """Restore deterministic battery state and sample it when selected."""

    del stage  # Battery is currently not curriculum staged; preserve a uniform feature API.
    env.battery_initial_voltage[env_ids] = env.cfg.battery_voltage
    env.battery_voltage[env_ids] = env.cfg.battery_voltage
    env.battery_voltage_drop_per_s[env_ids] = env.cfg.battery_voltage_drop_per_s
    if not domain_randomization_feature_enabled(env, "battery"):
        return
    voltage_min, voltage_max = env.cfg.domain_randomization.battery_voltage_range
    drop_min, drop_max = env.cfg.domain_randomization.battery_voltage_drop_per_s_range
    sampled = sample_bounded_normal(voltage_min, voltage_max, (len(env_ids), 1), env.device)
    env.battery_initial_voltage[env_ids] = sampled
    env.battery_voltage[env_ids] = sampled
    env.battery_voltage_drop_per_s[env_ids] = sample_bounded_normal(
        drop_min, drop_max, (len(env_ids), 1), env.device
    )

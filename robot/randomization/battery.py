"""Battery initial-voltage and voltage-sag DR feature."""

from __future__ import annotations

from environment.profiles.random_sampling import sample_bounded_normal


def reset_battery(state, cfg, env_ids, stage: int, *, enabled: bool) -> None:
    """Restore deterministic battery state and sample it when selected."""

    del stage  # Battery is currently not curriculum staged; preserve a uniform feature API.
    state.battery_initial_voltage[env_ids] = cfg.battery_voltage
    state.battery_voltage[env_ids] = cfg.battery_voltage
    state.battery_voltage_drop_per_s[env_ids] = cfg.battery_voltage_drop_per_s
    if not enabled:
        return
    voltage_min, voltage_max = cfg.domain_randomization.battery_voltage_range
    drop_min, drop_max = cfg.domain_randomization.battery_voltage_drop_per_s_range
    sampled = sample_bounded_normal(voltage_min, voltage_max, (len(env_ids), 1), state.device)
    state.battery_initial_voltage[env_ids] = sampled
    state.battery_voltage[env_ids] = sampled
    state.battery_voltage_drop_per_s[env_ids] = sample_bounded_normal(
        drop_min, drop_max, (len(env_ids), 1), state.device
    )

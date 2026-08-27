"""Kinematic limits, phase retiming, and time-domain reference sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import torch
from pydantic import Field

from common.schema import PositiveFloat, StrictFrozenModel

from .catalog import (
    AXIS_SINE,
    CHIRP,
    LATERAL_WAVE,
    LISSAJOUS,
    RACETRACK,
    REVERSE_SPATIAL_HELIX,
    SPATIAL_HELIX,
    VERTICAL_WAVE,
)
from .geometry import (
    _geometry_derivatives,
    _geometry_derivatives_through_third,
    _yaw_rate_per_phase,
)


_INTERPOLATION_PROBE_SUBSTEPS = 4

class TrajectoryKinematicLimits(StrictFrozenModel):
    """Explicit kinematic envelope for one reference-generator configuration."""

    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_yaw_rate_radps: PositiveFloat
    max_jerk_mps3: PositiveFloat
    retime_samples: Annotated[int, Field(ge=32)]


@dataclass
class RetimedTrajectoryTables:
    """Per-environment phase tables used to invert elapsed time to phase."""

    phase: torch.Tensor
    elapsed_s: torch.Tensor
    phase_rate: torch.Tensor
    phase_acceleration: torch.Tensor
    requested_period_s: torch.Tensor
    effective_period_s: torch.Tensor
    retimed: torch.Tensor


def smooth_startup_time(
    elapsed_s: torch.Tensor,
    duration_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map episode time to trajectory time with a C2 speed ramp.

    Returns trajectory time, ``d(trajectory_time)/dt``, and its time
    derivative. The quintic smootherstep rate starts at zero and reaches one
    with zero endpoint acceleration. After the ramp, trajectory time advances
    at ordinary wall-clock speed without a phase discontinuity.
    """
    duration = float(duration_s)
    if duration < 0.0:
        raise ValueError("trajectory startup duration must be non-negative.")
    if duration == 0.0:
        return elapsed_s, torch.ones_like(elapsed_s), torch.zeros_like(elapsed_s)

    nonnegative_time = elapsed_s.clamp_min(0.0)
    u = (nonnegative_time / duration).clamp(0.0, 1.0)
    u2 = u.square()
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u

    speed_scale = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    speed_scale_rate = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) / duration
    ramp_time = duration * (2.5 * u4 - 3.0 * u5 + u6)
    # The ramp integrates to duration / 2. Continuing with t-duration/2
    # preserves both trajectory time and unit speed at the join.
    trajectory_time = torch.where(
        nonnegative_time < duration,
        ramp_time,
        nonnegative_time - 0.5 * duration,
    )
    return trajectory_time, speed_scale, speed_scale_rate

def _closed_time_derivatives(position: torch.Tensor, elapsed_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Numerically estimate periodic velocity, acceleration, and jerk."""

    period = elapsed_s[:, -1:]
    elapsed_nodes = elapsed_s[:, :-1]
    previous_time = torch.cat((elapsed_nodes[:, -1:] - period, elapsed_nodes[:, :-1]), dim=1)
    next_time = torch.cat((elapsed_nodes[:, 1:], elapsed_nodes[:, :1] + period), dim=1)
    denominator = (next_time - previous_time).clamp_min(1.0e-6).unsqueeze(-1)
    velocity = (torch.roll(position, -1, 1) - torch.roll(position, 1, 1)) / denominator
    acceleration = (torch.roll(velocity, -1, 1) - torch.roll(velocity, 1, 1)) / denominator
    jerk = (torch.roll(acceleration, -1, 1) - torch.roll(acceleration, 1, 1)) / denominator
    return velocity, acceleration, jerk


def _closed_scalar_time_derivative(values: torch.Tensor, elapsed_s: torch.Tensor) -> torch.Tensor:
    """Return a central periodic derivative for one scalar field per phase."""

    period = elapsed_s[:, -1:]
    nodes = elapsed_s[:, :-1]
    previous_time = torch.cat((nodes[:, -1:] - period, nodes[:, :-1]), dim=1)
    next_time = torch.cat((nodes[:, 1:], nodes[:, :1] + period), dim=1)
    return (torch.roll(values, -1, 1) - torch.roll(values, 1, 1)) / (next_time - previous_time).clamp_min(1.0e-6)


def _validated_speed_targets(
    trajectory_type: torch.Tensor,
    requested_period_s: torch.Tensor,
    target_speed_mps: torch.Tensor | None,
    limits: TrajectoryKinematicLimits,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.any(requested_period_s <= 0.0):
        raise ValueError("requested_period_s must be positive.")
    controlled = (
        (trajectory_type == AXIS_SINE)
        | (trajectory_type == LISSAJOUS)
        | (trajectory_type == LATERAL_WAVE)
        | (trajectory_type == VERTICAL_WAVE)
        | (trajectory_type == SPATIAL_HELIX)
        | (trajectory_type == REVERSE_SPATIAL_HELIX)
    )
    if target_speed_mps is None:
        if bool(torch.any(controlled)):
            raise ValueError("speed-controlled trajectories require target_speed_mps.")
        target_speed_mps = torch.zeros_like(requested_period_s)
    if target_speed_mps.shape != requested_period_s.shape:
        raise ValueError("target_speed_mps must match requested_period_s shape.")
    if bool(torch.any(target_speed_mps[controlled] <= 0.0)):
        raise ValueError("speed-controlled trajectory targets must be positive.")
    if bool(torch.any(target_speed_mps[controlled] > limits.max_speed_mps)):
        raise ValueError("target_speed_mps exceeds the configured kinematic speed limit.")
    return controlled, target_speed_mps


def _phase_geometry(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    *,
    radius_min: float,
    radius_max: float,
    harmonic_ratio: float,
    samples: int,
    device,
    dtype,
) -> dict[str, torch.Tensor | float]:
    phase_step = 2.0 * torch.pi / samples
    phase_1d = torch.arange(samples, device=device, dtype=dtype) * phase_step
    phase = phase_1d.unsqueeze(0).expand(trajectory_type.numel(), -1)
    position, first, second = _geometry_derivatives(
        trajectory_type,
        axis,
        wave_count,
        amp_x,
        amp_y,
        amp_z,
        phase_x,
        phase_y,
        phase,
        radius_min=radius_min,
        radius_max=radius_max,
        harmonic_ratio=harmonic_ratio,
    )
    phase_speed = torch.linalg.vector_norm(first, dim=-1).clamp_min(1.0e-6)
    curvature = (
        torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1)
        / phase_speed.pow(3)
    )
    yaw_rate_per_phase = _yaw_rate_per_phase(first, second)
    stopped_sines = (trajectory_type == AXIS_SINE).unsqueeze(-1)
    return {
        "trajectory_type": trajectory_type,
        "phase_step": phase_step,
        "phase": phase,
        "position": position,
        "first": first,
        "second": second,
        "phase_speed": phase_speed,
        "curvature": curvature,
        "yaw_rate_per_phase": torch.where(
            stopped_sines,
            torch.zeros_like(yaw_rate_per_phase),
            yaw_rate_per_phase,
        ),
    }


def _phase_interval_duration(rate: torch.Tensor, phase_step: float) -> torch.Tensor:
    inverse_rate = rate.clamp_min(1.0e-6).reciprocal()
    inverse_rate_midpoint = (
        -torch.roll(inverse_rate, 1, dims=1)
        + 9.0 * inverse_rate
        + 9.0 * torch.roll(inverse_rate, -1, dims=1)
        - torch.roll(inverse_rate, -2, dims=1)
    ) / 16.0
    return (phase_step / 6.0) * (
        inverse_rate
        + 4.0 * inverse_rate_midpoint
        + torch.roll(inverse_rate, -1, dims=1)
    )


def _elapsed_for_rate(rate: torch.Tensor, phase_step: float) -> tuple[torch.Tensor, torch.Tensor]:
    dt = _phase_interval_duration(rate, phase_step)
    elapsed = torch.cat(
        (
            torch.zeros(rate.shape[0], 1, dtype=rate.dtype, device=rate.device),
            torch.cumsum(dt, dim=1),
        ),
        dim=1,
    )
    return dt, elapsed


def _phase_acceleration_for_rate(
    rate: torch.Tensor,
    elapsed: torch.Tensor,
    trajectory_type: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    phase_speed: torch.Tensor,
) -> torch.Tensor:
    finite_difference = _closed_scalar_time_derivative(rate, elapsed)
    tangent_growth = torch.sum(first * second, dim=-1)
    arc_length_acceleration = -rate.square() * tangent_growth / phase_speed.square()
    return torch.where(
        (
            (trajectory_type == LISSAJOUS)
            | (trajectory_type == LATERAL_WAVE)
            | (trajectory_type == VERTICAL_WAVE)
            | (trajectory_type == SPATIAL_HELIX)
            | (trajectory_type == REVERSE_SPATIAL_HELIX)
        ).unsqueeze(-1),
        arc_length_acceleration,
        finite_difference,
    )


def _candidate_phase_rate(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    requested_period_s: torch.Tensor,
    target_speed_mps: torch.Tensor,
    speed_controlled: torch.Tensor,
    geometry: dict,
    chirp_rate: float,
    limits: TrajectoryKinematicLimits,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = geometry["phase"]
    phase_speed = geometry["phase_speed"]
    curvature = geometry["curvature"]
    yaw_rate_per_phase = geometry["yaw_rate_per_phase"]
    phase_step = geometry["phase_step"]
    period_rate = (2.0 * torch.pi / requested_period_s).unsqueeze(-1).expand_as(phase)
    amplitudes = torch.stack((amp_x, amp_y, amp_z), dim=-1)
    axis_amplitude = torch.gather(amplitudes, 1, axis.unsqueeze(-1)).squeeze(-1)
    axis_rate = (target_speed_mps / axis_amplitude.clamp_min(1.0e-6)).unsqueeze(-1).expand_as(phase)
    arc_length_rate = target_speed_mps.unsqueeze(-1) / phase_speed
    rate = torch.where((trajectory_type == AXIS_SINE).unsqueeze(-1), axis_rate, period_rate)
    rate = torch.where((trajectory_type == LISSAJOUS).unsqueeze(-1), arc_length_rate, rate)
    rate = torch.where((trajectory_type == LATERAL_WAVE).unsqueeze(-1), arc_length_rate, rate)
    rate = torch.where((trajectory_type == VERTICAL_WAVE).unsqueeze(-1), arc_length_rate, rate)
    rate = torch.where((trajectory_type == SPATIAL_HELIX).unsqueeze(-1), arc_length_rate, rate)
    rate = torch.where(
        (trajectory_type == REVERSE_SPATIAL_HELIX).unsqueeze(-1),
        arc_length_rate,
        rate,
    )
    speed_period = torch.sum(_phase_interval_duration(rate, phase_step), dim=1)
    table_period = torch.where(speed_controlled, speed_period, requested_period_s)
    axis_acceleration_cap = torch.sqrt(
        torch.full_like(axis_amplitude, limits.max_acceleration_mps2)
        / axis_amplitude.clamp_min(1.0e-6)
    ).unsqueeze(-1)
    axis_jerk_cap = torch.pow(
        torch.full_like(axis_amplitude, limits.max_jerk_mps3)
        / axis_amplitude.clamp_min(1.0e-6),
        1.0 / 3.0,
    ).unsqueeze(-1)
    axis_limited_rate = torch.minimum(rate, torch.minimum(axis_acceleration_cap, axis_jerk_cap))
    rate = torch.where(
        (trajectory_type == AXIS_SINE).unsqueeze(-1),
        axis_limited_rate,
        rate,
    )
    chirp_shape = 1.0 + 0.5 * max(0.0, float(chirp_rate) - 1.0) * (1.0 - torch.cos(phase))
    rate = torch.where((trajectory_type == CHIRP).unsqueeze(-1), rate * chirp_shape, rate)

    speed_cap = torch.full_like(rate, limits.max_speed_mps) / phase_speed
    normal_cap = torch.sqrt(
        torch.full_like(rate, limits.max_acceleration_mps2)
        / (curvature * phase_speed.square()).clamp_min(1.0e-8)
    )
    yaw_cap = torch.full_like(rate, limits.max_yaw_rate_radps) / yaw_rate_per_phase.clamp_min(1.0e-6)
    rate = torch.minimum(torch.minimum(rate, speed_cap), torch.minimum(normal_cap, yaw_cap))
    smoothed = (
        torch.roll(rate, 2, 1)
        + 4.0 * torch.roll(rate, 1, 1)
        + 6.0 * rate
        + 4.0 * torch.roll(rate, -1, 1)
        + torch.roll(rate, -2, 1)
    ) / 16.0
    rate = torch.where(speed_controlled.unsqueeze(-1), rate, smoothed)
    rate = torch.minimum(torch.minimum(rate, speed_cap), torch.minimum(normal_cap, yaw_cap))
    uniform = rate.amin(dim=1, keepdim=True).expand_as(rate)
    arc_length_controlled = (
        (trajectory_type == LISSAJOUS)
        | (trajectory_type == LATERAL_WAVE)
        | (trajectory_type == VERTICAL_WAVE)
        | (trajectory_type == SPATIAL_HELIX)
        | (trajectory_type == REVERSE_SPATIAL_HELIX)
    ).unsqueeze(-1)
    rate = torch.where(arc_length_controlled, rate, uniform)
    chirp_progress = 0.5 + 0.25 * (1.0 - torch.cos(phase))
    rate = torch.where((trajectory_type == CHIRP).unsqueeze(-1), rate * chirp_progress, rate)
    sharp_ood = (trajectory_type == CHIRP).unsqueeze(-1)
    rate = torch.where(sharp_ood, 0.60 * rate, rate)
    rate = torch.where((trajectory_type == RACETRACK).unsqueeze(-1), 0.95 * rate, rate)
    return rate, table_period


def _kinematic_scale(
    max_speed: torch.Tensor,
    max_acceleration: torch.Tensor,
    max_jerk: torch.Tensor,
    limits: TrajectoryKinematicLimits,
    max_yaw_rate: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = torch.ones_like(max_speed)
    scale = torch.minimum(scale, torch.full_like(scale, limits.max_speed_mps) / max_speed.clamp_min(1.0e-6))
    scale = torch.minimum(
        scale,
        torch.sqrt(torch.full_like(scale, limits.max_acceleration_mps2) / max_acceleration.clamp_min(1.0e-6)),
    )
    scale = torch.minimum(
        scale,
        torch.pow(torch.full_like(scale, limits.max_jerk_mps3) / max_jerk.clamp_min(1.0e-6), 1.0 / 3.0),
    )
    if max_yaw_rate is not None:
        scale = torch.minimum(
            scale,
            torch.full_like(scale, limits.max_yaw_rate_radps) / max_yaw_rate.clamp_min(1.0e-6),
        )
    return torch.clamp(scale, max=1.0)


def _limit_discrete_rate(
    rate: torch.Tensor,
    geometry: dict,
    limits: TrajectoryKinematicLimits,
) -> torch.Tensor:
    for _ in range(3):
        _, elapsed = _elapsed_for_rate(rate, geometry["phase_step"])
        velocity, acceleration, jerk = _closed_time_derivatives(geometry["position"], elapsed)
        scale = _kinematic_scale(
            torch.linalg.vector_norm(velocity, dim=-1).amax(dim=1),
            torch.linalg.vector_norm(acceleration, dim=-1).amax(dim=1),
            torch.linalg.vector_norm(jerk, dim=-1).amax(dim=1),
            limits,
            (geometry["yaw_rate_per_phase"] * rate).amax(dim=1),
        )
        scaled_rate = rate * scale.unsqueeze(-1)
        rate = torch.where(
            (geometry["trajectory_type"] == AXIS_SINE).unsqueeze(-1),
            rate,
            scaled_rate,
        )
    return rate


def _probe_tables(
    rate: torch.Tensor,
    elapsed: torch.Tensor,
    geometry: dict,
    trajectory_type: torch.Tensor,
    table_requested_period: torch.Tensor,
) -> RetimedTrajectoryTables:
    phase = geometry["phase"]
    phase_nodes = torch.cat(
        (
            phase,
            torch.full((phase.shape[0], 1), 2.0 * torch.pi, dtype=phase.dtype, device=phase.device),
        ),
        dim=1,
    )
    acceleration = _phase_acceleration_for_rate(
        rate,
        elapsed,
        trajectory_type,
        geometry["first"],
        geometry["second"],
        geometry["phase_speed"],
    )
    return RetimedTrajectoryTables(
        phase=phase_nodes,
        elapsed_s=elapsed,
        phase_rate=torch.cat((rate, rate[:, :1]), dim=1),
        phase_acceleration=torch.cat((acceleration, acceleration[:, :1]), dim=1),
        requested_period_s=table_requested_period,
        effective_period_s=elapsed[:, -1],
        retimed=torch.ones(rate.shape[0], dtype=torch.bool, device=rate.device),
    )


def _limit_interpolated_rate(
    rate: torch.Tensor,
    geometry: dict,
    trajectory_type: torch.Tensor,
    target_speed_mps: torch.Tensor,
    speed_controlled: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    table_requested_period: torch.Tensor,
    *,
    radius_min: float,
    radius_max: float,
    harmonic_ratio: float,
    limits: TrajectoryKinematicLimits,
) -> torch.Tensor:
    analytic_only = bool(torch.all(speed_controlled))
    probe_fractions = (
        torch.arange(
            _INTERPOLATION_PROBE_SUBSTEPS,
            dtype=rate.dtype,
            device=rate.device,
        )
    ) / _INTERPOLATION_PROBE_SUBSTEPS
    for _ in range(4):
        dt, elapsed = _elapsed_for_rate(rate, geometry["phase_step"])
        tables = _probe_tables(rate, elapsed, geometry, trajectory_type, table_requested_period)
        interpolated_max_speed = torch.zeros_like(target_speed_mps)
        interpolated_max_acceleration = torch.zeros_like(target_speed_mps)
        interpolated_max_jerk = torch.zeros_like(target_speed_mps)
        interpolated_max_yaw_rate = torch.zeros_like(target_speed_mps)
        for fraction in probe_fractions:
            probe_time = elapsed[:, :-1] + dt * fraction
            probe_phase, probe_rate, probe_acceleration, probe_jerk = (
                _sample_retimed_phase_kinematics(tables, probe_time)
            )
            _, first, second, third = _geometry_derivatives_through_third(
                trajectory_type,
                axis,
                wave_count,
                amp_x,
                amp_y,
                amp_z,
                phase_x,
                phase_y,
                probe_phase,
                analytic_only=analytic_only,
                radius_min=radius_min,
                radius_max=radius_max,
                harmonic_ratio=harmonic_ratio,
            )
            velocity = first * probe_rate.unsqueeze(-1)
            acceleration = (
                second * probe_rate.square().unsqueeze(-1)
                + first * probe_acceleration.unsqueeze(-1)
            )
            jerk = (
                third * probe_rate.pow(3).unsqueeze(-1)
                + 3.0
                * second
                * probe_rate.unsqueeze(-1)
                * probe_acceleration.unsqueeze(-1)
                + first * probe_jerk.unsqueeze(-1)
            )
            interpolated_max_speed = torch.maximum(
                interpolated_max_speed,
                torch.linalg.vector_norm(velocity, dim=-1).amax(dim=1),
            )
            interpolated_max_acceleration = torch.maximum(
                interpolated_max_acceleration,
                torch.linalg.vector_norm(acceleration, dim=-1).amax(dim=1),
            )
            interpolated_max_jerk = torch.maximum(
                interpolated_max_jerk,
                torch.linalg.vector_norm(jerk, dim=-1).amax(dim=1),
            )
            interpolated_max_yaw_rate = torch.maximum(
                interpolated_max_yaw_rate,
                (_yaw_rate_per_phase(first, second) * probe_rate).amax(dim=1),
            )
        scale = _kinematic_scale(
            interpolated_max_speed,
            interpolated_max_acceleration,
            interpolated_max_jerk,
            limits,
            interpolated_max_yaw_rate,
        )
        requested_speed_scale = target_speed_mps / interpolated_max_speed.clamp_min(1.0e-6)
        scale = torch.where(
            speed_controlled,
            torch.minimum(scale, requested_speed_scale),
            scale,
        )
        scaled_rate = rate * scale.unsqueeze(-1)
        rate = torch.where(
            (trajectory_type == AXIS_SINE).unsqueeze(-1),
            rate,
            scaled_rate,
        )
    return rate


def build_retimed_tables(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    requested_period_s: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    target_speed_mps: torch.Tensor | None = None,
    *,
    radius_min: float,
    radius_max: float,
    chirp_rate: float,
    harmonic_ratio: float,
    limits: TrajectoryKinematicLimits,
) -> RetimedTrajectoryTables:
    """Construct deterministic, locally speed-limited phase tables at reset."""

    speed_controlled, target_speed_mps = _validated_speed_targets(
        trajectory_type,
        requested_period_s,
        target_speed_mps,
        limits,
    )
    geometry = _phase_geometry(
        trajectory_type,
        axis,
        wave_count,
        amp_x,
        amp_y,
        amp_z,
        phase_x,
        phase_y,
        radius_min=radius_min,
        radius_max=radius_max,
        harmonic_ratio=harmonic_ratio,
        samples=int(limits.retime_samples),
        device=requested_period_s.device,
        dtype=requested_period_s.dtype,
    )
    rate, table_period = _candidate_phase_rate(
        trajectory_type,
        axis,
        amp_x,
        amp_y,
        amp_z,
        requested_period_s,
        target_speed_mps,
        speed_controlled,
        geometry,
        chirp_rate,
        limits,
    )
    rate = _limit_discrete_rate(rate, geometry, limits)
    rate = _limit_interpolated_rate(
        rate,
        geometry,
        trajectory_type,
        target_speed_mps,
        speed_controlled,
        axis,
        wave_count,
        amp_x,
        amp_y,
        amp_z,
        phase_x,
        phase_y,
        table_period,
        radius_min=radius_min,
        radius_max=radius_max,
        harmonic_ratio=harmonic_ratio,
        limits=limits,
    )
    conservative = 0.90 * rate
    conservative = torch.where(
        (trajectory_type == RACETRACK).unsqueeze(-1),
        0.90 * conservative,
        conservative,
    )
    rate = torch.where(speed_controlled.unsqueeze(-1), rate, conservative)
    _, elapsed = _elapsed_for_rate(rate, geometry["phase_step"])
    tables = _probe_tables(rate, elapsed, geometry, trajectory_type, table_period)
    effective_period = elapsed[:, -1]
    tables.retimed = effective_period > table_period * (1.0 + 1.0e-4)
    return tables

def _sample_retimed_phase_kinematics(
    tables: RetimedTrajectoryTables,
    elapsed_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert time and return phase through its third time derivative."""

    period = tables.effective_period_s.clamp_min(1.0e-6)
    if elapsed_s.ndim > 1:
        period = period.reshape(-1, *([1] * (elapsed_s.ndim - 1)))
    wrapped = torch.remainder(elapsed_s, period)
    search_values = wrapped.unsqueeze(-1) if wrapped.ndim == 1 else wrapped
    indices = torch.searchsorted(tables.elapsed_s.contiguous(), search_values, right=True)
    if wrapped.ndim == 1:
        indices = indices.squeeze(-1)
    indices = indices.clamp(min=1, max=tables.elapsed_s.shape[1] - 1)
    left = indices - 1
    env = torch.arange(tables.elapsed_s.shape[0], device=left.device)
    if left.ndim > 1:
        env = env.reshape(-1, *([1] * (left.ndim - 1))).expand_as(left)
    t0 = tables.elapsed_s[env, left]
    t1 = tables.elapsed_s[env, indices]
    interval = (t1 - t0).clamp_min(1.0e-6)
    alpha = ((wrapped - t0) / interval).clamp(0.0, 1.0)
    q0 = tables.phase[env, left]
    q1 = tables.phase[env, indices]
    rate0 = tables.phase_rate[env, left]
    rate1 = tables.phase_rate[env, indices]
    acc0 = tables.phase_acceleration[env, left]
    acc1 = tables.phase_acceleration[env, indices]
    s = alpha
    s2 = s.square()
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    # Numerically stable quintic Hermite polynomial.  Expanding around q0
    # avoids subtracting O(q/interval) endpoint terms when differentiating;
    # that cancellation previously injected artificial acceleration/jerk into
    # otherwise constant-rate speed commands in float32.
    c1 = interval * rate0
    c2 = 0.5 * interval.square() * acc0
    displacement_residual = (q1 - q0) - c1 - c2
    rate_residual = interval * rate1 - c1 - 2.0 * c2
    acceleration_residual = interval.square() * acc1 - 2.0 * c2
    c3 = 10.0 * displacement_residual - 4.0 * rate_residual + 0.5 * acceleration_residual
    c4 = -15.0 * displacement_residual + 7.0 * rate_residual - acceleration_residual
    c5 = 6.0 * displacement_residual - 3.0 * rate_residual + 0.5 * acceleration_residual
    q = q0 + c1 * s + c2 * s2 + c3 * s3 + c4 * s4 + c5 * s5
    rate = (c1 + 2.0 * c2 * s + 3.0 * c3 * s2 + 4.0 * c4 * s3 + 5.0 * c5 * s4) / interval
    acceleration = (2.0 * c2 + 6.0 * c3 * s + 12.0 * c4 * s2 + 20.0 * c5 * s3) / interval.square()
    jerk = (6.0 * c3 + 24.0 * c4 * s + 60.0 * c5 * s2) / interval.pow(3)
    return q, rate, acceleration, jerk


def sample_retimed_phase(
    tables: RetimedTrajectoryTables,
    elapsed_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert periodic elapsed time with a periodic C² quintic interpolant."""

    phase, rate, acceleration, _ = _sample_retimed_phase_kinematics(
        tables,
        elapsed_s,
    )
    return phase, rate, acceleration


def evaluate_retimed_reference(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    phase: torch.Tensor,
    phase_rate: torch.Tensor,
    phase_acceleration: torch.Tensor,
    *,
    radius_min: float,
    radius_max: float,
    harmonic_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local position, velocity, acceleration, and curvature at phase."""

    single_sample = phase.ndim == 1
    phase_2d = phase.unsqueeze(-1) if single_sample else phase
    position, first, second = _geometry_derivatives(
        trajectory_type,
        axis,
        wave_count,
        amp_x,
        amp_y,
        amp_z,
        phase_x,
        phase_y,
        phase_2d,
        radius_min=radius_min,
        radius_max=radius_max,
        harmonic_ratio=harmonic_ratio,
    )
    if single_sample:
        position = position[:, 0]
        first = first[:, 0]
        second = second[:, 0]
    velocity = first * phase_rate.unsqueeze(-1)
    acceleration = second * phase_rate.square().unsqueeze(-1) + first * phase_acceleration.unsqueeze(-1)
    phase_speed = torch.linalg.vector_norm(first, dim=-1).clamp_min(1.0e-6)
    curvature = torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1) / phase_speed.pow(3)
    return position, velocity, acceleration, curvature

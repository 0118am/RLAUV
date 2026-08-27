"""Closed trajectory geometry and phase derivatives."""

from __future__ import annotations

import torch

from .catalog import (
    AXIS_SINE,
    BREATHING_LOOP,
    CHIRP,
    CIRCLE,
    LATERAL_WAVE,
    LISSAJOUS,
    RACETRACK,
    RANDOM_SMOOTH,
    REVERSE_SPATIAL_HELIX,
    SPATIAL_HELIX,
    VERTICAL_WAVE,
    WAVY_LOOP,
)


_TRAVELING_WAVE_RETURN_LANE_RATIO = 0.25

def _expand(value: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1).expand_as(phase)


def _racetrack_turn_position(
    progress: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate one right-hand G² turn with a cubic smoothstep curvature law.

    The turn begins with a straight heading and ends facing the opposite
    direction.  Each curvature transition has length ``0.20 * radius``;
    the middle arc supplies the remaining heading change.  Fixed, vectorised
    trapezoidal quadrature is used only for this compact geometry primitive,
    making it deterministic and differentiable in the sampled phase.
    """

    progress = progress.clamp(0.0, 1.0)
    transition_length = 0.20 * radius
    middle_length = (torch.pi - 0.20) * radius
    turn_length = middle_length + 2.0 * transition_length
    distance = progress * turn_length
    # Gauss-Legendre integration is split at both curvature joins.  Unlike a
    # whole-turn trapezoid, no moving integration sample crosses a join, so
    # numerical derivatives preserve the intended G² continuity.
    nodes = torch.tensor(
        (-0.9815606342, -0.9041172564, -0.7699026742, -0.5873179543, -0.3678314989, -0.1252334085,
         0.1252334085, 0.3678314989, 0.5873179543, 0.7699026742, 0.9041172564, 0.9815606342),
        dtype=progress.dtype,
        device=progress.device,
    )
    weights = torch.tensor(
        (0.0471753364, 0.1069393259, 0.1600783285, 0.2031674267, 0.2334925365, 0.2491470458,
         0.2491470458, 0.2334925365, 0.2031674267, 0.1600783285, 0.1069393259, 0.0471753364),
        dtype=progress.dtype,
        device=progress.device,
    )

    def integrate_transition(length: torch.Tensor, *, entering: bool) -> tuple[torch.Tensor, torch.Tensor]:
        local = 0.5 * length.unsqueeze(-1) * (nodes + 1.0)
        u = local / transition_length.unsqueeze(-1).clamp_min(1.0e-6)
        if entering:
            heading = -0.20 * (u.pow(3) - 0.5 * u.pow(4))
        else:
            heading = -(torch.pi - 0.10) - 0.20 * (u - u.pow(3) + 0.5 * u.pow(4))
        scale = 0.5 * length
        return (
            scale * torch.sum(torch.cos(heading) * weights, dim=-1),
            scale * torch.sum(torch.sin(heading) * weights, dim=-1),
        )

    entry_length = torch.minimum(distance, transition_length)
    entry_x, entry_y = integrate_transition(entry_length, entering=True)
    middle_distance = torch.minimum((distance - transition_length).clamp_min(0.0), middle_length)
    middle_angle = middle_distance / radius.clamp_min(1.0e-6)
    # The entry transition contributes exactly -0.10 rad.  The middle part is
    # an ordinary circular arc and therefore has an exact integral.
    middle_x = radius * (torch.sin(torch.as_tensor(-0.10, dtype=progress.dtype, device=progress.device)) - torch.sin(-0.10 - middle_angle))
    middle_y = radius * (torch.cos(-0.10 - middle_angle) - torch.cos(
        torch.as_tensor(-0.10, dtype=progress.dtype, device=progress.device)
    ))
    exit_length = torch.minimum(
        (distance - transition_length - middle_length).clamp_min(0.0), transition_length
    )
    exit_x, exit_y = integrate_transition(exit_length, entering=False)
    return entry_x + middle_x + exit_x, entry_y + middle_y + exit_y


def _smooth_racetrack(phase: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor) -> torch.Tensor:
    """Return the closed G² racetrack used by the curriculum.

    The former straight/semicircle switch was only C¹.  This construction has
    two straight sections and two G² turns whose curvature follows a cubic
    smoothstep over ``0.20 * radius`` at every line/arc join.
    """

    radius = torch.clamp(_expand(ay, phase), min=1.0e-4)
    half_straight = torch.clamp(_expand(ax, phase) - radius, min=0.10)
    turn_length = (torch.pi + 0.20) * radius
    total_length = 4.0 * half_straight + 2.0 * turn_length
    arc_s = torch.remainder(phase, 2.0 * torch.pi) * total_length / (2.0 * torch.pi)
    turn_dx, turn_dy = _racetrack_turn_position(torch.ones_like(arc_s), radius)
    top_y = -0.5 * turn_dy

    top_end = 2.0 * half_straight
    right_end = top_end + turn_length
    bottom_end = right_end + 2.0 * half_straight
    on_top = arc_s < top_end
    on_right = (arc_s >= top_end) & (arc_s < right_end)
    on_bottom = (arc_s >= right_end) & (arc_s < bottom_end)
    right_progress = ((arc_s - top_end) / turn_length).clamp(0.0, 1.0)
    left_progress = ((arc_s - bottom_end) / turn_length).clamp(0.0, 1.0)
    right_dx, right_dy = _racetrack_turn_position(right_progress, radius)
    left_dx, left_dy = _racetrack_turn_position(left_progress, radius)
    top_x = -half_straight + arc_s
    bottom_x = half_straight + turn_dx - (arc_s - right_end)
    right_x = half_straight + right_dx
    right_y = top_y + right_dy
    left_x = -half_straight + turn_dx - left_dx
    left_y = top_y + turn_dy - left_dy
    x = torch.where(on_top, top_x, torch.where(on_right, right_x, torch.where(on_bottom, bottom_x, left_x)))
    y = torch.where(on_top, top_y, torch.where(on_right, right_y, torch.where(on_bottom, top_y + turn_dy, left_y)))
    return torch.stack((x, y, torch.zeros_like(x)), dim=-1)


def evaluate_geometry(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    phase: torch.Tensor,
    *,
    radius_min: float,
    radius_max: float,
    harmonic_ratio: float,
) -> torch.Tensor:
    """Evaluate closed trajectory geometry at a phase tensor ``[env, sample]``.

    The phase offsets are reset-time parameters rather than a separate time
    state. The returned position is a signed offset from the trajectory center;
    the simulator adds the positive pool-center coordinate before publishing a
    pool-local FLU target.
    """

    if phase.ndim != 2:
        raise ValueError("phase must have shape [num_envs, num_samples].")
    harmonic_ratio = float(harmonic_ratio)
    px = _expand(phase_x, phase)
    py = _expand(phase_y, phase)
    ax = _expand(amp_x, phase)
    ay = _expand(amp_y, phase)
    az = _expand(amp_z, phase)
    wave_harmonic = 2.0 * _expand(wave_count.to(dtype=phase.dtype), phase)
    types = trajectory_type.unsqueeze(-1)

    base = phase + px
    harmonic = 2.0 * phase + py
    zeros = torch.zeros_like(phase)
    result = torch.zeros(*phase.shape, 3, dtype=phase.dtype, device=phase.device)

    circle = torch.stack((ax * torch.cos(base), ay * torch.sin(base), zeros), dim=-1)
    result = torch.where((types == CIRCLE).unsqueeze(-1), circle, result)

    # Keep the 1:2:3 harmonic phases coherent. Independent random offsets can
    # create a zero-tangent cusp, which no finite heading rate can track; the
    # common offset still randomises the starting point. All three amplitudes
    # are half-extents, so this is a genuinely spatial Lissajous curve.
    lissajous_harmonic = 2.0 * base
    lissajous_vertical = 3.0 * base
    lissajous = torch.stack(
        (
            ax * torch.sin(base),
            ay * torch.sin(lissajous_harmonic),
            az * torch.sin(lissajous_vertical),
        ),
        dim=-1,
    )
    result = torch.where((types == LISSAJOUS).unsqueeze(-1), lissajous, result)

    sine_amplitude = torch.where(_expand(axis, phase) == 1, ay, ax)
    sine_amplitude = torch.where(_expand(axis, phase) == 2, az, sine_amplitude)
    sine = torch.nn.functional.one_hot(axis, num_classes=3).to(dtype=phase.dtype).unsqueeze(1)
    sine = sine * (sine_amplitude * torch.sin(base)).unsqueeze(-1)
    result = torch.where((types == AXIS_SINE).unsqueeze(-1), sine, result)

    wavy_loop = torch.stack(
        (ax * torch.cos(base), ay * torch.sin(base), az * torch.sin(harmonic)), dim=-1
    )
    result = torch.where((types == WAVY_LOOP).unsqueeze(-1), wavy_loop, result)

    # This is a closed, breathing loop rather than a monotone spiral. One
    # requested period is one complete closed reference cycle.
    radius_floor = torch.as_tensor(radius_min, dtype=phase.dtype, device=phase.device)
    radius_span = torch.as_tensor(radius_max - radius_min, dtype=phase.dtype, device=phase.device)
    radius = radius_floor + 0.5 * radius_span * (1.0 - torch.cos(phase))
    breathing_loop = torch.stack((radius * torch.cos(base), radius * torch.sin(base), zeros), dim=-1)
    result = torch.where((types == BREATHING_LOOP).unsqueeze(-1), breathing_loop, result)

    chirp = torch.stack((ax * torch.sin(base), ay * torch.sin(lissajous_harmonic), zeros), dim=-1)
    result = torch.where((types == CHIRP).unsqueeze(-1), chirp, result)

    racetrack = _smooth_racetrack(phase, amp_x, amp_y)
    result = torch.where((types == RACETRACK).unsqueeze(-1), racetrack, result)

    vertical = phase + py
    random_smooth = torch.stack(
        (
            ax * (torch.cos(base) + harmonic_ratio * torch.cos(harmonic)),
            ay * (torch.sin(base) + harmonic_ratio * torch.sin(harmonic)),
            az * (0.70 * torch.sin(vertical) + 0.15 * torch.sin(harmonic)),
        ),
        dim=-1,
    )
    result = torch.where((types == RANDOM_SMOOTH).unsqueeze(-1), random_smooth, result)

    # During either half-cycle, x moves monotonically from one end of the
    # command envelope to the other while the transverse coordinate completes
    # ``wave_count`` full sine waves. A quarter-width cosine return lane keeps
    # the outbound and inbound branches separate and gives vertical waves a
    # horizontal U-turn. The resulting closed curve has no zero tangent or
    # vertical-heading singularity.
    wave_phase = wave_harmonic * base
    return_lane_ratio = torch.as_tensor(
        _TRAVELING_WAVE_RETURN_LANE_RATIO,
        dtype=phase.dtype,
        device=phase.device,
    )
    wave_ratio = 1.0 - return_lane_ratio
    return_lane = return_lane_ratio * ay * torch.cos(base)
    lateral_wave = torch.stack(
        (
            ax * torch.sin(base),
            return_lane + wave_ratio * ay * torch.sin(wave_phase),
            zeros,
        ),
        dim=-1,
    )
    result = torch.where((types == LATERAL_WAVE).unsqueeze(-1), lateral_wave, result)

    vertical_wave = torch.stack(
        (ax * torch.sin(base), return_lane, az * torch.sin(wave_phase)),
        dim=-1,
    )
    result = torch.where((types == VERTICAL_WAVE).unsqueeze(-1), vertical_wave, result)

    # A bounded, closed spatial helix.  Closing the vertical component after
    # two coils avoids the discontinuity of an unbounded z=pitch*q helix at an
    # episode/table wrap while preserving genuinely three-dimensional motion.
    spatial_helix = torch.stack(
        (ax * torch.cos(base), ay * torch.sin(base), az * torch.sin(2.0 * base)), dim=-1
    )
    result = torch.where((types == SPATIAL_HELIX).unsqueeze(-1), spatial_helix, result)

    # Traverse exactly the same closed spatial-helix geometry in reverse from
    # the sampled start phase.  With q increasing, the original curve is
    # evaluated at ``phase_x - q`` instead of ``phase_x + q``; this reverses
    # velocity and every odd time derivative without inventing a mirrored task.
    reverse_base = px - phase
    reverse_spatial_helix = torch.stack(
        (
            ax * torch.cos(reverse_base),
            ay * torch.sin(reverse_base),
            az * torch.sin(2.0 * reverse_base),
        ),
        dim=-1,
    )
    result = torch.where(
        (types == REVERSE_SPATIAL_HELIX).unsqueeze(-1),
        reverse_spatial_helix,
        result,
    )
    return result


def _analytic_speed_controlled_derivatives(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact position through third phase derivative for training curves."""

    types = trajectory_type.unsqueeze(-1)
    condition_axis = (types == AXIS_SINE).unsqueeze(-1)
    condition_lissajous = (types == LISSAJOUS).unsqueeze(-1)
    condition_lateral = (types == LATERAL_WAVE).unsqueeze(-1)
    condition_vertical = (types == VERTICAL_WAVE).unsqueeze(-1)
    condition_helix = (types == SPATIAL_HELIX).unsqueeze(-1)
    condition_reverse_helix = (types == REVERSE_SPATIAL_HELIX).unsqueeze(-1)
    base = phase + _expand(phase_x, phase)
    reverse_base = _expand(phase_x, phase) - phase
    zeros = torch.zeros_like(phase)
    ax = _expand(amp_x, phase)
    ay = _expand(amp_y, phase)
    az = _expand(amp_z, phase)
    wave_harmonic = 2.0 * _expand(wave_count.to(dtype=phase.dtype), phase)
    wave_phase = wave_harmonic * base
    sin_1 = torch.sin(base)
    cos_1 = torch.cos(base)
    sin_2 = torch.sin(2.0 * base)
    cos_2 = torch.cos(2.0 * base)
    sin_3 = torch.sin(3.0 * base)
    cos_3 = torch.cos(3.0 * base)
    reverse_sin_1 = torch.sin(reverse_base)
    reverse_cos_1 = torch.cos(reverse_base)
    reverse_sin_2 = torch.sin(2.0 * reverse_base)
    reverse_cos_2 = torch.cos(2.0 * reverse_base)
    sin_wave = torch.sin(wave_phase)
    cos_wave = torch.cos(wave_phase)

    shape = (*phase.shape, 3)
    position = torch.zeros(shape, dtype=phase.dtype, device=phase.device)
    first = torch.zeros_like(position)
    second = torch.zeros_like(position)
    third = torch.zeros_like(position)
    return_lane_ratio = torch.as_tensor(
        _TRAVELING_WAVE_RETURN_LANE_RATIO,
        dtype=phase.dtype,
        device=phase.device,
    )
    wave_ratio = 1.0 - return_lane_ratio

    sine_amplitude = torch.where(_expand(axis, phase) == 1, ay, ax)
    sine_amplitude = torch.where(_expand(axis, phase) == 2, az, sine_amplitude)
    sine_axis = torch.nn.functional.one_hot(axis, num_classes=3).to(dtype=phase.dtype).unsqueeze(1)
    axis_sine = (
        sine_axis * (sine_amplitude * sin_1).unsqueeze(-1),
        sine_axis * (sine_amplitude * cos_1).unsqueeze(-1),
        sine_axis * (-sine_amplitude * sin_1).unsqueeze(-1),
        sine_axis * (-sine_amplitude * cos_1).unsqueeze(-1),
    )

    lissajous = (
        torch.stack((ax * sin_1, ay * sin_2, az * sin_3), dim=-1),
        torch.stack((ax * cos_1, 2.0 * ay * cos_2, 3.0 * az * cos_3), dim=-1),
        torch.stack((-ax * sin_1, -4.0 * ay * sin_2, -9.0 * az * sin_3), dim=-1),
        torch.stack((-ax * cos_1, -8.0 * ay * cos_2, -27.0 * az * cos_3), dim=-1),
    )
    lateral = (
        torch.stack(
            (ax * sin_1, return_lane_ratio * ay * cos_1 + wave_ratio * ay * sin_wave, zeros),
            dim=-1,
        ),
        torch.stack(
            (
                ax * cos_1,
                -return_lane_ratio * ay * sin_1 + wave_ratio * wave_harmonic * ay * cos_wave,
                zeros,
            ),
            dim=-1,
        ),
        torch.stack(
            (
                -ax * sin_1,
                -return_lane_ratio * ay * cos_1
                - wave_ratio * wave_harmonic.square() * ay * sin_wave,
                zeros,
            ),
            dim=-1,
        ),
        torch.stack(
            (
                -ax * cos_1,
                return_lane_ratio * ay * sin_1
                - wave_ratio * wave_harmonic.pow(3) * ay * cos_wave,
                zeros,
            ),
            dim=-1,
        ),
    )
    vertical = (
        torch.stack((ax * sin_1, return_lane_ratio * ay * cos_1, az * sin_wave), dim=-1),
        torch.stack(
            (ax * cos_1, -return_lane_ratio * ay * sin_1, wave_harmonic * az * cos_wave),
            dim=-1,
        ),
        torch.stack(
            (-ax * sin_1, -return_lane_ratio * ay * cos_1, -wave_harmonic.square() * az * sin_wave),
            dim=-1,
        ),
        torch.stack(
            (-ax * cos_1, return_lane_ratio * ay * sin_1, -wave_harmonic.pow(3) * az * cos_wave),
            dim=-1,
        ),
    )
    helix = (
        torch.stack((ax * cos_1, ay * sin_1, az * sin_2), dim=-1),
        torch.stack((-ax * sin_1, ay * cos_1, 2.0 * az * cos_2), dim=-1),
        torch.stack((-ax * cos_1, -ay * sin_1, -4.0 * az * sin_2), dim=-1),
        torch.stack((ax * sin_1, -ay * cos_1, -8.0 * az * cos_2), dim=-1),
    )
    reverse_helix = (
        torch.stack(
            (ax * reverse_cos_1, ay * reverse_sin_1, az * reverse_sin_2),
            dim=-1,
        ),
        torch.stack(
            (ax * reverse_sin_1, -ay * reverse_cos_1, -2.0 * az * reverse_cos_2),
            dim=-1,
        ),
        torch.stack(
            (-ax * reverse_cos_1, -ay * reverse_sin_1, -4.0 * az * reverse_sin_2),
            dim=-1,
        ),
        torch.stack(
            (-ax * reverse_sin_1, ay * reverse_cos_1, 8.0 * az * reverse_cos_2),
            dim=-1,
        ),
    )
    for condition, values in (
        (condition_axis, axis_sine),
        (condition_lissajous, lissajous),
        (condition_lateral, lateral),
        (condition_vertical, vertical),
        (condition_helix, helix),
        (condition_reverse_helix, reverse_helix),
    ):
        position = torch.where(condition, values[0], position)
        first = torch.where(condition, values[1], first)
        second = torch.where(condition, values[2], second)
        third = torch.where(condition, values[3], third)
    return position, first, second, third


def _geometry_derivatives(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    phase: torch.Tensor,
    *,
    radius_min: float,
    radius_max: float,
    harmonic_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return geometry and first two phase derivatives via central differences."""

    # A 1e-2 rad central stencil is accurate for these low-order Fourier
    # curves while avoiding float32 cancellation in the second derivative.
    step = torch.as_tensor(1.0e-2, dtype=phase.dtype, device=phase.device)
    common = dict(
        radius_min=radius_min,
        radius_max=radius_max,
        harmonic_ratio=harmonic_ratio,
    )
    position = evaluate_geometry(
        trajectory_type, axis, wave_count, amp_x, amp_y, amp_z, phase_x, phase_y, phase, **common
    )
    forward = evaluate_geometry(
        trajectory_type, axis, wave_count, amp_x, amp_y, amp_z, phase_x, phase_y, phase + step, **common
    )
    backward = evaluate_geometry(
        trajectory_type, axis, wave_count, amp_x, amp_y, amp_z, phase_x, phase_y, phase - step, **common
    )
    first = (forward - backward) / (2.0 * step)
    second = (forward - 2.0 * position + backward) / (step * step)

    # The speed-controlled training curves have compact analytic derivatives.
    # Using them avoids float32 second-difference noise leaking into the
    # arc-length reparameterization and its jerk estimate.
    analytic_position, analytic_first, analytic_second, _ = (
        _analytic_speed_controlled_derivatives(
            trajectory_type,
            axis,
            wave_count,
            amp_x,
            amp_y,
            amp_z,
            phase_x,
            phase,
        )
    )
    analytic = (
        (trajectory_type == AXIS_SINE)
        | (trajectory_type == LISSAJOUS)
        | (trajectory_type == LATERAL_WAVE)
        | (trajectory_type == VERTICAL_WAVE)
        | (trajectory_type == SPATIAL_HELIX)
        | (trajectory_type == REVERSE_SPATIAL_HELIX)
    ).reshape(-1, 1, 1)
    position = torch.where(analytic, analytic_position, position)
    first = torch.where(analytic, analytic_first, first)
    second = torch.where(analytic, analytic_second, second)
    return position, first, second


def _geometry_derivatives_through_third(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
    wave_count: torch.Tensor,
    amp_x: torch.Tensor,
    amp_y: torch.Tensor,
    amp_z: torch.Tensor,
    phase_x: torch.Tensor,
    phase_y: torch.Tensor,
    phase: torch.Tensor,
    *,
    analytic_only: bool,
    radius_min: float,
    radius_max: float,
    harmonic_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return geometry and its first three phase derivatives at reset time."""

    common = dict(
        radius_min=radius_min,
        radius_max=radius_max,
        harmonic_ratio=harmonic_ratio,
    )
    analytic = (
        (trajectory_type == AXIS_SINE)
        | (trajectory_type == LISSAJOUS)
        | (trajectory_type == LATERAL_WAVE)
        | (trajectory_type == VERTICAL_WAVE)
        | (trajectory_type == SPATIAL_HELIX)
        | (trajectory_type == REVERSE_SPATIAL_HELIX)
    )
    analytic_values = _analytic_speed_controlled_derivatives(
        trajectory_type,
        axis,
        wave_count,
        amp_x,
        amp_y,
        amp_z,
        phase_x,
        phase,
    )
    if analytic_only:
        return analytic_values

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
        **common,
    )
    third = torch.zeros_like(position)
    if bool(torch.any(~analytic)):
        step = torch.as_tensor(1.0e-2, dtype=phase.dtype, device=phase.device)
        _, _, second_forward = _geometry_derivatives(
            trajectory_type,
            axis,
            wave_count,
            amp_x,
            amp_y,
            amp_z,
            phase_x,
            phase_y,
            phase + step,
            **common,
        )
        _, _, second_backward = _geometry_derivatives(
            trajectory_type,
            axis,
            wave_count,
            amp_x,
            amp_y,
            amp_z,
            phase_x,
            phase_y,
            phase - step,
            **common,
        )
        third = (second_forward - second_backward) / (2.0 * step)

    analytic_mask = analytic.reshape(-1, 1, 1)
    return tuple(
        torch.where(analytic_mask, analytic_value, numeric_value)
        for analytic_value, numeric_value in zip(
            analytic_values,
            (position, first, second, third),
            strict=True,
        )
    )


def _yaw_rate_per_phase(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return absolute horizontal-heading rate per unit phase."""

    vx, vy = first[..., :2].unbind(dim=-1)
    ax, ay = second[..., :2].unbind(dim=-1)
    horizontal_speed_sq = (vx.square() + vy.square()).clamp_min(1.0e-8)
    yaw_rate = (vx * ay - vy * ax) / horizontal_speed_sq
    return torch.abs(yaw_rate)

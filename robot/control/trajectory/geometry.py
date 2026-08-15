"""Closed trajectory geometry and phase derivatives."""

from __future__ import annotations

import torch

from .catalog import (
    AXIS_SINE,
    BREATHING_LOOP,
    CHIRP,
    CIRCLE,
    LATERAL_SINE,
    LISSAJOUS,
    RACETRACK,
    RANDOM_SMOOTH,
    SPATIAL_HELIX,
    VERTICAL_SINE,
    WAVY_LOOP,
)

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
    harmonic_ratio = max(0.0, min(float(harmonic_ratio), 0.10))
    px = _expand(phase_x, phase)
    py = _expand(phase_y, phase)
    ax = _expand(amp_x, phase)
    ay = _expand(amp_y, phase)
    az = _expand(amp_z, phase)
    types = trajectory_type.unsqueeze(-1)

    base = phase + px
    harmonic = 2.0 * phase + py
    zeros = torch.zeros_like(phase)
    result = torch.zeros(*phase.shape, 3, dtype=phase.dtype, device=phase.device)

    circle = torch.stack((ax * torch.cos(base), ay * torch.sin(base), zeros), dim=-1)
    result = torch.where((types == CIRCLE).unsqueeze(-1), circle, result)

    # Keep the 2:1 harmonic phase coherent with the fundamental.  Independent
    # random offsets can create a zero-tangent cusp, which no finite heading
    # rate can track; the common offset still randomises the starting point.
    lissajous_harmonic = 2.0 * base
    lissajous = torch.stack((ax * torch.sin(base), ay * torch.sin(lissajous_harmonic), zeros), dim=-1)
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

    lateral_sine = torch.stack((zeros, ay * torch.sin(base), zeros), dim=-1)
    result = torch.where((types == LATERAL_SINE).unsqueeze(-1), lateral_sine, result)

    vertical_sine = torch.stack((zeros, zeros, az * torch.sin(base)), dim=-1)
    result = torch.where((types == VERTICAL_SINE).unsqueeze(-1), vertical_sine, result)

    # A bounded, closed spatial helix.  Closing the vertical component after
    # two coils avoids the discontinuity of an unbounded z=pitch*q helix at an
    # episode/table wrap while preserving genuinely three-dimensional motion.
    spatial_helix = torch.stack(
        (ax * torch.cos(base), ay * torch.sin(base), az * torch.sin(2.0 * base)), dim=-1
    )
    result = torch.where((types == SPATIAL_HELIX).unsqueeze(-1), spatial_helix, result)
    return result


def _geometry_derivatives(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
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
        trajectory_type, axis, amp_x, amp_y, amp_z, phase_x, phase_y, phase, **common
    )
    forward = evaluate_geometry(
        trajectory_type, axis, amp_x, amp_y, amp_z, phase_x, phase_y, phase + step, **common
    )
    backward = evaluate_geometry(
        trajectory_type, axis, amp_x, amp_y, amp_z, phase_x, phase_y, phase - step, **common
    )
    first = (forward - backward) / (2.0 * step)
    second = (forward - 2.0 * position + backward) / (step * step)

    # The speed-controlled training curves have compact analytic derivatives.
    # Using them avoids float32 second-difference noise leaking into the
    # arc-length reparameterization and its jerk estimate.
    types = trajectory_type.unsqueeze(-1)
    base = phase + _expand(phase_x, phase)
    zeros = torch.zeros_like(phase)
    ay = _expand(amp_y, phase)
    az = _expand(amp_z, phase)
    lateral_first = torch.stack((zeros, ay * torch.cos(base), zeros), dim=-1)
    lateral_second = torch.stack((zeros, -ay * torch.sin(base), zeros), dim=-1)
    first = torch.where((types == LATERAL_SINE).unsqueeze(-1), lateral_first, first)
    second = torch.where((types == LATERAL_SINE).unsqueeze(-1), lateral_second, second)

    vertical_first = torch.stack((zeros, zeros, az * torch.cos(base)), dim=-1)
    vertical_second = torch.stack((zeros, zeros, -az * torch.sin(base)), dim=-1)
    first = torch.where((types == VERTICAL_SINE).unsqueeze(-1), vertical_first, first)
    second = torch.where((types == VERTICAL_SINE).unsqueeze(-1), vertical_second, second)

    ax = _expand(amp_x, phase)
    helix_first = torch.stack(
        (-ax * torch.sin(base), ay * torch.cos(base), 2.0 * az * torch.cos(2.0 * base)), dim=-1
    )
    helix_second = torch.stack(
        (-ax * torch.cos(base), -ay * torch.sin(base), -4.0 * az * torch.sin(2.0 * base)), dim=-1
    )
    first = torch.where((types == SPATIAL_HELIX).unsqueeze(-1), helix_first, first)
    second = torch.where((types == SPATIAL_HELIX).unsqueeze(-1), helix_second, second)
    return position, first, second


def _orientation_rate_per_phase(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    speed_per_phase = torch.linalg.vector_norm(first, dim=-1).clamp_min(1.0e-6)
    tangent = first / speed_per_phase.unsqueeze(-1)
    tangent_rate = second / speed_per_phase.unsqueeze(-1)
    tangent_rate = tangent_rate - tangent * torch.sum(tangent * tangent_rate, dim=-1, keepdim=True)
    # ``tangent_rate`` is d(tangent)/dq. Multiplying it by dq/dt gives the
    # commanded heading/pitch rate; dividing by phase speed here would apply
    # curvature once more and severely over-slow otherwise benign curves.
    return torch.linalg.vector_norm(tangent_rate, dim=-1)

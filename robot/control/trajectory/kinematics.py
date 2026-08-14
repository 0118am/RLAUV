"""Kinematically bounded closed reference trajectories for the T60 controller.

Simulator tasks keep their reset sampling logic, while this robot-owned module
provides reusable geometry and time re-parameterization. All functions are
Torch-only so the same reference generator can be deployed outside Isaac.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


TRAJECTORY_GENERATOR_VERSION = "curve_v2"

# Explicit speed-controlled training commands.  The period-based single-axis
# sine (type 2) randomly chooses x/y/z and the wavy loop (type 3) is also
# period-driven, so neither can express the requested Cartesian training
# matrix unambiguously.
LATERAL_SINE = 8
VERTICAL_SINE = 9
SPATIAL_HELIX = 10
SPEED_CONTROLLED_TYPES = (LATERAL_SINE, VERTICAL_SINE, SPATIAL_HELIX)


@dataclass(frozen=True)
class TrajectoryKinematicLimits:
    """Temporary simulator limits for a physically trackable reference.

    They intentionally match the final-stage ``random_smooth`` command
    envelope with modest headroom.  Replace them with limits identified from
    pool data before deployment.
    """

    max_speed_mps: float = 0.60
    max_acceleration_mps2: float = 0.45
    max_orientation_rate_radps: float = 0.80
    max_jerk_mps3: float = 0.36
    retime_samples: int = 256

    def validate(self) -> None:
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive.")
        if self.max_acceleration_mps2 <= 0.0:
            raise ValueError("max_acceleration_mps2 must be positive.")
        if self.max_orientation_rate_radps <= 0.0:
            raise ValueError("max_orientation_rate_radps must be positive.")
        if self.max_jerk_mps3 <= 0.0:
            raise ValueError("max_jerk_mps3 must be positive.")
        if self.retime_samples < 32:
            raise ValueError("retime_samples must be at least 32.")


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
    state.  The returned position is local to the trajectory center.
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
    result = torch.where((types == 0).unsqueeze(-1), circle, result)

    # Keep the 2:1 harmonic phase coherent with the fundamental.  Independent
    # random offsets can create a zero-tangent cusp, which no finite heading
    # rate can track; the common offset still randomises the starting point.
    lissajous_harmonic = 2.0 * base
    lissajous = torch.stack((ax * torch.sin(base), ay * torch.sin(lissajous_harmonic), zeros), dim=-1)
    result = torch.where((types == 1).unsqueeze(-1), lissajous, result)

    sine_amplitude = torch.where(_expand(axis, phase) == 1, ay, ax)
    sine_amplitude = torch.where(_expand(axis, phase) == 2, az, sine_amplitude)
    sine = torch.nn.functional.one_hot(axis, num_classes=3).to(dtype=phase.dtype).unsqueeze(1)
    sine = sine * (sine_amplitude * torch.sin(base)).unsqueeze(-1)
    result = torch.where((types == 2).unsqueeze(-1), sine, result)

    wavy_loop = torch.stack(
        (ax * torch.cos(base), ay * torch.sin(base), az * torch.sin(harmonic)), dim=-1
    )
    result = torch.where((types == 3).unsqueeze(-1), wavy_loop, result)

    # This is a closed, breathing loop rather than a monotone spiral. One
    # requested period is one complete closed reference cycle.
    radius_floor = torch.as_tensor(radius_min, dtype=phase.dtype, device=phase.device)
    radius_span = torch.as_tensor(radius_max - radius_min, dtype=phase.dtype, device=phase.device)
    radius = radius_floor + 0.5 * radius_span * (1.0 - torch.cos(phase))
    breathing_loop = torch.stack((radius * torch.cos(base), radius * torch.sin(base), zeros), dim=-1)
    result = torch.where((types == 4).unsqueeze(-1), breathing_loop, result)

    chirp = torch.stack((ax * torch.sin(base), ay * torch.sin(lissajous_harmonic), zeros), dim=-1)
    result = torch.where((types == 5).unsqueeze(-1), chirp, result)

    racetrack = _smooth_racetrack(phase, amp_x, amp_y)
    result = torch.where((types == 6).unsqueeze(-1), racetrack, result)

    vertical = phase + py
    random_smooth = torch.stack(
        (
            ax * (torch.cos(base) + harmonic_ratio * torch.cos(harmonic)),
            ay * (torch.sin(base) + harmonic_ratio * torch.sin(harmonic)),
            az * (0.70 * torch.sin(vertical) + 0.15 * torch.sin(harmonic)),
        ),
        dim=-1,
    )
    result = torch.where((types == 7).unsqueeze(-1), random_smooth, result)

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


def build_retimed_tables(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
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

    limits.validate()
    if torch.any(requested_period_s <= 0.0):
        raise ValueError("requested_period_s must be positive.")
    speed_controlled = (
        (trajectory_type == LATERAL_SINE)
        | (trajectory_type == VERTICAL_SINE)
        | (trajectory_type == SPATIAL_HELIX)
    )
    if target_speed_mps is None:
        if bool(torch.any(speed_controlled)):
            raise ValueError("speed-controlled trajectories require target_speed_mps.")
        target_speed_mps = torch.zeros_like(requested_period_s)
    if target_speed_mps.shape != requested_period_s.shape:
        raise ValueError("target_speed_mps must match requested_period_s shape.")
    if bool(torch.any(target_speed_mps[speed_controlled] <= 0.0)):
        raise ValueError("speed-controlled trajectory targets must be positive.")
    if bool(torch.any(target_speed_mps[speed_controlled] > limits.max_speed_mps)):
        raise ValueError("target_speed_mps exceeds the configured kinematic speed limit.")
    num_envs = trajectory_type.numel()
    samples = int(limits.retime_samples)
    device = requested_period_s.device
    dtype = requested_period_s.dtype
    phase_1d = torch.arange(samples, device=device, dtype=dtype) * (2.0 * torch.pi / samples)
    phase = phase_1d.unsqueeze(0).expand(num_envs, -1)
    position, first, second = _geometry_derivatives(
        trajectory_type,
        axis,
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
    phase_step = 2.0 * torch.pi / samples
    phase_speed = torch.linalg.vector_norm(first, dim=-1).clamp_min(1.0e-6)
    curvature = torch.linalg.vector_norm(torch.linalg.cross(first, second, dim=-1), dim=-1) / phase_speed.pow(3)
    orientation_per_phase = _orientation_rate_per_phase(first, second)

    def phase_acceleration_for_rate(rate: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        """Return dq-dot/dt, using the exact arc-speed identity for the helix."""

        finite_difference = _closed_scalar_time_derivative(rate, elapsed)
        tangent_growth = torch.sum(first * second, dim=-1)
        constant_arc_speed = -rate.square() * tangent_growth / phase_speed.square()
        return torch.where(
            (trajectory_type == SPATIAL_HELIX).unsqueeze(-1),
            constant_arc_speed,
            finite_difference,
        )

    def phase_interval_duration(rate: torch.Tensor) -> torch.Tensor:
        """Integrate dt=dq/q-dot over each periodic phase interval."""

        inverse_rate = rate.clamp_min(1.0e-6).reciprocal()
        return 0.5 * phase_step * (inverse_rate + torch.roll(inverse_rate, -1, dims=1))
    # A single-axis sine intentionally stops before reversing. Its velocity
    # direction is undefined at the stop and the environment retains the prior
    # attitude command, so it must not impose a fictitious infinite turn-rate
    # constraint on the phase schedule.
    orientation_per_phase = torch.where(
        (
            (trajectory_type == 2)
            | (trajectory_type == LATERAL_SINE)
            | (trajectory_type == VERTICAL_SINE)
        ).unsqueeze(-1),
        torch.zeros_like(orientation_per_phase),
        orientation_per_phase,
    )
    period_rate = (2.0 * torch.pi / requested_period_s).unsqueeze(-1).expand_as(phase)
    lateral_rate = (target_speed_mps / amp_y.clamp_min(1.0e-6)).unsqueeze(-1).expand_as(phase)
    vertical_rate = (target_speed_mps / amp_z.clamp_min(1.0e-6)).unsqueeze(-1).expand_as(phase)
    helix_rate = target_speed_mps.unsqueeze(-1) / phase_speed
    base_rate = torch.where((trajectory_type == LATERAL_SINE).unsqueeze(-1), lateral_rate, period_rate)
    base_rate = torch.where((trajectory_type == VERTICAL_SINE).unsqueeze(-1), vertical_rate, base_rate)
    base_rate = torch.where((trajectory_type == SPATIAL_HELIX).unsqueeze(-1), helix_rate, base_rate)
    requested_speed_period = torch.sum(phase_step / base_rate.clamp_min(1.0e-6), dim=1)
    table_requested_period = torch.where(speed_controlled, requested_speed_period, requested_period_s)
    # Preserve chirp variation while making the cycle periodic at q=0/2pi.
    chirp_shape = 1.0 + 0.5 * max(0.0, float(chirp_rate) - 1.0) * (1.0 - torch.cos(phase))
    base_rate = torch.where((trajectory_type == 5).unsqueeze(-1), base_rate * chirp_shape, base_rate)
    phase_rate = torch.minimum(base_rate, torch.full_like(base_rate, limits.max_speed_mps) / phase_speed)
    normal_rate_cap = torch.sqrt(
        torch.full_like(phase_rate, limits.max_acceleration_mps2)
        / (curvature * phase_speed.square()).clamp_min(1.0e-8)
    )
    orientation_rate_cap = torch.full_like(phase_rate, limits.max_orientation_rate_radps) / orientation_per_phase.clamp_min(1.0e-6)
    phase_rate = torch.minimum(phase_rate, normal_rate_cap)
    phase_rate = torch.minimum(phase_rate, orientation_rate_cap)
    # Circular smoothing removes small finite-difference kinks while reapplying
    # the local ceilings afterwards so no limit is relaxed.
    smoothed_phase_rate = (
        torch.roll(phase_rate, 2, 1)
        + 4.0 * torch.roll(phase_rate, 1, 1)
        + 6.0 * phase_rate
        + 4.0 * torch.roll(phase_rate, -1, 1)
        + torch.roll(phase_rate, -2, 1)
    ) / 16.0
    # The speed-controlled schedules are already analytic and smooth.  Do not
    # blur their four requested speed levels before enforcing physical caps.
    phase_rate = torch.where(speed_controlled.unsqueeze(-1), phase_rate, smoothed_phase_rate)
    phase_rate = torch.minimum(phase_rate, normal_rate_cap)
    phase_rate = torch.minimum(phase_rate, orientation_rate_cap)
    phase_rate = torch.minimum(phase_rate, torch.full_like(phase_rate, limits.max_speed_mps) / phase_speed)

    # The lookup is sampled with a periodic C² interpolant.  A single local
    # trough would otherwise force a large, non-physical phase acceleration at
    # a neighbouring knot.  Use the tightest locally admissible rate as the
    # C² baseline for the closed cycle: it never violates a local ceiling and
    # is much safer than a discontinuous speed schedule.  Chirp retains its
    # diagnostic gradual speed variation while its peak remains below that
    # same admissible baseline.
    uniform_rate = phase_rate.amin(dim=1, keepdim=True)
    uniform_phase_rate = uniform_rate.expand_as(phase_rate)
    # A spatial helix needs q-dot proportional to 1/|dp/dq| to maintain a
    # constant path speed.  Sine commands intentionally use a uniform q-dot,
    # making their configured speed the peak of the smooth reversal cycle.
    phase_rate = torch.where((trajectory_type == SPATIAL_HELIX).unsqueeze(-1), phase_rate, uniform_phase_rate)
    chirp_progress = 0.5 + 0.25 * (1.0 - torch.cos(phase))
    phase_rate = torch.where((trajectory_type == 5).unsqueeze(-1), phase_rate * chirp_progress, phase_rate)
    # Figure-eights and chirps can approach a velocity-direction reversal for
    # arbitrary held-out phase offsets.  Keep a conservative margin for their
    # between-knot turn-rate/jerk peak; they are OOD diagnostics, not training
    # command families.
    sharp_ood = ((trajectory_type == 1) | (trajectory_type == 5)).unsqueeze(-1)
    phase_rate = torch.where(sharp_ood, 0.60 * phase_rate, phase_rate)
    phase_rate = torch.where((trajectory_type == 6).unsqueeze(-1), 0.95 * phase_rate, phase_rate)

    # A uniform final scale constrains total acceleration/jerk generated by
    # changes in the local schedule.  Three passes are sufficient because
    # acceleration and jerk scale approximately with rate^2 and rate^3.
    for _ in range(3):
        dt = phase_interval_duration(phase_rate)
        elapsed = torch.cat((torch.zeros(num_envs, 1, dtype=dtype, device=device), torch.cumsum(dt, dim=1)), dim=1)
        velocity, acceleration, jerk = _closed_time_derivatives(position, elapsed)
        max_speed = torch.linalg.vector_norm(velocity, dim=-1).amax(dim=1)
        max_acceleration = torch.linalg.vector_norm(acceleration, dim=-1).amax(dim=1)
        max_jerk = torch.linalg.vector_norm(jerk, dim=-1).amax(dim=1)
        max_orientation = (orientation_per_phase * phase_rate).amax(dim=1)
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
        scale = torch.minimum(
            scale,
            torch.full_like(scale, limits.max_orientation_rate_radps) / max_orientation.clamp_min(1.0e-6),
        )
        phase_rate = phase_rate * torch.clamp(scale, max=1.0).unsqueeze(-1)

    # The runtime sampler uses a C² quintic interpolant (rather than the
    # piecewise estimate above), so validate the actual sampled reference too.
    # This catches small interpolation and geometry-quadrature peaks before a
    # table is exposed to observations or rewards.
    for _ in range(4):
        dt = phase_interval_duration(phase_rate)
        elapsed = torch.cat((torch.zeros(num_envs, 1, dtype=dtype, device=device), torch.cumsum(dt, dim=1)), dim=1)
        phase_nodes = torch.cat((phase, torch.full((num_envs, 1), 2.0 * torch.pi, dtype=dtype, device=device)), dim=1)
        phase_rate_nodes = torch.cat((phase_rate, phase_rate[:, :1]), dim=1)
        phase_acceleration = phase_acceleration_for_rate(phase_rate, elapsed)
        phase_acceleration_nodes = torch.cat((phase_acceleration, phase_acceleration[:, :1]), dim=1)
        probe_tables = RetimedTrajectoryTables(
            phase=phase_nodes,
            elapsed_s=elapsed,
            phase_rate=phase_rate_nodes,
            phase_acceleration=phase_acceleration_nodes,
            requested_period_s=table_requested_period,
            effective_period_s=elapsed[:, -1],
            retimed=torch.ones(num_envs, dtype=torch.bool, device=device),
        )
        probe_time = elapsed[:, :-1] + 0.5 * dt
        probe_phase, probe_rate, probe_acceleration = sample_retimed_phase(probe_tables, probe_time)
        _, probe_velocity, probe_linear_acceleration, _ = evaluate_retimed_reference(
            trajectory_type,
            axis,
            amp_x,
            amp_y,
            amp_z,
            phase_x,
            phase_y,
            probe_phase,
            probe_rate,
            probe_acceleration,
            radius_min=radius_min,
            radius_max=radius_max,
            harmonic_ratio=harmonic_ratio,
        )
        previous_time = torch.cat((probe_time[:, -1:] - elapsed[:, -1:], probe_time[:, :-1]), dim=1)
        next_time = torch.cat((probe_time[:, 1:], probe_time[:, :1] + elapsed[:, -1:]), dim=1)
        probe_jerk = (
            torch.roll(probe_linear_acceleration, -1, 1) - torch.roll(probe_linear_acceleration, 1, 1)
        ) / (next_time - previous_time).clamp_min(1.0e-6).unsqueeze(-1)
        max_speed = torch.linalg.vector_norm(probe_velocity, dim=-1).amax(dim=1)
        max_acceleration = torch.linalg.vector_norm(probe_linear_acceleration, dim=-1).amax(dim=1)
        max_jerk = torch.linalg.vector_norm(probe_jerk, dim=-1).amax(dim=1)
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
        phase_rate = phase_rate * torch.clamp(scale, max=1.0).unsqueeze(-1)

    # Keep a small final numerical margin for finite-difference derivatives at
    # the policy rate; the racetrack receives extra margin at its two G² joins.
    conservative_phase_rate = 0.90 * phase_rate
    conservative_phase_rate = torch.where(
        (trajectory_type == 6).unsqueeze(-1), 0.90 * conservative_phase_rate, conservative_phase_rate
    )
    phase_rate = torch.where(speed_controlled.unsqueeze(-1), phase_rate, conservative_phase_rate)
    dt = phase_interval_duration(phase_rate)
    elapsed = torch.cat((torch.zeros(num_envs, 1, dtype=dtype, device=device), torch.cumsum(dt, dim=1)), dim=1)
    phase_nodes = torch.cat((phase, torch.full((num_envs, 1), 2.0 * torch.pi, dtype=dtype, device=device)), dim=1)
    phase_rate_nodes = torch.cat((phase_rate, phase_rate[:, :1]), dim=1)
    phase_acceleration = phase_acceleration_for_rate(phase_rate, elapsed)
    phase_acceleration_nodes = torch.cat((phase_acceleration, phase_acceleration[:, :1]), dim=1)
    effective_period = elapsed[:, -1]
    return RetimedTrajectoryTables(
        phase=phase_nodes,
        elapsed_s=elapsed,
        phase_rate=phase_rate_nodes,
        phase_acceleration=phase_acceleration_nodes,
        requested_period_s=table_requested_period,
        effective_period_s=effective_period,
        retimed=effective_period > table_requested_period * (1.0 + 1.0e-4),
    )


def sample_retimed_phase(tables: RetimedTrajectoryTables, elapsed_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert periodic elapsed time with a periodic C² quintic interpolant."""

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
    return q, rate, acceleration


def evaluate_retimed_reference(
    trajectory_type: torch.Tensor,
    axis: torch.Tensor,
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

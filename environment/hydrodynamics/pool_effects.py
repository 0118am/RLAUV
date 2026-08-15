"""Pool boundary, free-surface proximity, and linear sloshing models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class RectangularSloshingState:
    surface_z: torch.Tensor
    elevation_up_m: torch.Tensor
    orbital_velocity_w: torch.Tensor
    angular_frequencies_rad_s: torch.Tensor


@dataclass(frozen=True)
class _SloshingInputs:
    bounds: torch.Tensor
    frequencies: torch.Tensor
    modes: torch.Tensor
    amplitudes: torch.Tensor
    phases: torch.Tensor
    time: torch.Tensor


def rectangular_sloshing_mode_frequencies(
    pool_bounds: torch.Tensor | Sequence[float],
    water_depth: float,
    mode_numbers: torch.Tensor | Sequence[Sequence[int]],
    gravity_magnitude: float = 9.81,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Return finite-depth rectangular-tank sloshing frequencies."""

    bounds = torch.as_tensor(pool_bounds, dtype=dtype, device=device)
    if validate and bounds.numel() not in (4, 6):
        raise ValueError("pool_bounds must contain x/y bounds or full 3D box bounds.")
    bounds = bounds.reshape(-1)
    length_x = bounds[1] - bounds[0]
    length_y = bounds[3] - bounds[2]
    if validate and (length_x <= 0.0 or length_y <= 0.0):
        raise ValueError("pool_bounds x/y limits must be ordered min < max.")
    if validate and float(water_depth) <= 0.0:
        raise ValueError("water_depth must be positive.")
    if validate and float(gravity_magnitude) <= 0.0:
        raise ValueError("gravity_magnitude must be positive.")
    modes = torch.as_tensor(mode_numbers, device=bounds.device)
    if validate and (modes.ndim != 2 or modes.shape[1] != 2 or modes.shape[0] == 0):
        raise ValueError("mode_numbers must have shape (M, 2).")
    if validate and (torch.any(modes < 0) or torch.any(modes != torch.round(modes))):
        raise ValueError("mode_numbers must contain non-negative integers.")
    if validate and torch.any(torch.sum(modes, dim=-1) == 0):
        raise ValueError("Each sloshing mode must have m > 0 or n > 0.")
    modes = modes.to(dtype=bounds.dtype)
    kx = modes[:, 0] * torch.pi / length_x
    ky = modes[:, 1] * torch.pi / length_y
    wave_number = torch.sqrt(kx.square() + ky.square())
    return torch.sqrt(
        float(gravity_magnitude)
        * wave_number
        * torch.tanh(wave_number * float(water_depth))
    )


def _prepare_sloshing_inputs(
    positions: torch.Tensor,
    time_s: torch.Tensor | float,
    pool_bounds: torch.Tensor | Sequence[float],
    water_depth: float,
    mode_numbers: torch.Tensor | Sequence[Sequence[int]],
    amplitudes_m: torch.Tensor | Sequence[float],
    phases_rad: torch.Tensor | Sequence[float],
    gravity_magnitude: float,
    angular_frequencies_rad_s: torch.Tensor | None,
    validate: bool,
) -> _SloshingInputs:
    bounds = torch.as_tensor(
        pool_bounds,
        dtype=positions.dtype,
        device=positions.device,
    ).reshape(-1)
    if validate and bounds.numel() not in (4, 6):
        raise ValueError("pool_bounds must contain x/y bounds or full 3D box bounds.")
    frequencies = (
        rectangular_sloshing_mode_frequencies(
            bounds,
            water_depth,
            mode_numbers,
            gravity_magnitude,
            dtype=positions.dtype,
            device=positions.device,
            validate=validate,
        )
        if angular_frequencies_rad_s is None
        else torch.as_tensor(
            angular_frequencies_rad_s,
            dtype=positions.dtype,
            device=positions.device,
        ).reshape(-1)
    )
    modes = torch.as_tensor(mode_numbers, dtype=positions.dtype, device=positions.device)
    amplitudes = torch.as_tensor(
        amplitudes_m,
        dtype=positions.dtype,
        device=positions.device,
    ).reshape(-1)
    phases = torch.as_tensor(phases_rad, dtype=positions.dtype, device=positions.device).reshape(-1)
    mode_count = modes.shape[0]
    if validate and (amplitudes.shape != (mode_count,) or phases.shape != (mode_count,)):
        raise ValueError("amplitudes_m and phases_rad must contain one value per mode.")
    if validate and (not torch.all(torch.isfinite(amplitudes)) or torch.any(amplitudes < 0.0)):
        raise ValueError("amplitudes_m must contain finite non-negative values.")
    if validate and not torch.all(torch.isfinite(phases)):
        raise ValueError("phases_rad must contain only finite values.")
    time = torch.as_tensor(time_s, dtype=positions.dtype, device=positions.device)
    if time.ndim == 0:
        time = time.reshape(1).repeat(positions.shape[0])
    elif time.ndim == 2 and time.shape == (positions.shape[0], 1):
        time = time[:, 0]
    elif validate and (time.ndim != 1 or time.shape[0] != positions.shape[0]):
        raise ValueError("time_s must be scalar, shape (N,), or shape (N, 1).")
    if validate and not torch.all(torch.isfinite(time)):
        raise ValueError("time_s must contain only finite values.")
    return _SloshingInputs(bounds, frequencies, modes, amplitudes, phases, time)


def _sloshing_spatial_terms(
    positions: torch.Tensor,
    inputs: _SloshingInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length_x = inputs.bounds[1] - inputs.bounds[0]
    length_y = inputs.bounds[3] - inputs.bounds[2]
    kx = inputs.modes[:, 0] * torch.pi / length_x
    ky = inputs.modes[:, 1] * torch.pi / length_y
    wave_number = torch.sqrt(kx.square() + ky.square())
    x_phase = (positions[:, 0:1] - inputs.bounds[0]) * kx.reshape(1, -1)
    y_phase = (positions[:, 1:2] - inputs.bounds[2]) * ky.reshape(1, -1)
    spatial = torch.cos(x_phase) * torch.cos(y_phase)
    return kx, ky, wave_number, x_phase, y_phase, spatial


def _sloshing_orbital_velocity(
    inputs: _SloshingInputs,
    spatial_terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    sine_time: torch.Tensor,
    submerged_depth: torch.Tensor,
    water_depth: float,
    gravity_magnitude: float,
    depth_axis_sign: float,
) -> torch.Tensor:
    kx, ky, wave_number, x_phase, y_phase, spatial = spatial_terms
    denominator = wave_number.reshape(1, -1) * float(water_depth)
    numerator = wave_number.reshape(1, -1) * (float(water_depth) - submerged_depth)
    horizontal_depth_factor = _stable_cosh_ratio(numerator, denominator)
    vertical_depth_factor = _stable_sinh_over_cosh(numerator, denominator)
    potential_scale = (
        float(gravity_magnitude)
        * inputs.amplitudes
        / torch.clamp(inputs.frequencies, min=1.0e-8)
    ).reshape(1, -1)
    horizontal_common = potential_scale * horizontal_depth_factor * sine_time
    velocity_x = torch.sum(
        horizontal_common * kx.reshape(1, -1) * torch.sin(x_phase) * torch.cos(y_phase),
        dim=-1,
        keepdim=True,
    )
    velocity_y = torch.sum(
        horizontal_common * ky.reshape(1, -1) * torch.cos(x_phase) * torch.sin(y_phase),
        dim=-1,
        keepdim=True,
    )
    velocity_up = torch.sum(
        -potential_scale
        * wave_number.reshape(1, -1)
        * vertical_depth_factor
        * spatial
        * sine_time,
        dim=-1,
        keepdim=True,
    )
    return torch.cat((velocity_x, velocity_y, -float(depth_axis_sign) * velocity_up), dim=-1)


def calculate_rectangular_pool_sloshing_state(
    positions: torch.Tensor,
    time_s: torch.Tensor | float,
    base_surface_z: float,
    pool_bounds: torch.Tensor | Sequence[float],
    water_depth: float,
    mode_numbers: torch.Tensor | Sequence[Sequence[int]],
    amplitudes_m: torch.Tensor | Sequence[float],
    phases_rad: torch.Tensor | Sequence[float],
    gravity_magnitude: float = 9.81,
    depth_axis_sign: float = 1.0,
    *,
    angular_frequencies_rad_s: torch.Tensor | None = None,
    validate: bool = True,
) -> RectangularSloshingState:
    """Evaluate linear standing-wave modes in a rectangular pool.

    ``depth_axis_sign=-1`` is used by this repository's Isaac/PhysX z-up pool
    world; ``+1`` remains available for imported positive-down datasets.
    Amplitudes are positive upward free-surface elevations, independent of
    world-axis convention.
    """

    if validate and (positions.ndim != 2 or positions.shape[1] != 3):
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")
    if validate and not torch.all(torch.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    if validate and float(depth_axis_sign) not in (-1.0, 1.0):
        raise ValueError("depth_axis_sign must be -1 or 1.")
    inputs = _prepare_sloshing_inputs(
        positions,
        time_s,
        pool_bounds,
        water_depth,
        mode_numbers,
        amplitudes_m,
        phases_rad,
        gravity_magnitude,
        angular_frequencies_rad_s,
        validate,
    )
    spatial_terms = _sloshing_spatial_terms(positions, inputs)
    spatial = spatial_terms[-1]
    temporal_phase = (
        inputs.time.reshape(-1, 1) * inputs.frequencies.reshape(1, -1)
        + inputs.phases.reshape(1, -1)
    )
    cosine_time = torch.cos(temporal_phase)
    sine_time = torch.sin(temporal_phase)

    elevation_up = torch.sum(
        inputs.amplitudes.reshape(1, -1) * spatial * cosine_time,
        dim=-1,
        keepdim=True,
    )
    surface_z = float(base_surface_z) - float(depth_axis_sign) * elevation_up
    submerged_depth = float(depth_axis_sign) * (positions[:, 2:3] - float(base_surface_z))
    submerged_depth = torch.clamp(submerged_depth, min=0.0, max=float(water_depth))
    orbital_velocity_w = _sloshing_orbital_velocity(
        inputs,
        spatial_terms,
        sine_time,
        submerged_depth,
        water_depth,
        gravity_magnitude,
        depth_axis_sign,
    )
    return RectangularSloshingState(
        surface_z,
        elevation_up,
        orbital_velocity_w,
        inputs.frequencies,
    )


def _stable_cosh_ratio(numerator_argument: torch.Tensor, denominator_argument: torch.Tensor) -> torch.Tensor:
    return torch.exp(numerator_argument - denominator_argument) * (
        1.0 + torch.exp(-2.0 * numerator_argument)
    ) / (1.0 + torch.exp(-2.0 * denominator_argument))


def _stable_sinh_over_cosh(numerator_argument: torch.Tensor, denominator_argument: torch.Tensor) -> torch.Tensor:
    return torch.exp(numerator_argument - denominator_argument) * (
        1.0 - torch.exp(-2.0 * numerator_argument)
    ) / (1.0 + torch.exp(-2.0 * denominator_argument))


def calculate_pool_boundary_scales(
    positions: torch.Tensor,
    bounds: torch.Tensor | list[float] | tuple[float, ...],
    effect_distance: float,
    damping_scale_at_boundary: float,
    added_mass_scale_at_boundary: float,
    thrust_scale_at_boundary: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return damping, added-mass, and thrust scales from box-boundary proximity.

    ``positions`` are expressed in the pool-local/world-aligned frame.  Bounds
    are ``[x_min, x_max, y_min, y_max, z_min, z_max]`` in the same frame.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")

    bounds_tensor = torch.as_tensor(bounds, dtype=positions.dtype, device=positions.device)
    if bounds_tensor.shape != (6,):
        raise ValueError(f"bounds must be [x_min, x_max, y_min, y_max, z_min, z_max], got {tuple(bounds_tensor.shape)}.")

    distance = torch.stack(
        [
            positions[:, 0] - bounds_tensor[0],
            bounds_tensor[1] - positions[:, 0],
            positions[:, 1] - bounds_tensor[2],
            bounds_tensor[3] - positions[:, 1],
            positions[:, 2] - bounds_tensor[4],
            bounds_tensor[5] - positions[:, 2],
        ],
        dim=-1,
    )
    min_distance = torch.min(distance, dim=-1).values
    effect_distance = max(float(effect_distance), 1.0e-6)
    proximity = torch.clamp((effect_distance - min_distance) / effect_distance, min=0.0, max=1.0)
    proximity = proximity.reshape(-1, 1)

    damping_scale = 1.0 + proximity * (float(damping_scale_at_boundary) - 1.0)
    added_mass_scale = 1.0 + proximity * (float(added_mass_scale_at_boundary) - 1.0)
    thrust_scale = 1.0 + proximity * (float(thrust_scale_at_boundary) - 1.0)
    return damping_scale, added_mass_scale, torch.clamp(thrust_scale, min=0.0)


def calculate_free_surface_scales(
    positions: torch.Tensor,
    surface_z: torch.Tensor | float,
    effect_distance: float,
    heave_damping_scale_at_surface: float,
    roll_pitch_damping_scale_at_surface: float,
    added_mass_scale_at_surface: float,
    buoyancy_scale_at_surface: float,
    thrust_scale_at_surface: float,
    depth_axis_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hydrodynamic scales for proximity to a flat free surface.

    The model is local and empirical: within ``effect_distance`` of
    ``surface_z``, heave/roll/pitch damping and added mass are scaled while
    buoyancy and thrust may be reduced to approximate partial surfacing and
    thruster ventilation. ``depth_axis_sign`` selects whether increasing world
    z means increasing depth; points above the surface retain the surface
    scale instead of regaining the fully submerged model.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")

    effect_distance = max(float(effect_distance), 1.0e-6)
    surface = torch.as_tensor(surface_z, dtype=positions.dtype, device=positions.device)
    if surface.ndim == 0:
        surface = surface.reshape(1).repeat(positions.shape[0])
    elif surface.ndim == 2 and surface.shape == (positions.shape[0], 1):
        surface = surface[:, 0]
    elif surface.ndim != 1 or surface.shape[0] != positions.shape[0]:
        raise ValueError("surface_z must be scalar, shape (N,), or shape (N, 1).")
    if float(depth_axis_sign) not in (-1.0, 1.0):
        raise ValueError("depth_axis_sign must be -1 or 1.")
    depth_from_surface = float(depth_axis_sign) * (positions[:, 2] - surface)
    # Above the surface stays at the configured surface scale instead of
    # silently regaining full underwater physics outside the local band.
    proximity = torch.clamp((effect_distance - depth_from_surface) / effect_distance, min=0.0, max=1.0)
    proximity = proximity * proximity * (3.0 - 2.0 * proximity)
    proximity = proximity.reshape(-1, 1)

    damping_scale = torch.ones((positions.shape[0], 6), dtype=positions.dtype, device=positions.device)
    added_mass_scale = torch.ones_like(damping_scale)

    heave_damping = 1.0 + proximity[:, 0] * (float(heave_damping_scale_at_surface) - 1.0)
    roll_pitch_damping = 1.0 + proximity[:, 0] * (float(roll_pitch_damping_scale_at_surface) - 1.0)
    damping_scale[:, 2] = heave_damping
    damping_scale[:, 3] = roll_pitch_damping
    damping_scale[:, 4] = roll_pitch_damping

    added_mass = 1.0 + proximity[:, 0] * (float(added_mass_scale_at_surface) - 1.0)
    added_mass_scale[:, 2] = added_mass
    added_mass_scale[:, 3] = added_mass
    added_mass_scale[:, 4] = added_mass

    buoyancy_scale = 1.0 + proximity * (float(buoyancy_scale_at_surface) - 1.0)
    thrust_scale = 1.0 + proximity * (float(thrust_scale_at_surface) - 1.0)
    return damping_scale, added_mass_scale, torch.clamp(buoyancy_scale, min=0.0), torch.clamp(thrust_scale, min=0.0)

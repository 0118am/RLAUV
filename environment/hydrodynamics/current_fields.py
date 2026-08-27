"""Measured or prescribed pool water-current fields."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def calculate_periodic_water_current(
    time_s: torch.Tensor | float,
    amplitude_w: torch.Tensor | list[float] | tuple[float, float, float],
    period_s: torch.Tensor | list[float] | tuple[float, float, float],
    phase_rad: torch.Tensor | list[float] | tuple[float, float, float],
) -> torch.Tensor:
    """Evaluate ``A * sin(2*pi*t/T + phase)`` on world x/y/z."""

    amplitude = torch.as_tensor(amplitude_w)
    period = torch.as_tensor(period_s, dtype=amplitude.dtype, device=amplitude.device)
    phase = torch.as_tensor(phase_rad, dtype=amplitude.dtype, device=amplitude.device)
    time = torch.as_tensor(time_s, dtype=amplitude.dtype, device=amplitude.device)
    if amplitude.shape != (3,) or period.shape != (3,) or phase.shape != (3,):
        raise ValueError("Periodic current amplitude, period, and phase must each have shape (3,).")
    if not bool(torch.all(torch.isfinite(amplitude))) or not bool(torch.all(torch.isfinite(period))) or not bool(
        torch.all(torch.isfinite(phase))
    ):
        raise ValueError("Periodic current amplitude, period, and phase must be finite.")
    if bool(torch.any(period <= 0.0)):
        raise ValueError("Periodic current periods must be positive.")
    return amplitude * torch.sin(2.0 * torch.pi * time.unsqueeze(-1) / period + phase)


def calculate_trilinear_current_field(
    positions: torch.Tensor,
    bounds: torch.Tensor | list[float] | tuple[float, ...],
    grid_shape: torch.Tensor | list[int] | tuple[int, int, int],
    grid_values: torch.Tensor | list,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Interpolate a world-frame current field on a regular pool-local grid.

    ``bounds`` are ``[x_min, x_max, y_min, y_max, z_min, z_max]``.  Grid values
    may be shaped ``(nx, ny, nz, 3)`` or flattened as ``(nx * ny * nz, 3)`` in
    x-major, then y, then z order.
    """

    if validate and (positions.ndim != 2 or positions.shape[1] != 3):
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")

    bounds_tensor = torch.as_tensor(bounds, dtype=positions.dtype, device=positions.device)
    if validate and bounds_tensor.shape != (6,):
        raise ValueError(f"bounds must have shape (6,), got {tuple(bounds_tensor.shape)}.")
    if validate and not (
        bounds_tensor[0] < bounds_tensor[1]
        and bounds_tensor[2] < bounds_tensor[3]
        and bounds_tensor[4] < bounds_tensor[5]
    ):
        raise ValueError("bounds must be ordered as min < max on each axis.")

    shape_values = tuple(int(value) for value in grid_shape)
    if validate and (len(shape_values) != 3 or any(value <= 0 for value in shape_values)):
        raise ValueError(f"grid_shape must be three positive integers, got {shape_values}.")
    nx, ny, nz = shape_values

    values = torch.as_tensor(grid_values, dtype=positions.dtype, device=positions.device)
    if values.ndim == 2 and values.shape == (nx * ny * nz, 3):
        values = values.reshape(nx, ny, nz, 3)
    if validate and values.shape != (nx, ny, nz, 3):
        raise ValueError(
            f"grid_values must have shape ({nx}, {ny}, {nz}, 3) or ({nx * ny * nz}, 3), "
            f"got {tuple(values.shape)}."
        )

    coordinates = torch.stack(
        (
            2.0 * (positions[:, 0] - bounds_tensor[0]) / (bounds_tensor[1] - bounds_tensor[0]) - 1.0,
            2.0 * (positions[:, 1] - bounds_tensor[2]) / (bounds_tensor[3] - bounds_tensor[2]) - 1.0,
            2.0 * (positions[:, 2] - bounds_tensor[4]) / (bounds_tensor[5] - bounds_tensor[4]) - 1.0,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)
    # grid_sample expects (N, C, D, H, W) and xyz-normalized query points.
    field = values.permute(3, 2, 1, 0).unsqueeze(0)
    query = coordinates.reshape(1, -1, 1, 1, 3)
    sampled = functional.grid_sample(
        field,
        query,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0, 0].transpose(0, 1)

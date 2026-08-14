"""Gaussian sampling helpers for physically bounded domain parameters."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_bounded_normal(
    lower: float,
    upper: float,
    shape: Sequence[int],
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a normal centered in ``[lower, upper]`` with bounds at ±3σ."""

    lower_value = float(lower)
    upper_value = float(upper)
    if upper_value <= lower_value:
        return torch.full(tuple(shape), lower_value, dtype=dtype, device=device)
    mean = 0.5 * (lower_value + upper_value)
    standard_deviation = (upper_value - lower_value) / 6.0
    sample = mean + standard_deviation * torch.randn(tuple(shape), dtype=dtype, device=device)
    return torch.clamp(sample, min=lower_value, max=upper_value)


def sample_bounded_normal_integer(
    lower: int,
    upper: int,
    shape: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Sample rounded bounded-normal integer parameters."""

    return torch.round(
        sample_bounded_normal(lower, upper, shape, device)
    ).to(dtype=torch.long)


def sample_symmetric_bounded_normal(
    half_range: float | torch.Tensor,
    shape: Sequence[int],
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ``N(0, (half_range/3)^2)`` and clip to ±``half_range``."""

    bound = torch.as_tensor(half_range, dtype=dtype, device=device)
    sample = torch.randn(tuple(shape), dtype=dtype, device=device) * bound / 3.0
    return torch.clamp(sample, min=-bound, max=bound)


def sample_isotropic_bounded_normal(
    radius: float,
    count: int,
    dimensions: int,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample isotropic Gaussian vectors and project rare tails to a radius."""

    radius_value = float(radius)
    if radius_value <= 0.0:
        return torch.zeros((count, dimensions), dtype=dtype, device=device)
    sample = torch.randn((count, dimensions), dtype=dtype, device=device) * radius_value / 3.0
    norm = torch.linalg.vector_norm(sample, dim=-1, keepdim=True)
    projection = torch.clamp(radius_value / torch.clamp(norm, min=1.0e-8), max=1.0)
    return sample * projection

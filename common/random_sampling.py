"""Gaussian sampling helpers for physically bounded parameters."""

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
    """Sample a normal centered in ``[lower, upper]`` and truncated at +/-3 sigma."""

    lower_value = float(lower)
    upper_value = float(upper)
    if upper_value < lower_value:
        raise ValueError("upper must be greater than or equal to lower")
    if upper_value == lower_value:
        return torch.full(tuple(shape), lower_value, dtype=dtype, device=device)
    mean = 0.5 * (lower_value + upper_value)
    standard_deviation = (upper_value - lower_value) / 6.0
    return torch.nn.init.trunc_normal_(
        torch.empty(tuple(shape), dtype=dtype, device=device),
        mean=mean,
        std=standard_deviation,
        a=lower_value,
        b=upper_value,
    )


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
    half_range: float,
    shape: Sequence[int],
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ``N(0, (half_range/3)^2)`` truncated at +/-``half_range``."""

    bound = float(half_range)
    if bound < 0.0:
        raise ValueError("half_range must be non-negative")
    if bound == 0.0:
        return torch.zeros(tuple(shape), dtype=dtype, device=device)
    return torch.nn.init.trunc_normal_(
        torch.empty(tuple(shape), dtype=dtype, device=device),
        mean=0.0,
        std=bound / 3.0,
        a=-bound,
        b=bound,
    )


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

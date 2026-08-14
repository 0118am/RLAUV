"""Regression tests for Gaussian domain-parameter sampling."""

from __future__ import annotations

import torch

from environment.profiles.random_sampling import (
    sample_bounded_normal,
    sample_bounded_normal_integer,
    sample_isotropic_bounded_normal,
    sample_symmetric_bounded_normal,
)


def test_bounded_normal_uses_midpoint_mean_and_three_sigma_bounds() -> None:
    torch.manual_seed(421)
    samples = sample_bounded_normal(-3.0, 3.0, (200_000,), "cpu")

    assert abs(float(samples.mean())) < 0.015
    assert abs(float(samples.std(unbiased=False)) - 1.0) < 0.015
    assert float(samples.min()) >= -3.0
    assert float(samples.max()) <= 3.0


def test_degenerate_normal_range_is_deterministic() -> None:
    samples = sample_bounded_normal(2.5, 2.5, (32,), "cpu")

    assert torch.equal(samples, torch.full((32,), 2.5))


def test_symmetric_normal_supports_per_parameter_bounds() -> None:
    torch.manual_seed(422)
    bounds = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    samples = sample_symmetric_bounded_normal(bounds, bounds.shape, "cpu")

    assert torch.all(torch.abs(samples) <= bounds)


def test_isotropic_normal_respects_vector_radius() -> None:
    torch.manual_seed(423)
    samples = sample_isotropic_bounded_normal(0.2, 100_000, 3, "cpu")

    assert torch.all(torch.linalg.vector_norm(samples, dim=1) <= 0.200001)
    assert torch.all(torch.abs(samples.mean(dim=0)) < 0.001)


def test_integer_normal_is_centered_and_bounded() -> None:
    torch.manual_seed(424)
    samples = sample_bounded_normal_integer(0, 6, (100_000,), "cpu")

    assert samples.dtype == torch.long
    assert abs(float(samples.float().mean()) - 3.0) < 0.02
    assert int(samples.min()) == 0
    assert int(samples.max()) == 6

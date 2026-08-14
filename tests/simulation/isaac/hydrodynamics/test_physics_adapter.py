"""Explicit high-order wrench manager regression checks."""

import torch

from environment.hydrodynamics.models import HydrodynamicForceModels
from simulation.isaac.physics_adapter import PhysxHydrodynamicWrenchCfg, PhysxHydrodynamicWrenchManager


def test_physx_manager_applies_and_resets_a_cached_residual_wrench() -> None:
    model = HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    manager = PhysxHydrodynamicWrenchManager(
        model,
        PhysxHydrodynamicWrenchCfg(enabled=True, base_scale=1.0),
        added_mass_factor=torch.zeros(6),
        linear_damping_factor=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        quadratic_damping_factor=torch.zeros(6),
        cubic_damping_factor=torch.zeros(6),
    )
    wrench = manager.compute_wrench(torch.tensor([[1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0]]), None, 0.0)
    assert torch.all(wrench[:, 0] < 0.0)
    manager.reset(torch.tensor([1]))
    assert torch.all(manager.last_wrench_b[1] == 0.0)
    manager.set_enabled(False)
    assert torch.all(manager.compute_wrench(torch.ones(2, 6), None, 0.0) == 0.0)

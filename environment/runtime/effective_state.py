"""Environment-owned hydrodynamic state records and tensor conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

def _nominal_hydro_coeff_tensor(values, device: torch.device, name: str) -> torch.Tensor:
    """Normalize 6-DOF hydrodynamic coefficients to a single-env tensor."""

    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1 and tensor.shape[0] == 6:
        return tensor.reshape(1, 6)
    if tensor.ndim == 2:
        if tensor.shape == (6, 6):
            return tensor.reshape(1, 6, 6)
        if tensor.shape == (1, 6):
            return tensor
    if tensor.ndim == 3 and tensor.shape == (1, 6, 6):
        return tensor
    raise ValueError(f"{name} must be a 6-vector or 6x6 matrix, got shape {tuple(tensor.shape)}.")


def _repeat_hydro_coeff_for_envs(nominal: torch.Tensor, num_envs: int) -> torch.Tensor:
    repeats = (num_envs,) + tuple(1 for _ in range(nominal.ndim - 1))
    return nominal.repeat(repeats)


@dataclass
class EffectiveHydrodynamicState:
    """Effective quantities shared by the force path and asymmetric Critic."""

    water_current_w: torch.Tensor
    relative_velocity_b: torch.Tensor
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    added_mass: torch.Tensor
    buoyancy_scale: torch.Tensor
    thruster_scale: torch.Tensor


@dataclass(frozen=True)
class BodyKinematics:
    """Simulator state copied at the assembly boundary for pure runtime models."""

    root_position_w: torch.Tensor
    root_position_local_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_linear_velocity_w: torch.Tensor
    root_linear_velocity_b: torch.Tensor
    root_angular_velocity_b: torch.Tensor
    scene_origins_w: torch.Tensor

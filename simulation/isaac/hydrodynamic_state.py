"""Hydrodynamic state records and optional residual-wrench manager."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from environment.hydrodynamics.models import HydrodynamicForceModels

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
class PhysxHydrodynamicWrenchCfg:
    """Configuration for the optional high-order wrench sent to PhysX."""

    enabled: bool = False
    base_scale: float = 1.0
    modulation_amplitude: float = 0.0
    modulation_frequency_hz: float = 0.0
    modulation_phase_rad: float = 0.0


class PhysxHydrodynamicWrenchManager:
    """Own the state of a residual fluid wrench applied by the adapter."""

    def __init__(
        self,
        force_model: HydrodynamicForceModels,
        cfg: PhysxHydrodynamicWrenchCfg,
        *,
        added_mass_factor: torch.Tensor,
        linear_damping_factor: torch.Tensor,
        quadratic_damping_factor: torch.Tensor,
        cubic_damping_factor: torch.Tensor,
    ) -> None:
        self.force_model = force_model
        self.cfg = cfg
        self.added_mass_factor = added_mass_factor
        self.linear_damping_factor = linear_damping_factor
        self.quadratic_damping_factor = quadratic_damping_factor
        self.cubic_damping_factor = cubic_damping_factor
        self.enabled = bool(cfg.enabled)
        self._manual_scale = torch.ones(
            force_model.num_envs,
            1,
            dtype=torch.float32,
            device=force_model.device,
        )
        self.last_scale = torch.zeros_like(self._manual_scale)
        self.last_wrench_b = torch.zeros(
            force_model.num_envs,
            6,
            dtype=torch.float32,
            device=force_model.device,
        )

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.enabled and not enabled:
            self.last_scale.zero_()
            self.last_wrench_b.zero_()
        self.enabled = enabled

    def set_environment_scale(self, env_ids: torch.Tensor, scale: torch.Tensor | float) -> None:
        values = torch.as_tensor(
            scale,
            dtype=self._manual_scale.dtype,
            device=self._manual_scale.device,
        )
        self._manual_scale[env_ids] = values.reshape(-1, 1) if values.numel() > 1 else values

    def reset(self, env_ids: torch.Tensor) -> None:
        self._manual_scale[env_ids] = 1.0
        self.last_scale[env_ids] = 0.0
        self.last_wrench_b[env_ids] = 0.0

    def scale_at(self, physics_time_s: float | torch.Tensor) -> torch.Tensor:
        """Return the active non-negative gain for every environment."""

        time = torch.as_tensor(
            physics_time_s,
            dtype=self.last_scale.dtype,
            device=self.last_scale.device,
        )
        phase = (
            2.0 * torch.pi * self.cfg.modulation_frequency_hz * time
            + self.cfg.modulation_phase_rad
        )
        scheduled = self.cfg.base_scale + self.cfg.modulation_amplitude * torch.sin(phase)
        return (self._manual_scale * torch.clamp(scheduled, min=0.0)).reshape(-1, 1)

    def compute_wrench(
        self,
        nu_relative_b: torch.Tensor,
        relative_acceleration_b: torch.Tensor | None,
        physics_time_s: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate and cache a 6-D body wrench for the PhysX force path."""

        if not self.enabled:
            return self.last_wrench_b
        self.last_scale[:] = self.scale_at(physics_time_s)
        residual = self.force_model.calculate_high_order_residual_wrench(
            nu_relative_b,
            relative_acceleration_b,
            added_mass_factor=self.added_mass_factor,
            linear_damping_factor=self.linear_damping_factor,
            quadratic_damping_factor=self.quadratic_damping_factor,
            cubic_damping_factor=self.cubic_damping_factor,
        )
        self.last_wrench_b[:] = residual * self.last_scale
        return self.last_wrench_b


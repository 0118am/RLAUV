"""Explicit manager for high-order hydrodynamic wrenches applied to PhysX.

PhysX remains responsible for rigid-body integration.  This manager owns only
the residual fluid wrench, exposes its time-varying gain, and returns a body
wrench for the environment's permanent PhysX wrench composer.  Keeping this
stateful part outside the nominal Fossen function makes dynamic-model changes
auditable and prevents an implicit, per-call parameter mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from environment.hydrodynamics.models import HydrodynamicForceModels


@dataclass(frozen=True)
class PhysxHydrodynamicWrenchCfg:
    """Configuration of the explicit high-order wrench overlay.

    ``modulation_amplitude`` is clamped to keep the gain non-negative.  A
    frequency of zero gives a static calibrated residual.  The manager uses
    the existing PSD factor parameterisation, so its damping component stays
    passive for every modulation phase.
    """

    enabled: bool = False
    base_scale: float = 1.0
    modulation_amplitude: float = 0.0
    modulation_frequency_hz: float = 0.0
    modulation_phase_rad: float = 0.0


class PhysxHydrodynamicWrenchManager:
    """Vectorized state and API for a residual wrench sent to PhysX."""

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
            force_model.num_envs, 1, dtype=torch.float32, device=force_model.device
        )
        self.last_scale = torch.zeros_like(self._manual_scale)
        self.last_wrench_b = torch.zeros(
            force_model.num_envs, 6, dtype=torch.float32, device=force_model.device
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the residual without changing its calibration."""

        self.enabled = bool(enabled)

    def set_environment_scale(self, env_ids: torch.Tensor, scale: torch.Tensor | float) -> None:
        """Set an explicit per-environment scale for a curriculum or ablation."""

        values = torch.as_tensor(scale, dtype=self._manual_scale.dtype, device=self._manual_scale.device)
        self._manual_scale[env_ids] = values.reshape(-1, 1) if values.numel() > 1 else values

    def reset(self, env_ids: torch.Tensor) -> None:
        self._manual_scale[env_ids] = 1.0
        self.last_scale[env_ids] = 0.0
        self.last_wrench_b[env_ids] = 0.0

    def scale_at(self, physics_time_s: float | torch.Tensor) -> torch.Tensor:
        """Return the active non-negative gain for every PhysX environment."""

        time = torch.as_tensor(physics_time_s, dtype=self.last_scale.dtype, device=self.last_scale.device)
        phase = 2.0 * torch.pi * self.cfg.modulation_frequency_hz * time + self.cfg.modulation_phase_rad
        scheduled = self.cfg.base_scale + self.cfg.modulation_amplitude * torch.sin(phase)
        return (self._manual_scale * torch.clamp(scheduled, min=0.0)).reshape(-1, 1)

    def compute_wrench(
        self,
        nu_relative_b: torch.Tensor,
        relative_acceleration_b: torch.Tensor | None,
        physics_time_s: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate and cache a 6-D body wrench to add to the PhysX command."""

        if not self.enabled:
            self.last_scale.zero_()
            self.last_wrench_b.zero_()
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

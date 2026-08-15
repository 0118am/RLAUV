"""Battery, inflow, wake-interaction, and reaction-torque effects."""

from __future__ import annotations

import torch

def calculate_voltage_thrust_scale(
    voltage: torch.Tensor | float,
    nominal_voltage: float,
    exponent: float = 2.0,
    min_voltage: float = 0.0,
) -> torch.Tensor:
    """Return thrust scaling from battery voltage relative to nominal voltage."""

    if isinstance(voltage, torch.Tensor):
        voltage_tensor = voltage.to(dtype=torch.float32)
    else:
        voltage_tensor = torch.tensor(voltage, dtype=torch.float32)
    nominal = max(float(nominal_voltage), 1.0e-6)
    voltage_tensor = torch.clamp(voltage_tensor, min=float(min_voltage))
    return torch.pow(torch.clamp(voltage_tensor / nominal, min=0.0), float(exponent))


def calculate_axial_inflow_thrust_scale(
    axial_inflow_speed: torch.Tensor,
    loss_coefficient: float,
    reference_speed: float,
    min_scale: float,
) -> torch.Tensor:
    """Return thrust-loss scale from positive axial inflow speed.

    Positive axial inflow means water is moving into the propeller along its
    thrust axis.  The simple model reduces thrust with a quadratic factor and
    clamps to ``min_scale``; negative axial inflow never boosts thrust.
    """

    if loss_coefficient <= 0.0:
        return torch.ones_like(axial_inflow_speed)
    reference = max(float(reference_speed), 1.0e-6)
    inflow_ratio = torch.clamp(axial_inflow_speed, min=0.0) / reference
    scale = 1.0 - float(loss_coefficient) * inflow_ratio * inflow_ratio
    return torch.clamp(scale, min=float(min_scale), max=1.0)


def _wake_loss_coefficient(
    loss_coefficient: torch.Tensor | float,
    thrust: torch.Tensor,
) -> torch.Tensor:
    coefficient = torch.as_tensor(
        loss_coefficient,
        dtype=thrust.dtype,
        device=thrust.device,
    )
    if coefficient.ndim == 0:
        return coefficient.reshape(1, 1, 1)
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1, 1, 1)
    elif coefficient.ndim == 2 and coefficient.shape[1] == 1:
        coefficient = coefficient.reshape(-1, 1, 1)
    else:
        raise ValueError("loss_coefficient must be a scalar or per-env tensor.")
    if coefficient.shape[0] not in (1, thrust.shape[0]):
        raise ValueError(
            "loss_coefficient must be scalar or have one value per environment, got "
            f"{tuple(coefficient.shape)} for {thrust.shape[0]} environments."
        )
    return coefficient


def _validated_wake_geometry_inputs(
    thruster_positions_b: torch.Tensor,
    thruster_axes_b: torch.Tensor,
    thrust: torch.Tensor,
) -> torch.Tensor:
    if thruster_positions_b.ndim == 2:
        thruster_positions_b = thruster_positions_b.reshape(1, *thruster_positions_b.shape).repeat(
            thrust.shape[0],
            1,
            1,
        )
    if thruster_positions_b.shape != thruster_axes_b.shape:
        raise ValueError(
            "thruster_positions_b and thruster_axes_b must have matching "
            f"(num_envs, num_thrusters, 3) shapes, got {tuple(thruster_positions_b.shape)} "
            f"and {tuple(thruster_axes_b.shape)}."
        )
    if thrust.shape != thruster_axes_b.shape[:2]:
        raise ValueError(
            f"thrust must have shape {tuple(thruster_axes_b.shape[:2])}, got {tuple(thrust.shape)}."
        )
    return thruster_positions_b


def _wake_mask_and_profile(
    thruster_positions_b: torch.Tensor,
    thruster_axes_b: torch.Tensor,
    thrust: torch.Tensor,
    wake_length: float,
    wake_radius: float,
    expansion_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative_position = thruster_positions_b.unsqueeze(1) - thruster_positions_b.unsqueeze(2)
    direction = torch.sign(thrust).unsqueeze(-1) * thruster_axes_b
    axial_distance = torch.sum(relative_position * direction.unsqueeze(2), dim=-1)
    radial_vector = relative_position - axial_distance.unsqueeze(-1) * direction.unsqueeze(2)
    radial_distance = torch.linalg.norm(radial_vector, dim=-1)
    radius_at_target = float(wake_radius) + torch.clamp(axial_distance, min=0.0) * max(
        float(expansion_rate),
        0.0,
    )
    in_wake = (
        (axial_distance > 0.0)
        & (axial_distance <= float(wake_length))
        & (radial_distance <= radius_at_target)
        & (torch.abs(thrust).unsqueeze(-1) > 1.0e-6)
    )
    identity = torch.eye(thrust.shape[1], dtype=torch.bool, device=thrust.device).reshape(
        1,
        thrust.shape[1],
        thrust.shape[1],
    )
    radial_ratio = radial_distance / torch.clamp(radius_at_target, min=1.0e-6)
    axial_fade = 1.0 - torch.clamp(axial_distance / float(wake_length), min=0.0, max=1.0)
    return in_wake & ~identity, torch.exp(-(radial_ratio * radial_ratio)) * axial_fade


def _wake_reference_thrust(
    reference_thrust: torch.Tensor | float | None,
    thrust: torch.Tensor,
) -> torch.Tensor:
    if reference_thrust is None:
        return torch.clamp(torch.max(torch.abs(thrust), dim=1, keepdim=True).values, min=1.0e-6)
    reference = torch.as_tensor(reference_thrust, dtype=thrust.dtype, device=thrust.device)
    if reference.ndim == 0:
        reference = reference.reshape(1, 1)
    elif reference.ndim == 1:
        if reference.shape[0] == thrust.shape[0]:
            reference = reference.reshape(thrust.shape[0], 1)
        elif reference.shape[0] == thrust.shape[1]:
            reference = reference.reshape(1, thrust.shape[1])
    return torch.clamp(reference, min=1.0e-6)


def calculate_thruster_wake_interaction_scale(
    thruster_positions_b: torch.Tensor,
    thruster_axes_b: torch.Tensor,
    thrust: torch.Tensor,
    wake_length: float,
    wake_radius: float,
    loss_coefficient: torch.Tensor | float,
    expansion_rate: float = 0.0,
    min_scale: float = 0.7,
    reference_thrust: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return thrust scales from simplified propeller wake interference.

    A source thruster sheds a wake in its signed thrust direction.  Any other
    thruster inside that expanding cylinder/cone receives a thrust-loss scale.
    This is a compact empirical model, not a blade-resolved propeller solver.
    """

    if wake_length <= 0.0 or wake_radius <= 0.0:
        return torch.ones_like(thrust)
    raw_loss_coefficient = torch.as_tensor(
        loss_coefficient,
        dtype=thrust.dtype,
        device=thrust.device,
    )
    if raw_loss_coefficient.ndim == 0 and float(raw_loss_coefficient.item()) <= 0.0:
        return torch.ones_like(thrust)
    loss_coefficient_tensor = _wake_loss_coefficient(loss_coefficient, thrust)
    thruster_positions_b = _validated_wake_geometry_inputs(
        thruster_positions_b,
        thruster_axes_b,
        thrust,
    )
    in_wake, wake_profile = _wake_mask_and_profile(
        thruster_positions_b,
        thruster_axes_b,
        thrust,
        wake_length,
        wake_radius,
        expansion_rate,
    )
    reference = _wake_reference_thrust(reference_thrust, thrust)
    source_strength = torch.clamp(torch.abs(thrust) / reference, min=0.0, max=1.0)
    loss = torch.clamp(loss_coefficient_tensor, min=0.0) * source_strength.unsqueeze(-1) * wake_profile
    loss = torch.where(in_wake, loss, torch.zeros_like(loss))

    total_loss = torch.sum(loss, dim=1)
    return torch.clamp(1.0 - total_loss, min=float(min_scale), max=1.0)

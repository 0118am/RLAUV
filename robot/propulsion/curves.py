"""Measured T60 command-to-force curves and wrench reduction."""

from __future__ import annotations

import torch

from robot.dynamics.parameters import AUV

def get_thruster_positions(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the canonical T1...T8 COM-relative installation centers."""

    return torch.as_tensor(AUV.thruster_positions_body_m, dtype=dtype, device=device)


def normalized_command_to_pwm_us(command: torch.Tensor) -> torch.Tensor:
    """Map an already bounded normalized command to physical PWM."""

    return AUV.thruster_pwm_center_us + AUV.thruster_pwm_half_range_us * command


def thruster_body_forces_from_pwm_us(
    pwm_us: torch.Tensor,
    coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the canonical absolute-PWM to FLU vector-force curve.

    ``pwm_us`` has shape ``(..., 8)`` and must already be in 1300...1700 us.
    The coefficient layout is ``(8, 4, 3)`` with rows
    ``(a_negative, b_negative, a_positive, b_positive)`` and final components
    ``(Fx, Fy, Fz)``.  PWM values in the inclusive 1475...1525 us dead zone
    produce an exact zero vector.
    """

    if pwm_us.shape[-1] != len(AUV.thruster_labels):
        raise ValueError(f"pwm_us must have {len(AUV.thruster_labels)} T1...T8 values.")
    coeff = (
        torch.as_tensor(AUV.thruster_force_curve_coefficients, dtype=pwm_us.dtype, device=pwm_us.device)
        if coefficients is None
        else coefficients.to(dtype=pwm_us.dtype, device=pwm_us.device)
    )
    if coeff.shape != (len(AUV.thruster_labels), 4, 3):
        raise ValueError("coefficients must have shape (8, 4, 3).")

    offset_us = pwm_us - AUV.thruster_pwm_center_us
    q_negative = torch.clamp(-offset_us - AUV.thruster_pwm_deadband_us, min=0.0).unsqueeze(-1)
    q_positive = torch.clamp(offset_us - AUV.thruster_pwm_deadband_us, min=0.0).unsqueeze(-1)
    return (
        coeff[:, 0, :] * q_negative.square()
        + coeff[:, 1, :] * q_negative
        + coeff[:, 2, :] * q_positive.square()
        + coeff[:, 3, :] * q_positive
    )


def measured_thruster_body_forces(
    command: torch.Tensor,
    coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the measured FLU force vector of every T60.

    ``command`` has shape ``(..., 8)`` and the result has shape ``(..., 8, 3)``.
    Coefficients are stored in physical-PWM branch order. All measured FLU
    components and their signs are preserved together.
    """

    return thruster_body_forces_from_pwm_us(
        normalized_command_to_pwm_us(command),
        coefficients,
    )


def measured_thruster_force_jacobian(
    command: torch.Tensor,
    coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return d(Fx,Fy,Fz)/d(action) with shape ``(..., 8, 3)``."""

    if command.shape[-1] != len(AUV.thruster_labels):
        raise ValueError(f"command must have {len(AUV.thruster_labels)} T1...T8 values.")
    coeff = (
        torch.as_tensor(AUV.thruster_force_curve_coefficients, dtype=command.dtype, device=command.device)
        if coefficients is None
        else coefficients.to(dtype=command.dtype, device=command.device)
    )
    if coeff.shape != (len(AUV.thruster_labels), 4, 3):
        raise ValueError("coefficients must have shape (8, 4, 3).")
    offset_us = AUV.thruster_pwm_half_range_us * command
    q_positive = torch.clamp(offset_us - AUV.thruster_pwm_deadband_us, min=0.0).unsqueeze(-1)
    q_negative = torch.clamp(-offset_us - AUV.thruster_pwm_deadband_us, min=0.0).unsqueeze(-1)
    positive_slope = AUV.thruster_pwm_half_range_us * (
        2.0 * coeff[:, 2, :] * q_positive + coeff[:, 3, :]
    )
    negative_slope = -AUV.thruster_pwm_half_range_us * (
        2.0 * coeff[:, 0, :] * q_negative + coeff[:, 1, :]
    )
    branch_slope = torch.where(
        (offset_us > AUV.thruster_pwm_deadband_us).unsqueeze(-1),
        positive_slope,
        torch.where(
            (offset_us < -AUV.thruster_pwm_deadband_us).unsqueeze(-1),
            negative_slope,
            torch.zeros_like(positive_slope),
        ),
    )
    return branch_slope


def reduce_point_forces_to_wrench(positions_b: torch.Tensor, forces_b: torch.Tensor) -> torch.Tensor:
    """Reduce T1...T8 body-frame point forces to one COM wrench."""

    if positions_b.shape[-2:] != forces_b.shape[-2:] or forces_b.shape[-2:] != (len(AUV.thruster_labels), 3):
        raise ValueError("positions_b and forces_b must end in matching (8, 3) dimensions.")
    if positions_b.ndim < forces_b.ndim:
        positions_b = positions_b.reshape((1,) * (forces_b.ndim - positions_b.ndim) + positions_b.shape)
    positions_b = positions_b.expand_as(forces_b)
    force_b = forces_b.sum(dim=-2)
    torque_b = torch.cross(positions_b, forces_b, dim=-1).sum(dim=-2)
    return torch.cat((force_b, torque_b), dim=-1)

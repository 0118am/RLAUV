"""Simplified tether force model for pool experiments."""

from __future__ import annotations

import torch


def calculate_tether_wrench(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_linvel_w: torch.Tensor,
    water_current_w: torch.Tensor,
    anchor_pos_w: torch.Tensor | list[float] | tuple[float, ...],
    attach_offset_b: torch.Tensor | list[float] | tuple[float, ...],
    slack_length: torch.Tensor | float,
    stiffness: float,
    damping: float,
    drag_coeff: float,
    quat_conjugate_fn,
    quat_apply_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body-frame force/torque from a slack tether.

    The model is intentionally compact: a one-segment tether applies tension
    only when stretched beyond ``slack_length`` and adds a quadratic drag term
    opposite relative water velocity.  It is not a cable dynamics solver, but
    it captures the dominant pool-test bias from a safety line.
    """

    anchor = torch.as_tensor(anchor_pos_w, dtype=body_pos_w.dtype, device=body_pos_w.device)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, 3).repeat(body_pos_w.shape[0], 1)
    attach_offset = torch.as_tensor(attach_offset_b, dtype=body_pos_w.dtype, device=body_pos_w.device)
    if attach_offset.ndim == 1:
        attach_offset = attach_offset.reshape(1, 3).repeat(body_pos_w.shape[0], 1)

    attach_pos_w = body_pos_w + quat_apply_fn(body_quat_w, attach_offset)
    tether_vec_w = anchor - attach_pos_w
    tether_length = torch.linalg.norm(tether_vec_w, dim=-1, keepdim=True)
    direction_w = tether_vec_w / torch.clamp(tether_length, min=1.0e-6)

    slack = _as_env_column(slack_length, body_pos_w, "slack_length")
    stretch = torch.clamp(tether_length - slack, min=0.0)
    relative_vel_w = body_linvel_w - water_current_w
    vel_along_tether = torch.sum(relative_vel_w * direction_w, dim=-1, keepdim=True)
    tension = float(stiffness) * stretch + float(damping) * torch.clamp(-vel_along_tether, min=0.0)
    spring_force_w = tension * direction_w

    drag_force_w = torch.zeros_like(spring_force_w)
    if drag_coeff > 0.0:
        rel_speed = torch.linalg.norm(relative_vel_w, dim=-1, keepdim=True)
        drag_force_w = -float(drag_coeff) * rel_speed * relative_vel_w

    force_w = spring_force_w + drag_force_w
    force_b = quat_apply_fn(quat_conjugate_fn(body_quat_w), force_w)
    torque_b = torch.cross(attach_offset, force_b, dim=-1)
    return force_b, torque_b


def calculate_multisegment_tether_wrench(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_linvel_w: torch.Tensor,
    water_current_w: torch.Tensor,
    anchor_pos_w: torch.Tensor | list[float] | tuple[float, ...],
    attach_offset_b: torch.Tensor | list[float] | tuple[float, ...],
    slack_length: torch.Tensor | float,
    stiffness: float,
    damping: float,
    drag_coeff: float,
    num_segments: int,
    segment_diameter: float,
    segment_density: float,
    segment_buoyancy_density: float,
    gravity_w: torch.Tensor | list[float] | tuple[float, ...],
    quat_conjugate_fn,
    quat_apply_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body-frame wrench from a quasi-static multi-segment tether.

    This helper approximates the cable as straight segments between the anchor
    and vehicle attachment point.  It preserves the same end-to-end stretch
    spring as ``calculate_tether_wrench`` while adding distributed apparent
    weight/buoyancy and quadratic drag accumulated along the line.  The body
    receives the negative of the summed external loads plus the axial tension at
    the attachment point.
    """

    if num_segments <= 1:
        return calculate_tether_wrench(
            body_pos_w,
            body_quat_w,
            body_linvel_w,
            water_current_w,
            anchor_pos_w,
            attach_offset_b,
            slack_length,
            stiffness,
            damping,
            drag_coeff,
            quat_conjugate_fn,
            quat_apply_fn,
        )

    anchor = torch.as_tensor(anchor_pos_w, dtype=body_pos_w.dtype, device=body_pos_w.device)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, 3).repeat(body_pos_w.shape[0], 1)
    attach_offset = torch.as_tensor(attach_offset_b, dtype=body_pos_w.dtype, device=body_pos_w.device)
    if attach_offset.ndim == 1:
        attach_offset = attach_offset.reshape(1, 3).repeat(body_pos_w.shape[0], 1)
    gravity = torch.as_tensor(gravity_w, dtype=body_pos_w.dtype, device=body_pos_w.device)
    if gravity.ndim == 1:
        gravity = gravity.reshape(1, 3).repeat(body_pos_w.shape[0], 1)

    attach_pos_w = body_pos_w + quat_apply_fn(body_quat_w, attach_offset)
    cable_vec_w = anchor - attach_pos_w
    cable_length = torch.linalg.norm(cable_vec_w, dim=-1, keepdim=True)
    direction_to_anchor_w = cable_vec_w / torch.clamp(cable_length, min=1.0e-6)

    slack = _as_env_column(slack_length, body_pos_w, "slack_length")
    stretch = torch.clamp(cable_length - slack, min=0.0)
    relative_vel_w = body_linvel_w - water_current_w
    vel_along_tether = torch.sum(relative_vel_w * direction_to_anchor_w, dim=-1, keepdim=True)
    tension = float(stiffness) * stretch + float(damping) * torch.clamp(-vel_along_tether, min=0.0)
    axial_force_w = tension * direction_to_anchor_w

    segment_length = cable_length / float(num_segments)
    segment_volume = torch.pi * (0.5 * float(segment_diameter)) ** 2 * segment_length
    apparent_mass = (float(segment_density) - float(segment_buoyancy_density)) * segment_volume
    distributed_weight_w = apparent_mass * gravity
    body_force_from_weight_w = 0.5 * distributed_weight_w * float(num_segments)

    body_force_from_drag_w = torch.zeros_like(axial_force_w)
    if drag_coeff > 0.0:
        rel_speed = torch.linalg.norm(relative_vel_w, dim=-1, keepdim=True)
        segment_drag_w = -float(drag_coeff) * rel_speed * relative_vel_w * segment_length
        body_force_from_drag_w = 0.5 * segment_drag_w * float(num_segments)

    force_w = axial_force_w + body_force_from_weight_w + body_force_from_drag_w
    force_b = quat_apply_fn(quat_conjugate_fn(body_quat_w), force_w)
    torque_b = torch.cross(attach_offset, force_b, dim=-1)
    return force_b, torque_b


def update_rate_limited_winch_slack_length(
    current_slack_length: torch.Tensor,
    target_slack_length: torch.Tensor | float,
    reel_speed_mps: float,
    dt_s: float,
    min_length: float,
    max_length: float,
) -> torch.Tensor:
    """Move slack length toward a winch target with a symmetric rate limit."""

    if float(reel_speed_mps) < 0.0:
        raise ValueError("reel_speed_mps must be non-negative.")
    if float(dt_s) < 0.0:
        raise ValueError("dt_s must be non-negative.")
    if float(min_length) < 0.0 or float(max_length) < float(min_length):
        raise ValueError("winch length bounds must satisfy 0 <= min_length <= max_length.")
    target = torch.as_tensor(
        target_slack_length,
        dtype=current_slack_length.dtype,
        device=current_slack_length.device,
    )
    if target.ndim == 0:
        target = target.reshape(1, 1).repeat(current_slack_length.shape[0], 1)
    elif target.ndim == 1:
        target = target.reshape(-1, 1)
    if target.shape[0] == 1 and current_slack_length.shape[0] > 1:
        target = target.repeat(current_slack_length.shape[0], 1)
    if target.shape != current_slack_length.shape:
        raise ValueError(
            f"target_slack_length must broadcast to {tuple(current_slack_length.shape)}, got {tuple(target.shape)}."
        )
    target = torch.clamp(target, min=float(min_length), max=float(max_length))
    max_step = float(reel_speed_mps) * float(dt_s)
    delta = torch.clamp(target - current_slack_length, min=-max_step, max=max_step)
    return torch.clamp(current_slack_length + delta, min=float(min_length), max=float(max_length))


def _as_env_column(value: torch.Tensor | float, reference: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).repeat(reference.shape[0], 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    if tensor.shape[0] == 1 and reference.shape[0] > 1:
        tensor = tensor.repeat(reference.shape[0], 1)
    if tensor.shape != (reference.shape[0], 1):
        raise ValueError(f"{name} must be scalar or shaped ({reference.shape[0]}, 1), got {tuple(tensor.shape)}.")
    return tensor

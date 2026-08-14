"""Measured T60 body-force curves and actuator transport effects."""

from __future__ import annotations

import torch

from robot.dynamics.parameters import AUV


def get_thruster_positions(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the canonical T1...T8 COM-relative installation centers."""

    return torch.as_tensor(AUV.thruster_positions_body_m, dtype=dtype, device=device)


def normalized_command_to_pwm_us(command: torch.Tensor) -> torch.Tensor:
    """Map a normalized action to the measured 1300...1700 µs interval."""

    return AUV.thruster_pwm_center_us + AUV.thruster_pwm_half_range_us * command.clamp(-1.0, 1.0)


def thruster_body_forces_from_pwm_us(
    pwm_us: torch.Tensor,
    coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the canonical absolute-PWM to FLU vector-force curve.

    ``pwm_us`` has shape ``(..., 8)`` and is clamped to 1300...1700 us before
    evaluation.  The coefficient layout is ``(8, 4, 3)`` with rows
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

    minimum_pwm_us = AUV.thruster_pwm_center_us - AUV.thruster_pwm_half_range_us
    maximum_pwm_us = AUV.thruster_pwm_center_us + AUV.thruster_pwm_half_range_us
    offset_us = pwm_us.clamp(minimum_pwm_us, maximum_pwm_us) - AUV.thruster_pwm_center_us
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
    The negative-branch signs and off-axis components are part of the measured
    coefficients and must not be modified by a separate polarity or direction.
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
    offset_us = AUV.thruster_pwm_half_range_us * command.clamp(-1.0, 1.0)
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
    inside_command_range = ((command >= -1.0) & (command <= 1.0)).unsqueeze(-1)
    return torch.where(inside_command_range, branch_slope, torch.zeros_like(branch_slope))


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


class FirstOrderThrusterResponse:
    """Filter T60 body-force targets with a per-environment motor time constant."""

    def __init__(
        self,
        num_envs: int,
        num_thrusters: int,
        time_constant_s: torch.Tensor | float,
        device: torch.device,
    ) -> None:
        if num_envs <= 0 or num_thrusters <= 0:
            raise ValueError("num_envs and num_thrusters must be positive.")
        self.num_envs = int(num_envs)
        self.num_thrusters = int(num_thrusters)
        self.device = torch.device(device)
        self.output_forces_b = torch.zeros(
            (self.num_envs, self.num_thrusters, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_update_time_s = torch.full(
            (self.num_envs,),
            torch.nan,
            dtype=torch.float32,
            device=self.device,
        )
        self.set_time_constants(time_constant_s)

    def set_time_constants(self, time_constant_s: torch.Tensor | float) -> None:
        """Set one shared constant or one constant per vectorized environment."""

        value = torch.as_tensor(time_constant_s, dtype=torch.float32, device=self.device)
        if value.ndim > 1 or (value.ndim == 1 and value.shape != (self.num_envs,)):
            raise ValueError("time_constant_s must be scalar or have shape (num_envs,).")
        self.time_constant_s = value.clone()

    def reset(self, env_ids: list | torch.Tensor | None = None) -> None:
        """Clear the realized force and timing state for selected environments."""

        selected = slice(None) if env_ids is None else env_ids
        self.output_forces_b[selected] = 0.0
        self.last_update_time_s[selected] = torch.nan

    def advance(self, target_forces_b: torch.Tensor, time_s: torch.Tensor | float) -> torch.Tensor:
        """Advance the exact discrete first-order response to ``time_s``."""

        expected_shape = (self.num_envs, self.num_thrusters, 3)
        if target_forces_b.shape != expected_shape:
            raise ValueError(f"target_forces_b must have shape {expected_shape}.")

        current_time_s = torch.as_tensor(
            time_s,
            dtype=target_forces_b.dtype,
            device=target_forces_b.device,
        )
        if current_time_s.ndim == 0:
            current_time_s = current_time_s.repeat(self.num_envs)
        elif current_time_s.shape != (self.num_envs,):
            raise ValueError("time_s must be scalar or have shape (num_envs,).")

        previous_time_s = self.last_update_time_s.to(
            dtype=target_forces_b.dtype,
            device=target_forces_b.device,
        )
        elapsed_s = torch.where(
            torch.isfinite(previous_time_s),
            torch.clamp(current_time_s - previous_time_s, min=0.0),
            torch.zeros_like(current_time_s),
        )
        time_constant_s = self.time_constant_s.to(
            dtype=target_forces_b.dtype,
            device=target_forces_b.device,
        )
        if time_constant_s.ndim == 0:
            time_constant_s = time_constant_s.repeat(self.num_envs)
        decay = torch.exp(-elapsed_s / torch.clamp(time_constant_s, min=1.0e-6))
        decay = torch.where(time_constant_s <= 0.0, torch.zeros_like(decay), decay)
        blend = decay.reshape(self.num_envs, 1, 1)
        self.output_forces_b = self.output_forces_b * blend + (1.0 - blend) * target_forces_b
        self.last_update_time_s[:] = current_time_s
        return self.output_forces_b


class ThrusterCommandProcessor:
    """Apply command delay, dropouts, rate limits, and quantization."""

    def __init__(
        self,
        num_envs: int,
        num_thrusters: int,
        max_delay_steps: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.num_thrusters = num_thrusters
        self.device = device
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.history_length = self.max_delay_steps + 1
        self.history_index = 0
        self.history = torch.zeros(
            (self.history_length, self.num_envs, self.num_thrusters),
            dtype=torch.float32,
            device=self.device,
        )
        self.rate_limited_state = torch.zeros(
            (self.num_envs, self.num_thrusters),
            dtype=torch.float32,
            device=self.device,
        )
        self._env_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: list | torch.Tensor | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self.history[:, selected, :] = 0.0
        self.rate_limited_state[selected, :] = 0.0
        if env_ids is None:
            self.history_index = 0

    def process(
        self,
        commands: torch.Tensor,
        delay_steps: torch.Tensor | int,
        max_rate: torch.Tensor | float,
        dt: torch.Tensor | float,
        command_resolution: torch.Tensor | float = 0.0,
        dropout_probability: torch.Tensor | float = 0.0,
        *,
        dropout_enabled: bool | None = None,
    ) -> torch.Tensor:
        expected_shape = (self.num_envs, self.num_thrusters)
        if commands.shape != expected_shape:
            raise ValueError(f"commands must have shape {expected_shape}.")
        self.history[self.history_index, :, :] = commands

        delay_steps = torch.as_tensor(delay_steps, dtype=torch.long, device=commands.device)
        if delay_steps.ndim == 0:
            delay_steps = delay_steps.repeat(self.num_envs)
        delay_steps = torch.clamp(delay_steps.reshape(self.num_envs), min=0, max=self.max_delay_steps)

        delayed_indices = (self.history_index - delay_steps) % self.history_length
        delayed_cmd = self.history[delayed_indices, self._env_indices, :]
        self.history_index = (self.history_index + 1) % self.history_length

        dropout_probability = torch.clamp(
            _expand_env_thruster_value(dropout_probability, commands),
            min=0.0,
            max=1.0,
        )
        if dropout_enabled is None:
            dropout_enabled = bool(torch.any(dropout_probability > 0.0))
        if dropout_enabled:
            dropout_mask = torch.rand_like(commands) < dropout_probability
            delayed_cmd = torch.where(dropout_mask, self.rate_limited_state, delayed_cmd)

        rate = _expand_env_thruster_value(max_rate, commands)
        dt_tensor = torch.as_tensor(dt, dtype=commands.dtype, device=commands.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.reshape(1, 1)
        elif dt_tensor.ndim == 1:
            dt_tensor = dt_tensor.reshape(self.num_envs, 1)
        max_delta = torch.clamp(rate, min=0.0) * dt_tensor

        delta = delayed_cmd - self.rate_limited_state
        limited_cmd = self.rate_limited_state + torch.clamp(delta, -max_delta, max_delta)
        processed_cmd = torch.where(rate <= 0.0, delayed_cmd, limited_cmd)

        resolution = torch.clamp(_expand_env_thruster_value(command_resolution, commands), min=0.0)
        quantized_cmd = torch.round(processed_cmd / torch.clamp(resolution, min=1.0e-6)) * resolution
        self.rate_limited_state = torch.where(resolution > 0.0, quantized_cmd, processed_cmd)
        self.rate_limited_state = torch.clamp(self.rate_limited_state, min=-1.0, max=1.0)
        return self.rate_limited_state


def _expand_env_thruster_value(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        if tensor.shape[0] == reference.shape[0]:
            return tensor.reshape(reference.shape[0], 1)
        if tensor.shape[0] == reference.shape[1]:
            return tensor.reshape(1, reference.shape[1])
    if tensor.ndim == 2:
        if tensor.shape == (reference.shape[0], 1):
            return tensor
        if tensor.shape == (1, reference.shape[1]):
            return tensor
    if tensor.shape == reference.shape:
        return tensor
    raise ValueError(f"Cannot broadcast value with shape {tuple(tensor.shape)} to {tuple(reference.shape)}.")


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
    loss_coefficient_tensor = torch.as_tensor(loss_coefficient, dtype=thrust.dtype, device=thrust.device)
    if loss_coefficient_tensor.ndim == 0:
        if float(loss_coefficient_tensor.item()) <= 0.0:
            return torch.ones_like(thrust)
        loss_coefficient_tensor = loss_coefficient_tensor.reshape(1, 1, 1)
    elif loss_coefficient_tensor.ndim == 1:
        loss_coefficient_tensor = loss_coefficient_tensor.reshape(-1, 1, 1)
    elif loss_coefficient_tensor.ndim == 2 and loss_coefficient_tensor.shape[1] == 1:
        loss_coefficient_tensor = loss_coefficient_tensor.reshape(-1, 1, 1)
    else:
        raise ValueError("loss_coefficient must be a scalar or per-env tensor.")

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
    if loss_coefficient_tensor.shape[0] not in (1, thrust.shape[0]):
        raise ValueError(
            "loss_coefficient must be scalar or have one value per environment, got "
            f"{tuple(loss_coefficient_tensor.shape)} for {thrust.shape[0]} environments."
        )

    source_pos = thruster_positions_b.unsqueeze(2)
    target_pos = thruster_positions_b.unsqueeze(1)
    rel_source_to_target = target_pos - source_pos

    signed_direction = torch.sign(thrust).unsqueeze(-1) * thruster_axes_b
    axial_distance = torch.sum(rel_source_to_target * signed_direction.unsqueeze(2), dim=-1)
    radial_vector = rel_source_to_target - axial_distance.unsqueeze(-1) * signed_direction.unsqueeze(2)
    radial_distance = torch.linalg.norm(radial_vector, dim=-1)

    wake_radius_at_target = float(wake_radius) + torch.clamp(axial_distance, min=0.0) * max(
        float(expansion_rate),
        0.0,
    )
    in_wake = (
        (axial_distance > 0.0)
        & (axial_distance <= float(wake_length))
        & (radial_distance <= wake_radius_at_target)
        & (torch.abs(thrust).unsqueeze(-1) > 1.0e-6)
    )
    num_thrusters = thrust.shape[1]
    source_is_target = torch.eye(num_thrusters, dtype=torch.bool, device=thrust.device).reshape(
        1,
        num_thrusters,
        num_thrusters,
    )
    in_wake = in_wake & ~source_is_target

    if reference_thrust is None:
        reference = torch.clamp(torch.max(torch.abs(thrust), dim=1, keepdim=True).values, min=1.0e-6)
    else:
        reference = torch.as_tensor(reference_thrust, dtype=thrust.dtype, device=thrust.device)
        if reference.ndim == 0:
            reference = reference.reshape(1, 1)
        elif reference.ndim == 1:
            if reference.shape[0] == thrust.shape[0]:
                reference = reference.reshape(thrust.shape[0], 1)
            elif reference.shape[0] == thrust.shape[1]:
                reference = reference.reshape(1, thrust.shape[1])
        reference = torch.clamp(reference, min=1.0e-6)

    source_strength = torch.clamp(torch.abs(thrust) / reference, min=0.0, max=1.0)
    radial_ratio = radial_distance / torch.clamp(wake_radius_at_target, min=1.0e-6)
    axial_fade = 1.0 - torch.clamp(axial_distance / float(wake_length), min=0.0, max=1.0)
    wake_profile = torch.exp(-(radial_ratio * radial_ratio)) * axial_fade
    loss = torch.clamp(loss_coefficient_tensor, min=0.0) * source_strength.unsqueeze(-1) * wake_profile
    loss = torch.where(in_wake, loss, torch.zeros_like(loss))

    total_loss = torch.sum(loss, dim=1)
    return torch.clamp(1.0 - total_loss, min=float(min_scale), max=1.0)


def calculate_reaction_torques(
    thrust: torch.Tensor,
    thruster_axes_b: torch.Tensor,
    torque_coeff: torch.Tensor | float,
    spin_directions: torch.Tensor | list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Return body-frame reaction torques from propeller spin."""

    coefficient = torch.as_tensor(torque_coeff, dtype=thrust.dtype, device=thrust.device)
    if coefficient.ndim == 0:
        if float(coefficient.item()) == 0.0:
            return torch.zeros_like(thruster_axes_b)
        coefficient = coefficient.reshape(1, 1, 1)
    elif coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1, 1, 1)
    elif coefficient.ndim == 2 and coefficient.shape[1] == 1:
        coefficient = coefficient.reshape(-1, 1, 1)
    else:
        raise ValueError("torque_coeff must be a scalar or per-env tensor.")
    if coefficient.shape[0] not in (1, thrust.shape[0]):
        raise ValueError(
            f"torque_coeff must be scalar or have one value per environment, got {tuple(coefficient.shape)}."
        )

    spin = torch.as_tensor(spin_directions, dtype=thrust.dtype, device=thrust.device)
    if spin.ndim == 1:
        spin = spin.reshape(1, -1)
    if spin.shape[0] == 1:
        spin = spin.repeat(thrust.shape[0], 1)
    if spin.shape != thrust.shape:
        raise ValueError(f"spin_directions must broadcast to thrust shape {tuple(thrust.shape)}.")
    return -torch.clamp(coefficient, min=0.0) * spin.unsqueeze(-1) * thrust.unsqueeze(-1) * thruster_axes_b

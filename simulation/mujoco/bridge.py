"""Simulator-independent policy bridge used by the MuJoCo validator.

The code in this module deliberately depends only on NumPy.  It mirrors the
deployed Actor contract from :mod:`simulation.isaac.observations` without importing
IsaacLab or MuJoCo, which keeps the bridge easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


BASE_OBSERVATION_DIM = 30
ACTION_DIM = 8
OBSERVATION_GROUP_SLICES = {
    "position_error_b": slice(0, 3),
    "target_linear_velocity_b": slice(3, 6),
    "linear_velocity_error_b": slice(6, 9),
    "attitude_error_quat": slice(9, 13),
    "angular_velocity_b": slice(13, 16),
    "target_angular_velocity_b": slice(16, 19),
    "target_linear_acceleration_b": slice(19, 22),
    "actions": slice(22, 30),
    "applied_action": slice(22, 30),
}


def _vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def coefficient_matrix(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return a 6x6 coefficient matrix from a diagonal or full matrix."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape == (6,):
        return np.diag(result)
    if result.shape == (6, 6):
        return result
    raise ValueError(f"{name} must be a 6-vector or 6x6 matrix, got {result.shape}.")


def quaternion_normalize(quaternion_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    quaternion = _vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-12:
        raise ValueError("quaternion_wxyz must be nonzero.")
    return quaternion / norm


def quaternion_conjugate(quaternion_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    quaternion = quaternion_normalize(quaternion_wxyz)
    return quaternion * np.array((1.0, -1.0, -1.0, -1.0))


def quaternion_multiply(
    left_wxyz: Sequence[float] | np.ndarray,
    right_wxyz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    left = quaternion_normalize(left_wxyz)
    right = quaternion_normalize(right_wxyz)
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def quaternion_rotate(
    quaternion_wxyz: Sequence[float] | np.ndarray,
    vector: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Rotate a vector from body coordinates into world coordinates."""

    quaternion = quaternion_normalize(quaternion_wxyz)
    value = _vector(vector, 3, "vector")
    xyz = quaternion[1:]
    intermediate = 2.0 * np.cross(xyz, value)
    return value + quaternion[0] * intermediate + np.cross(xyz, intermediate)


def quaternion_rotate_inverse(
    quaternion_wxyz: Sequence[float] | np.ndarray,
    vector: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Rotate a vector from world coordinates into body coordinates."""

    return quaternion_rotate(quaternion_conjugate(quaternion_wxyz), vector)


def unique_quaternion(quaternion_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    """Match IsaacLab's positive-real quaternion representative."""

    quaternion = quaternion_normalize(quaternion_wxyz)
    return -quaternion if quaternion[0] < 0.0 else quaternion


def align_body_x_with_velocity(
    velocity_w: Sequence[float] | np.ndarray,
    previous_quaternion_wxyz: Sequence[float] | np.ndarray,
    min_speed: float = 1.0e-3,
) -> np.ndarray:
    """Return the zero-roll attitude whose body +X follows world velocity."""

    velocity = _vector(velocity_w, 3, "velocity_w")
    previous = quaternion_normalize(previous_quaternion_wxyz)
    horizontal_speed = np.linalg.norm(velocity[:2])
    if np.linalg.norm(velocity) <= float(min_speed):
        return previous
    yaw = np.arctan2(velocity[1], velocity[0])
    pitch = -np.arctan2(velocity[2], horizontal_speed)
    half_yaw = 0.5 * yaw
    half_pitch = 0.5 * pitch
    candidate = np.array(
        (
            np.cos(half_yaw) * np.cos(half_pitch),
            -np.sin(half_yaw) * np.sin(half_pitch),
            np.cos(half_yaw) * np.sin(half_pitch),
            np.sin(half_yaw) * np.cos(half_pitch),
        ),
        dtype=np.float64,
    )
    if np.dot(candidate, previous) < 0.0:
        candidate = -candidate
    return quaternion_normalize(candidate)


def quaternion_step_angular_velocity_world(
    previous_quaternion_wxyz: Sequence[float] | np.ndarray,
    current_quaternion_wxyz: Sequence[float] | np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return shortest-step target angular velocity in world coordinates."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    previous = quaternion_normalize(previous_quaternion_wxyz)
    current = quaternion_normalize(current_quaternion_wxyz)
    relative = quaternion_normalize(quaternion_multiply(quaternion_conjugate(previous), current))
    if relative[0] < 0.0:
        relative = -relative
    vector_norm = np.linalg.norm(relative[1:])
    if vector_norm < 1.0e-10:
        angular_velocity_previous_b = np.zeros(3, dtype=np.float64)
    else:
        angle = 2.0 * np.arctan2(vector_norm, max(relative[0], 0.0))
        angular_velocity_previous_b = relative[1:] * angle / (vector_norm * dt_s)
    return quaternion_rotate(previous, angular_velocity_previous_b)


@dataclass(frozen=True)
class VehicleState:
    position_w: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_b: np.ndarray
    angular_velocity_b: np.ndarray

    def validate(self) -> None:
        _vector(self.position_w, 3, "position_w")
        quaternion_normalize(self.quaternion_wxyz)
        _vector(self.linear_velocity_b, 3, "linear_velocity_b")
        _vector(self.angular_velocity_b, 3, "angular_velocity_b")


@dataclass(frozen=True)
class ReferenceState:
    position_w: np.ndarray
    linear_velocity_w: np.ndarray
    linear_acceleration_w: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity_w: np.ndarray

    def validate(self) -> None:
        _vector(self.position_w, 3, "position_w")
        _vector(self.linear_velocity_w, 3, "linear_velocity_w")
        _vector(self.linear_acceleration_w, 3, "linear_acceleration_w")
        quaternion_normalize(self.quaternion_wxyz)
        _vector(self.angular_velocity_w, 3, "angular_velocity_w")


@dataclass(frozen=True)
class TrajectoryConfig:
    kind: str = "lissajous"
    center_w: tuple[float, float, float] = (0.0, 0.0, -3.0)
    amplitude_x_m: float = 0.75
    amplitude_y_m: float = 0.65
    amplitude_z_m: float = 0.16
    period_s: float = 12.0

    def validate(self) -> None:
        if self.kind not in {"hold", "circle", "lissajous", "helix"}:
            raise ValueError(f"Unsupported trajectory kind {self.kind!r}.")
        _vector(self.center_w, 3, "center_w")
        if min(self.amplitude_x_m, self.amplitude_y_m, self.amplitude_z_m) < 0.0:
            raise ValueError("Trajectory amplitudes must be non-negative.")
        if self.period_s <= 0.0:
            raise ValueError("period_s must be positive.")


class ReferenceGenerator:
    """Generate deterministic references matching the fixed Isaac evaluation shapes."""

    def __init__(self, config: TrajectoryConfig, policy_dt_s: float):
        config.validate()
        if policy_dt_s <= 0.0:
            raise ValueError("policy_dt_s must be positive.")
        self.config = config
        self.policy_dt_s = float(policy_dt_s)
        self._previous_quaternion = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._has_previous = False

    def reset(self) -> None:
        self._previous_quaternion = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._has_previous = False

    def _kinematics(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        omega = 2.0 * np.pi / cfg.period_s
        phase = omega * float(time_s)
        if cfg.kind == "hold":
            zeros = np.zeros(3, dtype=np.float64)
            return zeros.copy(), zeros.copy(), zeros.copy()
        if cfg.kind == "circle":
            position = np.array(
                (cfg.amplitude_x_m * np.cos(phase), cfg.amplitude_y_m * np.sin(phase), 0.0)
            )
            velocity = np.array(
                (-cfg.amplitude_x_m * omega * np.sin(phase), cfg.amplitude_y_m * omega * np.cos(phase), 0.0)
            )
            acceleration = np.array(
                (-cfg.amplitude_x_m * omega**2 * np.cos(phase), -cfg.amplitude_y_m * omega**2 * np.sin(phase), 0.0)
            )
            return position, velocity, acceleration
        if cfg.kind == "lissajous":
            position = np.array(
                (cfg.amplitude_x_m * np.sin(phase), cfg.amplitude_y_m * np.sin(2.0 * phase), 0.0)
            )
            velocity = np.array(
                (
                    cfg.amplitude_x_m * omega * np.cos(phase),
                    2.0 * cfg.amplitude_y_m * omega * np.cos(2.0 * phase),
                    0.0,
                )
            )
            acceleration = np.array(
                (
                    -cfg.amplitude_x_m * omega**2 * np.sin(phase),
                    -4.0 * cfg.amplitude_y_m * omega**2 * np.sin(2.0 * phase),
                    0.0,
                )
            )
            return position, velocity, acceleration
        position = np.array(
            (
                cfg.amplitude_x_m * np.cos(phase),
                cfg.amplitude_y_m * np.sin(phase),
                cfg.amplitude_z_m * np.sin(2.0 * phase),
            )
        )
        velocity = np.array(
            (
                -cfg.amplitude_x_m * omega * np.sin(phase),
                cfg.amplitude_y_m * omega * np.cos(phase),
                2.0 * cfg.amplitude_z_m * omega * np.cos(2.0 * phase),
            )
        )
        acceleration = np.array(
            (
                -cfg.amplitude_x_m * omega**2 * np.cos(phase),
                -cfg.amplitude_y_m * omega**2 * np.sin(phase),
                -4.0 * cfg.amplitude_z_m * omega**2 * np.sin(2.0 * phase),
            )
        )
        return position, velocity, acceleration

    def sample(self, time_s: float) -> ReferenceState:
        offset, velocity, acceleration = self._kinematics(time_s)
        position = _vector(self.config.center_w, 3, "center_w") + offset
        quaternion = align_body_x_with_velocity(velocity, self._previous_quaternion)
        angular_velocity = (
            quaternion_step_angular_velocity_world(
                self._previous_quaternion,
                quaternion,
                self.policy_dt_s,
            )
            if self._has_previous
            else np.zeros(3, dtype=np.float64)
        )
        self._previous_quaternion = quaternion.copy()
        self._has_previous = True
        return ReferenceState(position, velocity, acceleration, quaternion, angular_velocity)


class PolicyObservationAdapter:
    """Build the exact 30-D Actor sample and optional causal MLP history."""

    def __init__(
        self,
        *,
        history_steps: int,
        history_fields: Sequence[str],
        position_scale_m: float = 2.0,
        linear_velocity_scale_mps: float = 1.0,
        angular_velocity_scale_radps: float = 1.0,
        linear_acceleration_scale_mps2: float = 0.5,
    ):
        if history_steps < 0:
            raise ValueError("history_steps must be non-negative.")
        unknown = set(history_fields) - set(OBSERVATION_GROUP_SLICES)
        if unknown:
            raise ValueError("Unknown history fields: " + ", ".join(sorted(unknown)))
        scales = (position_scale_m, linear_velocity_scale_mps, angular_velocity_scale_radps, linear_acceleration_scale_mps2)
        if min(scales) <= 0.0:
            raise ValueError("Observation scales must be positive.")
        self.history_steps = int(history_steps)
        self.history_fields = tuple(history_fields)
        self._normalization_scale = np.ones(BASE_OBSERVATION_DIM, dtype=np.float64)
        self._normalization_scale[0:3] = position_scale_m
        self._normalization_scale[3:9] = linear_velocity_scale_mps
        self._normalization_scale[13:19] = angular_velocity_scale_radps
        self._normalization_scale[19:22] = linear_acceleration_scale_mps2
        self._history_indices = np.concatenate(
            [
                np.arange(OBSERVATION_GROUP_SLICES[name].start, OBSERVATION_GROUP_SLICES[name].stop)
                for name in self.history_fields
            ]
        ).astype(np.int64) if self.history_fields else np.empty(0, dtype=np.int64)
        self._history = np.zeros(
            (self.history_steps, len(self._history_indices)),
            dtype=np.float64,
        )

    @property
    def observation_dim(self) -> int:
        return BASE_OBSERVATION_DIM + self._history.size

    def reset(self) -> None:
        self._history.fill(0.0)

    def current_sample(
        self,
        vehicle: VehicleState,
        reference: ReferenceState,
        applied_action: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        vehicle.validate()
        reference.validate()
        action = _vector(applied_action, ACTION_DIM, "applied_action")
        attitude_error = unique_quaternion(
            quaternion_multiply(
                quaternion_conjugate(vehicle.quaternion_wxyz),
                reference.quaternion_wxyz,
            )
        )
        target_velocity_b = quaternion_rotate_inverse(
            vehicle.quaternion_wxyz,
            reference.linear_velocity_w,
        )
        raw = np.concatenate(
            (
                quaternion_rotate_inverse(
                    vehicle.quaternion_wxyz,
                    reference.position_w - vehicle.position_w,
                ),
                target_velocity_b,
                target_velocity_b - vehicle.linear_velocity_b,
                attitude_error,
                vehicle.angular_velocity_b,
                quaternion_rotate_inverse(
                    vehicle.quaternion_wxyz,
                    reference.angular_velocity_w,
                ),
                quaternion_rotate_inverse(
                    vehicle.quaternion_wxyz,
                    reference.linear_acceleration_w,
                ),
                action,
            )
        )
        if raw.shape != (BASE_OBSERVATION_DIM,):
            raise RuntimeError(f"Internal observation contract produced {raw.shape}.")
        return raw / self._normalization_scale

    def build(
        self,
        vehicle: VehicleState,
        reference: ReferenceState,
        applied_action: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        current = self.current_sample(vehicle, reference, applied_action)
        actor_observation = np.concatenate((current, self._history.reshape(-1)))
        if self.history_steps:
            if self.history_steps > 1:
                self._history[1:] = self._history[:-1].copy()
            self._history[0] = current[self._history_indices]
        return actor_observation.astype(np.float32)


@dataclass(frozen=True)
class ThrusterParameters:
    positions_b: np.ndarray
    force_curve_coefficients: np.ndarray
    time_constant_s: float
    max_command_rate_per_s: float
    command_delay_steps: int
    command_resolution: float
    dropout_probability: float
    pwm_center_us: float
    pwm_half_range_us: float
    pwm_deadband_us: float

    def validate(self) -> None:
        positions = np.asarray(self.positions_b, dtype=np.float64)
        coefficients = np.asarray(self.force_curve_coefficients, dtype=np.float64)
        if positions.shape != (ACTION_DIM, 3):
            raise ValueError("Thruster positions must have shape (8, 3).")
        if coefficients.shape != (ACTION_DIM, 4, 3):
            raise ValueError("Thruster force coefficients must have shape (8, 4, 3).")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Thruster positions and force coefficients must be finite.")
        if self.time_constant_s < 0.0 or self.max_command_rate_per_s < 0.0:
            raise ValueError("Thruster time constant and command rate must be non-negative.")
        if self.command_delay_steps < 0 or self.command_resolution < 0.0:
            raise ValueError("Thruster delay and resolution must be non-negative.")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1].")
        if not np.isfinite(self.pwm_center_us):
            raise ValueError("pwm_center_us must be finite.")
        if not np.isfinite(self.pwm_half_range_us) or self.pwm_half_range_us <= 0.0:
            raise ValueError("pwm_half_range_us must be finite and positive.")
        if not np.isfinite(self.pwm_deadband_us) or not 0.0 <= self.pwm_deadband_us < self.pwm_half_range_us:
            raise ValueError("pwm_deadband_us must be finite and lie in [0, pwm_half_range_us).")


@dataclass(frozen=True)
class ThrusterStep:
    applied_command: np.ndarray
    forces_b_n: np.ndarray
    wrench_b: np.ndarray


def thruster_body_forces_from_pwm_us(
    pwm_us: Sequence[float] | np.ndarray,
    parameters: ThrusterParameters,
) -> np.ndarray:
    """Evaluate the canonical clamped absolute-PWM vector-force curve."""

    pwm = _vector(pwm_us, ACTION_DIM, "pwm_us")
    cfg = parameters
    minimum_pwm_us = cfg.pwm_center_us - cfg.pwm_half_range_us
    maximum_pwm_us = cfg.pwm_center_us + cfg.pwm_half_range_us
    offset_us = np.clip(pwm, minimum_pwm_us, maximum_pwm_us) - cfg.pwm_center_us
    negative_effective_us = np.maximum(-offset_us - cfg.pwm_deadband_us, 0.0)[:, None]
    positive_effective_us = np.maximum(offset_us - cfg.pwm_deadband_us, 0.0)[:, None]
    coefficients = np.asarray(cfg.force_curve_coefficients, dtype=np.float64)
    return (
        coefficients[:, 0, :] * negative_effective_us**2
        + coefficients[:, 1, :] * negative_effective_us
        + coefficients[:, 2, :] * positive_effective_us**2
        + coefficients[:, 3, :] * positive_effective_us
    )


class ThrusterModel:
    """Command transport and measured T1--T8 body-force response."""

    def __init__(self, parameters: ThrusterParameters, seed: int = 0):
        parameters.validate()
        self.parameters = parameters
        self._rng = np.random.default_rng(seed)
        self._history = np.zeros((parameters.command_delay_steps + 1, ACTION_DIM))
        self._history_index = 0
        self.applied_command = np.zeros(ACTION_DIM)
        self.forces_b_n = np.zeros((ACTION_DIM, 3))

    def reset(self) -> None:
        self._history.fill(0.0)
        self._history_index = 0
        self.applied_command.fill(0.0)
        self.forces_b_n.fill(0.0)

    def _command_to_body_forces(self, command: np.ndarray) -> np.ndarray:
        """Map normalized T1--T8 commands to measured FLU force vectors."""

        cfg = self.parameters
        pwm_us = cfg.pwm_center_us + cfg.pwm_half_range_us * np.clip(command, -1.0, 1.0)
        return thruster_body_forces_from_pwm_us(pwm_us, cfg)

    def step(self, requested_command: Sequence[float] | np.ndarray, dt_s: float) -> ThrusterStep:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        cfg = self.parameters
        requested = np.clip(_vector(requested_command, ACTION_DIM, "requested_command"), -1.0, 1.0)
        self._history[self._history_index] = requested
        delayed_index = (self._history_index - cfg.command_delay_steps) % len(self._history)
        delayed = self._history[delayed_index].copy()
        self._history_index = (self._history_index + 1) % len(self._history)
        if cfg.dropout_probability > 0.0:
            dropout = self._rng.random(ACTION_DIM) < cfg.dropout_probability
            delayed = np.where(dropout, self.applied_command, delayed)
        if cfg.max_command_rate_per_s > 0.0:
            max_delta = cfg.max_command_rate_per_s * dt_s
            delayed = self.applied_command + np.clip(
                delayed - self.applied_command,
                -max_delta,
                max_delta,
            )
        if cfg.command_resolution > 0.0:
            delayed = np.round(delayed / cfg.command_resolution) * cfg.command_resolution
        self.applied_command = np.clip(delayed, -1.0, 1.0)
        commanded_forces_b_n = self._command_to_body_forces(self.applied_command)
        alpha = 0.0 if cfg.time_constant_s <= 0.0 else np.exp(-dt_s / cfg.time_constant_s)
        self.forces_b_n = alpha * self.forces_b_n + (1.0 - alpha) * commanded_forces_b_n
        force_b = np.sum(self.forces_b_n, axis=0)
        torque_b = np.sum(np.cross(np.asarray(cfg.positions_b), self.forces_b_n), axis=0)
        return ThrusterStep(
            self.applied_command.copy(),
            self.forces_b_n.copy(),
            np.concatenate((force_b, torque_b)),
        )


@dataclass(frozen=True)
class HydrodynamicsParameters:
    fluid_density_kg_m3: float
    displaced_volume_m3: float
    center_of_buoyancy_from_com_b: np.ndarray
    linear_damping: np.ndarray
    quadratic_damping: np.ndarray
    added_mass: np.ndarray
    added_mass_inertia_scale: float
    added_mass_acceleration_filter_alpha: float
    water_current_w: np.ndarray
    periodic_current_enabled: bool
    periodic_current_amplitude_w: np.ndarray
    periodic_current_period_s: np.ndarray
    periodic_current_phase_rad: np.ndarray

    def validate(self) -> None:
        if self.fluid_density_kg_m3 <= 0.0 or self.displaced_volume_m3 <= 0.0:
            raise ValueError("Fluid density and displaced volume must be positive.")
        _vector(self.center_of_buoyancy_from_com_b, 3, "center_of_buoyancy_from_com_b")
        coefficient_matrix(self.linear_damping, "linear_damping")
        coefficient_matrix(self.quadratic_damping, "quadratic_damping")
        coefficient_matrix(self.added_mass, "added_mass")
        if not 0.0 <= self.added_mass_acceleration_filter_alpha <= 1.0:
            raise ValueError("added_mass_acceleration_filter_alpha must be in [0, 1].")
        _vector(self.water_current_w, 3, "water_current_w")
        _vector(self.periodic_current_amplitude_w, 3, "periodic_current_amplitude_w")
        periods = _vector(self.periodic_current_period_s, 3, "periodic_current_period_s")
        if np.any(periods <= 0.0):
            raise ValueError("Periodic current periods must be positive.")
        _vector(self.periodic_current_phase_rad, 3, "periodic_current_phase_rad")


class HydrodynamicsModel:
    """Fossen damping, buoyancy, and added-mass model."""

    def __init__(self, parameters: HydrodynamicsParameters, gravity_mps2: float = 9.81):
        parameters.validate()
        self.parameters = parameters
        self.gravity_mps2 = float(gravity_mps2)
        self._previous_nu_r = np.zeros(6)
        self._filtered_nu_r_dot = np.zeros(6)
        self._has_previous = False

    def reset(self) -> None:
        self._previous_nu_r.fill(0.0)
        self._filtered_nu_r_dot.fill(0.0)
        self._has_previous = False

    def water_current_w(self, time_s: float) -> np.ndarray:
        cfg = self.parameters
        current = np.asarray(cfg.water_current_w, dtype=np.float64).copy()
        if cfg.periodic_current_enabled:
            current += np.asarray(cfg.periodic_current_amplitude_w) * np.sin(
                2.0 * np.pi * float(time_s) / np.asarray(cfg.periodic_current_period_s)
                + np.asarray(cfg.periodic_current_phase_rad)
            )
        return current

    def step(self, state: VehicleState, time_s: float, dt_s: float) -> np.ndarray:
        state.validate()
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        cfg = self.parameters
        current_b = quaternion_rotate_inverse(state.quaternion_wxyz, self.water_current_w(time_s))
        nu_r = np.concatenate((state.linear_velocity_b - current_b, state.angular_velocity_b))
        linear = coefficient_matrix(cfg.linear_damping, "linear_damping")
        quadratic = coefficient_matrix(cfg.quadratic_damping, "quadratic_damping")
        wrench = -(linear @ nu_r + quadratic @ (np.abs(nu_r) * nu_r))

        added_mass = coefficient_matrix(cfg.added_mass, "added_mass")
        added_momentum = added_mass @ nu_r
        linear_momentum = added_momentum[:3]
        angular_momentum = added_momentum[3:]
        coriolis = np.concatenate(
            (
                -np.cross(linear_momentum, nu_r[3:]),
                -np.cross(linear_momentum, nu_r[:3]) - np.cross(angular_momentum, nu_r[3:]),
            )
        )
        wrench -= coriolis
        if self._has_previous and cfg.added_mass_inertia_scale > 0.0:
            raw_acceleration = (nu_r - self._previous_nu_r) / dt_s
            filter_alpha = cfg.added_mass_acceleration_filter_alpha
            self._filtered_nu_r_dot = (
                filter_alpha * raw_acceleration
                + (1.0 - filter_alpha) * self._filtered_nu_r_dot
            )
            wrench -= cfg.added_mass_inertia_scale * (added_mass @ self._filtered_nu_r_dot)
        self._previous_nu_r = nu_r.copy()
        self._has_previous = True

        buoyancy_force_w = np.array(
            (0.0, 0.0, cfg.fluid_density_kg_m3 * cfg.displaced_volume_m3 * self.gravity_mps2)
        )
        buoyancy_force_b = quaternion_rotate_inverse(state.quaternion_wxyz, buoyancy_force_w)
        wrench[:3] += buoyancy_force_b
        wrench[3:] += np.cross(cfg.center_of_buoyancy_from_com_b, buoyancy_force_b)
        return wrench


def summarize_validation(
    position_errors_m: Sequence[float],
    raw_actions: Sequence[Sequence[float]],
    *,
    max_position_rmse_m: float,
    max_action_clip_fraction: float,
) -> dict[str, float | bool | list[str]]:
    """Compute deterministic gates for a completed cross-simulator rollout."""

    errors = np.asarray(position_errors_m, dtype=np.float64)
    actions = np.asarray(raw_actions, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("position_errors_m must contain at least one scalar.")
    if actions.ndim != 2 or actions.shape[0] != errors.size or actions.shape[1] != ACTION_DIM:
        raise ValueError("raw_actions must have shape (num_samples, 8).")
    finite = bool(np.all(np.isfinite(errors)) and np.all(np.isfinite(actions)))
    position_rmse = float(np.sqrt(np.mean(errors**2))) if finite else float("inf")
    position_p95 = float(np.percentile(errors, 95.0)) if finite else float("inf")
    action_rms = float(np.sqrt(np.mean(actions**2))) if finite else float("inf")
    clip_fraction = float(np.mean(np.abs(actions) > 1.0)) if finite else 1.0
    failures: list[str] = []
    if not finite:
        failures.append("rollout contains non-finite observations, actions, or errors")
    if position_rmse > max_position_rmse_m:
        failures.append(
            f"position RMSE {position_rmse:.4f} m exceeds {max_position_rmse_m:.4f} m"
        )
    if clip_fraction > max_action_clip_fraction:
        failures.append(
            f"raw action clip fraction {clip_fraction:.4f} exceeds {max_action_clip_fraction:.4f}"
        )
    return {
        "passed": not failures,
        "position_rmse_m": position_rmse,
        "position_error_p95_m": position_p95,
        "raw_action_rms": action_rms,
        "raw_action_clip_fraction": clip_fraction,
        "sample_count": int(errors.size),
        "failures": failures,
    }

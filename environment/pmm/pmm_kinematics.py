"""Fourier motion fitting and PMM frame transformations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FourierFit:
    """Constant/trend plus harmonic motion representation."""

    frequency_hz: float
    harmonics: int
    t_center: float
    coefficients: np.ndarray
    r2: float

    def evaluate(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = np.asarray(time_s, dtype=float)
        omega = 2.0 * np.pi * self.frequency_hz
        value = np.full_like(time, self.coefficients[0], dtype=float)
        velocity = np.full_like(time, self.coefficients[1], dtype=float)
        acceleration = np.zeros_like(time, dtype=float)
        value += self.coefficients[1] * (time - self.t_center)
        coefficient_index = 2
        for harmonic in range(1, self.harmonics + 1):
            sine = self.coefficients[coefficient_index]
            cosine = self.coefficients[coefficient_index + 1]
            harmonic_omega = harmonic * omega
            phase = harmonic_omega * time
            value += sine * np.sin(phase) + cosine * np.cos(phase)
            velocity += harmonic_omega * (
                sine * np.cos(phase) - cosine * np.sin(phase)
            )
            acceleration -= harmonic_omega**2 * (
                sine * np.sin(phase) + cosine * np.cos(phase)
            )
            coefficient_index += 2
        return value, velocity, acceleration


def fit_fourier(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
    harmonics: int,
) -> FourierFit:
    """Fit the motion representation used by the PMM identification."""

    time = np.asarray(time_s, dtype=float)
    response = np.asarray(values, dtype=float)
    centre = float(np.mean(time))
    columns = [np.ones_like(time), time - centre]
    omega = 2.0 * np.pi * frequency_hz
    for harmonic in range(1, harmonics + 1):
        phase = harmonic * omega * time
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    prediction = design @ coefficients
    return FourierFit(
        frequency_hz,
        harmonics,
        centre,
        coefficients,
        r2_score(response, prediction),
    )


def r2_score(values: np.ndarray, prediction: np.ndarray) -> float:
    """Return the ordinary coefficient of determination."""

    response = np.asarray(values, dtype=float)
    fitted = np.asarray(prediction, dtype=float)
    denominator = float(np.sum((response - np.mean(response)) ** 2))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.allclose(response, fitted) else float("-inf")
    return 1.0 - float(np.sum((response - fitted) ** 2)) / denominator


def rotation_x(angle_rad: float) -> np.ndarray:
    """Return the active FLU rotation about +x, mapping body vectors to H."""

    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=float,
    )


def validate_rotation(matrix: np.ndarray, name: str) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be a proper rotation")


def skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotate_twist_H_to_B(
    linear_H: np.ndarray,
    angular_H: np.ndarray,
    body_to_H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate vector histories from apparatus FLU H into body FLU B."""

    return linear_H @ body_to_H, angular_H @ body_to_H


def rotate_wrench_H_to_B(wrench_H: np.ndarray, body_to_H: np.ndarray) -> np.ndarray:
    """Rotate ``[force, moment]`` histories from apparatus H into body B."""

    force_B = wrench_H[:, :3] @ body_to_H
    moment_B = wrench_H[:, 3:] @ body_to_H
    return np.column_stack((force_B, moment_B))


def translate_wrench_origin_to_com(
    wrench_at_origin_B: np.ndarray,
    com_from_origin_B: np.ndarray,
) -> np.ndarray:
    """Translate moment from PMM origin O to COM G.

    ``r_OG`` points from the PMM motion/wrench origin to COM.  Therefore
    ``M_G = M_O - r_OG x F``.
    """

    force = wrench_at_origin_B[:, :3]
    moment_com = wrench_at_origin_B[:, 3:] - np.cross(com_from_origin_B, force)
    return np.column_stack((force, moment_com))


def motion_origin_to_com_kinematics(
    linear_velocity_origin_B: np.ndarray,
    linear_derivative_origin_B: np.ndarray,
    angular_velocity_B: np.ndarray,
    angular_acceleration_B: np.ndarray,
    com_from_origin_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move body-frame linear kinematics from PMM origin O to COM G.

    With the constant body vector ``r_OG``, ``v_G=v_O+omega x r_OG`` and
    the body-coordinate derivative is
    ``v_G_dot=v_O_dot+omega_dot x r_OG``.
    """

    velocity_com = linear_velocity_origin_B + np.cross(angular_velocity_B, com_from_origin_B)
    derivative_com = linear_derivative_origin_B + np.cross(
        angular_acceleration_B, com_from_origin_B
    )
    return velocity_com, derivative_com


def rigid_body_wrench_at_com(
    mass_kg: float,
    inertia_at_com_B: np.ndarray,
    linear_velocity_B: np.ndarray,
    linear_acceleration_B: np.ndarray,
    angular_velocity_B: np.ndarray,
    angular_acceleration_B: np.ndarray,
) -> np.ndarray:
    """Return Newton-Euler inertial wrench at COM in rotating FLU axes."""

    force = mass_kg * (
        linear_acceleration_B + np.cross(angular_velocity_B, linear_velocity_B)
    )
    angular_momentum = angular_velocity_B @ inertia_at_com_B.T
    moment = angular_acceleration_B @ inertia_at_com_B.T + np.cross(
        angular_velocity_B, angular_momentum
    )
    return np.column_stack((force, moment))

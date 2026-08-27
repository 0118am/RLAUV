"""Pure tensor bridge between modeled 6-DOF dynamics and PhysX integration."""

from __future__ import annotations

import torch

from environment.hydrodynamics.tensor_ops import expand_6d_matrix


def calculate_total_inertia_physx_wrench(
    external_wrench_b: torch.Tensor,
    velocity_b: torch.Tensor,
    gravity_force_b: torch.Tensor,
    rigid_mass_kg: torch.Tensor,
    rigid_inertia_body_kg_m2: torch.Tensor,
    fluid_added_mass: torch.Tensor,
    current_acceleration_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the relative-current equation and return the PhysX rigid wrench.

    The sampled fluid matrix is used once in
    ``(M_RB + M_A) nu_dot = tau + M_A nu_c_dot``. PhysX then receives only the
    equivalent rigid-body wrench, so it does not integrate ``M_A`` a second time.
    """

    batch_size = external_wrench_b.shape[0]
    mass = rigid_mass_kg.reshape(batch_size, 1)
    inertia = rigid_inertia_body_kg_m2.reshape(batch_size, 3, 3)
    velocity = velocity_b.reshape(batch_size, 6)
    rigid_mass = torch.zeros(
        (batch_size, 6, 6),
        dtype=external_wrench_b.dtype,
        device=external_wrench_b.device,
    )
    rigid_mass[:, :3, :3] = mass.reshape(batch_size, 1, 1) * torch.eye(
        3, dtype=external_wrench_b.dtype, device=external_wrench_b.device
    )
    rigid_mass[:, 3:, 3:] = inertia
    fluid_added_mass_matrix = expand_6d_matrix(fluid_added_mass, batch_size)

    linear_velocity = velocity[:, :3]
    angular_velocity = velocity[:, 3:]
    angular_momentum = torch.bmm(inertia, angular_velocity.unsqueeze(-1)).squeeze(-1)
    rigid_coriolis = torch.cat(
        (
            torch.cross(angular_velocity, mass * linear_velocity, dim=-1),
            torch.cross(angular_velocity, angular_momentum, dim=-1),
        ),
        dim=-1,
    )
    gravity_wrench = torch.cat(
        (gravity_force_b, torch.zeros_like(gravity_force_b)), dim=-1
    )
    fluid_current_inertia = torch.bmm(
        fluid_added_mass_matrix, current_acceleration_b.unsqueeze(-1)
    ).squeeze(-1)
    right_hand_side = (
        external_wrench_b
        + gravity_wrench
        - rigid_coriolis
        + fluid_current_inertia
    )
    generalized_acceleration = torch.linalg.solve(
        rigid_mass + fluid_added_mass_matrix, right_hand_side.unsqueeze(-1)
    ).squeeze(-1)
    physx_wrench = (
        torch.bmm(rigid_mass, generalized_acceleration.unsqueeze(-1)).squeeze(-1)
        + rigid_coriolis
        - gravity_wrench
    )
    return physx_wrench, generalized_acceleration

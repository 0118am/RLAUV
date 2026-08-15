"""
Fossen-style hydrodynamic wrenches for a rigid AUV body.

Isaac/PhysX integrates the rigid-body inertia, gyroscopic terms, and gravity.
This module therefore returns only the external fluid wrench to apply in the
body/link frame: buoyancy, relative-velocity damping, and optional added-mass
Coriolis terms.  Keeping that boundary explicit prevents double-counting the
rigid-body part of Fossen's 6-DOF model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .tensor_ops import (
    calculate_speed_dependent_damping_scale,
    expand_6d_matrix,
    mean_one_lognormal_scale,
    multiply_6d_matrix,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    scale_hydrodynamic_coefficients,
    skew_symmetric,
)




















@dataclass
class HydrodynamicForceModels:
    num_envs: int
    device: torch.device
    debug: bool = False

    def calculate_buoyancy_forces(
        self,
        root_quats_w: torch.Tensor,
        gravity_w: torch.Tensor,
        fluid_density: float,
        volumes: torch.Tensor,
        com_to_cob_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute buoyancy in body frame.

        PhysX already applies gravity at the COM.  We only add buoyancy here,
        but use the same world gravity vector so neutral buoyancy can be checked
        in one frame: F_g^w + F_b^w ~= 0 when m = rho * V.
        """

        if gravity_w.ndim == 1:
            gravity_w = gravity_w.reshape(1, 3).repeat(self.num_envs, 1)

        buoyancy_forces_w = -fluid_density * volumes * gravity_w
        buoyancy_forces_b = quat_apply_wxyz(quat_conjugate_wxyz(root_quats_w), buoyancy_forces_w)
        buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_b, dim=-1)

        if self.debug:
            print(f"buoyancy_forces_b={buoyancy_forces_b}, buoyancy_torques_b={buoyancy_torques_b}")

        return buoyancy_forces_b, buoyancy_torques_b

    def calculate_fossen_fluid_forces(
        self,
        root_quats_w: torch.Tensor,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        gravity_w: torch.Tensor,
        fluid_density: float,
        volumes: torch.Tensor,
        com_to_cob_offsets: torch.Tensor,
        linear_damping: torch.Tensor,
        quadratic_damping: torch.Tensor,
        water_current_w: torch.Tensor,
        added_mass_diag: torch.Tensor | None = None,
        relative_acceleration_b: torch.Tensor | None = None,
        *,
        added_mass_enabled: bool | None = None,
        relative_velocity_b: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the body-frame fluid wrench.

        Damping is evaluated with relative velocity nu_r = nu - nu_c.  The
        dissipation check should therefore use nu_r^T tau_damping <= 0, not
        nu^T tau_damping, because moving water can do work on the vehicle.
        ``added_mass_diag`` is the runtime config key and accepts either a
        6-vector of diagonal coefficients or a full 6x6 matrix.
        """

        buoyancy_forces_b, buoyancy_torques_b = self.calculate_buoyancy_forces(
            root_quats_w,
            gravity_w,
            fluid_density,
            volumes,
            com_to_cob_offsets,
        )

        nu_r = relative_velocity_b
        if nu_r is None:
            nu_r = self.calculate_relative_velocity(
                root_quats_w,
                root_linvels_b,
                root_angvels_b,
                water_current_w,
            )

        damping_wrench = self.calculate_relative_damping_wrench(nu_r, linear_damping, quadratic_damping)

        fluid_wrench = torch.cat((buoyancy_forces_b, buoyancy_torques_b), dim=-1)
        fluid_wrench = fluid_wrench + damping_wrench

        if added_mass_enabled is None:
            added_mass_enabled = added_mass_diag is not None and bool(torch.any(added_mass_diag != 0.0))
        if added_mass_diag is not None and added_mass_enabled:
            fluid_wrench = fluid_wrench - self.calculate_added_mass_coriolis_wrench(nu_r, added_mass_diag)
            if relative_acceleration_b is not None:
                fluid_wrench = fluid_wrench + self.calculate_added_mass_inertia_wrench(
                    relative_acceleration_b,
                    added_mass_diag,
                )

        if self.debug:
            power = torch.sum(nu_r * damping_wrench, dim=-1)
            print(f"relative damping power={power}")

        return fluid_wrench[:, 0:3], fluid_wrench[:, 3:6]

    def calculate_relative_velocity(
        self,
        root_quats_w: torch.Tensor,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        water_current_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return body-frame relative velocity ``nu_r = nu - nu_current``."""

        if water_current_w.ndim == 1:
            water_current_w = water_current_w.reshape(1, 3).repeat(self.num_envs, 1)
        water_current_b = quat_apply_wxyz(quat_conjugate_wxyz(root_quats_w), water_current_w)

        nu = torch.cat((root_linvels_b, root_angvels_b), dim=-1)
        nu_current = torch.zeros_like(nu)
        nu_current[:, 0:3] = water_current_b
        return nu - nu_current

    def calculate_added_mass_coriolis_wrench(self, nu_r: torch.Tensor, added_mass_diag: torch.Tensor) -> torch.Tensor:
        """Compute ``C_A(nu_r) nu_r`` for diagonal or full added mass.

        The environment applies ``-C_A nu_r`` as an external wrench.  The helper
        returns ``C_A nu_r`` so tests can directly check skew-symmetry and power.
        """

        added_momentum = multiply_6d_matrix(added_mass_diag, nu_r)
        v = nu_r[:, 0:3]
        omega = nu_r[:, 3:6]
        a_linear = added_momentum[:, 0:3]
        a_angular = added_momentum[:, 3:6]

        c_top = -torch.bmm(skew_symmetric(a_linear), omega.unsqueeze(-1)).squeeze(-1)
        c_bottom = (
            -torch.bmm(skew_symmetric(a_linear), v.unsqueeze(-1)).squeeze(-1)
            - torch.bmm(skew_symmetric(a_angular), omega.unsqueeze(-1)).squeeze(-1)
        )
        return torch.cat((c_top, c_bottom), dim=-1)

    def calculate_added_mass_inertia_wrench(
        self,
        relative_acceleration_b: torch.Tensor,
        added_mass_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Return the external wrench ``-M_A dot(nu_r)``."""

        return -multiply_6d_matrix(added_mass_diag, relative_acceleration_b)

    def calculate_relative_damping_wrench(
        self,
        nu_r: torch.Tensor,
        linear_damping: torch.Tensor,
        quadratic_damping: torch.Tensor,
    ) -> torch.Tensor:
        """Standalone damping helper used by tests and diagnostics."""

        linear_wrench = multiply_6d_matrix(linear_damping, nu_r)
        quadratic_wrench = multiply_6d_matrix(quadratic_damping, torch.abs(nu_r) * nu_r)
        return -(linear_wrench + quadratic_wrench)


if __name__ == "__main__":
    from robot.dynamics.parameters import AUV

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HydrodynamicForceModels(num_envs=1, device=device)

    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    gravity_w = torch.tensor([0.0, 0.0, -9.81], device=device)
    volume = torch.tensor([[AUV.displaced_volume_m3]], device=device)
    rho = AUV.water_density_kg_m3
    cob = torch.tensor([AUV.center_of_buoyancy_from_com_m], device=device)
    force_b, torque_b = model.calculate_buoyancy_forces(q_identity, gravity_w, rho, volume, cob)
    expected = torch.tensor([[0.0, 0.0, rho * volume.item() * 9.81]], device=device)
    assert torch.allclose(force_b, expected, atol=1.0e-5), (force_b, expected)
    assert torch.allclose(torque_b, torch.zeros_like(torque_b), atol=1.0e-5), torque_b

    nu_r = torch.tensor([[0.2, -0.1, 0.3, 0.04, -0.02, 0.01]], device=device)
    linear = torch.ones(6, device=device) * 0.1
    quadratic = torch.ones(6, device=device) * 2.0
    damping = model.calculate_relative_damping_wrench(nu_r, linear, quadratic)
    assert torch.all(torch.sum(nu_r * damping, dim=-1) <= 0.0)
    print("Hydrodynamic sanity checks passed.")

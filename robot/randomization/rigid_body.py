"""Payload, mass, buoyancy, and centre-of-mass DR feature."""

from __future__ import annotations

import torch

from environment.hydrodynamics.models import scale_hydrodynamic_coefficients
from environment.profiles.random_sampling import sample_bounded_normal, sample_isotropic_bounded_normal
from robot.dynamics.rigid_body import physx_principal_inertia_and_com_quat_xyzw


def initialize_payload_domain(env) -> None:
    """Prepare categorical payload tensors once at environment construction."""

    samples = list(getattr(env.cfg.domain_randomization, "payload_samples", []))
    env.payload_sample_indices = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    env._payload_sample_count = len(samples)
    if not samples:
        return

    def six_scale(sample, name: str) -> torch.Tensor:
        value = torch.as_tensor(sample.get(name, 1.0), dtype=torch.float32, device=env.device).reshape(-1)
        return value.repeat(6) if value.numel() == 1 else value

    env._payload_weights = torch.as_tensor(
        [sample.get("weight", 1.0) for sample in samples], dtype=torch.float32, device=env.device
    )
    env._payload_masses = torch.as_tensor(
        [sample["mass"] for sample in samples], dtype=torch.float32, device=env.device
    )
    env._payload_volumes = torch.as_tensor(
        [sample["volume"] for sample in samples], dtype=torch.float32, device=env.device
    )
    principal_properties = [
        physx_principal_inertia_and_com_quat_xyzw(sample["inertia"], env.device) for sample in samples
    ]
    env._payload_principal_moments = torch.stack([properties[0] for properties in principal_properties])
    env._payload_principal_axes_xyzw = torch.stack([properties[1] for properties in principal_properties])
    env._payload_center_of_mass_offsets = torch.as_tensor(
        [sample["center_of_mass_offset"] for sample in samples], dtype=torch.float32, device=env.device
    )
    env._payload_com_to_cob_offsets = torch.as_tensor(
        [sample["com_to_cob_offset"] for sample in samples], dtype=torch.float32, device=env.device
    )
    env._payload_linear_damping_scales = torch.stack(
        [six_scale(sample, "linear_damping_scale") for sample in samples]
    )
    env._payload_quadratic_damping_scales = torch.stack(
        [six_scale(sample, "quadratic_damping_scale") for sample in samples]
    )
    env._payload_added_mass_scales = torch.stack(
        [six_scale(sample, "added_mass_scale") for sample in samples]
    )


def reset_rigid_body(env, env_ids: torch.Tensor, *, enabled: bool) -> bool:
    """Restore nominal rigid-body state and sample the selected feature.

    Returns whether a correlated payload sample was selected so its
    hydrodynamic factors can be composed after hydro DR is reset.
    """

    randomized = enabled
    payload_enabled = randomized and env._payload_sample_count > 0
    env.payload_sample_indices[env_ids] = -1

    if payload_enabled:
        payload_indices = torch.multinomial(env._payload_weights, len(env_ids), replacement=True)
        env.payload_sample_indices[env_ids] = payload_indices
        env.masses[env_ids, 0] = env._payload_masses[payload_indices]
        env.volumes[env_ids, 0] = env._payload_volumes[payload_indices]
        env.inertia_principal_moments[env_ids] = env._payload_principal_moments[payload_indices]
        env.inertia_principal_axes_xyzw[env_ids] = env._payload_principal_axes_xyzw[payload_indices]
        env.center_of_mass_offsets[env_ids] = env._payload_center_of_mass_offsets[payload_indices]
        env.com_to_cob_offsets[env_ids] = env._payload_com_to_cob_offsets[payload_indices]
    else:
        env.center_of_mass_offsets[env_ids] = torch.as_tensor(
            env.cfg.center_of_mass_offset, dtype=torch.float32, device=env.device
        )
        if randomized:
            mass_lower, mass_upper = env.cfg.domain_randomization.mass_range
            env.masses[env_ids] = sample_bounded_normal(
                mass_lower, mass_upper, env.masses[env_ids].shape, env.device
            )
        else:
            env.masses[env_ids] = float(env.cfg.mass)
        mass_ratio = env.masses[env_ids].reshape(-1, 1) / float(env.cfg.mass)
        env.inertia_principal_moments[env_ids] = env._nominal_principal_inertia.reshape(1, 3) * mass_ratio
        env.inertia_principal_axes_xyzw[env_ids] = env._nominal_principal_axes_xyzw.reshape(1, 4)

        nominal_offset = torch.as_tensor(
            env.cfg.com_to_cob_offset, device=env.device, dtype=env.com_to_cob_offsets.dtype
        ).reshape(1, 3)
        env.com_to_cob_offsets[env_ids] = nominal_offset
        if randomized:
            env.com_to_cob_offsets[env_ids] += sample_isotropic_bounded_normal(
                env.cfg.domain_randomization.com_to_cob_offset_radius,
                len(env_ids),
                3,
                env.device,
            )

        if randomized:
            volume_lower, volume_upper = env.cfg.domain_randomization.volume_range
            env.volumes[env_ids] = sample_bounded_normal(
                volume_lower, volume_upper, env.volumes[env_ids].shape, env.device
            )
        else:
            env.volumes[env_ids] = float(env.cfg.volume)

    env._apply_runtime_mass_properties(env_ids)
    env._apply_runtime_center_of_mass(env_ids)
    return payload_enabled


def apply_payload_hydrodynamics(env, env_ids: torch.Tensor) -> None:
    """Compose selected payload's correlated hydro factors after baseline DR."""

    payload_indices = env.payload_sample_indices[env_ids]
    env.linear_damping[env_ids] = scale_hydrodynamic_coefficients(
        env.linear_damping[env_ids], env._payload_linear_damping_scales[payload_indices]
    )
    env.quadratic_damping[env_ids] = scale_hydrodynamic_coefficients(
        env.quadratic_damping[env_ids], env._payload_quadratic_damping_scales[payload_indices]
    )
    env.added_mass_diag[env_ids] = scale_hydrodynamic_coefficients(
        env.added_mass_diag[env_ids], env._payload_added_mass_scales[payload_indices]
    )

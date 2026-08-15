"""Payload, mass, buoyancy, and centre-of-mass DR feature."""

from __future__ import annotations

import torch

from environment.profiles.random_sampling import sample_bounded_normal, sample_isotropic_bounded_normal
from robot.dynamics.rigid_body import physx_principal_inertia_and_com_quat_xyzw


def initialize_payload_domain(state, cfg) -> None:
    """Prepare categorical payload tensors once at environment construction."""

    samples = list(getattr(cfg.domain_randomization, "payload_samples", []))
    state.payload_sample_indices = torch.full(
        (state.num_envs,), -1, dtype=torch.long, device=state.device
    )
    state.payload_sample_count = len(samples)
    if not samples:
        return

    def six_scale(sample, name: str) -> torch.Tensor:
        value = torch.as_tensor(sample.get(name, 1.0), dtype=torch.float32, device=state.device).reshape(-1)
        return value.repeat(6) if value.numel() == 1 else value

    state.payload_weights = torch.as_tensor(
        [sample.get("weight", 1.0) for sample in samples], dtype=torch.float32, device=state.device
    )
    state.payload_masses = torch.as_tensor(
        [sample["mass"] for sample in samples], dtype=torch.float32, device=state.device
    )
    state.payload_volumes = torch.as_tensor(
        [sample["volume"] for sample in samples], dtype=torch.float32, device=state.device
    )
    principal_properties = [
        physx_principal_inertia_and_com_quat_xyzw(sample["inertia"], state.device) for sample in samples
    ]
    state.payload_principal_moments = torch.stack([properties[0] for properties in principal_properties])
    state.payload_principal_axes_xyzw = torch.stack([properties[1] for properties in principal_properties])
    state.payload_center_of_mass_offsets = torch.as_tensor(
        [sample["center_of_mass_offset"] for sample in samples], dtype=torch.float32, device=state.device
    )
    state.payload_com_to_cob_offsets = torch.as_tensor(
        [sample["com_to_cob_offset"] for sample in samples], dtype=torch.float32, device=state.device
    )
    state.payload_linear_damping_scales = torch.stack(
        [six_scale(sample, "linear_damping_scale") for sample in samples]
    )
    state.payload_quadratic_damping_scales = torch.stack(
        [six_scale(sample, "quadratic_damping_scale") for sample in samples]
    )
    state.payload_added_mass_scales = torch.stack(
        [six_scale(sample, "added_mass_scale") for sample in samples]
    )


def reset_rigid_body(state, cfg, env_ids: torch.Tensor, stage: int, *, enabled: bool):
    """Restore nominal rigid-body state and sample the selected feature.

    Returns whether a correlated payload sample was selected so its
    hydrodynamic factors can be composed after hydro DR is reset.
    """

    del stage  # Rigid-body sampling is not curriculum staged yet.
    randomized = enabled
    payload_enabled = randomized and state.payload_sample_count > 0
    state.payload_sample_indices[env_ids] = -1

    if payload_enabled:
        payload_indices = torch.multinomial(state.payload_weights, len(env_ids), replacement=True)
        state.payload_sample_indices[env_ids] = payload_indices
        state.masses[env_ids, 0] = state.payload_masses[payload_indices]
        state.volumes[env_ids, 0] = state.payload_volumes[payload_indices]
        state.inertia_principal_moments[env_ids] = state.payload_principal_moments[payload_indices]
        state.inertia_principal_axes_xyzw[env_ids] = state.payload_principal_axes_xyzw[payload_indices]
        state.center_of_mass_offsets[env_ids] = state.payload_center_of_mass_offsets[payload_indices]
        state.com_to_cob_offsets[env_ids] = state.payload_com_to_cob_offsets[payload_indices]
    else:
        state.center_of_mass_offsets[env_ids] = torch.as_tensor(
            cfg.center_of_mass_offset, dtype=torch.float32, device=state.device
        )
        if randomized:
            mass_lower, mass_upper = cfg.domain_randomization.mass_range
            state.masses[env_ids] = sample_bounded_normal(
                mass_lower, mass_upper, state.masses[env_ids].shape, state.device
            )
        else:
            state.masses[env_ids] = float(cfg.mass)
        mass_ratio = state.masses[env_ids].reshape(-1, 1) / float(cfg.mass)
        state.inertia_principal_moments[env_ids] = state.nominal_principal_inertia.reshape(1, 3) * mass_ratio
        state.inertia_principal_axes_xyzw[env_ids] = state.nominal_principal_axes_xyzw.reshape(1, 4)

        nominal_offset = torch.as_tensor(
            cfg.com_to_cob_offset, device=state.device, dtype=state.com_to_cob_offsets.dtype
        ).reshape(1, 3)
        state.com_to_cob_offsets[env_ids] = nominal_offset
        if randomized:
            state.com_to_cob_offsets[env_ids] += sample_isotropic_bounded_normal(
                cfg.domain_randomization.com_to_cob_offset_radius,
                len(env_ids),
                3,
                state.device,
            )

        if randomized:
            volume_lower, volume_upper = cfg.domain_randomization.volume_range
            state.volumes[env_ids] = sample_bounded_normal(
                volume_lower, volume_upper, state.volumes[env_ids].shape, state.device
            )
        else:
            state.volumes[env_ids] = float(cfg.volume)

    if not payload_enabled:
        return None
    from robot.runtime_state import PayloadHydrodynamicScale

    payload_indices = state.payload_sample_indices[env_ids]
    return PayloadHydrodynamicScale(
        state.payload_linear_damping_scales[payload_indices],
        state.payload_quadratic_damping_scales[payload_indices],
        state.payload_added_mass_scales[payload_indices],
    )

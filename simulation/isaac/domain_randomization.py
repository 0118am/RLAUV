"""Domain-randomization state, curriculum, reset, and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from environment.profiles.features import domain_randomization_feature_enabled
from environment.randomization import reset_current, reset_hydrodynamics
from robot.randomization import reset_actuators, reset_battery
from robot.randomization.rigid_body import (
    apply_payload_hydrodynamics,
    initialize_payload_domain,
    reset_rigid_body,
)


class AUVDomainRandomizationMixin:
    """Own the randomized physics domain independently of trajectory commands."""

    def _init_payload_domain(self) -> None:
        """Prepare a categorical ensemble of physically correlated payloads."""

        initialize_payload_domain(self)

    def _reset_domain(self, env_ids: Sequence[int]):
        env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        payload_enabled = reset_rigid_body(
            self,
            env_ids_device,
            enabled=self._domain_randomization_feature_enabled("rigid_body"),
        )
        self._reset_disturbance_domain(env_ids_device)
        if payload_enabled:
            apply_payload_hydrodynamics(self, env_ids_device)
        self._log_domain_randomization_state()

    def _domain_randomization_enabled(self) -> bool:
        return bool(self.cfg.domain_randomization.use_custom_randomization) and (
            not self.cfg.eval_mode or bool(getattr(self.cfg, "eval_domain_randomization", False))
        )

    def _domain_randomization_feature_enabled(self, feature: str) -> bool:
        """Return whether one independently managed DR feature is active."""

        return domain_randomization_feature_enabled(self, feature)

    def _log_domain_randomization_state(self) -> None:
        """Expose effective sampled domains to RSL-RL/TensorBoard.

        Statistics cover the full active vectorized environment after each
        reset.  This makes it possible to audit the distribution that actually
        reached PhysX instead of relying only on configured bounds.
        """

        interval = max(1, int(getattr(self.cfg, "domain_randomization_log_interval_steps", 250)))
        last_step = getattr(self, "_last_domain_randomization_log_step", None)
        if last_step is not None and self.common_step_counter - last_step < interval:
            return
        self._last_domain_randomization_log_step = self.common_step_counter

        log = self.extras.setdefault("log", {})
        # Keep terminal and TensorBoard fields compact. The surrounding
        # episode/log context already identifies these as randomized-domain
        # diagnostics, so a repeated ``DomainRandomization/`` namespace only
        # makes the rollout summary harder to scan.
        log["enabled"] = float(self._domain_randomization_enabled())
        for feature in (
            "rigid_body",
            "current",
            "hydrodynamics",
            "actuators",
            "battery",
        ):
            log[f"feature_{feature}_enabled"] = float(
                self._domain_randomization_feature_enabled(feature)
            )
        if hasattr(self.cfg.domain_randomization, "water_current_max_by_stage"):
            log["curriculum_stage"] = float(
                self._get_disturbance_curriculum_stage()
            )
            log["curriculum_global_step"] = float(
                self._disturbance_curriculum_global_step()
            )
            log["additional_hydrodynamics_scale"] = float(
                self._additional_hydrodynamics_scale()
            )

        def add_stats(name: str, values: torch.Tensor) -> None:
            flat = values.detach().to(dtype=torch.float32).reshape(-1)
            if flat.numel() == 0:
                return
            log[f"{name}_mean"] = flat.mean()
            log[f"{name}_std"] = flat.std(unbiased=False)
            log[f"{name}_min"] = flat.min()
            log[f"{name}_max"] = flat.max()

        add_stats("mass_kg", self.masses)
        add_stats("volume_m3", self.volumes)
        add_stats("center_of_mass_offset_m", torch.linalg.vector_norm(self.center_of_mass_offsets, dim=1))
        add_stats("com_to_cob_offset_m", torch.linalg.vector_norm(self.com_to_cob_offsets, dim=1))
        add_stats("principal_inertia_kg_m2", self.inertia_principal_moments)
        add_stats("added_mass_randomization_scale", self.added_mass_randomization_scale)
        add_stats("added_mass_coefficient", self.added_mass_diag)
        if self._payload_sample_count > 0:
            add_stats("payload_sample_index", self.payload_sample_indices)
        add_stats("water_current_mps", torch.linalg.vector_norm(self.water_current_w, dim=1))
        add_stats("thruster_force_scale", self.thruster_force_scale)
        add_stats("thruster_time_constant_s", self.thruster_time_constant)
        add_stats("thruster_delay_steps", self.thruster_delay_steps)
        add_stats("battery_voltage_v", self.battery_voltage)

    def _disturbance_curriculum_global_step(self) -> int:
        """Return the current policy-step count used by the DR curriculum."""

        return int(self.common_step_counter)

    def _get_disturbance_curriculum_stage(self) -> int:
        forced_eval_stage = int(getattr(self.cfg, "eval_disturbance_stage", -1))
        if self.cfg.eval_mode and forced_eval_stage >= 0:
            return min(forced_eval_stage, len(self.cfg.domain_randomization.water_current_max_by_stage) - 1)
        if not getattr(self.cfg.domain_randomization, "disturbance_curriculum", False):
            return len(self.cfg.domain_randomization.water_current_max_by_stage) - 1

        stage = 0
        for step_boundary in self.cfg.domain_randomization.disturbance_curriculum_stage_steps:
            if self._disturbance_curriculum_global_step() >= step_boundary:
                stage += 1
        return min(stage, len(self.cfg.domain_randomization.water_current_max_by_stage) - 1)

    def _reset_disturbance_domain(self, env_ids: Sequence[int]) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        stage = self._get_disturbance_curriculum_stage()
        reset_current(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("current"),
        )
        reset_hydrodynamics(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("hydrodynamics"),
        )
        reset_actuators(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("actuators"),
        )
        reset_battery(
            self,
            env_ids_device,
            stage,
            enabled=self._domain_randomization_feature_enabled("battery"),
        )
        self.tether_slack_length[env_ids_device] = self.cfg.tether_slack_length

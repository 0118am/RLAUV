"""RSL-RL runner variant with rollout-batched episode bookkeeping.

The upstream runner calls ``nonzero()`` and copies completed episode values to
the CPU after every environment step. CUDA ``nonzero`` requires a host
synchronization. This class preserves the training and logging contract while
collecting fixed-shape completion records on the GPU and transferring them once
per rollout.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state


class GpuBatchedOnPolicyRunner(OnPolicyRunner):
    """On-policy runner without per-step GPU-to-CPU episode bookkeeping."""

    @staticmethod
    def _git_backed_repositories(repository_paths: list[str]) -> list[str]:
        """Return unique paths that actually belong to a Git worktree."""
        valid = []
        for repository_path in repository_paths:
            path = Path(repository_path).resolve()
            search_root = path if path.is_dir() else path.parent
            if not any((parent / ".git").exists() for parent in (search_root, *search_root.parents)):
                continue
            value = str(path)
            if value not in valid:
                valid.append(value)
        return valid

    @staticmethod
    def _flush_completed_episodes(
        completed_episode_data: list[torch.Tensor],
        rewbuffer: deque,
        lenbuffer: deque,
    ) -> None:
        if not completed_episode_data:
            return
        completed_cpu = torch.stack(completed_episode_data).reshape(
            -1,
            completed_episode_data[0].shape[-1],
        ).cpu()
        completed_cpu = completed_cpu[completed_cpu[:, -1] > 0.0]
        rewbuffer.extend(completed_cpu[:, 0].tolist())
        lenbuffer.extend(completed_cpu[:, 1].tolist())

    def _collect_rollout(
        self,
        obs: torch.Tensor,
        *,
        logging_enabled: bool,
        ep_infos: list,
        cur_reward_sum: torch.Tensor,
        cur_episode_length: torch.Tensor,
        rewbuffer: deque,
        lenbuffer: deque,
    ) -> tuple[torch.Tensor, float]:
        started = time.time()
        completed_episode_data = []
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                actions = self.alg.act(obs)
                obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                self.alg.process_env_step(obs, rewards, dones, extras)

                if not logging_enabled:
                    continue
                if "episode" in extras:
                    ep_infos.append(extras["episode"])
                elif "log" in extras:
                    ep_infos.append(extras["log"])
                cur_reward_sum += rewards
                cur_episode_length += 1
                done_mask = dones > 0
                completed_episode_data.append(
                    torch.stack(
                        [cur_reward_sum, cur_episode_length, done_mask.to(dtype=torch.float32)],
                        dim=-1,
                    )
                )
                keep_mask = ~done_mask
                cur_reward_sum *= keep_mask
                cur_episode_length *= keep_mask

            if logging_enabled:
                self._flush_completed_episodes(completed_episode_data, rewbuffer, lenbuffer)
            self.alg.compute_returns(obs)
        return obs, time.time() - started

    def _store_initial_code_state(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        repositories = self._git_backed_repositories([*self.git_status_repos, str(project_root)])
        git_file_paths = store_code_state(self.log_dir, repositories)
        if self.logger_type in ["wandb", "neptune"]:
            for path in git_file_paths:
                self.writer.save_file(path)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        logging_enabled = self.log_dir is not None and not self.disable_logs

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            obs, collection_time = self._collect_rollout(
                obs,
                logging_enabled=logging_enabled,
                ep_infos=ep_infos,
                cur_reward_sum=cur_reward_sum,
                cur_episode_length=cur_episode_length,
                rewbuffer=rewbuffer,
                lenbuffer=lenbuffer,
            )
            start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            if logging_enabled:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and logging_enabled:
                self._store_initial_code_state()

        if logging_enabled:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

"""Training runner that binds the AUV Actor-curvature PPO implementation."""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

from simulation.training.ppo.networks import initialize_ppo_mlp
from simulation.training.ppo.squashed_actor_critic import AUVTanhGaussianActorCritic
from simulation.training.ppo.smooth_ppo import AUVSmoothPPO


class AUVOnPolicyRunner(OnPolicyRunner):
    """Construct the bounded feed-forward policy and :class:`AUVSmoothPPO`."""

    def _construct_algorithm(self, obs: TensorDict) -> AUVSmoothPPO:
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name")
        actor_critic = AUVTanhGaussianActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)
        initialize_ppo_mlp(actor_critic.actor, output_gain=0.01)
        initialize_ppo_mlp(actor_critic.critic, output_gain=1.0)

        algorithm_cfg = dict(self.alg_cfg)
        algorithm_cfg.pop("class_name")
        algorithm = AUVSmoothPPO(
            actor_critic,
            device=self.device,
            **algorithm_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

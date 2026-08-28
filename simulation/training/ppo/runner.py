"""Training runner that binds the AUV Actor-curvature PPO implementation."""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner

from simulation.training.ppo.networks import initialize_ppo_mlp
from simulation.training.ppo.smooth_ppo import AUVSmoothPPO


class AUVOnPolicyRunner(OnPolicyRunner):
    """Construct the current feed-forward ActorCritic and :class:`AUVSmoothPPO`."""

    def _construct_algorithm(self, obs: TensorDict) -> AUVSmoothPPO:
        self.policy_cfg.pop("class_name")
        actor_critic = ActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        initialize_ppo_mlp(actor_critic.actor, output_gain=0.01)
        initialize_ppo_mlp(actor_critic.critic, output_gain=1.0)

        self.alg_cfg.pop("class_name")
        algorithm = AUVSmoothPPO(
            actor_critic,
            device=self.device,
            **self.alg_cfg,
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

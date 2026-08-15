"""Isaac-compatible PPO configuration selected by the training system."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from simulation.training.ppo.networks import MLP_30D


@configclass
class AUVRolloutAdaptivePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO settings for one KL/LR decision after each rollout."""

    class_name: str = "RolloutAdaptivePPO"
    schedule: str = "rollout_adaptive"
    rollout_kl_stop: float = 0.015
    rollout_kl_low: float = 0.005
    rollout_lr_up_factor: float = 1.1
    rollout_lr_down_factor: float = 1.2
    rollout_lr_min: float = 1.0e-4
    rollout_lr_max: float = 5.0e-4


@configclass
class AUVTrajPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO profile for the trajectory curriculum.

    The campaign chooses the environment count; the repository notebook
    currently defaults to 1,024. Learning rate stays fixed for every rollout,
    KL stops excessive PPO updates early, and a bounded controller selects the
    rate for the next rollout.
    """

    # At 50 Hz, 256 rollout steps provide a 5.12 s collection window;
    # gamma/lambda determine the shorter effective GAE credit-assignment horizon.
    num_steps_per_env = 256
    max_iterations = 500
    save_interval = 50
    # The environment exposes a deployable ``policy`` group and a separate
    # simulator-only ``critic`` group.  Make this explicit instead of relying
    # on RSL-RL's fallback that reuses Actor observations for the value model.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    # The default agent entry is feed-forward. The named profile selected by
    # simulation.training overrides its widths and input layout together.
    experiment_name = "auv_traj_mlp"
    policy = RslRlPpoActorCriticCfg(
        # Keep the initial unsquashed Gaussian mostly inside the actuator's
        # normalized range; execution still clamps and reports any overflow.
        init_noise_std=0.5,
        # The environment applies fixed physical observation scales, so a
        # second running normalizer is intentionally disabled for deployment
        # parity.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=list(MLP_30D.actor_hidden_dims),
        critic_hidden_dims=list(MLP_30D.critic_hidden_dims),
        activation="elu",
    )
    algorithm = AUVRolloutAdaptivePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        gamma=0.997,
        lam=0.98,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

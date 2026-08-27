"""Isaac-compatible PPO configuration selected by the training system."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from simulation.training.ppo.networks import MLP_33D


@configclass
class AUVTrajPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Native RSL-RL PPO profile for the trajectory curriculum."""

    # At 25 Hz, 128 rollout steps provide a 5.12 s collection window;
    # gamma/lambda determine the shorter effective GAE credit-assignment horizon.
    num_steps_per_env = 128
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
        # Positive log-standard-deviation parameterization for the native
        # Gaussian policy.
        noise_std_type="log",
        # The environment applies fixed physical observation scales, so a
        # second running normalizer is intentionally disabled for deployment
        # parity.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=list(MLP_33D.actor_hidden_dims),
        critic_hidden_dims=list(MLP_33D.critic_hidden_dims),
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.994009,
        lam=0.9604,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

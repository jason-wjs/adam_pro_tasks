"""RL configuration for Adam Pro tracking task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

try:
  from mjlab.rl import RslRlModelCfg

  _USE_MODEL_CFG = True
except ImportError:
  from mjlab.rl import RslRlPpoActorCriticCfg

  _USE_MODEL_CFG = False


def adam_pro_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  if _USE_MODEL_CFG:
    model_cfg = {
      "actor": RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        obs_normalization=True,
        stochastic=True,
      ),
      "critic": RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        obs_normalization=True,
      ),
    }
  else:
    model_cfg = {
      "policy": RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
      )
    }

  return RslRlOnPolicyRunnerCfg(
    **model_cfg,
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="adam_pro_tracking",
    wandb_project="adam_pro_tasks",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )

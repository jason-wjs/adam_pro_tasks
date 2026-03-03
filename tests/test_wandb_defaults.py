from adam_pro_tasks.tasks.tracking.rl_cfg import adam_pro_tracking_ppo_runner_cfg
from adam_pro_tasks.tasks.velocity.rl_cfg import adam_pro_velocity_ppo_runner_cfg


def test_tracking_default_wandb_project() -> None:
  cfg = adam_pro_tracking_ppo_runner_cfg()
  assert cfg.wandb_project == "adam_pro_tasks"


def test_velocity_default_wandb_project() -> None:
  cfg = adam_pro_velocity_ppo_runner_cfg()
  assert cfg.wandb_project == "adam_pro_tasks"

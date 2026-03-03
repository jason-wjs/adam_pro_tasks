from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from adam_pro_tasks.tasks.velocity.env_cfgs import adam_pro_flat_env_cfg
from adam_pro_tasks.tasks.velocity.rl_cfg import adam_pro_velocity_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Adam-Pro",
  env_cfg=adam_pro_flat_env_cfg(),
  play_env_cfg=adam_pro_flat_env_cfg(play=True),
  rl_cfg=adam_pro_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from adam_pro_tasks.tasks.tracking.env_cfgs import adam_pro_flat_tracking_env_cfg
from adam_pro_tasks.tasks.tracking.rl_cfg import adam_pro_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Adam-Pro",
  env_cfg=adam_pro_flat_tracking_env_cfg(),
  play_env_cfg=adam_pro_flat_tracking_env_cfg(play=True),
  rl_cfg=adam_pro_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

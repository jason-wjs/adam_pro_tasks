"""Robot exports for Adam Pro tasks."""

from adam_pro_tasks.robots.adam_pro.adam_pro_29_constants import (
  ADAM_PRO_29_ACTION_SCALE,
  get_adam_pro_29_robot_cfg,
)

ADAM_PRO_ACTION_SCALE = ADAM_PRO_29_ACTION_SCALE
get_adam_pro_robot_cfg = get_adam_pro_29_robot_cfg

__all__ = [
  "ADAM_PRO_ACTION_SCALE",
  "get_adam_pro_29_robot_cfg",
  "get_adam_pro_robot_cfg",
]

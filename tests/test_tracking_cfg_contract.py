from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def test_tracking_env_cfg_contract() -> None:
  import adam_pro_tasks  # noqa: F401

  cfg = load_env_cfg("Mjlab-Tracking-Flat-Adam-Pro")
  assert cfg.commands is not None
  assert "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.anchor_body_name == "torso"
  assert len(motion_cmd.body_names) > 0

  assert cfg.scene.sensors is not None
  sensor_names = {sensor.name for sensor in cfg.scene.sensors}
  assert "self_collision" in sensor_names

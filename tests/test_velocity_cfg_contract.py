from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_velocity_env_cfg_contract() -> None:
  import adam_pro_tasks  # noqa: F401

  cfg = load_env_cfg("Mjlab-Velocity-Flat-Adam-Pro")
  assert cfg.commands is not None
  assert "twist" in cfg.commands
  assert isinstance(cfg.commands["twist"], UniformVelocityCommandCfg)

  assert "joint_pos" in cfg.actions
  assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)

  assert cfg.scene.sensors is not None
  sensor_names = {sensor.name for sensor in cfg.scene.sensors}
  assert "feet_ground_contact" in sensor_names
  assert "nonfoot_ground_touch" in sensor_names

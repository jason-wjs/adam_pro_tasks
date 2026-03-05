"""Adam Pro tracking environment configuration."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from adam_pro_tasks.robots.adam_pro.adam_pro_29_constants import (
  ADAM_PRO_29_ACTION_SCALE,
  get_adam_pro_29_robot_cfg,
)
from adam_pro_tasks.tasks.tracking.motion_io import (
  TRACKING_ANCHOR_BODY,
  TRACKING_BODY_NAMES,
)

ADAM_PRO_ACTION_RATE_L2_WEIGHT = -0.08

ADAM_PRO_TRACKING_TERMINATION_THRESHOLDS: dict[str, float] = {
  "anchor_pos": 0.35,
  "ee_body_pos": 0.35,
}

ADAM_PRO_PLAY_SAMPLING_MODE_ENV_VAR = "ADAM_PRO_PLAY_SAMPLING_MODE"
ADAM_PRO_PLAY_SAMPLING_MODE_DEFAULT = "uniform"
ADAM_PRO_PLAY_SAMPLING_MODES = ("start", "uniform", "adaptive")


def _resolve_play_sampling_mode() -> str:
  mode = os.environ.get(
    ADAM_PRO_PLAY_SAMPLING_MODE_ENV_VAR, ADAM_PRO_PLAY_SAMPLING_MODE_DEFAULT
  )
  resolved_mode = mode.strip().lower()
  if resolved_mode not in ADAM_PRO_PLAY_SAMPLING_MODES:
    allowed_modes = ", ".join(ADAM_PRO_PLAY_SAMPLING_MODES)
    raise ValueError(
      f"{ADAM_PRO_PLAY_SAMPLING_MODE_ENV_VAR} must be one of: {allowed_modes}; "
      f"got: {mode!r}"
    )
  return resolved_mode


def _policy_obs_group_name(cfg: ManagerBasedRlEnvCfg) -> str:
  if "policy" in cfg.observations:
    return "policy"
  if "actor" in cfg.observations:
    return "actor"
  raise KeyError("Expected observations to contain 'policy' or 'actor'.")


def _remap_base_imu_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  """Map base velocity observations to sensor names present in Adam Pro XML."""
  for group_name in ("policy", "actor", "critic"):
    if group_name not in cfg.observations:
      continue
    terms = cfg.observations[group_name].terms
    if "base_lin_vel" in terms:
      terms["base_lin_vel"].params["sensor_name"] = "robot/BodyVel"
    if "base_ang_vel" in terms:
      terms["base_ang_vel"].params["sensor_name"] = "robot/BodyGyro"


def _apply_adam_pro_action_rate_override(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply minimal R1 reward override for action-rate penalty."""
  cfg.rewards["action_rate_l2"].weight = ADAM_PRO_ACTION_RATE_L2_WEIGHT


def _apply_adam_pro_termination_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Relax selected termination thresholds for Adam Pro morphology."""
  cfg.terminations["anchor_pos"].params["threshold"] = (
    ADAM_PRO_TRACKING_TERMINATION_THRESHOLDS["anchor_pos"]
  )
  cfg.terminations["ee_body_pos"].params["threshold"] = (
    ADAM_PRO_TRACKING_TERMINATION_THRESHOLDS["ee_body_pos"]
  )


def adam_pro_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Adam Pro flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_adam_pro_29_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ADAM_PRO_29_ACTION_SCALE

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = ""
  motion_cmd.anchor_body_name = TRACKING_ANCHOR_BODY
  motion_cmd.body_names = TRACKING_BODY_NAMES

  _remap_base_imu_sensors(cfg)
  _apply_adam_pro_action_rate_override(cfg)
  _apply_adam_pro_termination_overrides(cfg)

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-5]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "toeLeft",
    "toeRight",
    "wristRollLeft",
    "wristRollRight",
  )

  cfg.viewer.body_name = "torso"

  if play:
    cfg.episode_length_s = int(1e9)
    policy_group = _policy_obs_group_name(cfg)
    cfg.observations[policy_group].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = _resolve_play_sampling_mode()

  return cfg

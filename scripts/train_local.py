#!/usr/bin/env python3
"""Train tracking task from a local motion npz file.

This bypasses mjlab's train.py registry-name requirement for tracking tasks.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

import adam_pro_tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task-id",
    default="Mjlab-Tracking-Flat-Adam-Pro",
    help="Tracking task id to train",
  )
  parser.add_argument("--motion-file", required=True, help="Local motion npz path")
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--max-iters", type=int, default=30000)
  parser.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu")
  parser.add_argument("--run-name", default=None)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()

  motion_path = Path(args.motion_file).expanduser().resolve()
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")

  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  env_cfg = load_env_cfg(args.task_id)
  agent_cfg = load_rl_cfg(args.task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )
  if not is_tracking_task:
    raise ValueError(f"Task {args.task_id} is not a tracking task.")

  assert env_cfg.commands is not None
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion_path)

  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iters
  if args.run_name:
    agent_cfg.run_name = args.run_name

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_name += f"_{agent_cfg.run_name}"
  log_dir = log_root / log_name

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device)

  add_wandb_tags(agent_cfg.wandb_tags)
  runner.add_git_repo_to_log(__file__)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations,
    init_at_random_ep_len=True,
  )

  env.close()


if __name__ == "__main__":
  main()

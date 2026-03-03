#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$(dirname "${BASH_SOURCE[0]}")"

## tracking (registry)
# uv run train Mjlab-Tracking-Flat-Adam-Pro \
#   --registry-name "humanoid/motions/adam-pro-motion:latest" \
#   --env.scene.num-envs 4096 \
#   --agent.max-iterations 30000 \
#   "$@"

## tracking (local motion file)
uv run python scripts/train_local.py \
  --task-id Mjlab-Tracking-Flat-Adam-Pro \
  --motion-file /home/humanoid/Downloads/Data/adam_pro_29dof_retarget_bm/BOXING4_Skeleton+004_z_up_x_forward_gym_bm.npz \
  --device cuda:0 \
  --num-envs 8192 \
  --max-iters 10000 \
  "$@"

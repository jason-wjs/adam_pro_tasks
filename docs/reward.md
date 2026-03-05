# Adam Pro Tracking Reward & Termination Tuning Log

## Scope & Rules

- This file is the single source of truth for tracking reward/termination tuning history.
- `mjlab` defaults are read-only reference values and must not be edited in-place.
- Every round must record:
  - what changed from default,
  - what stayed unchanged,
  - run/log path,
  - observed metrics,
  - postmortem and next action.
- Round naming:
  - `R0`: baseline run with fully reused `mjlab` tracking defaults on Adam Pro.
  - `R1`, `R2`, ...: subsequent tuning rounds.

## Read-Only mjlab Defaults (Reference)

Reference source (read-only):
- `/home/humanoid/Projects/Junsong_WU/adam_reference/adam_pro_tasks/.venv/lib/python3.10/site-packages/mjlab/tasks/tracking/tracking_env_cfg.py`

### Reward Defaults

| Term | Default Weight | Default Std |
|---|---:|---:|
| `motion_global_root_pos` | `0.5` | `0.3` |
| `motion_global_root_ori` | `0.5` | `0.4` |
| `motion_body_pos` | `1.0` | `0.3` |
| `motion_body_ori` | `1.0` | `0.4` |
| `motion_body_lin_vel` | `1.0` | `1.0` |
| `motion_body_ang_vel` | `1.0` | `3.14` |
| `action_rate_l2` | `-0.1` | N/A |
| `joint_limit` | `-10.0` | N/A |
| `self_collisions` | `-10.0` | N/A |

### Termination Defaults

| Term | Default Threshold |
|---|---:|
| `anchor_pos` | `0.25` |
| `anchor_ori` | `0.8` |
| `ee_body_pos` | `0.25` |

### Motion Command Randomization Defaults

| Field | Default |
|---|---|
| `pose_range` | `x/y: ±0.05, z: ±0.01, roll/pitch: ±0.1, yaw: ±0.2` |
| `velocity_range` | `x/y: ±0.5, z: ±0.2, roll/pitch: ±0.52, yaw: ±0.78` |
| `joint_position_range` | `(-0.1, 0.1)` |

## Baseline Round (R0): Adam + Fully Reused mjlab Tracking Defaults

### Run Info

- Log path:
  - `/home/humanoid/Projects/Junsong_WU/adam_reference/adam_pro_tasks/logs/rsl_rl/adam_pro_tracking/2026-03-03_17-59-12`
- Motion file:
  - `/home/humanoid/Downloads/Data/adam_pro_29dof_retarget_bm/BOXING4_Skeleton+004_z_up_x_forward_gym_bm.npz`
- Config snapshot:
  - `action_rate_l2=-0.1`
  - `anchor_pos=0.25`
  - `anchor_ori=0.8`
  - `ee_body_pos=0.25`
  - 6 tracking reward terms all at `mjlab` defaults.

### Metrics Snapshot (Raw)

```text
Episode_Reward/action_rate_l2: -0.8691542148590088
Episode_Reward/joint_limit: -0.013260445557534696
Episode_Reward/motion_body_ang_vel: 0.4276978373527527
Episode_Reward/motion_body_lin_vel: 0.41057533025741577
Episode_Reward/motion_body_ori: 0.482542484998703
Episode_Reward/motion_body_pos: 0.6750295162200928
Episode_Reward/motion_global_root_ori: 0.33062291145324707
Episode_Reward/motion_global_root_pos: 0.16450917720794678
Episode_Reward/self_collisions: -0.024409063160419464
Episode_Termination/anchor_ori: 0
Episode_Termination/anchor_pos: 0.8333333730697632
Episode_Termination/ee_body_pos: 8.125
Episode_Termination/time_out: 13.708333969116213
Loss/entropy: 18.871556186676024
Loss/learning_rate: 0.00005062500000000002
Loss/surrogate: -0.006214211261249147
Loss/value: 0.07619956471025943
Metrics/motion/error_anchor_ang_vel: 2.017754316329956
Metrics/motion/error_anchor_lin_vel: 0.9053330421447754
Metrics/motion/error_anchor_pos: 0.3424443006515503
Metrics/motion/error_anchor_rot: 0.22569775581359863
Metrics/motion/error_body_ang_vel: 3.102552890777588
Metrics/motion/error_body_lin_vel: 1.106835126876831
Metrics/motion/error_body_pos: 0.11601941287517548
Metrics/motion/error_body_rot: 0.3513205051422119
Metrics/motion/error_joint_pos: 1.324202060699463
Metrics/motion/error_joint_vel: 11.61037254333496
Metrics/motion/sampling_entropy: 0.7418785095214844
Metrics/motion/sampling_top1_bin: 0.7096773982048035
Metrics/motion/sampling_top1_prob: 0.20563754439353943
Perf/collection_time: 1.7769432067871094
Perf/learning_time: 0.13893914222717285
Perf/total_fps: 102,620
Policy/mean_noise_std: 0.4701199531555176
Train/mean_episode_length: 380.62
Train/mean_reward: 16.530836123158224
```

### R0 Problem Summary

- Early termination pressure is high, dominated by `ee_body_pos`, then `anchor_pos`.
- Root global position tracking is weak (`motion_global_root_pos` low, `error_anchor_pos` high).
- Velocity tracking errors (`body_lin_vel/body_ang_vel/joint_vel`) are still high.
- Reward/termination defaults reused from `mjlab` are functional but not yet tuned for Adam Pro morphology.

## Round Log

## R1 (Planned): Minimal Stabilization Round

### Objective

Improve training stability and episode survivability with minimal intervention.

### Changes from Default

| Item | Default | R1 Planned |
|---|---:|---:|
| `terminations.anchor_pos.threshold` | `0.25` | `0.30` |
| `terminations.ee_body_pos.threshold` | `0.25` | `0.35` |
| `rewards.action_rate_l2.weight` | `-0.10` | `-0.08` |

### Kept Unchanged

- `terminations.anchor_ori.threshold = 0.8`
- All six tracking reward terms keep default `weight/std`.
- `joint_limit` and `self_collisions` stay unchanged.
- PPO hyperparameters unchanged.

### Codebase Consistency Check Before Running R1

- R1 intent is "change only 3 knobs" (two terminations + `action_rate_l2`).
- Current working tree should be checked to ensure no extra reward overrides are active.
- If extra overrides exist in `src/adam_pro_tasks/tasks/tracking/env_cfgs.py`, remove/disable them before launching R1.

### Expected Effect

- Reduce premature resets from end-effector and anchor-z mismatch.
- Increase effective episode length and improve usable training signal.
- Keep behavior conservative by limiting changes to three knobs only.

### Run Command / Log Path

- Command: `./train.sh` (R1 config with 3 changes only)
- Log path: `TBD (fill exact run folder after archiving)`

### Result Metrics

- `Episode_Reward/action_rate_l2`: `-0.8990692496299744`
- `Episode_Reward/joint_limit`: `-0.015419535338878632`
- `Episode_Reward/motion_body_ang_vel`: `0.46789973974227905`
- `Episode_Reward/motion_body_lin_vel`: `0.46027061343193054`
- `Episode_Reward/motion_body_ori`: `0.523145854473114`
- `Episode_Reward/motion_body_pos`: `0.7439004778862`
- `Episode_Reward/motion_global_root_ori`: `0.3672536611557007`
- `Episode_Reward/motion_global_root_pos`: `0.20539610087871552`
- `Episode_Reward/self_collisions`: `-0.014295997098088264`
- `Episode_Termination/anchor_ori`: `0`
- `Episode_Termination/anchor_pos`: `0.875`
- `Episode_Termination/ee_body_pos`: `4.625`
- `Episode_Termination/time_out`: `13.791666984558104`
- `Loss/entropy`: `21.12652940750122`
- `Loss/learning_rate`: `0.00011390625000000003`
- `Loss/surrogate`: `-0.008314565941691398`
- `Loss/value`: `0.06049388442188501`
- `Metrics/motion/error_anchor_ang_vel`: `1.9101788997650144`
- `Metrics/motion/error_anchor_lin_vel`: `0.8509263396263123`
- `Metrics/motion/error_anchor_pos`: `0.3385317027568817`
- `Metrics/motion/error_anchor_rot`: `0.22671012580394745`
- `Metrics/motion/error_body_ang_vel`: `3.0500800609588623`
- `Metrics/motion/error_body_lin_vel`: `1.0624194145202637`
- `Metrics/motion/error_body_pos`: `0.119429811835289`
- `Metrics/motion/error_body_rot`: `0.36265426874160767`
- `Metrics/motion/error_joint_pos`: `1.325148582458496`
- `Metrics/motion/error_joint_vel`: `11.156824111938477`
- `Metrics/motion/sampling_entropy`: `0.7380456328392029`
- `Metrics/motion/sampling_top1_bin`: `0.774193525314331`
- `Metrics/motion/sampling_top1_prob`: `0.2690756618976593`
- `Perf/collection_time`: `1.768047332763672`
- `Perf/learning_time`: `0.13854765892028809`
- `Perf/total_fps`: `103,119`
- `Policy/mean_noise_std`: `0.5080177187919617`
- `Train/mean_episode_length`: `426.38`
- `Train/mean_reward`: `19.372882544919847`

### Postmortem

- R1 converges faster and to higher reward than R0.
- `ee_body_pos` termination is significantly reduced, confirming threshold/action-rate changes are effective for early reset pressure.
- Mean episode length improved but still below target near 500; `anchor_pos` termination remains non-trivial and root alignment is still the weak link.
- Adaptive sampling concentration increased (`sampling_top1_prob` up), which may slow global-error cleanup.

### Next-Round Proposal

- Keep R1 changes.
- Prioritize root-related tracking tuning before broad changes:
  - Option A (safer): increase only `motion_global_root_pos` weight/std slightly.
  - Option B: keep rewards unchanged, but further relax `anchor_pos` by a small step.
- Avoid large multi-knob changes in one round; keep R2 as a narrow A/B test.

## R2 (Done): Anchor-Only Relaxation

### Objective

Try a single additional change to reduce root-related early resets without touching reward structure.

### Changes from R1

| Item | R1 | R2 Planned |
|---|---:|---:|
| `terminations.anchor_pos.threshold` | `0.30` | `0.35` |

### Kept Unchanged

- `terminations.ee_body_pos.threshold = 0.35`
- `terminations.anchor_ori.threshold = 0.8`
- Six tracking reward terms unchanged from default.
- Three regularization terms unchanged from R1 (`action_rate_l2=-0.08`, `joint_limit`, `self_collisions`).
- PPO hyperparameters unchanged.

### Run Command / Log Path

- Command: `./train.sh`
- Log path: `TBD (fill exact run folder after archiving)`

### Result Metrics

- `Episode_Reward/action_rate_l2`: `-0.8694533705711365`
- `Episode_Reward/joint_limit`: `-0.014380661770701408`
- `Episode_Reward/motion_body_ang_vel`: `0.4696875512599945`
- `Episode_Reward/motion_body_lin_vel`: `0.45426124334335327`
- `Episode_Reward/motion_body_ori`: `0.5371050834655762`
- `Episode_Reward/motion_body_pos`: `0.7427890300750732`
- `Episode_Reward/motion_global_root_ori`: `0.3672854602336883`
- `Episode_Reward/motion_global_root_pos`: `0.21030926704406736`
- `Episode_Reward/self_collisions`: `-0.01373525895178318`
- `Episode_Termination/anchor_ori`: `0`
- `Episode_Termination/anchor_pos`: `0.0416666679084301`
- `Episode_Termination/ee_body_pos`: `6.333333492279053`
- `Episode_Termination/time_out`: `15.791666984558104`
- `Loss/entropy`: `20.585213661193848`
- `Loss/learning_rate`: `0.00011390625000000003`
- `Loss/surrogate`: `-0.007626400503795594`
- `Loss/value`: `0.06596574615687131`
- `Metrics/motion/error_anchor_ang_vel`: `2.047815322875977`
- `Metrics/motion/error_anchor_lin_vel`: `0.8309947848320007`
- `Metrics/motion/error_anchor_pos`: `0.31704872846603394`
- `Metrics/motion/error_anchor_rot`: `0.23168997466564176`
- `Metrics/motion/error_body_ang_vel`: `3.0357675552368164`
- `Metrics/motion/error_body_lin_vel`: `1.0542948246002195`
- `Metrics/motion/error_body_pos`: `0.11522099375724792`
- `Metrics/motion/error_body_rot`: `0.35004734992980957`
- `Metrics/motion/error_joint_pos`: `1.2921209335327148`
- `Metrics/motion/error_joint_vel`: `10.915472984313965`
- `Metrics/motion/sampling_entropy`: `0.7122475504875183`
- `Metrics/motion/sampling_top1_bin`: `0.7741936445236206`
- `Metrics/motion/sampling_top1_prob`: `0.31884199380874634`
- `Perf/collection_time`: `1.781224012374878`
- `Perf/learning_time`: `0.13982582092285156`
- `Perf/total_fps`: `102,344`
- `Policy/mean_noise_std`: `0.49888119101524353`
- `Train/mean_episode_length`: `400.05`
- `Train/mean_reward`: `18.712668476793915`

### Postmortem

- Anchor-only relaxation achieved its direct target:
  - `anchor_pos` termination dropped sharply (`0.875 -> 0.0417` vs R1).
- But overall training quality regressed from R1:
  - `mean_episode_length` decreased (`426.38 -> 400.05`),
  - `mean_reward` decreased (`19.37 -> 18.71`),
  - `ee_body_pos` termination increased (`4.625 -> 6.333`).
- Root positional tracking improved (`error_anchor_pos` reduced), but this gain did not convert to better final rollout quality.
- Sampling concentration increased further (`sampling_top1_prob` up), suggesting optimization may be over-focused on difficult bins while shifting failures to end-effectors.

### Next-Round Proposal

- Revert `anchor_pos` threshold from `0.35` back to `0.30` (restore R1 baseline behavior).
- Keep `action_rate_l2=-0.08` and `ee_body_pos=0.35` unchanged.
- Start tuning one reward knob at a time from the R1 baseline (prioritize root-related term first), instead of further termination relaxation.

## Round Template

Copy this section for each new round (`R2`, `R3`, ...):

```markdown
## RX (Planned/Done): <short title>

### Objective
- <goal>

### Changes from Default
| Item | Default | RX |
|---|---:|---:|
| <key> | <v0> | <vx> |

### Kept Unchanged
- <unchanged key list>

### Run Command / Log Path
- Command: <command>
- Log path: <log path>

### Result Metrics
- Train/mean_reward: <value>
- Train/mean_episode_length: <value>
- Episode_Termination/anchor_pos: <value>
- Episode_Termination/ee_body_pos: <value>
- Metrics/motion/error_anchor_pos: <value>
- Metrics/motion/error_body_lin_vel: <value>
- Metrics/motion/error_body_ang_vel: <value>

### Postmortem
- <what improved>
- <what regressed>
- <most likely reason>

### Next-Round Proposal
- <next changes>
```

## Decision Rules for Next Round

- If `ee_body_pos` termination remains dominant, prioritize termination/range calibration over PPO changes.
- If episode length increases but root errors remain high, prioritize root reward tuning before body fine-tuning.
- If velocity errors remain high after stability improves, adjust velocity-related reward `std` before large weight changes.
- Keep one round focused: avoid changing too many dimensions at once unless emergency recovery is needed.

# adam_pro_tasks

`adam_pro_tasks` is an external mjlab task package for Adam Pro.

## Included tasks

- `Mjlab-Velocity-Flat-Adam-Pro`
- `Mjlab-Tracking-Flat-Adam-Pro`

## Usage

```bash
# List discovered tasks
uv run list_envs

# Play velocity task
uv run play Mjlab-Velocity-Flat-Adam-Pro --agent zero

# Train tracking task from local motion file (recommended)
./train.sh /path/to/motion.npz

# Train tracking task from existing artifact (compatible with mjlab flow)
./train.sh --registry-name <entity/project/artifact:latest>
```

By default, Adam Pro runs log to W&B project `adam_pro_tasks`.
You can still override project name via CLI (for example:
`--agent.wandb_project <other_project>` when using registry mode).

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

# Train tracking task (requires motion artifact)
uv run train Mjlab-Tracking-Flat-Adam-Pro --registry-name <wandb-artifact>
```

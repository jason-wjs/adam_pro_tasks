from mjlab.tasks.registry import list_tasks


def test_adam_pro_tasks_are_registered() -> None:
  import adam_pro_tasks  # noqa: F401

  task_ids = list_tasks()
  assert "Mjlab-Velocity-Flat-Adam-Pro" in task_ids
  assert "Mjlab-Tracking-Flat-Adam-Pro" in task_ids

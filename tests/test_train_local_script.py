from pathlib import Path


def test_train_local_script_exists() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  assert (repo_root / "scripts" / "train_local.py").exists()

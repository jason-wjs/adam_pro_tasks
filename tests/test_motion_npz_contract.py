from pathlib import Path

import numpy as np
import pytest

from adam_pro_tasks.tasks.tracking.motion_io import validate_motion_npz


def test_validate_motion_npz_rejects_missing_keys(tmp_path: Path) -> None:
  bad_npz = tmp_path / "bad_motion.npz"
  np.savez_compressed(bad_npz, joint_pos=np.zeros((2, 29), dtype=np.float32))

  with pytest.raises(ValueError, match="missing keys"):
    validate_motion_npz(bad_npz)

#!/usr/bin/env python3
"""Offline motion conversion helper for Adam Pro tracking tasks."""

from __future__ import annotations

from pathlib import Path

import tyro

from adam_pro_tasks.tasks.tracking.motion_io import prepare_motion_npz


class Args(tyro.conf.FlagConversionOff):
  input: Path
  cache_dir: Path | None = None


def main() -> None:
  args = tyro.cli(Args)
  output = prepare_motion_npz(args.input, cache_dir=args.cache_dir)
  print(output)


if __name__ == "__main__":
  main()

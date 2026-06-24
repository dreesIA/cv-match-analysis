#!/usr/bin/env python
"""CLI entrypoint for the updt.pro CV MVP pipeline.

Examples:
  python run.py --source data/raw_matches/sample.mp4 --max-frames 750
  python run.py --source data/raw_matches/sample.mp4 --stride 2 --device cpu
"""
from __future__ import annotations

import argparse

from src.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="updt.pro CV MVP — assisted tagging pipeline")
    ap.add_argument("--source", required=True, help="path to match video")
    ap.add_argument("--output-dir", default="outputs", help="where to write events + summary")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="cap processed frames (debug)")
    ap.add_argument("--device", default=None, help="override device (mps|cpu|cuda)")
    args = ap.parse_args()

    run(
        source=args.source,
        output_dir=args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        device=args.device,
    )


if __name__ == "__main__":
    main()

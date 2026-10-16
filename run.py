#!/usr/bin/env python
"""CLI entrypoint for the CV Match Analysis pipeline.

Examples:
  python run.py --source data/raw_matches/sample.mp4 --max-frames 750
  python run.py --source data/raw_matches/sample.mp4 --stride 2 --device cpu
"""
from __future__ import annotations

import argparse

from src.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="CV Match Analysis — assisted tagging pipeline")
    ap.add_argument("--source", required=True, help="path to match video")
    ap.add_argument("--output-dir", default="outputs", help="where to write events + summary")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="cap processed frames (debug)")
    ap.add_argument("--device", default=None, help="override device (mps|cpu|cuda)")
    ap.add_argument("--ball-inference", choices=("sliced", "full_frame"), default=None,
                    help="dedicated-ball inference mode (default: configured sliced mode)")
    ap.add_argument("--ball-imgsz", type=int, default=None,
                    help="ball-model input size for full-frame inference")
    ap.add_argument("--player-imgsz", type=int, default=None,
                    help="override player-model input size")
    ap.add_argument("--classify-every", type=int, default=1,
                    help="run visual team classification every N processed frames")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch local model inference while preserving sequential tracking")
    ap.add_argument("--ball-every", type=int, default=1,
                    help="run the ball detector every N processed frames")
    ap.add_argument("--pitch-every", type=int, default=1,
                    help="refresh pitch calibration every N processed frames")
    ap.add_argument("--warmup-every", type=int, default=30,
                    help="sample every Nth source frame for team-cluster fitting")
    ap.add_argument("--warmup-cap", type=int, default=600,
                    help="maximum player crops used to fit team clusters")
    args = ap.parse_args()

    run(
        source=args.source,
        output_dir=args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        device=args.device,
        ball_inference=args.ball_inference,
        ball_imgsz=args.ball_imgsz,
        player_imgsz=args.player_imgsz,
        classify_every=args.classify_every,
        batch_size=args.batch_size,
        ball_every=args.ball_every,
        pitch_every=args.pitch_every,
        warmup_every=args.warmup_every,
        warmup_cap=args.warmup_cap,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download pretrained weights + a sample clip (no Roboflow API key needed).

Pulls the trained YOLOv8 weights and DFL sample videos that the roboflow/sports
soccer example hosts on Google Drive, saving weights under data/models/ with the
names configs/detection.yaml expects (players.pt / ball.pt / pitch.pt).

Usage:
  python scripts/download_assets.py            # weights + one sample clip
  python scripts/download_assets.py --all-videos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gdown

REPO = Path(__file__).resolve().parents[1]

WEIGHTS = {  # gdrive id -> local filename (matches detection.yaml local_weights)
    "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q": "players.pt",
    "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V": "ball.pt",
    "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf": "pitch.pt",
}

VIDEOS = {  # gdrive id -> filename
    "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf": "2e57b9_0.mp4",   # default smoke-test clip
    "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF": "0bfacc_0.mp4",
    "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-": "08fd33_0.mp4",
    "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU": "573e61_0.mp4",
    "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu": "121364_0.mp4",
}
DEFAULT_VIDEO = "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"


def _get(file_id: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} already present")
        return
    gdown.download(id=file_id, output=str(out), quiet=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-videos", action="store_true", help="download all 5 sample clips")
    args = ap.parse_args()

    for fid, name in WEIGHTS.items():
        _get(fid, REPO / "data" / "models" / name)

    ids = VIDEOS if args.all_videos else {DEFAULT_VIDEO: VIDEOS[DEFAULT_VIDEO]}
    for fid, name in ids.items():
        _get(fid, REPO / "data" / "raw_matches" / name)

    print("[done] assets ready under data/models/ and data/raw_matches/")


if __name__ == "__main__":
    main()

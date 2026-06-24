"""Golden-set format + loader (plan §10.1).

A golden clip is a directory under data/golden_set/<clip_id>/ that is NEVER used for
training and is version-controlled. Annotations are optional per type, so you can
grow coverage incrementally.

    data/golden_set/<clip_id>/
      manifest.yaml          # metadata + conditions (kit_clash, degraded, set_piece...)
      video.mp4              # gitignored (media kept out-of-band)
      detections.json        # [{frame, boxes:[{xyxy:[x1,y1,x2,y2], class}]}]   (COCO-ish)
      tracks_gt.txt          # MOT-challenge rows (frame,id,l,t,w,h,conf,-1,-1,-1)
      team_labels.json       # {track_id: {"team": 0|1|null, "role": "player|goalkeeper|referee"}}
      keypoints.json         # [{frame, points:[{index, x, y}], pitch_m:[{index, x, y}]}]
      events_gt.json         # analyst event log (subset of the Event schema)

manifest.yaml example:
    clip_id: gs_2e57b9_0
    source: 2e57b9_0.mp4
    fps: 25
    conditions: [kit_clash, set_piece]
    annotated: [detections, tracks, team_labels, keypoints, events]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden_set"


@dataclass
class GoldenClip:
    clip_id: str
    path: Path
    manifest: dict[str, Any]

    def _load_json(self, name: str) -> Optional[Any]:
        p = self.path / name
        return json.loads(p.read_text()) if p.exists() else None

    def detections(self) -> Optional[Any]:
        return self._load_json("detections.json")

    def team_labels(self) -> Optional[dict[str, Any]]:
        return self._load_json("team_labels.json")

    def keypoints(self) -> Optional[Any]:
        return self._load_json("keypoints.json")

    def events(self) -> Optional[Any]:
        return self._load_json("events_gt.json")

    def tracks_mot(self) -> Optional[list[list[float]]]:
        p = self.path / "tracks_gt.txt"
        if not p.exists():
            return None
        rows = []
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append([float(x) for x in line.split(",")])
        return rows


def load_clip(clip_id: str) -> GoldenClip:
    path = GOLDEN_DIR / clip_id
    manifest = yaml.safe_load((path / "manifest.yaml").read_text())
    return GoldenClip(clip_id=clip_id, path=path, manifest=manifest)


def list_clips() -> list[str]:
    if not GOLDEN_DIR.exists():
        return []
    return sorted(p.name for p in GOLDEN_DIR.iterdir() if p.is_dir())

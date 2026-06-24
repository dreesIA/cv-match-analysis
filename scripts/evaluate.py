#!/usr/bin/env python
"""Evaluate a golden-set clip across every component (plan §10.2) and print a report.

Runs only the metrics whose ground-truth annotations exist for the clip, so you can
grow golden-set coverage incrementally.

Usage:
  python scripts/evaluate.py --clip gs_2e57b9_0
  python scripts/evaluate.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schema import Event
from src.evaluation.golden_set import list_clips, load_clip
from src.evaluation.tagging import evaluate_events
from src.evaluation.tracking import evaluate_mot


def _events_from_json(blob) -> list[Event]:
    out = []
    for e in blob:
        out.append(Event(
            type=e["type"], start_frame=e["start_frame"], end_frame=e["end_frame"],
            start_ts=e.get("start_ts", 0.0), end_ts=e.get("end_ts", 0.0),
            team_id=e.get("team_id"), player_ids=e.get("player_ids", []),
            attributes=e.get("attributes", {}),
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="golden-set clip id (dir under data/golden_set/)")
    ap.add_argument("--pred-events", help="path to predicted events.json (for tagging eval)")
    ap.add_argument("--pred-tracks", help="path to predicted MOT rows .txt (for tracking eval)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.clip:
        print("Golden-set clips:", list_clips() or "(none yet — see src/evaluation/golden_set.py for the format)")
        return

    clip = load_clip(args.clip)
    report: dict = {"clip": args.clip, "conditions": clip.manifest.get("conditions", [])}

    # ---- tagging ----
    gt_events = clip.events()
    if gt_events is not None and args.pred_events:
        pred = _events_from_json(json.loads(Path(args.pred_events).read_text()))
        report["tagging"] = evaluate_events(pred, _events_from_json(gt_events))

    # ---- tracking ----
    gt_tracks = clip.tracks_mot()
    if gt_tracks is not None and args.pred_tracks:
        pred_rows = [[float(x) for x in line.split(",")]
                     for line in Path(args.pred_tracks).read_text().splitlines() if line.strip()]
        report["tracking"] = evaluate_mot(pred_rows, gt_tracks)

    # detection / calibration / team eval call src.evaluation.{detection,calibration,team}
    # with predictions collected during a pipeline run; wire those once the matching
    # golden-set annotations exist (detections.json / keypoints.json / team_labels.json).

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

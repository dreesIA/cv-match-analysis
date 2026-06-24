"""End-to-end MVP pipeline: video -> FrameStates -> tagged event log.

Stages (plan §2):
  ingestion -> detection -> tracking -> team classification -> calibration
  -> coordinate projection -> possession/pass/carry/zone tagging
  -> confidence aggregation + review routing -> structured event log.

Two passes over the video:
  1) warmup: sample player crops and fit the team classifier.
  2) main:   per-frame perception -> FrameStates; team labels backfilled via the
             per-track majority vote after the pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from src.common.config import load_all
from src.common.schema import BallObservation, FrameState, PlayerObservation
from src.calibration.pitch import SegmentCalibrator
from src.detection.detector import Detector
from src.ingestion.video import frames, get_video_info
from src.tagging.events import tag
from src.team_classification.classifier import (
    TeamClassifierWrapper,
    crops_from,
    resolve_goalkeeper_team,
)
from src.tracking.tracker import PlayerTracker


@dataclass
class PipelineResult:
    frame_states: list
    events: list
    summary: dict[str, Any]


def _role_of(class_id: int, class_map: dict[str, int]) -> str:
    inv = {v: k for k, v in class_map.items()}
    return inv.get(int(class_id), "player")


def run(
    source: str,
    output_dir: str = "outputs",
    stride: int = 1,
    max_frames: Optional[int] = None,
    warmup_every: int = 30,
    warmup_cap: int = 600,
    device: Optional[str] = None,
) -> PipelineResult:
    cfg = load_all()
    if device:
        cfg["detection"]["device"] = device
    class_map = cfg["detection"]["class_map"]
    info = get_video_info(source)
    cfg["tracking"]["byte_track"]["frame_rate"] = round(info.fps / max(1, stride))

    detector = Detector(cfg["detection"])
    tracker = PlayerTracker(cfg["tracking"])
    team_clf = TeamClassifierWrapper(device=detector.device, batch_size=32)
    calibrator = SegmentCalibrator()

    # ---------- pass 1: warmup the team classifier ----------
    print("[warmup] sampling player crops to fit team classifier...")
    for fp in frames(source, stride=warmup_every, max_frames=None):
        det = detector.players_only(fp.image)
        players = det[det.class_id == class_map["player"]]
        if len(players):
            team_clf.collect(crops_from(fp.image, players), cap=warmup_cap)
        if len(team_clf._warmup_crops) >= warmup_cap:
            break
    team_clf.fit()
    print(f"[warmup] fitted on {len(team_clf._votes)} crops-worth of samples.")

    # ---------- pass 2: per-frame perception ----------
    frame_states: list[FrameState] = []
    gk_votes: dict[int, dict[int, int]] = {}
    n_total = (max_frames or (info.total_frames // max(1, stride)))
    for fp in tqdm(frames(source, stride=stride, max_frames=max_frames), total=n_total, desc="[perception]"):
        fdet = detector.detect(fp.image)
        tracked = tracker.update(fdet.players, fp.frame_idx)
        calib = calibrator.update(fdet.keypoints, fp.segment_id)
        calib_conf = calib.confidence if calib else 0.0

        anchors = (
            tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            if len(tracked) else np.empty((0, 2))
        )

        # per-frame team labels for outfield players via the fitted classifier
        is_player = tracked.class_id == class_map["player"]
        player_dets = tracked[is_player]
        player_labels: dict[int, int] = {}
        if len(player_dets):
            labels = team_clf.assign(crops_from(fp.image, player_dets), [int(t) for t in player_dets.tracker_id])
            player_labels = {int(t): int(l) for t, l in zip(player_dets.tracker_id, labels)}

        observations: list[PlayerObservation] = []
        gks: list[tuple[int, tuple[float, float]]] = []  # (track_id, anchor for resolution)
        team_xy: dict[int, list] = {0: [], 1: []}
        for det_i in range(len(tracked)):
            tid = int(tracked.tracker_id[det_i])
            role = _role_of(tracked.class_id[det_i], class_map)
            px = (float(anchors[det_i][0]), float(anchors[det_i][1]))
            pitch_xy = calib.project(px) if calib else None
            obs = PlayerObservation(
                frame_idx=fp.frame_idx, track_id=tid, xy_pixel=px,
                det_confidence=float(tracked.confidence[det_i]), role=role, xy_pitch_m=pitch_xy,
            )
            if role == "player":
                obs.team_id = player_labels.get(tid)
                if obs.team_id is not None:
                    team_xy[obs.team_id].append(pitch_xy or px)
            elif role == "goalkeeper":
                gks.append((tid, pitch_xy or px))
            observations.append(obs)

        # resolve goalkeepers to a team via centroid proximity (accumulate votes)
        t0 = np.array(team_xy[0], dtype=float) if team_xy[0] else np.empty((0, 2))
        t1 = np.array(team_xy[1], dtype=float) if team_xy[1] else np.empty((0, 2))
        for tid, gxy in gks:
            team = resolve_goalkeeper_team(gxy, t0, t1)
            if team is not None:
                gk_votes.setdefault(tid, {0: 0, 1: 0})[team] += 1

        # ball observation (project if calibrated)
        if fdet.ball_xy is not None:
            ball = BallObservation(
                frame_idx=fp.frame_idx, valid=True, confidence=fdet.ball_conf,
                xy_pixel=fdet.ball_xy, xy_pitch_m=(calib.project(fdet.ball_xy) if calib else None),
            )
        else:
            ball = BallObservation(frame_idx=fp.frame_idx, valid=False, confidence=fdet.ball_conf)

        frame_states.append(FrameState(
            frame_idx=fp.frame_idx, timestamp_s=fp.timestamp_s, segment_id=fp.segment_id,
            calibration_confidence=calib_conf, players=observations, ball=ball,
        ))

    # ---------- backfill smoothed team labels ----------
    team_map = team_clf.finalize()  # track_id -> (team, conf)
    gk_map = {tid: (max(v, key=v.get), v[max(v, key=v.get)] / sum(v.values()))
              for tid, v in gk_votes.items() if sum(v.values()) > 0}
    for fs in frame_states:
        for p in fs.players:
            if p.role == "player" and p.track_id in team_map:
                p.team_id, p.team_confidence = team_map[p.track_id]
            elif p.role == "goalkeeper" and p.track_id in gk_map:
                p.team_id, p.team_confidence = gk_map[p.track_id]

    # ---------- tagging ----------
    track_conf = {tid: tracker.track_confidence(tid) for tid in tracker.stability}
    events = tag(frame_states, track_conf, cfg["tagging"])

    summary = _summarize(frame_states, events, calibrator, team_clf)
    _write_outputs(output_dir, events, summary)
    return PipelineResult(frame_states=frame_states, events=events, summary=summary)


def _summarize(frame_states, events, calibrator, team_clf) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for e in events:
        by_type[e.type] = by_type.get(e.type, 0) + 1
        by_status[e.review_status] = by_status.get(e.review_status, 0) + 1
    valid_ball = sum(1 for fs in frame_states if fs.ball and fs.ball.valid)
    return {
        "frames_processed": len(frame_states),
        "events_total": len(events),
        "events_by_type": by_type,
        "events_by_review_status": by_status,
        "ball_valid_frames": valid_ball,
        "ball_valid_fraction": valid_ball / len(frame_states) if frame_states else 0.0,
        "calibration_coverage_fraction": calibrator.coverage_fraction(),
        "team_label_changes": {tid: team_clf.label_changes(tid) for tid in team_clf._votes},
    }


def _write_outputs(output_dir: str, events, summary) -> None:
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [e.to_row() for e in events]
    df = pd.DataFrame(rows)
    df.to_parquet(out / "events.parquet") if len(df) else None
    df.to_json(out / "events.json", orient="records", indent=2)
    with open(out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[output] wrote {len(events)} events to {out}/events.json")
    print(f"[output] summary: {json.dumps(summary, indent=2)}")

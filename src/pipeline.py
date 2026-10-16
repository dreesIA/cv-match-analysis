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
import math
from itertools import islice
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


def _scale_temporal_thresholds(cfg: dict[str, Any], stride: int) -> None:
    """Preserve configured wall-clock evidence windows under frame sampling.

    The YAML expresses temporal hysteresis in source-frame counts.  When every
    Nth frame is processed, dividing those counts by N keeps their approximate
    duration intact instead of making a sampled run N times more conservative.
    """
    factor = max(1, int(stride))
    for section, key in (
        ("possession", "control_min_frames"),
        ("possession", "hysteresis_frames"),
        ("possession_change", "min_control_frames"),
        ("carry", "min_frames"),
        ("no_ball", "max_gap_frames"),
    ):
        if key in cfg.get(section, {}):
            original = int(cfg[section][key])
            cfg[section][key] = max(1, math.ceil(original / factor))


def _role_of(class_id: int, class_map: dict[str, int]) -> str:
    inv = {v: k for k, v in class_map.items()}
    return inv.get(int(class_id), "player")


def _batches(values, size: int):
    """Yield small lists without materialising a video in memory."""
    iterator = iter(values)
    while batch := list(islice(iterator, max(1, size))):
        yield batch


def _interpolate_sampled_balls(
    frame_states: list[FrameState],
    observation_interval: int,
) -> None:
    """Fill short gaps between real ball measurements for tagging and display.

    This is fixed-lag interpolation between two valid detector observations, not
    a fabricated detection: interpolated samples carry ``observed=False`` and a
    conservative confidence penalty. Gaps across cuts or failed measurements
    remain invalid.
    """
    if observation_interval <= 1:
        return
    observed = [
        i for i, state in enumerate(frame_states)
        if state.ball is not None and state.ball.observed
    ]
    for left_i, right_i in zip(observed, observed[1:]):
        left, right = frame_states[left_i], frame_states[right_i]
        lb, rb = left.ball, right.ball
        gap = right.frame_idx - left.frame_idx
        if (
            gap <= 1
            or gap > observation_interval
            or left.segment_id != right.segment_id
            or lb is None or rb is None
            or not lb.valid or not rb.valid
            or lb.xy_pixel is None or rb.xy_pixel is None
        ):
            continue
        for state in frame_states[left_i + 1:right_i]:
            alpha = (state.frame_idx - left.frame_idx) / gap
            pitch_xy = None
            if lb.xy_pitch_m is not None and rb.xy_pitch_m is not None:
                pitch_xy = tuple(
                    float(a + (b - a) * alpha)
                    for a, b in zip(lb.xy_pitch_m, rb.xy_pitch_m)
                )
            state.ball = BallObservation(
                frame_idx=state.frame_idx,
                valid=True,
                confidence=0.9 * min(lb.confidence, rb.confidence),
                xy_pixel=tuple(
                    float(a + (b - a) * alpha)
                    for a, b in zip(lb.xy_pixel, rb.xy_pixel)
                ),
                xy_pitch_m=pitch_xy,
                observed=False,
            )


def run(
    source: str,
    output_dir: str = "outputs",
    stride: int = 1,
    max_frames: Optional[int] = None,
    warmup_every: int = 30,
    warmup_cap: int = 600,
    device: Optional[str] = None,
    ball_inference: Optional[str] = None,
    ball_imgsz: Optional[int] = None,
    player_imgsz: Optional[int] = None,
    classify_every: int = 1,
    batch_size: int = 1,
    ball_every: int = 1,
    pitch_every: int = 1,
) -> PipelineResult:
    cfg = load_all()
    if device:
        cfg["detection"]["device"] = device
    if ball_inference:
        cfg["detection"]["models"]["ball"]["inference"] = ball_inference
    if ball_imgsz:
        cfg["detection"]["models"]["ball"]["imgsz"] = int(ball_imgsz)
    if player_imgsz:
        cfg["detection"]["models"]["players"]["imgsz"] = int(player_imgsz)
    _scale_temporal_thresholds(cfg["tagging"], stride)
    class_map = cfg["detection"]["class_map"]
    info = get_video_info(source)
    cfg["tracking"]["byte_track"]["frame_rate"] = round(info.fps / max(1, stride))

    detector = Detector(cfg["detection"])
    tracker = PlayerTracker(cfg["tracking"])
    team_clf = TeamClassifierWrapper(device=detector.device, batch_size=32)
    calibrator = SegmentCalibrator()

    # ---------- pass 1: warmup the team classifier ----------
    print("[warmup] sampling player crops to fit team classifier...")
    warmup_packets = frames(source, stride=warmup_every, max_frames=None)
    for packet_batch in _batches(warmup_packets, batch_size):
        detections = detector.players_only_batch([fp.image for fp in packet_batch])
        for fp, det in zip(packet_batch, detections):
            players = det[det.class_id == class_map["player"]]
            if len(players):
                team_clf.collect(crops_from(fp.image, players), cap=warmup_cap)
        if len(team_clf._warmup_crops) >= warmup_cap:
            break
    team_clf.fit()
    print(f"[warmup] fitted on {team_clf.n_fit_samples} player crops.")

    # ---------- pass 2: per-frame perception ----------
    frame_states: list[FrameState] = []
    gk_votes: dict[int, dict[int, int]] = {}
    sampled_total = math.ceil(info.total_frames / max(1, stride))
    n_total = min(max_frames, sampled_total) if max_frames is not None else sampled_total
    main_packets = frames(source, stride=stride, max_frames=max_frames)
    sample_i = 0
    with tqdm(total=n_total, desc="[perception]") as progress:
        for packet_batch in _batches(main_packets, batch_size):
            images = [fp.image for fp in packet_batch]
            player_detections = detector.detect_players_batch(images)
            ball_slots = [
                i for i in range(len(packet_batch))
                if (sample_i + i) % max(1, ball_every) == 0
            ]
            pitch_slots = [
                i for i in range(len(packet_batch))
                if (sample_i + i) % max(1, pitch_every) == 0
            ]
            ball_values = detector.detect_balls_batch([images[i] for i in ball_slots])
            keypoint_values = detector.detect_keypoints_batch([images[i] for i in pitch_slots])
            balls_by_slot = dict(zip(ball_slots, ball_values))
            keypoints_by_slot = dict(zip(pitch_slots, keypoint_values))

            for batch_i, (fp, player_det) in enumerate(zip(packet_batch, player_detections)):
                tracked = tracker.update(player_det, fp.frame_idx)
                keypoints = keypoints_by_slot.get(batch_i)
                calib = calibrator.update(keypoints, fp.segment_id)
                calib_conf = calib.confidence if calib else 0.0

                anchors = (
                    tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    if len(tracked) else np.empty((0, 2))
                )

                # Per-frame team labels for outfield players via the fitted classifier.
                is_player = tracked.class_id == class_map["player"]
                player_dets = tracked[is_player]
                player_labels: dict[int, int] = {}
                if len(player_dets) and sample_i % max(1, classify_every) == 0:
                    track_ids = [int(t) for t in player_dets.tracker_id]
                    labels = team_clf.assign(crops_from(fp.image, player_dets), track_ids)
                    player_labels = {tid: int(label) for tid, label in zip(track_ids, labels)}

                observations: list[PlayerObservation] = []
                gks: list[tuple[int, tuple[float, float]]] = []
                team_xy: dict[int, list] = {0: [], 1: []}
                for det_i in range(len(tracked)):
                    tid = int(tracked.tracker_id[det_i])
                    role = _role_of(tracked.class_id[det_i], class_map)
                    px = (float(anchors[det_i][0]), float(anchors[det_i][1]))
                    pitch_xy = calib.project(px) if calib else None
                    obs = PlayerObservation(
                        frame_idx=fp.frame_idx, track_id=tid, xy_pixel=px,
                        det_confidence=float(tracked.confidence[det_i]),
                        role=role, xy_pitch_m=pitch_xy,
                    )
                    if role == "player":
                        obs.team_id = player_labels.get(tid)
                        if obs.team_id is not None:
                            team_xy[obs.team_id].append(pitch_xy or px)
                    elif role == "goalkeeper":
                        gks.append((tid, pitch_xy or px))
                    observations.append(obs)

                # Resolve goalkeepers to a team via centroid proximity.
                t0 = np.array(team_xy[0], dtype=float) if team_xy[0] else np.empty((0, 2))
                t1 = np.array(team_xy[1], dtype=float) if team_xy[1] else np.empty((0, 2))
                for tid, gxy in gks:
                    team = resolve_goalkeeper_team(gxy, t0, t1)
                    if team is not None:
                        gk_votes.setdefault(tid, {0: 0, 1: 0})[team] += 1

                ball_xy, ball_conf = balls_by_slot.get(batch_i, (None, 0.0))
                ball_observed = batch_i in balls_by_slot
                if ball_xy is not None:
                    ball = BallObservation(
                        frame_idx=fp.frame_idx, valid=True, confidence=ball_conf,
                        xy_pixel=ball_xy,
                        xy_pitch_m=(calib.project(ball_xy) if calib else None),
                        observed=True,
                    )
                else:
                    ball = BallObservation(
                        frame_idx=fp.frame_idx, valid=False, confidence=ball_conf,
                        observed=ball_observed,
                    )

                # Retain detected landmarks and homography for honest overlays.
                kp_xy, kp_conf = None, None
                kp = keypoints
                if kp is not None and kp.xy is not None and len(kp.xy) > 0:
                    kp_xy = np.asarray(kp.xy[0], dtype=float)
                    keypoint_conf = getattr(kp, "keypoint_confidence", None)
                    if keypoint_conf is not None and len(keypoint_conf) > 0:
                        kp_conf = np.asarray(keypoint_conf[0], dtype=float)
                calib_m = np.asarray(calib.transformer.m, dtype=float) if calib else None

                frame_states.append(FrameState(
                    frame_idx=fp.frame_idx, timestamp_s=fp.timestamp_s,
                    segment_id=fp.segment_id, calibration_confidence=calib_conf,
                    players=observations, ball=ball, pitch_keypoints_xy=kp_xy,
                    pitch_keypoints_conf=kp_conf, calibration_m=calib_m,
                ))
                sample_i += 1
            progress.update(len(packet_batch))

    _interpolate_sampled_balls(frame_states, max(1, ball_every))

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
    summary.update({
        "source": Path(source).name,
        "source_fps": info.fps,
        "source_frames": info.total_frames,
        "source_duration_s": info.total_frames / info.fps,
        "sampling_stride": max(1, stride),
        "effective_fps": info.fps / max(1, stride),
        "ball_inference": cfg["detection"]["models"]["ball"].get("inference", "sliced"),
        "ball_imgsz": cfg["detection"]["models"]["ball"].get("imgsz"),
        "player_imgsz": cfg["detection"]["models"]["players"].get("imgsz"),
        "team_classification_interval": max(1, classify_every),
        "inference_batch_size": max(1, batch_size),
        "ball_observation_interval": max(1, ball_every),
        "pitch_observation_interval": max(1, pitch_every),
    })
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
    # JSON + summary are the primary artifacts — write them first so a columnar
    # schema quirk can never abort the run or lose results.
    df.to_json(out / "events.json", orient="records", indent=2)
    with open(out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if len(df):
        try:
            df.to_parquet(out / "events.parquet")
        except Exception as e:  # e.g. all-empty `attributes` -> arrow empty-struct
            print(f"[output] skipped parquet ({type(e).__name__}: {e}); events.json written")
    print(f"[output] wrote {len(events)} events to {out}/events.json")
    print(f"[output] summary: {json.dumps(summary, indent=2)}")

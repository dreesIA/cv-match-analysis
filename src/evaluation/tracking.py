"""Tracking metrics (plan §6.3): MOTA / IDF1 (via py-motmetrics) and the path to
HOTA (via TrackEval). Needs ground-truth tracks for a subset of the golden set with
every player given a consistent id across the clip.

MOT-challenge row format (1-indexed frames):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
"""
from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv


def detections_to_mot_rows(detections: sv.Detections, frame: int) -> list[list[float]]:
    """One MOT row per tracked box for a frame (frame is 1-indexed)."""
    rows = []
    if detections.tracker_id is None:
        return rows
    for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id,
                               detections.confidence if detections.confidence is not None
                               else [1.0] * len(detections)):
        x1, y1, x2, y2 = xyxy
        rows.append([frame, int(tid), float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(conf), -1, -1, -1])
    return rows


def evaluate_mot(pred_rows: list[list[float]], gt_rows: list[list[float]], iou_threshold: float = 0.5) -> dict[str, Any]:
    """Compute MOTA / IDF1 / id-switches / fragmentation with py-motmetrics."""
    try:
        import motmetrics as mm
    except ImportError:
        return {"error": "pip install -r requirements-eval.txt (motmetrics) to compute MOT metrics"}

    acc = mm.MOTAccumulator(auto_id=False)
    pred = np.asarray(pred_rows, dtype=float)
    gt = np.asarray(gt_rows, dtype=float)
    frames = sorted(set(pred[:, 0].astype(int)) | set(gt[:, 0].astype(int))) if len(pred) or len(gt) else []
    for f in frames:
        g = gt[gt[:, 0] == f] if len(gt) else np.empty((0, 10))
        p = pred[pred[:, 0] == f] if len(pred) else np.empty((0, 10))
        gids = g[:, 1].astype(int).tolist()
        pids = p[:, 1].astype(int).tolist()
        dists = mm.distances.iou_matrix(g[:, 2:6], p[:, 2:6], max_iou=1 - iou_threshold) if len(g) and len(p) else np.empty((len(g), len(p)))
        acc.update(gids, pids, dists, frameid=f)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_fragmentations",
                                       "mostly_tracked", "mostly_lost"], name="overall")
    row = summary.loc["overall"].to_dict()
    return {
        "mota": float(row["mota"]), "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]), "fragmentations": int(row["num_fragmentations"]),
        "mostly_tracked": int(row["mostly_tracked"]), "mostly_lost": int(row["mostly_lost"]),
        "note": "HOTA: export these rows to MOT format and run TrackEval "
                "(pip install git+https://github.com/JonathonLuiten/TrackEval.git)",
    }

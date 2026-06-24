"""Tagging metrics (plan §10.2): agreement of predicted events with an analyst
ground-truth event log, plus possession-change precision/recall specifically.

Events are matched by type + team + temporal overlap (IoU on the frame interval).
"""
from __future__ import annotations

from typing import Any, Sequence

from src.common.schema import Event


def _interval_iou(a: Event, b: Event) -> float:
    lo = max(a.start_frame, b.start_frame)
    hi = min(a.end_frame, b.end_frame)
    inter = max(0, hi - lo)
    union = (a.end_frame - a.start_frame) + (b.end_frame - b.start_frame) - inter
    return inter / union if union > 0 else (1.0 if a.start_frame == b.start_frame else 0.0)


def _match(preds: list[Event], gts: list[Event], iou_threshold: float) -> tuple[int, int, int]:
    """Greedy 1-1 matching -> (tp, fp, fn)."""
    used = set()
    tp = 0
    for p in preds:
        best_j, best_iou = -1, iou_threshold
        for j, g in enumerate(gts):
            if j in used or g.team_id != p.team_id:
                continue
            iou = _interval_iou(p, g)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            used.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gts) - len(used)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def evaluate_events(
    pred_events: Sequence[Event],
    gt_events: Sequence[Event],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    types = sorted({e.type for e in list(pred_events) + list(gt_events)})
    out: dict[str, Any] = {"by_type": {}, "iou_threshold": iou_threshold}
    overall_tp = overall_fp = overall_fn = 0
    for t in types:
        preds = [e for e in pred_events if e.type == t]
        gts = [e for e in gt_events if e.type == t]
        tp, fp, fn = _match(preds, gts, iou_threshold)
        out["by_type"][t] = _prf(tp, fp, fn)
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
    out["overall"] = _prf(overall_tp, overall_fp, overall_fn)
    out["agreement_rate"] = out["overall"]["f1"]
    out["possession_change"] = out["by_type"].get("possession_change", _prf(0, 0, 0))
    return out

"""Detection metrics (plan §5.4, §10.2).

mAP@0.5 and @0.5:0.95 via supervision, plus a confusion matrix and per-class
precision/recall at the deploy threshold. Ball precision/recall is pulled out as a
headline number because an overall mAP of 0.85 can hide a ball mAP of 0.4 (§5.3).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision


def evaluate_detection(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    classes: list[str],
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """`classes` indexes class ids (classes[i] is the name of class id i)."""
    mapr = MeanAveragePrecision().update(predictions, targets).compute()

    cm = sv.ConfusionMatrix.from_detections(
        predictions=predictions, targets=targets, classes=classes,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
    )
    m = np.asarray(cm.matrix)  # (C+1, C+1); supervision rows=predicted, cols=true
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        tp = m[i, i]
        pred_total = m[i, :].sum()   # everything predicted as class i
        true_total = m[:, i].sum()   # everything truly class i
        per_class[name] = {
            "precision": float(tp / pred_total) if pred_total else 0.0,
            "recall": float(tp / true_total) if true_total else 0.0,
            "support": float(true_total),
        }

    return {
        "map50": float(mapr.map50),
        "map50_95": float(mapr.map50_95),
        "map75": float(mapr.map75),
        "per_class": per_class,
        "ball_precision": per_class.get("ball", {}).get("precision"),
        "ball_recall": per_class.get("ball", {}).get("recall"),
        "confusion_matrix": m.tolist(),
        "classes": classes,
    }

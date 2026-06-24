"""Calibration metrics (plan §8.2): reprojection error in METRES.

Apply the model-derived homography to held-out, manually annotated keypoints and
measure the distance, in real pitch units, to their known true positions. This is
the single most informative calibration metric — "the lines look right" can hide
multi-metre errors that corrupt zone and pass-distance stats.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.calibration.pitch import Calibration


def evaluate_calibration(
    calibration: Calibration,
    gt_image_pts: np.ndarray,   # (N, 2) pixel coords of annotated landmarks
    gt_pitch_pts_m: np.ndarray,  # (N, 2) their known pitch coords in metres
) -> dict[str, Any]:
    proj = calibration.transformer.transform_points(np.asarray(gt_image_pts, dtype=np.float32))
    err = np.linalg.norm(proj - np.asarray(gt_pitch_pts_m, dtype=np.float32), axis=1)
    return {
        "mean_error_m": float(err.mean()),
        "median_error_m": float(np.median(err)),
        "p90_error_m": float(np.percentile(err, 90)),
        "max_error_m": float(err.max()),
        "n_points": int(len(err)),
    }


def coverage_report(coverage: dict[int, list[int]]) -> dict[str, Any]:
    """`coverage` is SegmentCalibrator.coverage: segment_id -> [valid, total]."""
    valid = sum(v[0] for v in coverage.values())
    total = sum(v[1] for v in coverage.values())
    per_segment = {
        str(seg): {"valid": v[0], "total": v[1], "fraction": (v[0] / v[1] if v[1] else 0.0)}
        for seg, v in coverage.items()
    }
    return {
        "coverage_fraction": valid / total if total else 0.0,
        "n_segments": len(coverage),
        "per_segment": per_segment,
    }

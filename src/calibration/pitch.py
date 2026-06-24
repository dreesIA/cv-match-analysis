"""Pitch calibration & coordinate mapping (plan §8).

Detected pitch keypoints -> homography (via sports.ViewTransformer over
cv2.findHomography) -> project pixel positions to METRIC pitch coordinates.

Homography is estimated PER stable-camera segment (not once for the whole video):
the calibrator caches the last valid transform within a segment and resets on a new
segment id. Every projection carries a calibration confidence derived from keypoint
count + the fit's reprojection error in metres; low-confidence segments are flagged
rather than silently interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

_CONFIG = SoccerPitchConfiguration()
# pitch landmark targets in METRES (config is in cm): 120 m x 70 m
_TARGET_M = np.array(_CONFIG.vertices, dtype=np.float32) / 100.0
PITCH_LENGTH_M = _CONFIG.length / 100.0
PITCH_WIDTH_M = _CONFIG.width / 100.0


@dataclass
class Calibration:
    transformer: ViewTransformer
    confidence: float
    n_keypoints: int
    reprojection_error_m: float

    def project(self, xy_pixel: tuple[float, float]) -> tuple[float, float]:
        pts = np.array([xy_pixel], dtype=np.float32)
        out = self.transformer.transform_points(pts)[0]
        return float(out[0]), float(out[1])


def _confidence(n: int, err_m: float) -> float:
    """More keypoints + smaller reprojection error -> higher confidence."""
    n_score = min(1.0, n / 8.0)              # 8+ well-spread points = full credit
    err_score = float(np.exp(-err_m / 2.0))  # ~2 m characteristic scale
    return float(max(0.0, min(1.0, n_score * err_score)))


def build_calibration(keypoints: sv.KeyPoints, kp_conf_threshold: float = 0.5) -> Optional[Calibration]:
    """Build a Calibration from one frame's pitch keypoints, or None if too few."""
    if keypoints is None or len(keypoints.xy) == 0:
        return None
    xy = np.asarray(keypoints.xy[0], dtype=np.float32)  # (K, 2)
    conf = keypoints.confidence
    if conf is not None and len(conf) > 0:
        mask = np.asarray(conf[0]) > kp_conf_threshold
    else:  # fallback: undetected points come back as (0,0)
        mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)
    source = xy[mask]
    target = _TARGET_M[mask]
    if len(source) < 4:
        return None
    transformer = ViewTransformer(source=source.astype(np.float32), target=target.astype(np.float32))
    proj = transformer.transform_points(source.astype(np.float32))
    err_m = float(np.mean(np.linalg.norm(proj - target, axis=1)))
    return Calibration(
        transformer=transformer,
        confidence=_confidence(len(source), err_m),
        n_keypoints=int(len(source)),
        reprojection_error_m=err_m,
    )


class SegmentCalibrator:
    """Maintains the current segment's calibration and per-segment coverage stats."""

    def __init__(self, kp_conf_threshold: float = 0.5, min_confidence: float = 0.2):
        self.kp_conf_threshold = kp_conf_threshold
        self.min_confidence = min_confidence
        self._segment_id: Optional[int] = None
        self._current: Optional[Calibration] = None
        # coverage[segment_id] = [n_valid_frames, n_total_frames]
        self.coverage: dict[int, list[int]] = {}

    def update(self, keypoints: sv.KeyPoints, segment_id: int) -> Optional[Calibration]:
        if segment_id != self._segment_id:  # camera cut -> drop stale homography
            self._segment_id = segment_id
            self._current = None
        self.coverage.setdefault(segment_id, [0, 0])
        self.coverage[segment_id][1] += 1

        fresh = build_calibration(keypoints, self.kp_conf_threshold)
        if fresh is not None and fresh.confidence >= self.min_confidence:
            self._current = fresh  # refresh with a good fit
        if self._current is not None and self._current.confidence >= self.min_confidence:
            self.coverage[segment_id][0] += 1
            return self._current
        return None  # no trustworthy calibration this frame

    def coverage_fraction(self) -> float:
        valid = sum(v[0] for v in self.coverage.values())
        total = sum(v[1] for v in self.coverage.values())
        return valid / total if total else 0.0

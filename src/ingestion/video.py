"""Ingestion & normalization (plan §3 / §8.1).

Yields frames and assigns each a `segment_id` that increments on a detected camera
cut. Stable-camera segments are the unit over which homography is estimated, so the
cut detector here directly bounds calibration quality downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int


def get_video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


@dataclass
class FramePacket:
    frame_idx: int
    timestamp_s: float
    segment_id: int
    image: np.ndarray


class CameraCutDetector:
    """Cheap shot-boundary detector via HSV-histogram correlation between
    consecutive frames. A sharp drop in correlation = a cut/hard transition."""

    def __init__(self, corr_threshold: float = 0.6):
        self.corr_threshold = corr_threshold
        self._prev_hist: np.ndarray | None = None
        self.segment_id = 0

    @staticmethod
    def _hist(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def update(self, image: np.ndarray) -> int:
        """Return the segment id for this frame (incremented on a cut)."""
        hist = self._hist(image)
        if self._prev_hist is not None:
            corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < self.corr_threshold:
                self.segment_id += 1
        self._prev_hist = hist
        return self.segment_id


def frames(
    path: str,
    stride: int = 1,
    max_frames: int | None = None,
    cut_threshold: float = 0.6,
) -> Iterator[FramePacket]:
    """Stream frames with timestamps and stable-camera segment ids.

    `stride` lets you subsample (e.g. process every 2nd frame for speed); timestamps
    stay wall-clock-correct so velocities/durations remain in real seconds.
    """
    info = get_video_info(path)
    cuts = CameraCutDetector(cut_threshold)
    cap = cv2.VideoCapture(path)
    idx = -1
    emitted = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            idx += 1
            seg = cuts.update(image)  # update every frame so cuts aren't missed
            if idx % stride != 0:
                continue
            yield FramePacket(
                frame_idx=idx,
                timestamp_s=idx / info.fps,
                segment_id=seg,
                image=image,
            )
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        cap.release()

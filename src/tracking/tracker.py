"""Multi-object tracking (plan §6).

Adaptive ByteTrack wrapper: uses `sv.ByteTrack` when present (supervision < 0.30),
otherwise the external `trackers.ByteTrackTracker`. Pinning is enforced in
requirements.txt; this fallback keeps us running if/when we deliberately migrate.

Also tracks per-id stability signals (age, gap since last seen) that feed the
tracking-confidence component used in event routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import supervision as sv


@dataclass
class TrackStability:
    first_frame: int
    last_frame: int
    hits: int = 0
    gaps: int = 0  # frames the track was missing then re-appeared

    def confidence(self) -> float:
        """Crude stability score in [0,1]: more hits and fewer gaps -> higher."""
        span = max(1, self.last_frame - self.first_frame + 1)
        coverage = self.hits / span
        gap_penalty = self.gaps / span
        return float(max(0.0, min(1.0, coverage - 0.5 * gap_penalty)))


class PlayerTracker:
    def __init__(self, cfg: dict[str, Any]):
        bt = cfg["byte_track"]
        self.frame_rate = bt.get("frame_rate", 25)
        self._tracker, self._mode = self._make_tracker(bt)
        self.stability: dict[int, TrackStability] = {}
        self._last_seen: dict[int, int] = {}

    @staticmethod
    def _make_tracker(bt: dict[str, Any]):
        if hasattr(sv, "ByteTrack"):
            tracker = sv.ByteTrack(
                track_activation_threshold=bt.get("track_activation_threshold", 0.25),
                lost_track_buffer=bt.get("lost_track_buffer", 30),
                minimum_matching_threshold=bt.get("minimum_matching_threshold", 0.8),
                frame_rate=bt.get("frame_rate", 25),
            )
            return tracker, "sv"
        from trackers import ByteTrackTracker  # supervision >= 0.30 path

        return ByteTrackTracker(), "trackers"

    def update(self, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        if self._mode == "sv":
            tracked = self._tracker.update_with_detections(detections)
        else:
            tracked = self._tracker.update(detections)
        self._record_stability(tracked, frame_idx)
        return tracked

    def _record_stability(self, det: sv.Detections, frame_idx: int) -> None:
        if det.tracker_id is None:
            return
        for tid in det.tracker_id:
            tid = int(tid)
            st = self.stability.get(tid)
            if st is None:
                self.stability[tid] = TrackStability(first_frame=frame_idx, last_frame=frame_idx, hits=1)
            else:
                if frame_idx - self._last_seen.get(tid, frame_idx) > 1:
                    st.gaps += 1
                st.last_frame = frame_idx
                st.hits += 1
            self._last_seen[tid] = frame_idx

    def track_confidence(self, track_id: int) -> float:
        st = self.stability.get(int(track_id))
        return st.confidence() if st else 0.0

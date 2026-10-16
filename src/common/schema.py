"""Shared data structures passed between pipeline stages.

The pipeline is: video -> per-frame FrameState (perception) -> Events (tagging).
Every object carries the component confidences the review-queue routing needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ------------------------- per-frame perception output -------------------------

@dataclass
class BallObservation:
    """Ball state for a single frame. `valid=False` means we have NO trustworthy
    ball position — downstream tagging must treat this as unknown, never guess
    (plan §5.3)."""
    frame_idx: int
    valid: bool
    confidence: float = 0.0
    xy_pixel: Optional[tuple[float, float]] = None      # ball center, pixels
    xy_pitch_m: Optional[tuple[float, float]] = None     # projected metric coords
    observed: bool = True                               # detector measurement vs interpolation


@dataclass
class PlayerObservation:
    frame_idx: int
    track_id: int
    xy_pixel: tuple[float, float]                        # bottom-center anchor, pixels
    det_confidence: float
    role: str = "player"                                 # player | goalkeeper | referee
    team_id: Optional[int] = None                        # 0/1 for outfield+GK; None for referee
    team_confidence: float = 0.0
    xy_pitch_m: Optional[tuple[float, float]] = None


@dataclass
class FrameState:
    frame_idx: int
    timestamp_s: float
    segment_id: int                                      # stable-camera segment (between cuts)
    calibration_confidence: float                        # 0.0 if no valid homography
    players: list[PlayerObservation] = field(default_factory=list)
    ball: Optional[BallObservation] = None
    # detected pitch landmarks, kept for the keypoint overlay (visualization only).
    # plain arrays so FrameState stays pickle-safe and free of an sv dependency.
    pitch_keypoints_xy: Optional[Any] = None             # (K, 2) float, pixel coords
    pitch_keypoints_conf: Optional[Any] = None           # (K,) float, per-keypoint conf
    calibration_m: Optional[Any] = None                  # 3x3 homography, pixel -> metres
                                                         # (None if uncalibrated). inv() warps
                                                         # pitch overlays onto the frame.

    def player_by_track(self, track_id: int) -> Optional[PlayerObservation]:
        for p in self.players:
            if p.track_id == track_id:
                return p
        return None


# -------------------------------- tagging output --------------------------------

EVENT_TYPES = ("possession", "possession_change", "pass", "carry", "zone_entry")


@dataclass
class Event:
    type: str                                            # one of EVENT_TYPES
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    team_id: Optional[int] = None
    player_ids: list[int] = field(default_factory=list)
    start_xy_m: Optional[tuple[float, float]] = None
    end_xy_m: Optional[tuple[float, float]] = None
    length_m: Optional[float] = None
    direction_deg: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)   # e.g. outcome, zone
    # component confidences feeding the aggregate (plan §9)
    conf_components: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    review_status: str = "review"                        # accept | review | hidden

    def to_row(self) -> dict[str, Any]:
        """Flatten for a tabular event log (parquet/json)."""
        row = asdict(self)
        # flatten nested dicts/tuples for columnar storage
        for k in ("start_xy_m", "end_xy_m"):
            v = row.pop(k)
            row[f"{k}_x"] = v[0] if v else None
            row[f"{k}_y"] = v[1] if v else None
        for ck, cv in row.pop("conf_components").items():
            row[f"conf_{ck}"] = cv
        row["attributes"] = dict(row["attributes"])  # keep as struct
        row["player_ids"] = list(row["player_ids"])
        return row

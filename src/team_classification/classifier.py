"""Team & role classification (plan §7).

Wraps `sports.common.team.TeamClassifier` (SigLIP -> UMAP -> KMeans) for the two
outfield teams, and adds the production hardening the naive baseline lacks:

  * Temporal majority vote per track  -> one bad frame can't flip a player's team.
  * team_confidence = vote margin      -> feeds event routing; low margin = flag.
  * Goalkeeper team via centroid proximity (resolve_goalkeepers_team_id pattern).
  * Referees carry team_id=None by design.

Usage: warmup `collect()` over sampled frames -> `fit()` -> `assign()` per frame
(accumulates votes) -> `finalize()` to get the smoothed per-track label.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import supervision as sv


def crops_from(image: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    return [sv.crop_image(image=image, xyxy=xyxy) for xyxy in detections.xyxy]


class TeamClassifierWrapper:
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        from sports.common.team import TeamClassifier

        self.model = TeamClassifier(device=device, batch_size=batch_size)
        self._warmup_crops: list[np.ndarray] = []
        self._fitted = False
        self.n_fit_samples = 0
        # per-track vote tallies: track_id -> {team_id: count}
        self._votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # ---- warmup ----
    def collect(self, crops: list[np.ndarray], cap: int = 750) -> None:
        remaining = max(0, cap - len(self._warmup_crops))
        if remaining:
            self._warmup_crops.extend(crops[:remaining])

    def fit(self) -> None:
        if not self._warmup_crops:
            raise RuntimeError("No warmup crops collected; cannot fit team classifier.")
        self.n_fit_samples = len(self._warmup_crops)
        self.model.fit(self._warmup_crops)
        self._fitted = True
        self._warmup_crops = []  # free memory

    # ---- per-frame ----
    def assign(self, crops: list[np.ndarray], track_ids: list[int]) -> np.ndarray:
        """Predict team per crop and accumulate a vote for each track. Returns the
        per-frame team ids (array of 0/1)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before assign().")
        if not crops:
            return np.array([], dtype=int)
        labels = self.model.predict(crops)
        for tid, lab in zip(track_ids, labels):
            self._votes[int(tid)][int(lab)] += 1
        return labels

    def finalize(self) -> dict[int, tuple[int, float]]:
        """Smoothed team per track: (team_id, confidence=vote_margin)."""
        out: dict[int, tuple[int, float]] = {}
        for tid, tally in self._votes.items():
            total = sum(tally.values())
            team_id = max(tally, key=tally.get)
            out[tid] = (team_id, tally[team_id] / total if total else 0.0)
        return out

    def label_changes(self, track_id: int) -> int:
        """Stability diagnostic: distinct teams ever voted for this track minus 1."""
        return max(0, len(self._votes.get(int(track_id), {})) - 1)


def resolve_goalkeeper_team(
    gk_xy: tuple[float, float],
    team0_xy: np.ndarray,
    team1_xy: np.ndarray,
) -> Optional[int]:
    """Assign a goalkeeper to the nearer outfield-team centroid (plan §7.2).
    Coordinates may be pitch-metric or pixel; just be consistent. Returns None if
    neither team has players this frame."""
    if len(team0_xy) == 0 and len(team1_xy) == 0:
        return None
    if len(team0_xy) == 0:
        return 1
    if len(team1_xy) == 0:
        return 0
    c0 = team0_xy.mean(axis=0)
    c1 = team1_xy.mean(axis=0)
    p = np.asarray(gk_xy)
    return 0 if np.linalg.norm(p - c0) < np.linalg.norm(p - c1) else 1

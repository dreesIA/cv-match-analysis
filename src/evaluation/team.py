"""Team & role classification metrics (plan §7.3): per-track accuracy split by role,
plus a stability metric (how often a track's label changed — should be ~0 after the
temporal majority vote).
"""
from __future__ import annotations

from typing import Any, Optional


def evaluate_team(
    pred_track_team: dict[int, int],
    gt_track_team: dict[int, int],
    roles: Optional[dict[int, str]] = None,
    label_changes: Optional[dict[int, int]] = None,
) -> dict[str, Any]:
    roles = roles or {}
    correct: dict[str, list[bool]] = {"all": [], "player": [], "goalkeeper": [], "referee": []}
    for tid, gt in gt_track_team.items():
        if tid not in pred_track_team:
            continue
        ok = pred_track_team[tid] == gt
        correct["all"].append(ok)
        correct.setdefault(roles.get(tid, "player"), []).append(ok)

    def acc(xs: list[bool]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    out: dict[str, Any] = {f"accuracy_{k}": acc(v) for k, v in correct.items()}
    if label_changes is not None:
        changes = list(label_changes.values())
        out["mean_label_changes_per_track"] = sum(changes) / len(changes) if changes else 0.0
        out["tracks_with_label_change"] = sum(1 for c in changes if c > 0)
    return out

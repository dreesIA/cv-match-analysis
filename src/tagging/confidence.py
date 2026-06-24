"""Per-event confidence aggregation + review-queue routing (plan §9, §10.3).

Combines the four component confidences (detection, tracking, team, calibration)
into one score, then routes the event by per-event-type thresholds:
    score >= auto_accept -> "accept"
    score >= review      -> "review"   (analyst queue)
    else                 -> "hidden"

Thresholds are per-event-type because a 0.7 pass and a 0.7 possession-change do not
carry the same reliability. They live in configs/tagging.yaml and should be set from
the calibrated reliability curves in src/evaluation, not guessed.
"""
from __future__ import annotations

import math
from typing import Any


def aggregate(components: dict[str, float], cfg: dict[str, Any]) -> float:
    weights = cfg["weights"]
    method = cfg.get("combine", "weighted_geomean")
    keys = [k for k in weights if k in components]
    if not keys:
        return 0.0
    w = {k: weights[k] for k in keys}
    wsum = sum(w.values()) or 1.0
    vals = {k: max(1e-6, min(1.0, components[k])) for k in keys}
    if method == "min":
        return float(min(vals.values()))
    if method == "weighted_mean":
        return float(sum(w[k] * vals[k] for k in keys) / wsum)
    # weighted geometric mean: penalises any single weak component (default)
    log = sum(w[k] * math.log(vals[k]) for k in keys) / wsum
    return float(math.exp(log))


def route(event_type: str, score: float, cfg: dict[str, Any]) -> str:
    thr = cfg["thresholds"].get(event_type, {"auto_accept": 0.8, "review": 0.5})
    if score >= thr["auto_accept"]:
        return "accept"
    if score >= thr["review"]:
        return "review"
    return "hidden"


def score_and_route(event_type: str, components: dict[str, float], cfg: dict[str, Any]) -> tuple[float, str]:
    score = aggregate(components, cfg)
    return score, route(event_type, score, cfg)

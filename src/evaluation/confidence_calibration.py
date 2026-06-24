"""Confidence calibration (plan §10.3).

Raw model/event confidences are not automatically probabilities. Bin predictions by
confidence and compare each bin's mean confidence to its actual accuracy; if the
aggregate is mis-calibrated, fit a temperature so a 0.9 means ~90% correct. Set the
review-queue thresholds per event type from these curves, not by guessing.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def reliability_curve(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> dict[str, Any]:
    conf = np.clip(np.asarray(confidences, dtype=float), 1e-6, 1 - 1e-6)
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            bins.append({"range": [float(lo), float(hi)], "count": 0, "mean_conf": None, "accuracy": None})
            continue
        mean_conf = float(conf[mask].mean())
        accuracy = float(acc[mask].mean())
        weight = mask.sum() / len(conf)
        ece += weight * abs(mean_conf - accuracy)
        bins.append({"range": [float(lo), float(hi)], "count": int(mask.sum()),
                     "mean_conf": mean_conf, "accuracy": accuracy})
    return {"bins": bins, "ece": float(ece), "n": int(len(conf))}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(confidences: Sequence[float], correct: Sequence[bool], iters: int = 200) -> float:
    """Fit scalar temperature T (in logit space) minimising NLL via a simple grid +
    bisection. Returns T; apply with `apply_temperature`. T>1 softens overconfidence."""
    z = _logit(np.asarray(confidences, dtype=float))
    y = np.asarray(correct, dtype=float)

    def nll(T: float) -> float:
        p = 1.0 / (1.0 + np.exp(-z / T))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    grid = np.linspace(0.25, 5.0, 96)
    return float(grid[int(np.argmin([nll(T) for T in grid]))])


def apply_temperature(confidences: Sequence[float], temperature: float) -> np.ndarray:
    z = _logit(np.asarray(confidences, dtype=float))
    return 1.0 / (1.0 + np.exp(-z / temperature))

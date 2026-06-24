"""Metric geometry helpers. All inputs are pitch coordinates in metres."""
from __future__ import annotations

import math
from typing import Optional

Point = tuple[float, float]


def distance_m(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def direction_deg(a: Point, b: Point) -> float:
    """Bearing from a -> b in degrees, 0 = +x (toward attacking goal), CCW positive."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def speed_mps(a: Point, b: Point, dt_s: float) -> Optional[float]:
    if dt_s <= 0:
        return None
    return distance_m(a, b) / dt_s

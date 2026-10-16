"""Top-down tactical panel: 2D pitch map + Voronoi control regions + Delaunay team
structure, drawn from the pipeline's metric player/ball coordinates.

The pitch, points, and Voronoi come from `roboflow/sports` (already a dependency);
the Delaunay team-shape mesh is computed here with scipy and drawn with cv2 (neither
supervision nor sports ships a Delaunay helper). Everything consumes the same
`xy_pitch_m` the tagging stage uses, so this adds no perception cost.

Coordinates: the pipeline projects to METRES; the `sports` helpers and the pitch
config are in CENTIMETRES, so we scale by 100 on the way in. A panel is only
meaningful when the frame is calibrated — the caller gates it on confidence.
"""
from __future__ import annotations

import functools
from typing import Optional

import cv2
import numpy as np
from supervision.draw.color import Color

from sports.annotators.soccer import (
    draw_pitch,
    draw_pitch_voronoi_diagram,
    draw_points_on_pitch,
)
from sports.configs.soccer import SoccerPitchConfiguration
from src.common.schema import FrameState

_CONFIG = SoccerPitchConfiguration()

# team colours mirror annotate.TEAM_COLORS (BGR) as sv.Color (RGB), kept in sync by
# hand to avoid a circular import between the two visualization modules.
TEAM0 = Color(r=40, g=150, b=240)    # annotate BGR (240,150,40) — blue
TEAM1 = Color(r=235, g=60, b=48)     # annotate BGR (48,60,235)  — red
REF_COLOR = Color(r=245, g=205, b=40)
BALL_FACE = Color.WHITE
BALL_EDGE = Color.BLACK


def _team_arrays(fs: FrameState) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Per-team + referee pitch points (cm) and the ball point (cm), from projected
    metric coords. Only players/GKs with a valid projection contribute."""
    t0, t1, refs = [], [], []
    for p in fs.players:
        if p.xy_pitch_m is None:
            continue
        cm = (p.xy_pitch_m[0] * 100.0, p.xy_pitch_m[1] * 100.0)
        if p.role == "referee":
            refs.append(cm)
        elif p.team_id == 0:
            t0.append(cm)
        elif p.team_id == 1:
            t1.append(cm)
    ball = None
    if fs.ball is not None and fs.ball.valid and fs.ball.xy_pitch_m is not None:
        ball = np.array([[fs.ball.xy_pitch_m[0] * 100.0, fs.ball.xy_pitch_m[1] * 100.0]], dtype=np.float32)
    to_arr = lambda xs: np.array(xs, dtype=np.float32) if xs else np.empty((0, 2), np.float32)
    return to_arr(t0), to_arr(t1), to_arr(refs), ball


PITCH_LENGTH_M = _CONFIG.length / 100.0   # x, 120 m
PITCH_WIDTH_M = _CONFIG.width / 100.0     # y, 70 m


def team_points_metres(fs: FrameState) -> tuple[np.ndarray, np.ndarray]:
    """Per-team outfield+GK positions in METRES (for Delaunay adjacency / warping)."""
    t0, t1 = [], []
    for p in fs.players:
        if p.xy_pitch_m is None or p.role == "referee":
            continue
        if p.team_id == 0:
            t0.append(p.xy_pitch_m)
        elif p.team_id == 1:
            t1.append(p.xy_pitch_m)
    to_arr = lambda xs: np.array(xs, dtype=np.float32) if xs else np.empty((0, 2), np.float32)
    return to_arr(t0), to_arr(t1)


def team_delaunay_edges(pts: np.ndarray) -> list[tuple[int, int]]:
    """Unique Delaunay edges (index pairs) for a team, or [] if degenerate.

    Adjacency is coordinate-space agnostic — pass metric coords for a tactically
    correct mesh, or pixel coords to draw directly on the broadcast frame."""
    if len(pts) < 3:
        return []
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(pts)
    except Exception:
        return []  # QhullError on collinear/degenerate input
    seen: set[tuple[int, int]] = set()
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def _draw_delaunay(pitch: np.ndarray, pts_cm: np.ndarray, color: Color, scale: float, padding: int) -> None:
    """Draw one team's Delaunay mesh onto the top-down panel, in place."""
    edges = team_delaunay_edges(pts_cm)
    if not edges:
        return
    px = [(int(x * scale) + padding, int(y * scale) + padding) for x, y in pts_cm]
    bgr = color.as_bgr()
    for a, b in edges:
        cv2.line(pitch, px[a], px[b], bgr, 2, cv2.LINE_AA)


def voronoi_overlay_grid(
    t0_m: np.ndarray, t1_m: np.ndarray, res_m: float = 0.5
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Team-control regions as a small grid image in METRIC pitch space, for warping
    onto the broadcast frame. Returns (bgr, alpha, A) where A maps grid pixel
    (col,row,1) -> metres (x,y,1); None if either team is absent."""
    if len(t0_m) == 0 or len(t1_m) == 0:
        return None
    nx = int(round(PITCH_LENGTH_M / res_m))
    ny = int(round(PITCH_WIDTH_M / res_m))
    xs = (np.arange(nx) + 0.5) * res_m
    ys = (np.arange(ny) + 0.5) * res_m
    gx, gy = np.meshgrid(xs, ys)                         # (ny, nx)
    grid = np.stack([gx, gy], axis=-1).reshape(-1, 2)    # (ny*nx, 2)
    d0 = np.linalg.norm(grid[:, None, :] - t0_m[None, :, :], axis=2).min(axis=1)
    d1 = np.linalg.norm(grid[:, None, :] - t1_m[None, :, :], axis=2).min(axis=1)
    team0 = (d0 <= d1).reshape(ny, nx)
    bgr = np.empty((ny, nx, 3), np.uint8)
    bgr[team0] = TEAM0.as_bgr()
    bgr[~team0] = TEAM1.as_bgr()
    alpha = np.full((ny, nx), 255, np.uint8)
    A = np.array([[res_m, 0, 0.5 * res_m],
                  [0, res_m, 0.5 * res_m],
                  [0, 0, 1.0]], dtype=np.float64)
    return bgr, alpha, A


@functools.lru_cache(maxsize=8)
def _base_pitch(scale: float, padding: int) -> np.ndarray:
    """The empty pitch is identical every frame — draw its lines once and reuse."""
    return draw_pitch(_CONFIG, padding=padding, scale=scale)


def render_radar(
    fs: FrameState,
    scale: float = 0.06,           # native panel res; the inset is downscaled anyway
    padding: int = 50,
    voronoi_opacity: float = 0.45,
    draw_voronoi: bool = True,
    draw_delaunay: bool = True,
) -> Optional[np.ndarray]:
    """Return a BGR tactical panel for this frame, or None if nothing to plot.

    `draw_voronoi` / `draw_delaunay` select the layers, so the same function yields
    the Voronoi-only, Delaunay-only, and combined standalone videos."""
    t0, t1, refs, ball = _team_arrays(fs)
    if len(t0) == 0 and len(t1) == 0:
        return None

    pitch = _base_pitch(scale, padding).copy()

    # Voronoi control regions need both teams present
    if draw_voronoi and len(t0) and len(t1):
        pitch = draw_pitch_voronoi_diagram(
            _CONFIG, t0, t1, team_1_color=TEAM0, team_2_color=TEAM1,
            opacity=voronoi_opacity, padding=padding, scale=scale, pitch=pitch,
        )

    # Delaunay team-shape mesh, per team, over the Voronoi
    if draw_delaunay:
        _draw_delaunay(pitch, t0, TEAM0, scale, padding)
        _draw_delaunay(pitch, t1, TEAM1, scale, padding)

    # players / refs / ball on top
    if len(t0):
        pitch = draw_points_on_pitch(_CONFIG, t0, face_color=TEAM0, edge_color=Color.BLACK,
                                     radius=14, padding=padding, scale=scale, pitch=pitch)
    if len(t1):
        pitch = draw_points_on_pitch(_CONFIG, t1, face_color=TEAM1, edge_color=Color.BLACK,
                                     radius=14, padding=padding, scale=scale, pitch=pitch)
    if len(refs):
        pitch = draw_points_on_pitch(_CONFIG, refs, face_color=REF_COLOR, edge_color=Color.BLACK,
                                     radius=11, padding=padding, scale=scale, pitch=pitch)
    if ball is not None:
        pitch = draw_points_on_pitch(_CONFIG, ball, face_color=BALL_FACE, edge_color=BALL_EDGE,
                                     radius=9, thickness=2, padding=padding, scale=scale, pitch=pitch)
    return pitch

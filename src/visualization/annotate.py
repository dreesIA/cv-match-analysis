"""Render the pipeline's per-frame output into an annotated showcase video.

This is a *visualization* layer only: it takes the `FrameState`s and `Event`s the
pipeline already produces (foot anchors, team ids, ball, calibration confidence,
routed events) and draws them onto the source frames. It runs no models, so it can
be re-run cheaply to iterate on the look without touching perception.

What it draws (all from real pipeline output):
  * broadcast-style team-coloured ellipse under each tracked player, with track id
  * downward triangle marker on the ball (only when the ball is *valid*)
  * a "who's nearest the ball" highlight ring (pure geometry, a visual aid)
  * a HUD: clock, segment id, calibration-confidence bar, ball status
  * a live event log panel with per-event confidence + accept/review/hidden routing
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from supervision.draw.color import Color
from sports.configs.soccer import SoccerPitchConfiguration

from src.common.schema import BallObservation, Event, FrameState, PlayerObservation
from src.ingestion.video import frames, get_video_info
from src.visualization import radar
from src.visualization.radar import render_radar

# pitch skeleton for the on-video keypoint overlay (config edges are 1-indexed)
_PITCH_CONFIG = SoccerPitchConfiguration()
_PITCH_EDGES = [(a - 1, b - 1) for a, b in _PITCH_CONFIG.edges]
KEYPOINT_COLOR = Color(r=255, g=215, b=0)  # gold

# ------------------------------- palette (BGR) --------------------------------
TEAM_COLORS = {
    0: (240, 150, 40),    # team 0 — blue
    1: (48, 60, 235),     # team 1 — red
}
REFEREE_COLOR = (40, 205, 245)   # amber
UNKNOWN_COLOR = (180, 180, 180)  # grey — team not resolved
BALL_COLOR = (255, 255, 255)
BALL_OUTLINE = (20, 20, 20)
STATUS_COLORS = {
    "accept": (90, 205, 100),    # green
    "review": (60, 185, 240),    # amber
    "hidden": (140, 140, 140),   # grey
}
PANEL_BG = (28, 28, 28)
HUD_BG = (24, 24, 24)

# active-event banner priority (most "interesting" wins the center banner)
EVENT_PRIORITY = {
    "possession_change": 5,
    "pass": 4,
    "zone_entry": 3,
    "carry": 2,
    "possession": 1,
}


# ----------------------- display-only temporal interpolation -----------------

def _lerp(a: float, b: float, alpha: float) -> float:
    return float(a + (b - a) * alpha)


def _lerp_point(
    left: Optional[tuple[float, float]],
    right: Optional[tuple[float, float]],
    alpha: float,
) -> Optional[tuple[float, float]]:
    if left is None or right is None:
        return left if alpha < 0.5 else right
    return (_lerp(left[0], right[0], alpha), _lerp(left[1], right[1], alpha))


def _retime_state(fs: FrameState, frame_idx: int, timestamp_s: float) -> FrameState:
    """Copy a sampled state onto a display frame without changing its observations."""
    players = [PlayerObservation(
        frame_idx=frame_idx,
        track_id=p.track_id,
        xy_pixel=p.xy_pixel,
        det_confidence=p.det_confidence,
        role=p.role,
        team_id=p.team_id,
        team_confidence=p.team_confidence,
        xy_pitch_m=p.xy_pitch_m,
    ) for p in fs.players]
    ball = None if fs.ball is None else BallObservation(
        frame_idx=frame_idx,
        valid=fs.ball.valid,
        confidence=fs.ball.confidence,
        xy_pixel=fs.ball.xy_pixel,
        xy_pitch_m=fs.ball.xy_pitch_m,
        observed=fs.ball.observed,
    )
    return FrameState(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        segment_id=fs.segment_id,
        calibration_confidence=fs.calibration_confidence,
        players=players,
        ball=ball,
        pitch_keypoints_xy=fs.pitch_keypoints_xy,
        pitch_keypoints_conf=fs.pitch_keypoints_conf,
        calibration_m=fs.calibration_m,
    )


def _interpolate_state(
    left: FrameState,
    right: FrameState,
    frame_idx: int,
    timestamp_s: float,
) -> FrameState:
    """Interpolate matched tracks for smooth display; never used by event tagging."""
    span = right.frame_idx - left.frame_idx
    if span <= 0:
        return _retime_state(left, frame_idx, timestamp_s)
    alpha = min(1.0, max(0.0, (frame_idx - left.frame_idx) / span))
    nearest = left if alpha < 0.5 else right

    # Never interpolate through a detected camera cut.
    if left.segment_id != right.segment_id:
        return _retime_state(nearest, frame_idx, timestamp_s)

    left_players = {p.track_id: p for p in left.players}
    right_players = {p.track_id: p for p in right.players}
    players: list[PlayerObservation] = []
    for track_id in dict.fromkeys([*left_players, *right_players]):
        lp = left_players.get(track_id)
        rp = right_players.get(track_id)
        if lp is None or rp is None:
            chosen = lp if rp is None else rp
            if (lp is None and alpha < 0.5) or (rp is None and alpha >= 0.5):
                continue
            players.append(PlayerObservation(
                frame_idx=frame_idx,
                track_id=chosen.track_id,
                xy_pixel=chosen.xy_pixel,
                det_confidence=chosen.det_confidence,
                role=chosen.role,
                team_id=chosen.team_id,
                team_confidence=chosen.team_confidence,
                xy_pitch_m=chosen.xy_pitch_m,
            ))
            continue
        label_source = lp if alpha < 0.5 else rp
        players.append(PlayerObservation(
            frame_idx=frame_idx,
            track_id=track_id,
            xy_pixel=_lerp_point(lp.xy_pixel, rp.xy_pixel, alpha),
            det_confidence=_lerp(lp.det_confidence, rp.det_confidence, alpha),
            role=label_source.role,
            team_id=label_source.team_id,
            team_confidence=_lerp(lp.team_confidence, rp.team_confidence, alpha),
            xy_pitch_m=_lerp_point(lp.xy_pitch_m, rp.xy_pitch_m, alpha),
        ))

    lb, rb = left.ball, right.ball
    if lb is not None and rb is not None and lb.valid and rb.valid:
        ball = BallObservation(
            frame_idx=frame_idx,
            valid=True,
            confidence=_lerp(lb.confidence, rb.confidence, alpha),
            xy_pixel=_lerp_point(lb.xy_pixel, rb.xy_pixel, alpha),
            xy_pitch_m=_lerp_point(lb.xy_pitch_m, rb.xy_pitch_m, alpha),
            observed=lb.observed and rb.observed and frame_idx in {
                lb.frame_idx, rb.frame_idx
            },
        )
    else:
        sampled_ball = nearest.ball
        ball = None if sampled_ball is None else BallObservation(
            frame_idx=frame_idx,
            valid=sampled_ball.valid,
            confidence=sampled_ball.confidence,
            xy_pixel=sampled_ball.xy_pixel,
            xy_pitch_m=sampled_ball.xy_pitch_m,
            observed=sampled_ball.observed and frame_idx == sampled_ball.frame_idx,
        )

    return FrameState(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        segment_id=left.segment_id,
        calibration_confidence=_lerp(
            left.calibration_confidence, right.calibration_confidence, alpha
        ),
        players=players,
        ball=ball,
        # Landmark detections and homographies are sampled measurements. Use the
        # nearest real estimate rather than blending projective matrices.
        pitch_keypoints_xy=nearest.pitch_keypoints_xy,
        pitch_keypoints_conf=nearest.pitch_keypoints_conf,
        calibration_m=nearest.calibration_m,
    )


def densify_frame_states(
    frame_states: Iterable[FrameState],
    total_frames: int,
    fps: float,
) -> list[FrameState]:
    """Create native-rate visualization states from sampled CV observations.

    Players are matched only by their ByteTrack ids and linearly interpolated
    between adjacent observed states. The returned states are for rendering only;
    action tagging must always consume the original sampled observations.
    """
    sampled_by_frame = {fs.frame_idx: fs for fs in frame_states}
    sampled = sorted(sampled_by_frame.values(), key=lambda fs: fs.frame_idx)
    if not sampled or total_frames <= 0:
        return []
    indices = [fs.frame_idx for fs in sampled]
    dense: list[FrameState] = []
    for frame_idx in range(total_frames):
        timestamp_s = frame_idx / fps if fps > 0 else 0.0
        pos = bisect_right(indices, frame_idx)
        if pos == 0:
            dense.append(_retime_state(sampled[0], frame_idx, timestamp_s))
        elif pos == len(sampled):
            dense.append(_retime_state(sampled[-1], frame_idx, timestamp_s))
        else:
            dense.append(_interpolate_state(
                sampled[pos - 1], sampled[pos], frame_idx, timestamp_s
            ))
    return dense


def _color_for(role: str, team_id: Optional[int]) -> tuple[int, int, int]:
    if role == "referee":
        return REFEREE_COLOR
    if team_id in TEAM_COLORS:
        return TEAM_COLORS[team_id]
    return UNKNOWN_COLOR


def _team_label(team_id: Optional[int]) -> str:
    return f"T{team_id}" if team_id in (0, 1) else "T?"


def _blend_rect(img, x0, y0, x1, y1, color, alpha):
    """Alpha-blend a filled rect onto img in place (semi-transparent panels)."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, color, dtype=np.uint8), alpha, roi, 1 - alpha, 0, roi)


class _Text:
    """Font sizing anchored to frame height so the overlay reads at any resolution."""

    def __init__(self, height: int):
        self.s = max(0.5, height / 1080.0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def put(self, img, text, org, scale, color, thick=1, shadow=True):
        fs = scale * self.s
        th = max(1, int(round(thick * self.s)))
        if shadow:
            cv2.putText(img, text, (org[0] + 1, org[1] + 1), self.font, fs,
                        (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, org, self.font, fs, color, th, cv2.LINE_AA)

    def size(self, text, scale, thick=1):
        (w, h), _ = cv2.getTextSize(text, self.font, scale * self.s,
                                    max(1, int(round(thick * self.s))))
        return w, h


def _draw_chip(img, txt, org, bg, T: _Text, fg=(255, 255, 255), scale=0.6, pad=5):
    """A filled label chip with centered text; org is the top-left corner."""
    w, h = T.size(txt, scale, thick=2)
    p = int(pad * T.s)
    x0, y0 = int(org[0]), int(org[1])
    x1, y1 = x0 + w + 2 * p, y0 + h + 2 * p
    cv2.rectangle(img, (x0, y0), (x1, y1), bg, -1, cv2.LINE_AA)
    T.put(img, txt, (x0 + p, y1 - p), scale, fg, thick=2, shadow=False)
    return x1, y1


# ------------------------------ per-object drawing ----------------------------

def _draw_player(img, obs, T: _Text, highlight: bool):
    x, y = int(obs.xy_pixel[0]), int(obs.xy_pixel[1])
    color = _color_for(obs.role, obs.team_id)
    ax = int(22 * T.s)
    ay = int(9 * T.s)
    if highlight:  # white ring under the controlling (nearest-to-ball) player
        cv2.ellipse(img, (x, y), (ax + 4, ay + 3), 0, -45, 235,
                    (255, 255, 255), max(2, int(3 * T.s)), cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (ax, ay), 0, -45, 235, color,
                max(2, int(3 * T.s)), cv2.LINE_AA)
    tag = str(obs.track_id)
    if obs.role == "goalkeeper":
        tag = "GK" + tag
    elif obs.role == "referee":
        tag = "R" + tag
    w, h = T.size(tag, 0.5, thick=2)
    p = int(4 * T.s)
    bx0 = x - w // 2 - p
    by0 = y + ay + int(2 * T.s)
    cv2.rectangle(img, (bx0, by0), (bx0 + w + 2 * p, by0 + h + 2 * p), color, -1, cv2.LINE_AA)
    fg = (20, 20, 20) if obs.role == "referee" else (255, 255, 255)
    T.put(img, tag, (bx0 + p, by0 + h + p), 0.5, fg, thick=2, shadow=False)


def _draw_ball(img, ball, T: _Text):
    if ball is None or not ball.valid or ball.xy_pixel is None:
        return
    x, y = int(ball.xy_pixel[0]), int(ball.xy_pixel[1])
    m = int(11 * T.s)
    pts = np.array([[x, y], [x - m, y - int(1.7 * m)], [x + m, y - int(1.7 * m)]], np.int32)
    cv2.fillPoly(img, [pts], BALL_COLOR, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, BALL_OUTLINE, max(1, int(2 * T.s)), cv2.LINE_AA)


def _nearest_to_ball(fs: FrameState, max_px: float) -> Optional[int]:
    """Track id of the player closest to a valid ball within `max_px` (visual aid)."""
    if fs.ball is None or not fs.ball.valid or fs.ball.xy_pixel is None:
        return None
    bx, by = fs.ball.xy_pixel
    best, best_d = None, max_px
    for p in fs.players:
        d = float(np.hypot(p.xy_pixel[0] - bx, p.xy_pixel[1] - by))
        if d < best_d:
            best, best_d = p.track_id, d
    return best


# --------------------------------- HUD / panels -------------------------------

def _draw_hud(img, fs: FrameState, T: _Text):
    W = img.shape[1]
    bar_h = int(52 * T.s)
    _blend_rect(img, 0, 0, W, bar_h, HUD_BG, 0.55)
    y = int(34 * T.s)
    T.put(img, "updt.pro", (int(16 * T.s), y), 0.9, (255, 255, 255), thick=2)
    x = T.size("updt.pro", 0.9, thick=2)[0] + int(28 * T.s)
    T.put(img, "assisted tagging  |  pre-MVP", (x, y), 0.6, (170, 170, 170), thick=1)

    # clock + segment (right aligned)
    mm, ss = divmod(fs.timestamp_s, 60)
    clock = f"{int(mm):02d}:{ss:05.2f}  |  frame {fs.frame_idx}  |  seg {fs.segment_id}"
    cw = T.size(clock, 0.6, thick=1)[0]
    T.put(img, clock, (W - cw - int(16 * T.s), y), 0.6, (220, 220, 220), thick=1)

    # calibration confidence bar + ball status, just under the top bar
    cy = bar_h + int(22 * T.s)
    _draw_conf_bar(img, int(16 * T.s), cy, fs.calibration_confidence, "calib", T)
    ball_ok = fs.ball is not None and fs.ball.valid
    txt = f"BALL {fs.ball.confidence:.2f}" if ball_ok else "BALL none"
    _draw_chip(img, txt, (int(16 * T.s), cy + int(14 * T.s)),
               STATUS_COLORS["accept"] if ball_ok else STATUS_COLORS["hidden"], T, scale=0.55)


def _draw_conf_bar(img, x, y, val, label, T: _Text):
    w = int(150 * T.s)
    h = int(11 * T.s)
    val = max(0.0, min(1.0, float(val)))
    # red -> green ramp
    color = (int(60 + (1 - val) * 40), int(80 + val * 150), int(60 + (1 - val) * 180))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + int(w * val), y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1, cv2.LINE_AA)
    T.put(img, f"{label} {val:.2f}", (x + w + int(8 * T.s), y + h), 0.5, (220, 220, 220), thick=1)


def _draw_event_panel(img, events: list[Event], cur: int, T: _Text, max_rows: int = 9):
    started = [e for e in events if e.start_frame <= cur]
    if not started:
        return
    H, W = img.shape[:2]
    pw = int(340 * T.s)
    x0 = W - pw
    row_h = int(40 * T.s)
    ph = int(46 * T.s) + row_h * min(max_rows, len(started))
    y0 = int(80 * T.s)
    _blend_rect(img, x0, y0, W, y0 + ph, PANEL_BG, 0.6)
    T.put(img, "EVENT LOG", (x0 + int(14 * T.s), y0 + int(28 * T.s)), 0.7, (255, 255, 255), thick=2)
    ry = y0 + int(46 * T.s)
    for e in started[-max_rows:]:
        active = e.start_frame <= cur <= e.end_frame
        if active:
            _blend_rect(img, x0, ry, W, ry + row_h, (70, 70, 70), 0.5)
        cx = x0 + int(20 * T.s)
        cyc = ry + row_h // 2
        cv2.circle(img, (cx, cyc), int(7 * T.s), STATUS_COLORS.get(e.review_status, UNKNOWN_COLOR), -1, cv2.LINE_AA)
        name = e.type.replace("_", " ")
        T.put(img, name, (cx + int(16 * T.s), cyc + int(5 * T.s)), 0.55, (240, 240, 240), thick=1)
        # team chip + confidence, right-aligned in the row
        chip = _team_label(e.team_id)
        cw = T.size(chip, 0.5, thick=2)[0] + int(10 * T.s)
        confs = f"{e.confidence:.2f}"
        cw2 = T.size(confs, 0.55, thick=1)[0]
        T.put(img, confs, (W - cw2 - int(14 * T.s), cyc + int(5 * T.s)), 0.55,
              STATUS_COLORS.get(e.review_status, UNKNOWN_COLOR), thick=1)
        _draw_chip(img, chip, (W - cw2 - cw - int(26 * T.s), cyc - int(9 * T.s)),
                   _color_for("player", e.team_id), T, scale=0.5)
        if active:
            T.put(img, "LIVE", (cx + int(16 * T.s), ry + int(13 * T.s)), 0.4, (90, 205, 100), thick=1)
        ry += row_h


def _draw_active_banner(img, events: list[Event], cur: int, T: _Text):
    active = [e for e in events if e.start_frame <= cur <= e.end_frame]
    if not active:
        return
    e = max(active, key=lambda ev: EVENT_PRIORITY.get(ev.type, 0))
    label = e.type.replace("_", " ").upper()
    extra = ""
    if e.type == "possession_change":
        extra = f"  {_team_label(e.attributes.get('from_team'))} → {_team_label(e.attributes.get('to_team'))}"
    elif e.type == "pass":
        extra = f"  {e.attributes.get('outcome', '')}"
    elif e.type == "zone_entry":
        extra = f"  {e.attributes.get('zone', '')}"
    else:
        extra = f"  {_team_label(e.team_id)}"
    txt = label + extra
    W = img.shape[1]
    w = T.size(txt, 0.85, thick=2)[0]
    x0 = (W - w) // 2 - int(20 * T.s)
    y0 = int(64 * T.s)
    _blend_rect(img, x0, y0, x0 + w + int(40 * T.s), y0 + int(40 * T.s),
                STATUS_COLORS.get(e.review_status, UNKNOWN_COLOR), 0.85)
    T.put(img, txt, (x0 + int(20 * T.s), y0 + int(28 * T.s)), 0.85, (15, 15, 15), thick=2, shadow=False)


# ------------------------- keypoints + tactical inset -------------------------

def _draw_keypoints(img, fs: FrameState, T: _Text, conf_thr: float = 0.5):
    """Overlay detected pitch landmarks + skeleton (supervision annotators)."""
    xy = fs.pitch_keypoints_xy
    if xy is None or len(xy) == 0:
        return
    xy = np.asarray(xy, dtype=np.float32)
    conf = fs.pitch_keypoints_conf
    valid = (np.asarray(conf, dtype=float) > conf_thr) if conf is not None \
        else (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if not valid.any():
        return
    edges = [(a, b) for (a, b) in _PITCH_EDGES
             if a < len(xy) and b < len(xy) and valid[a] and valid[b]]
    if edges:  # draw skeleton on the full point set so indices line up
        sv.EdgeAnnotator(color=KEYPOINT_COLOR, thickness=max(1, int(2 * T.s)),
                         edges=edges).annotate(img, sv.KeyPoints(xy=xy[None, ...]))
    sv.VertexAnnotator(color=KEYPOINT_COLOR, radius=max(2, int(4 * T.s))).annotate(
        img, sv.KeyPoints(xy=xy[valid][None, ...]))


def _composite_radar(img, panel, T: _Text, calib_conf: float, calib_gate: float,
                     width_frac: float = 0.30):
    """Blend the tactical panel as a bottom-left inset; dim + label when the frame's
    calibration is too weak to trust the map (honesty guarantee)."""
    if panel is None:
        return
    H, W = img.shape[:2]
    tw = int(W * width_frac)
    ph, pw = panel.shape[:2]
    th = int(ph * tw / pw)
    panel = cv2.resize(panel, (tw, th), interpolation=cv2.INTER_AREA)
    low = calib_conf < calib_gate
    if low:
        panel = (panel.astype(np.float32) * 0.45).astype(np.uint8)
    m = int(16 * T.s)
    x0, y0 = m, H - th - m
    cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + tw + 3, y0 + th + 3), (20, 20, 20), 2, cv2.LINE_AA)
    img[y0:y0 + th, x0:x0 + tw] = panel
    _blend_rect(img, x0, y0, x0 + tw, y0 + int(26 * T.s), (20, 20, 20), 0.5)
    T.put(img, "TACTICAL MAP  (Voronoi + Delaunay)", (x0 + int(8 * T.s), y0 + int(18 * T.s)),
          0.5, (240, 240, 240), thick=1)
    label = f"calib {calib_conf:.2f}" + ("  LOW" if low else "")
    _draw_chip(img, label, (x0 + int(8 * T.s), y0 + th - int(26 * T.s)),
               STATUS_COLORS["review"] if low else STATUS_COLORS["accept"], T, scale=0.45)


# ------------------------- tactical overlays on the video ---------------------

def _draw_team_mesh(img, fs: FrameState, T: _Text):
    """Delaunay team-shape mesh drawn on the broadcast frame. Adjacency is computed
    in PIXEL space so it works with or without calibration."""
    teams: dict[int, list] = {0: [], 1: []}
    for p in fs.players:
        if p.role != "referee" and p.team_id in (0, 1):
            teams[p.team_id].append(p.xy_pixel)
    thick = max(2, int(2 * T.s))
    r = max(3, int(5 * T.s))
    for tid, pts in teams.items():
        if len(pts) < 3:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        color = TEAM_COLORS[tid]
        for a, b in radar.team_delaunay_edges(pts):
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     color, thick, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), r, (20, 20, 20), max(1, int(T.s)), cv2.LINE_AA)


def _warp_voronoi(img, fs: FrameState, calib_gate: float, opacity: float = 0.35):
    """Warp the metric Voronoi control regions onto the field via the calibration
    homography. Skipped (nothing drawn) when the frame isn't confidently calibrated."""
    M = fs.calibration_m
    if M is None or fs.calibration_confidence < calib_gate:
        return
    t0, t1 = radar.team_points_metres(fs)
    grid = radar.voronoi_overlay_grid(t0, t1)
    if grid is None:
        return
    bgr, alpha, A = grid
    try:
        Minv = np.linalg.inv(np.asarray(M, dtype=np.float64))  # metres -> pixels
    except np.linalg.LinAlgError:
        return
    H = Minv @ A                                               # grid px -> image px
    Himg, Wimg = img.shape[:2]
    wcol = cv2.warpPerspective(bgr, H, (Wimg, Himg), flags=cv2.INTER_NEAREST)
    wmask = cv2.warpPerspective(alpha, H, (Wimg, Himg), flags=cv2.INTER_NEAREST) > 0
    if not wmask.any():
        return
    img[wmask] = (img[wmask].astype(np.float32) * (1 - opacity)
                  + wcol[wmask].astype(np.float32) * opacity).astype(np.uint8)


def render_radar_video(
    frame_states: Iterable[FrameState],
    out_path: str,
    fps: float,
    scale: float = 0.1,
    padding: int = 50,
    draw_voronoi: bool = True,
    draw_delaunay: bool = True,
    title: str = "",
    fourcc: str = "mp4v",
) -> str:
    """Standalone top-down tactical video (no source frames needed) — used for the
    Voronoi-only, Delaunay-only and combined map outputs."""
    base = radar._base_pitch(scale, padding)
    h, w = base.shape[:2]
    T = _Text(h)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), max(1.0, fps), (w, h))
    n = 0
    for fs in tqdm(frame_states, desc=f"[radar:{title or 'map'}]"):
        panel = render_radar(fs, scale=scale, padding=padding,
                             draw_voronoi=draw_voronoi, draw_delaunay=draw_delaunay)
        if panel is None:
            panel = base.copy()
        if title:
            _blend_rect(panel, 0, 0, w, int(30 * T.s), (20, 20, 20), 0.5)
            T.put(panel, title, (int(10 * T.s), int(21 * T.s)), 0.6, (240, 240, 240), thick=1)
        writer.write(panel)
        n += 1
    writer.release()
    print(f"[render] wrote {n} frames -> {out_path}")
    return out_path


# ---------------------------------- driver ------------------------------------

def render_video(
    source: str,
    frame_states: Iterable[FrameState],
    events: list[Event],
    out_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    control_radius_px: float = 60.0,
    draw_players: bool = True,
    draw_ball: bool = True,
    draw_hud: bool = True,
    draw_events: bool = True,
    draw_keypoints: bool = False,
    draw_radar: bool = True,
    draw_triangulation: bool = False,
    draw_voronoi_overlay: bool = False,
    voronoi_overlay_opacity: float = 0.35,
    radar_width_frac: float = 0.30,
    radar_scale: float = 0.06,
    calib_gate: float = 0.5,
    desc: str = "render",
    fourcc: str = "mp4v",
) -> str:
    """Overlay pipeline output onto `source` and write an annotated video. The layer
    flags let one function emit the showcase, the triangulation overlay, and the
    Voronoi overlay.

    `frame_states` must have been produced from the same (source, stride, max_frames)
    so their `frame_idx` line up with the re-read frames.
    """
    states = {fs.frame_idx: fs for fs in frame_states}
    events = sorted(events, key=lambda e: (e.start_frame, e.type))
    info = get_video_info(source)
    out_fps = max(1.0, info.fps / max(1, stride))

    writer = None
    T = _Text(info.height)
    n = 0
    n_total = max_frames or (info.total_frames // max(1, stride))
    for fp in tqdm(frames(source, stride=stride, max_frames=max_frames),
                   total=n_total, desc=f"[{desc}]"):
        img = fp.image
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                                     out_fps, (img.shape[1], img.shape[0]))
        fs = states.get(fp.frame_idx)
        if fs is not None:
            if draw_voronoi_overlay:  # tint the field first, everything else on top
                _warp_voronoi(img, fs, calib_gate, voronoi_overlay_opacity)
            if draw_keypoints:
                _draw_keypoints(img, fs, T)
            if draw_triangulation:
                _draw_team_mesh(img, fs, T)
            if draw_players:
                controller = _nearest_to_ball(fs, control_radius_px)
                for p in fs.players:
                    _draw_player(img, p, T, highlight=(p.track_id == controller))
            if draw_ball:
                _draw_ball(img, fs.ball, T)
            if draw_hud:
                _draw_hud(img, fs, T)
            if draw_events:
                _draw_active_banner(img, events, fp.frame_idx, T)
                _draw_event_panel(img, events, fp.frame_idx, T)
            if draw_radar:
                panel = render_radar(fs, scale=radar_scale)
                _composite_radar(img, panel, T, fs.calibration_confidence,
                                 calib_gate, radar_width_frac)
        writer.write(img)
        n += 1
    if writer is not None:
        writer.release()
    print(f"[render] wrote {n} annotated frames @ {out_fps:.1f} fps -> {out_path}")
    return out_path

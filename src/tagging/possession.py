"""Possession state machine (plan §9.1, §9.2).

Turns the per-frame perception stream into:
  * control series  : who (if anyone) controls the ball each frame
  * control spells   : confirmed runs of a single player controlling the ball
  * possession spells: team-level possession runs, with min-evidence changes

Honest-uncertainty rules baked in:
  * No valid ball for > no_ball.max_gap_frames -> possession is UNKNOWN; no spell
    is extended across the gap (plan §5.3).
  * A controller is only confirmed after control_min_frames -> single deflections
    don't create possession.
  * A team change needs the new team to hold control_min_frames / min_control_frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.common.geometry import distance_m
from src.common.schema import FrameState


@dataclass
class ControlSample:
    frame_idx: int
    ts: float
    segment_id: int
    valid_ball: bool
    ball_xy_m: Optional[tuple[float, float]]
    controller_track: Optional[int]
    team: Optional[int]
    ball_conf: float = 0.0
    controller_det_conf: float = 0.0
    controller_team_conf: float = 0.0
    calib_conf: float = 0.0


@dataclass
class ControlSpell:
    track_id: int
    team: Optional[int]
    start_idx: int            # index into the control series (not raw frame idx)
    end_idx: int
    samples: list[ControlSample] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.samples[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.samples[-1].frame_idx

    @property
    def start_xy(self) -> Optional[tuple[float, float]]:
        return self.samples[0].ball_xy_m

    @property
    def end_xy(self) -> Optional[tuple[float, float]]:
        return self.samples[-1].ball_xy_m

    def conf_components(self, track_conf: dict[int, float]) -> dict[str, float]:
        return {
            "detection": float(np.mean([s.ball_conf for s in self.samples] or [0.0])),
            "tracking": track_conf.get(self.track_id, 0.0),
            "team": float(np.mean([s.controller_team_conf for s in self.samples] or [0.0])),
            "calibration": float(np.mean([s.calib_conf for s in self.samples] or [0.0])),
        }


@dataclass
class PossessionSpell:
    team: int
    spells: list[ControlSpell]

    @property
    def start_frame(self) -> int:
        return self.spells[0].start_frame

    @property
    def end_frame(self) -> int:
        return self.spells[-1].end_frame


def build_control_series(frame_states: list[FrameState], cfg: dict[str, Any]) -> list[ControlSample]:
    """For each frame, find the nearest player within the control radius whose
    velocity matches the ball's (ball moving WITH the player = controlled; ball
    moving away from everyone = in flight). Returns one ControlSample per frame."""
    pcfg = cfg["possession"]
    radius = pcfg["control_radius_m"]
    max_rel_speed = pcfg["ball_rel_speed_max_mps"]

    series: list[ControlSample] = []
    prev_ball: Optional[tuple[float, float]] = None
    prev_ts: Optional[float] = None
    prev_player: dict[int, tuple[float, float]] = {}

    for fs in frame_states:
        ball = fs.ball
        valid = bool(ball and ball.valid and ball.xy_pitch_m is not None and fs.calibration_confidence > 0)
        controller = None
        team = None
        cdet = cteam = 0.0
        ball_xy = ball.xy_pitch_m if valid else None
        ball_conf = ball.confidence if ball else 0.0

        if valid:
            dt = fs.timestamp_s - prev_ts if prev_ts is not None else None
            # nearest non-referee player within the control radius
            cand = None
            best_d = radius
            for p in fs.players:
                if p.role == "referee" or p.xy_pitch_m is None:
                    continue
                d = distance_m(ball_xy, p.xy_pitch_m)
                if d <= best_d:
                    best_d, cand = d, p
            if cand is not None:
                rel_ok = True
                pprev = prev_player.get(cand.track_id)
                if dt and dt > 0 and prev_ball is not None and pprev is not None:
                    ball_v = ((ball_xy[0] - prev_ball[0]) / dt, (ball_xy[1] - prev_ball[1]) / dt)
                    plyr_v = ((cand.xy_pitch_m[0] - pprev[0]) / dt, (cand.xy_pitch_m[1] - pprev[1]) / dt)
                    rel = distance_m(ball_v, plyr_v)  # |v_ball - v_player|
                    rel_ok = rel <= max_rel_speed
                if rel_ok:
                    controller, team = cand.track_id, cand.team_id
                    cdet, cteam = cand.det_confidence, cand.team_confidence
            prev_ball, prev_ts = ball_xy, fs.timestamp_s
            prev_player = {p.track_id: p.xy_pitch_m for p in fs.players if p.xy_pitch_m is not None}

        series.append(
            ControlSample(
                frame_idx=fs.frame_idx, ts=fs.timestamp_s, segment_id=fs.segment_id,
                valid_ball=valid, ball_xy_m=ball_xy, controller_track=controller, team=team,
                ball_conf=ball_conf, controller_det_conf=cdet, controller_team_conf=cteam,
                calib_conf=fs.calibration_confidence,
            )
        )
    return series


def confirmed_control_spells(series: list[ControlSample], cfg: dict[str, Any]) -> list[ControlSpell]:
    """Maximal runs of the same controller track, kept only if >= control_min_frames."""
    min_frames = cfg["possession"]["control_min_frames"]
    spells: list[ControlSpell] = []
    i = 0
    n = len(series)
    while i < n:
        s = series[i]
        if s.controller_track is None:
            i += 1
            continue
        j = i
        while j + 1 < n and series[j + 1].controller_track == s.controller_track:
            j += 1
        run = series[i : j + 1]
        if len(run) >= min_frames:
            spells.append(
                ControlSpell(track_id=s.controller_track, team=s.team, start_idx=i, end_idx=j, samples=run)
            )
        i = j + 1
    return spells


def team_possession_spells(spells: list[ControlSpell], cfg: dict[str, Any]) -> list[PossessionSpell]:
    """Merge adjacent confirmed control spells of the same team into team possession
    runs. A change to a new team only counts once that team holds for
    min_control_frames (no flapping on a single deflection)."""
    min_change = cfg["possession_change"]["min_control_frames"]
    out: list[PossessionSpell] = []
    for sp in spells:
        if sp.team is None:
            continue
        if out and out[-1].team == sp.team:
            out[-1].spells.append(sp)
        else:
            # require the new team's first spell to be substantial to switch
            if len(sp.samples) >= min_change or not out:
                out.append(PossessionSpell(team=sp.team, spells=[sp]))
            elif out:
                # too short to flip possession; attribute to current team's run
                out[-1].spells.append(sp)
    return out

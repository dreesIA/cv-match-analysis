"""Event derivation: possession, possession changes, passes, carries, zone entries
(plan §9). Each event is scored from its component confidences and routed to
accept / review / hidden.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Optional

import numpy as np

from src.common.geometry import direction_deg, distance_m
from src.common.schema import Event, FrameState
from src.tagging.confidence import score_and_route
from src.tagging.possession import (
    ControlSample,
    ControlSpell,
    PossessionSpell,
    build_control_series,
    confirmed_control_spells,
    team_possession_spells,
)

PITCH_LENGTH_M = 120.0


def _components(samples: list[ControlSample], track_ids: set[int], track_conf: dict[int, float]) -> dict[str, float]:
    if not samples:
        return {"detection": 0.0, "tracking": 0.0, "team": 0.0, "calibration": 0.0}
    return {
        "detection": float(np.mean([s.ball_conf for s in samples])),
        "tracking": float(np.mean([track_conf.get(t, 0.0) for t in track_ids])) if track_ids else 0.0,
        "team": float(np.mean([s.controller_team_conf for s in samples])),
        "calibration": float(np.mean([s.calib_conf for s in samples])),
    }


def _make_event(
    etype: str, start_s: ControlSample, end_s: ControlSample, team: Optional[int],
    player_ids: list[int], start_xy: Optional[tuple], end_xy: Optional[tuple],
    components: dict[str, float], cfg: dict[str, Any], attributes: Optional[dict] = None,
) -> Event:
    length = distance_m(start_xy, end_xy) if start_xy and end_xy else None
    direction = direction_deg(start_xy, end_xy) if start_xy and end_xy else None
    score, status = score_and_route(etype, components, cfg["confidence"])
    return Event(
        type=etype, start_frame=start_s.frame_idx, end_frame=end_s.frame_idx,
        start_ts=start_s.ts, end_ts=end_s.ts, team_id=team, player_ids=player_ids,
        start_xy_m=start_xy, end_xy_m=end_xy, length_m=length, direction_deg=direction,
        attributes=attributes or {}, conf_components=components, confidence=score, review_status=status,
    )


def _attacking_direction(passes: list[Event], poss: list[PossessionSpell]) -> dict[int, int]:
    """+1 if team attacks toward +x, -1 toward -x. Inferred from completed-pass dx,
    falling back to mean possession x vs midfield. Heuristic; documented in plan §9."""
    dxs: dict[int, list[float]] = {}
    for p in passes:
        if p.team_id is None or p.start_xy_m is None or p.end_xy_m is None:
            continue
        dxs.setdefault(p.team_id, []).append(p.end_xy_m[0] - p.start_xy_m[0])
    out: dict[int, int] = {}
    for team in {sp.team for sp in poss}:
        vals = dxs.get(team)
        if vals:
            out[team] = 1 if median(vals) >= 0 else -1
        else:
            out[team] = 1
    return out


def tag(frame_states: list[FrameState], track_conf: dict[int, float], cfg: dict[str, Any]) -> list[Event]:
    series = build_control_series(frame_states, cfg)
    spells = confirmed_control_spells(series, cfg)
    poss = team_possession_spells(spells, cfg)
    events: list[Event] = []

    # ---- possession spells ----
    for ps in poss:
        samples = [s for sp in ps.spells for s in sp.samples]
        tids = {sp.track_id for sp in ps.spells}
        comp = _components(samples, tids, track_conf)
        events.append(_make_event(
            "possession", samples[0], samples[-1], ps.team, sorted(tids),
            samples[0].ball_xy_m, samples[-1].ball_xy_m, comp, cfg,
        ))

    # ---- possession changes (between adjacent team spells) ----
    for prev, cur in zip(poss, poss[1:]):
        prev_last = prev.spells[-1].samples[-1]
        cur_first = cur.spells[0].samples[0]
        comp = _components([prev_last, cur_first], {prev.spells[-1].track_id, cur.spells[0].track_id}, track_conf)
        events.append(_make_event(
            "possession_change", prev_last, cur_first, cur.team,
            [prev.spells[-1].track_id, cur.spells[0].track_id],
            prev_last.ball_xy_m, cur_first.ball_xy_m, comp, cfg,
            attributes={"from_team": prev.team, "to_team": cur.team},
        ))

    # ---- passes (A releases -> teammate/opponent B controls) ----
    pcfg = cfg["pass"]
    passes: list[Event] = []
    for a, b in zip(spells, spells[1:]):
        if a.end_xy is None or b.start_xy is None:
            continue
        sep = distance_m(a.end_xy, b.start_xy)
        gap_s = b.samples[0].ts - a.samples[-1].ts
        if sep < pcfg["min_separation_m"] or gap_s > pcfg["receiver_window_s"] or gap_s < 0:
            continue
        outcome = "completed" if a.team == b.team else "intercepted"
        comp = _components(a.samples + b.samples, {a.track_id, b.track_id}, track_conf)
        ev = _make_event(
            "pass", a.samples[-1], b.samples[0], a.team, [a.track_id, b.track_id],
            a.end_xy, b.start_xy, comp, cfg, attributes={"outcome": outcome, "receiver_team": b.team},
        )
        passes.append(ev)
    events.extend(passes)

    # ---- carries (single controller moves the ball) ----
    ccfg = cfg["carry"]
    for sp in spells:
        if sp.start_xy is None or sp.end_xy is None:
            continue
        disp = distance_m(sp.start_xy, sp.end_xy)
        if disp >= ccfg["min_displacement_m"] and len(sp.samples) >= ccfg["min_frames"]:
            comp = sp.conf_components(track_conf)
            events.append(_make_event(
                "carry", sp.samples[0], sp.samples[-1], sp.team, [sp.track_id],
                sp.start_xy, sp.end_xy, comp, cfg, attributes={"displacement_m": disp},
            ))

    # ---- zone entries (final-third + halfway), inheriting calibration confidence ----
    atk = _attacking_direction(passes, poss)
    zcfg = cfg["zones"]
    for ps in poss:
        seg = series[ps.spells[0].start_idx : ps.spells[-1].end_idx + 1]
        valid = [s for s in seg if s.valid_ball and s.ball_xy_m is not None]
        d = atk.get(ps.team, 1)
        third_line = zcfg["attacking_third_x_m"] if d == 1 else PITCH_LENGTH_M - zcfg["attacking_third_x_m"]
        half_line = zcfg["halfway_x_m"] if d == 1 else PITCH_LENGTH_M - zcfg["halfway_x_m"]
        for line, name in ((half_line, "halfway_progression"), (third_line, "attacking_third_entry")):
            crossing = _first_crossing(valid, line, d)
            if crossing is not None:
                comp = _components([crossing], {crossing.controller_track} if crossing.controller_track else set(), track_conf)
                events.append(_make_event(
                    "zone_entry", crossing, crossing, ps.team,
                    [crossing.controller_track] if crossing.controller_track else [],
                    crossing.ball_xy_m, crossing.ball_xy_m, comp, cfg, attributes={"zone": name},
                ))

    events.sort(key=lambda e: (e.start_frame, e.type))
    return events


def _first_crossing(samples: list[ControlSample], line: float, direction: int) -> Optional[ControlSample]:
    """First sample whose ball crosses `line` in the attacking `direction`."""
    for prev, cur in zip(samples, samples[1:]):
        px, cx = prev.ball_xy_m[0], cur.ball_xy_m[0]
        if direction == 1 and px < line <= cx:
            return cur
        if direction == -1 and px > line >= cx:
            return cur
    return None

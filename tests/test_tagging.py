"""Synthetic tests for the tagging core (no models/video needed).

Validates the net-new MVP logic: possession spells, possession changes, completed
vs intercepted passes, carries, no-ball honesty, and confidence routing.
Run: ./.venv/bin/python tests/test_tagging.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.schema import BallObservation, FrameState, PlayerObservation
from src.tagging.confidence import score_and_route
from src.tagging.events import tag

FPS = 25.0


def _player(frame, tid, team, xy):
    return PlayerObservation(
        frame_idx=frame, track_id=tid, xy_pixel=xy, det_confidence=0.9,
        role="player", team_id=team, team_confidence=0.9, xy_pitch_m=xy,
    )


def _frame(i, ball_xy, players, ball_valid=True):
    ball = BallObservation(
        frame_idx=i, valid=ball_valid, confidence=0.9 if ball_valid else 0.1,
        xy_pixel=ball_xy, xy_pitch_m=ball_xy if ball_valid else None,
    )
    return FrameState(
        frame_idx=i, timestamp_s=i / FPS, segment_id=0,
        calibration_confidence=0.9, players=players, ball=ball,
    )


def build_scenario():
    """A(team0) controls -> pass to B(team0) -> B carries -> intercepted by C(team1)
    -> a no-ball gap (possession must go unknown)."""
    A, B, C = (30.0, 34.0), (50.0, 34.0), (55.0, 20.0)
    states = []
    # A controls frames 0-9
    for i in range(0, 10):
        states.append(_frame(i, A, [_player(i, 1, 0, A), _player(i, 2, 0, B), _player(i, 3, 1, C)]))
    # ball in flight 10-12 (fast, far from everyone)
    for k, i in enumerate(range(10, 13)):
        bx = (33.0 + 6 * k, 34.0)
        states.append(_frame(i, bx, [_player(i, 1, 0, A), _player(i, 2, 0, B), _player(i, 3, 1, C)]))
    # B controls + carries 13-25 (ball drifts 50->54)
    for k, i in enumerate(range(13, 26)):
        bx = (50.0 + 4 * k / 12.0, 34.0)
        states.append(_frame(i, bx, [_player(i, 1, 0, A), _player(i, 2, 0, (bx[0], 34.0)), _player(i, 3, 1, C)]))
    # flight 26-28
    for i in range(26, 29):
        states.append(_frame(i, (54.5, 27.0), [_player(i, 1, 0, A), _player(i, 2, 0, (54.0, 34.0)), _player(i, 3, 1, C)]))
    # C (team1) controls 29-45 -> interception + possession change
    for i in range(29, 46):
        states.append(_frame(i, C, [_player(i, 1, 0, A), _player(i, 2, 0, (54.0, 34.0)), _player(i, 3, 1, C)]))
    # no-ball gap 46-55
    for i in range(46, 56):
        states.append(_frame(i, None, [_player(i, 1, 0, A), _player(i, 2, 0, (54.0, 34.0)), _player(i, 3, 1, C)], ball_valid=False))
    return states


def test_tagging():
    cfg = load_config("tagging")
    states = build_scenario()
    track_conf = {1: 0.9, 2: 0.9, 3: 0.9}
    events = tag(states, track_conf, cfg)

    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    # possession changes: exactly one, team0 -> team1
    changes = by_type.get("possession_change", [])
    assert len(changes) == 1, f"expected 1 possession change, got {len(changes)}"
    assert changes[0].attributes["from_team"] == 0 and changes[0].attributes["to_team"] == 1

    # passes: a completed (A->B) and an intercepted (B->C)
    outcomes = {e.attributes.get("outcome") for e in by_type.get("pass", [])}
    assert "completed" in outcomes, f"missing completed pass; passes={by_type.get('pass')}"
    assert "intercepted" in outcomes, f"missing intercepted pass; outcomes={outcomes}"

    # carry: B carried the ball >=3 m
    assert any(e.attributes.get("displacement_m", 0) >= 3.0 for e in by_type.get("carry", [])), "missing carry"

    # no-ball honesty: NO event may start within the ball-gap frames 46-55
    assert all(not (46 <= e.start_frame <= 55) for e in events), "event fabricated during no-ball gap"

    # two team possession spells (team0 then team1)
    teams = [e.team_id for e in by_type.get("possession", [])]
    assert teams == [0, 1], f"expected possession [0,1], got {teams}"

    print(f"OK tagging: {len(events)} events -> "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_type.items())))


def test_routing():
    cfg = load_config("tagging")
    high = {"detection": 0.95, "tracking": 0.95, "team": 0.95, "calibration": 0.95}
    low = {"detection": 0.2, "tracking": 0.2, "team": 0.2, "calibration": 0.2}
    s_hi, r_hi = score_and_route("pass", high, cfg["confidence"])
    s_lo, r_lo = score_and_route("pass", low, cfg["confidence"])
    assert r_hi == "accept", f"high-conf pass should auto-accept, got {r_hi} ({s_hi:.2f})"
    assert r_lo == "hidden", f"low-conf pass should be hidden, got {r_lo} ({s_lo:.2f})"
    print(f"OK routing: high={s_hi:.2f}->{r_hi}, low={s_lo:.2f}->{r_lo}")


if __name__ == "__main__":
    test_tagging()
    test_routing()
    print("ALL TAGGING TESTS PASSED")

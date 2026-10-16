"""Synthetic checks for the dashboard's bundle and clip-window contract."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.showcase.data import (
    ClipWindow,
    build_clip_windows,
    discover_bundesliga_clips,
    discover_bundles,
    normalise_event,
)
from src.showcase.player import build_player_html


class ShowcaseDataTests(unittest.TestCase):
    def test_player_has_live_action_charts_without_playback_toggles(self):
        html = build_player_html(
            video_uri="data:video/mp4;base64,AA==",
            events=[],
            clip=ClipWindow("full", "Full", "", 0, 10),
            bundle_name="Bundesliga",
        )

        self.assertIn("Possession gains", html)
        self.assertIn("chart-pass", html)
        self.assertIn("chart-carry", html)
        self.assertIn('id="action-pitch"', html)
        self.assertIn("Action locations", html)
        self.assertIn("arrow-team-0", html)
        self.assertIn("arrow-team-1", html)
        self.assertIn("teamColors", html)
        self.assertIn("arrow-carry", html)
        self.assertNotIn("state.loop", html)
        self.assertNotIn("state.autoplay", html)

    def test_normalises_flat_pipeline_event(self):
        event = normalise_event({
            "type": "pass",
            "start_ts": 10,
            "end_ts": 11.25,
            "player_ids": [7, 9],
            "start_xy_m_x": 20,
            "start_xy_m_y": 30,
            "end_xy_m_x": 43,
            "end_xy_m_y": 28,
            "confidence": 1.4,
        }, 3)
        self.assertEqual(event["id"], 3)
        self.assertEqual(event["start_xy"], [20.0, 30.0])
        self.assertEqual(event["end_xy"], [43.0, 28.0])
        self.assertEqual(event["confidence"], 1.0)

    def test_builds_sequence_and_individual_moments(self):
        events = [
            {**normalise_event({"type": "pass", "start_ts": 20, "end_ts": 21}, 0)},
            {**normalise_event({"type": "zone_entry", "start_ts": 24, "end_ts": 24}, 1)},
            {**normalise_event({"type": "possession", "start_ts": 10, "end_ts": 40}, 2)},
        ]
        clips = build_clip_windows(events, 60)
        self.assertEqual(clips[0].key, "full")
        self.assertTrue(any(clip.key.startswith("sequence-") for clip in clips))
        self.assertEqual(sum(clip.key.startswith("event-") for clip in clips), 3)
        entry = next(clip for clip in clips if clip.key == "event-1")
        self.assertGreater(entry.end_s, entry.start_s)

    def test_discovers_only_complete_bundles(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            good = root / "good_bundle"
            good.mkdir()
            (good / "events.json").write_text(json.dumps([]), encoding="utf-8")
            (good / "tracking.mp4").write_bytes(b"not-a-real-video")
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "events.json").write_text(json.dumps([]), encoding="utf-8")
            with patch("src.showcase.data._probe_duration", return_value=1.0):
                bundles = discover_bundles([root])
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0].directory.name, "good_bundle")

    def test_bundesliga_discovery_prefers_curated_browser_assets(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "data" / "showcase"
            phase = root / "bundesliga" / "phase_20"
            phase.mkdir(parents=True)
            (phase / "showcase.m4v").write_bytes(b"browser-video")
            (phase / "radar_combined.m4v").write_bytes(b"browser-radar")
            (phase / "events.json").write_text(json.dumps([{
                "type": "pass",
                "start_ts": 4.0,
                "end_ts": 5.0,
                "confidence": 0.8,
            }]), encoding="utf-8")
            (phase / "run_summary.json").write_text(
                json.dumps({"source": "2e57b9_0.mp4"}), encoding="utf-8"
            )
            with patch("src.showcase.data._probe_duration", return_value=20.0):
                bundles = discover_bundesliga_clips([root])
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0].video_path.suffix, ".m4v")
            self.assertEqual(bundles[0].radar_path.name, "radar_combined.m4v")
            self.assertEqual(bundles[0].events[0]["type"], "pass")
            self.assertIn("2e57b9", bundles[0].name)
            self.assertTrue(all("match1" not in str(bundle.directory) for bundle in bundles))


if __name__ == "__main__":
    unittest.main()

"""Checks for sampled-inference time and confidence invariants."""
from __future__ import annotations

import unittest

from src.common.schema import BallObservation, FrameState
from src.pipeline import _interpolate_sampled_balls, _scale_temporal_thresholds


class SamplingTests(unittest.TestCase):
    def test_scales_frame_thresholds_to_keep_approximate_duration(self):
        cfg = {
            "possession": {"control_min_frames": 3, "hysteresis_frames": 5},
            "possession_change": {"min_control_frames": 8},
            "carry": {"min_frames": 8},
            "no_ball": {"max_gap_frames": 5},
        }

        _scale_temporal_thresholds(cfg, stride=5)

        self.assertEqual(cfg["possession"]["control_min_frames"], 1)
        self.assertEqual(cfg["possession"]["hysteresis_frames"], 1)
        self.assertEqual(cfg["possession_change"]["min_control_frames"], 2)
        self.assertEqual(cfg["carry"]["min_frames"], 2)
        self.assertEqual(cfg["no_ball"]["max_gap_frames"], 1)

    def test_ball_interpolation_is_marked_as_estimated(self):
        states = [
            FrameState(i, i / 25, 0, 1.0, ball=BallObservation(
                frame_idx=i,
                valid=i in (0, 5),
                confidence=0.8 if i in (0, 5) else 0.0,
                xy_pixel=(float(i), 10.0) if i in (0, 5) else None,
                xy_pitch_m=(float(i), 20.0) if i in (0, 5) else None,
                observed=i in (0, 5),
            ))
            for i in range(6)
        ]

        _interpolate_sampled_balls(states, observation_interval=5)

        self.assertTrue(all(state.ball.valid for state in states))
        self.assertFalse(states[3].ball.observed)
        self.assertEqual(states[3].ball.xy_pixel, (3.0, 10.0))
        self.assertAlmostEqual(states[3].ball.confidence, 0.72)


if __name__ == "__main__":
    unittest.main()

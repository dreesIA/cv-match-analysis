"""Checks for display-only interpolation of sampled CV observations."""
from __future__ import annotations

import unittest

from src.common.schema import BallObservation, FrameState, PlayerObservation
from src.visualization.annotate import densify_frame_states


def _state(frame_idx: int, x: float, *, segment_id: int = 0) -> FrameState:
    return FrameState(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 25,
        segment_id=segment_id,
        calibration_confidence=0.8,
        players=[PlayerObservation(
            frame_idx=frame_idx,
            track_id=7,
            xy_pixel=(x, 100.0),
            det_confidence=0.9,
            team_id=0,
            team_confidence=0.8,
            xy_pitch_m=(x / 10, 20.0),
        )],
        ball=BallObservation(
            frame_idx=frame_idx,
            valid=True,
            confidence=0.9,
            xy_pixel=(x + 5, 95.0),
            xy_pitch_m=(x / 10 + 0.5, 19.0),
        ),
    )


class VisualizationInterpolationTests(unittest.TestCase):
    def test_interpolates_same_bytetrack_id_at_native_rate(self):
        dense = densify_frame_states([_state(0, 0), _state(8, 80)], 9, 25)

        self.assertEqual(len(dense), 9)
        self.assertEqual(dense[4].frame_idx, 4)
        self.assertAlmostEqual(dense[4].timestamp_s, 0.16)
        self.assertEqual(dense[4].players[0].track_id, 7)
        self.assertEqual(dense[4].players[0].xy_pixel, (40.0, 100.0))
        self.assertEqual(dense[4].ball.xy_pixel, (45.0, 95.0))

    def test_does_not_interpolate_across_camera_cut(self):
        dense = densify_frame_states(
            [_state(0, 0, segment_id=1), _state(8, 80, segment_id=2)], 9, 25
        )

        self.assertEqual(dense[3].players[0].xy_pixel, (0, 100.0))
        self.assertEqual(dense[5].players[0].xy_pixel, (80, 100.0))


if __name__ == "__main__":
    unittest.main()

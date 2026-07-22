import unittest
from collections import deque

import numpy as np

from fish_activity.scoring import average_score_history, score_flow


class ScoringTest(unittest.TestCase):
    def test_score_flow_returns_zero_when_mask_too_small(self) -> None:
        mag = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        mask = np.array([[0, 255], [0, 0]], dtype=np.uint8)

        masked_mag, flow_activity, raw_activity = score_flow(
            mag,
            percentile=90.0,
            mask=mask,
            min_mask_pixels=2,
        )

        self.assertEqual(flow_activity, 0.0)
        self.assertGreater(raw_activity, 0.0)
        self.assertTrue(np.all(masked_mag == 0.0))

    def test_score_flow_uses_mask_area_weighted_percentile(self) -> None:
        mag = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        _masked_mag, flow_activity, _raw_activity = score_flow(
            mag,
            percentile=50.0,
            mask=mask,
            min_mask_pixels=1,
        )

        self.assertAlmostEqual(flow_activity, 75.0)

    def test_average_score_history(self) -> None:
        history = deque(
            [
                {
                    "segmentation_score": 1.0,
                    "optical_flow_score": 2.0,
                    "total_activity": 3.0,
                },
                {
                    "segmentation_score": 3.0,
                    "optical_flow_score": 4.0,
                    "total_activity": 7.0,
                },
            ]
        )

        averages = average_score_history(history)

        self.assertEqual(averages["previous_10_frame_count"], 2.0)
        self.assertEqual(averages["segmentation_score_prev10_avg"], 2.0)
        self.assertEqual(averages["optical_flow_score_prev10_avg"], 3.0)
        self.assertEqual(averages["total_activity_prev10_avg"], 5.0)


if __name__ == "__main__":
    unittest.main()

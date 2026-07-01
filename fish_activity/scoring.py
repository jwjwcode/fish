"""Activity score calculations."""

from __future__ import annotations

from collections import deque

import numpy as np


def average_score_history(score_history: deque[dict[str, float]]) -> dict[str, float]:
    if not score_history:
        return {
            "previous_10_frame_count": 0.0,
            "segmentation_score_prev10_avg": 0.0,
            "optical_flow_score_prev10_avg": 0.0,
            "total_activity_prev10_avg": 0.0,
        }

    count = float(len(score_history))
    return {
        "previous_10_frame_count": count,
        "segmentation_score_prev10_avg": sum(
            score["segmentation_score"] for score in score_history
        )
        / count,
        "optical_flow_score_prev10_avg": sum(
            score["optical_flow_score"] for score in score_history
        )
        / count,
        "total_activity_prev10_avg": sum(
            score["total_activity"] for score in score_history
        )
        / count,
    }


def score_flow(
    mag: np.ndarray,
    percentile: float,
    mask: np.ndarray | None,
    min_mask_pixels: int,
) -> tuple[np.ndarray, float, float]:
    raw_flow_activity = float(np.percentile(mag, percentile))

    if mask is None:
        return mag, raw_flow_activity, raw_flow_activity

    mask_bool = mask > 0
    if int(np.count_nonzero(mask_bool)) < min_mask_pixels:
        return np.zeros_like(mag, dtype=np.float32), 0.0, raw_flow_activity

    masked_mag = np.zeros_like(mag, dtype=np.float32)
    masked_mag[mask_bool] = mag[mask_bool]
    mask_area_pct = 100.0 * float(np.count_nonzero(mask_bool)) / float(mask_bool.size)
    masked_percentile = float(np.percentile(mag[mask_bool], percentile))
    flow_activity = masked_percentile * mask_area_pct
    return masked_mag, flow_activity, raw_flow_activity

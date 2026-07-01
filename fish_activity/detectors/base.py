"""Detector result contract shared by detector implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectorResult:
    mask: np.ndarray
    activity_score: float
    confidence: float = 1.0
    debug_maps: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

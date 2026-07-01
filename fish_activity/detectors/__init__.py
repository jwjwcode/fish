"""Detector implementations for feeding activity scoring."""

from fish_activity.detectors.base import DetectorResult
from fish_activity.detectors.unsupervised import UnsupervisedSplashDetector

__all__ = ["DetectorResult", "UnsupervisedSplashDetector"]

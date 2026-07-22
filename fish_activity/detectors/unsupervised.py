"""Unsupervised splash segmentation and optical-flow helpers."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from fish_activity.detectors.base import DetectorResult


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for component_id in range(1, component_count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == component_id] = 255
    return clean


def create_flow_estimator(method: str) -> tuple[object | None, str]:
    if method == "none":
        return None, "none"
    if method in {"auto", "dis"} and hasattr(cv2, "DISOpticalFlow_create"):
        preset = getattr(
            cv2,
            "DISOPTICAL_FLOW_PRESET_FINE",
            getattr(cv2, "DISOPTICAL_FLOW_PRESET_MEDIUM", 1),
        )
        return cv2.DISOpticalFlow_create(preset), "dis"
    if method == "dis":
        raise RuntimeError("DIS optical flow is not available in this OpenCV build.")
    return None, "farneback"


def compute_flow_magnitude(
    roi_gray_blur: np.ndarray,
    prev_gray_blur: np.ndarray | None,
    flow_estimator: object | None,
) -> np.ndarray:
    if prev_gray_blur is None:
        return np.zeros_like(roi_gray_blur, dtype=np.float32)

    if flow_estimator is not None:
        flow = flow_estimator.calc(prev_gray_blur, roi_gray_blur, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_blur,
            roi_gray_blur,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    mag, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    return mag


class AdaptiveSplashSegmenter:
    """Unsupervised water-splash segmenter with an adaptive normal-water model."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.means: dict[str, np.ndarray] | None = None
        self.vars: dict[str, np.ndarray] | None = None
        self.prev_filtered_mask: np.ndarray | None = None
        self.persistence: np.ndarray | None = None

    def _feature_maps(
        self,
        roi_bgr: np.ndarray,
        roi_gray_blur: np.ndarray,
        flow_mag: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        _hue, saturation_u8, value_u8 = cv2.split(hsv)
        saturation = saturation_u8.astype(np.float32)
        value = value_u8.astype(np.float32)
        white_score = value - 0.55 * saturation

        gray_f = roi_gray_blur.astype(np.float32)
        local_mean = cv2.blur(gray_f, (7, 7))
        local_sq_mean = cv2.blur(gray_f * gray_f, (7, 7))
        texture = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
        edge = np.abs(cv2.Laplacian(roi_gray_blur, cv2.CV_32F, ksize=3))

        features = {
            "saturation": saturation,
            "value": value,
            "white": white_score,
            "texture": texture,
            "edge": edge,
            "flow": flow_mag.astype(np.float32),
        }
        return features, saturation_u8, value_u8, white_score

    def _init_stats(self, features: dict[str, np.ndarray]) -> None:
        self.means = {name: feature.copy() for name, feature in features.items()}
        self.vars = {
            name: np.full_like(feature, 25.0, dtype=np.float32)
            for name, feature in features.items()
        }

    def _update_stats(
        self,
        features: dict[str, np.ndarray],
        update_mask: np.ndarray,
    ) -> None:
        if self.means is None or self.vars is None:
            self._init_stats(features)
            return

        alpha = float(np.clip(self.args.anomaly_learning_rate, 0.001, 1.0))
        idx = update_mask.astype(bool)
        if not np.any(idx):
            return

        for name, feature in features.items():
            mean = self.means[name]
            var = self.vars[name]
            delta = feature - mean
            mean[idx] += alpha * delta[idx]
            var[idx] = (1.0 - alpha) * (var[idx] + alpha * delta[idx] * delta[idx])
            np.maximum(var, 1.0, out=var)

    def _positive_z(self, name: str, feature: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.vars is not None
        z = (feature - self.means[name]) / np.sqrt(self.vars[name] + 1e-6)
        return np.maximum(z, 0.0)

    def _update_artifact_state(self, mask: np.ndarray) -> None:
        mask_bool = mask > 0
        if self.persistence is None or self.persistence.shape != mask.shape:
            self.persistence = np.zeros(mask.shape, dtype=np.float32)
        self.persistence[mask_bool] = np.minimum(self.persistence[mask_bool] + 1.0, 255.0)
        self.persistence[~mask_bool] *= 0.70
        self.prev_filtered_mask = mask_bool.copy()

    def _filter_artifacts(
        self,
        mask: np.ndarray,
        features: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.args.artifact_filter != "on":
            self._update_artifact_state(mask)
            return mask

        if cv2.countNonZero(mask) == 0:
            self._update_artifact_state(mask)
            return mask

        if self.persistence is None or self.persistence.shape != mask.shape:
            self.persistence = np.zeros(mask.shape, dtype=np.float32)
        if self.prev_filtered_mask is None or self.prev_filtered_mask.shape != mask.shape:
            prev_mask = np.zeros(mask.shape, dtype=bool)
        else:
            prev_mask = self.prev_filtered_mask

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        filtered = np.zeros_like(mask)
        roi_area = float(mask.size)

        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue

            component = labels == component_id
            area_pct = 100.0 * area / roi_area
            texture_values = features["texture"][component]
            edge_values = features["edge"][component]
            flow_values = features["flow"][component]
            saturation_values = features["saturation"][component]
            value_values = features["value"][component]
            white_values = features["white"][component]

            texture_mean = float(np.mean(texture_values)) if texture_values.size else 0.0
            edge_density = float(np.mean(edge_values >= self.args.splash_edge_threshold))
            texture_or_edge_density = float(
                np.mean(
                    (texture_values >= self.args.splash_texture_threshold)
                    | (edge_values >= self.args.splash_edge_threshold)
                )
            )
            flow_mean = float(np.mean(flow_values)) if flow_values.size else 0.0
            flow_std = float(np.std(flow_values)) if flow_values.size else 0.0
            saturation_mean = (
                float(np.mean(saturation_values)) if saturation_values.size else 0.0
            )
            value_mean = float(np.mean(value_values)) if value_values.size else 0.0
            white_mean = float(np.mean(white_values)) if white_values.size else 0.0
            flow_chaos = flow_std / (flow_mean + 1e-6)
            prev_overlap = float(np.mean(prev_mask[component]))
            new_ratio = 1.0 - prev_overlap
            mean_age = float(np.mean(self.persistence[component]))

            smooth_texture = (
                texture_mean < self.args.artifact_min_texture_mean
                or edge_density < self.args.artifact_min_edge_density
            )
            coherent_flow = (
                flow_mean >= self.args.anomaly_min_flow
                and flow_chaos < self.args.artifact_min_flow_chaos
            )
            persistent_static = (
                mean_age >= self.args.artifact_persistence_frames
                and new_ratio <= self.args.artifact_static_new_ratio
            )
            bright_white = (
                value_mean >= self.args.artifact_bright_min_value
                and white_mean >= self.args.artifact_bright_min_white_score
            )
            smooth_bright = (
                bright_white
                and texture_mean <= self.args.artifact_bright_max_texture_mean
                and edge_density <= self.args.artifact_bright_max_edge_density
            )
            static_like_bright = (
                mean_age >= self.args.artifact_bright_min_age
                or new_ratio <= self.args.artifact_static_new_ratio
                or flow_mean <= self.args.artifact_static_max_flow_mean
            )
            specular_bright = (
                value_mean >= self.args.artifact_specular_min_value
                and saturation_mean <= self.args.artifact_specular_max_saturation
                and white_mean >= self.args.artifact_specular_min_white_score
            )
            weak_texture_support = (
                texture_mean <= self.args.artifact_specular_max_texture_mean
                and edge_density <= self.args.artifact_specular_max_edge_density
                and texture_or_edge_density
                <= self.args.artifact_specular_max_texture_or_edge_density
            )
            specular_motion_is_not_splash_like = (
                flow_chaos <= self.args.artifact_specular_max_flow_chaos
                or mean_age >= self.args.artifact_bright_min_age
                or new_ratio <= self.args.artifact_static_new_ratio
            )

            reject_ripple_like = smooth_texture and coherent_flow and new_ratio < 0.65
            reject_persistent_bubble = (
                persistent_static
                and area_pct <= self.args.artifact_max_bubble_area_pct
                and (coherent_flow or smooth_texture)
            )
            reject_smooth_bright_bubble = (
                area_pct <= self.args.artifact_max_bubble_area_pct
                and smooth_bright
                and static_like_bright
            )
            reject_smooth_reflection = (
                persistent_static
                and area_pct >= self.args.artifact_min_reflection_area_pct
                and smooth_texture
            )
            reject_smooth_bright_reflection = (
                area_pct >= self.args.artifact_min_reflection_area_pct
                and smooth_bright
                and static_like_bright
            )
            reject_smooth_bright_artifact = (
                smooth_bright
                and static_like_bright
                and (area_pct >= self.args.artifact_min_reflection_area_pct * 0.25)
            )
            reject_specular_reflection = (
                area_pct >= self.args.artifact_specular_min_area_pct
                and specular_bright
                and weak_texture_support
                and specular_motion_is_not_splash_like
            )

            if (
                reject_ripple_like
                or reject_persistent_bubble
                or reject_smooth_bright_bubble
                or reject_smooth_reflection
                or reject_smooth_bright_reflection
                or reject_smooth_bright_artifact
                or reject_specular_reflection
            ):
                continue

            filtered[component] = 255

        self._update_artifact_state(mask)
        return filtered

    def compute(
        self,
        roi_bgr: np.ndarray,
        roi_gray_blur: np.ndarray,
        flow_mag: np.ndarray,
        fg_mask: np.ndarray,
        diff_mask: np.ndarray,
        processed_index: int,
    ) -> tuple[np.ndarray, float]:
        features, saturation, value, white_score = self._feature_maps(
            roi_bgr,
            roi_gray_blur,
            flow_mag,
        )

        if self.means is None or self.vars is None:
            self._init_stats(features)

        if processed_index < self.args.warmup_frames:
            self._update_stats(features, np.ones_like(value, dtype=bool))
            mask = np.zeros_like(value, dtype=np.uint8)
            return mask, 0.0

        z_value = self._positive_z("value", features["value"])
        z_white = self._positive_z("white", features["white"])
        z_texture = self._positive_z("texture", features["texture"])
        z_edge = self._positive_z("edge", features["edge"])
        z_flow = self._positive_z("flow", features["flow"])

        absolute_foam = (
            (value >= self.args.splash_min_value)
            & (saturation <= self.args.splash_max_saturation)
            & (white_score >= self.args.splash_white_score)
        )
        absolute_texture = (
            (features["texture"] >= self.args.splash_texture_threshold)
            | (features["edge"] >= self.args.splash_edge_threshold)
        )
        color_anomaly = (
            (z_white >= self.args.anomaly_color_z)
            | (z_value >= self.args.anomaly_color_z)
        )
        texture_anomaly = (
            (z_texture >= self.args.anomaly_texture_z)
            | (z_edge >= self.args.anomaly_texture_z)
        )
        flow_anomaly = (
            (z_flow >= self.args.anomaly_flow_z)
            & (features["flow"] >= self.args.anomaly_min_flow)
        )
        texture_flow_splash = np.zeros_like(flow_anomaly, dtype=bool)
        if self.args.anomaly_texture_flow_splash == "on":
            texture_flow_motion = (
                (z_flow >= self.args.anomaly_texture_flow_flow_z)
                & (features["flow"] >= self.args.anomaly_texture_flow_min_flow)
            )
            texture_flow_structure = (
                texture_anomaly
                & (
                    (features["texture"] >= self.args.anomaly_texture_flow_min_texture)
                    | (features["edge"] >= self.args.anomaly_texture_flow_min_edge)
                )
            )
            texture_flow_splash = texture_flow_structure & texture_flow_motion
        motion_support = fg_mask | diff_mask | flow_anomaly
        flow_supported_foam = absolute_foam & flow_anomaly
        if self.args.anomaly_flow_foam_requires_texture == "on":
            flow_supported_foam &= absolute_texture | texture_anomaly

        mask_bool = (
            (absolute_foam & absolute_texture & motion_support)
            | (color_anomaly & texture_anomaly & motion_support)
            | flow_supported_foam
            | texture_flow_splash
        )

        mask = (mask_bool.astype(np.uint8)) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = remove_small_components(mask, self.args.min_component_area)
        mask = self._filter_artifacts(mask, features)

        self._update_stats(features, mask == 0)
        activity_pct = 100.0 * float(cv2.countNonZero(mask)) / float(mask.size)
        return mask, activity_pct


class UnsupervisedSplashDetector:
    """Current V1 detector exposed through the common detector result contract."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=args.bg_history,
            varThreshold=args.bg_var_threshold,
            detectShadows=False,
        )
        self.anomaly_segmenter = AdaptiveSplashSegmenter(args)

    def compute(
        self,
        roi_bgr: np.ndarray,
        roi_gray_blur: np.ndarray,
        prev_gray_blur: np.ndarray | None,
        flow_mag: np.ndarray,
        processed_index: int,
    ) -> DetectorResult:
        mask, activity_pct = compute_segmentation_mask(
            roi_bgr,
            roi_gray_blur,
            prev_gray_blur,
            flow_mag,
            self.subtractor,
            self.anomaly_segmenter,
            self.args,
            processed_index,
        )
        confidence = 0.0 if processed_index < self.args.warmup_frames else 1.0
        return DetectorResult(
            mask=mask,
            activity_score=activity_pct,
            confidence=confidence,
            metrics={"segmentation_activity_pct": activity_pct},
        )


def compute_segmentation_mask(
    roi_bgr: np.ndarray,
    roi_gray_blur: np.ndarray,
    prev_gray_blur: np.ndarray | None,
    flow_mag: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    anomaly_segmenter: AdaptiveSplashSegmenter,
    args: argparse.Namespace,
    processed_index: int,
) -> tuple[np.ndarray, float]:
    fg = subtractor.apply(roi_bgr, learningRate=args.bg_learning_rate)
    fg_mask = fg > 127

    if prev_gray_blur is None:
        diff_mask = np.zeros_like(fg_mask)
    else:
        diff = cv2.absdiff(roi_gray_blur, prev_gray_blur)
        adaptive_threshold = float(np.percentile(diff, args.diff_percentile))
        threshold = max(args.diff_min_threshold, adaptive_threshold)
        diff_mask = diff > threshold

    if args.seg_method == "anomaly":
        return anomaly_segmenter.compute(
            roi_bgr,
            roi_gray_blur,
            flow_mag,
            fg_mask,
            diff_mask,
            processed_index,
        )

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)

    if args.seg_method == "motion":
        bright_mask = (value >= args.bright_value) & (
            saturation <= args.bright_saturation
        )
        mask_bool = (fg_mask & diff_mask) | (diff_mask & bright_mask)
    else:
        gray_f = roi_gray_blur.astype(np.float32)
        local_mean = cv2.blur(gray_f, (7, 7))
        local_sq_mean = cv2.blur(gray_f * gray_f, (7, 7))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
        texture_mask = local_std >= args.splash_texture_threshold

        laplacian = cv2.Laplacian(roi_gray_blur, cv2.CV_32F, ksize=3)
        edge_mask = np.abs(laplacian) >= args.splash_edge_threshold

        white_score = value.astype(np.float32) - 0.55 * saturation.astype(np.float32)
        white_mask = (
            (value >= args.splash_min_value)
            & (saturation <= args.splash_max_saturation)
            & (white_score >= args.splash_white_score)
        )
        texture_or_edge = texture_mask | edge_mask
        motion_support = fg_mask | diff_mask

        # Smooth wind ripples usually have motion but weak whiteness/foam texture.
        mask_bool = white_mask & texture_or_edge & motion_support
    if processed_index < args.warmup_frames:
        mask_bool[:] = False

    mask = (mask_bool.astype(np.uint8)) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = remove_small_components(mask, args.min_component_area)

    activity_pct = 100.0 * float(cv2.countNonZero(mask)) / float(mask.size)
    return mask, activity_pct

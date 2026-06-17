#!/usr/bin/env python3
"""Version 1 fish-feeding activity visualizer.

This script measures activity inside the middle 70% of the frame width and
full frame height. It writes an annotated video with:

- tinted non-ROI side bands
- ROI border
- a separate bottom panel for the segmentation/motion mask preview
- a separate bottom panel for the optical-flow magnitude preview
- raw segmentation, optical-flow, and total activity numbers

It also writes a CSV with the per-frame activity values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing startup guard
    raise SystemExit(
        "Missing dependency: "
        f"{exc.name}\nInstall dependencies with: python3 -m pip install -r requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotated fish-feeding activity video from one input video."
    )
    parser.add_argument("input", type=Path, help="Input video path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output annotated video path. Default: <input_stem>_activity_v1.mp4",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <output_stem>.csv",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=1280,
        help="Resize output/processing frames to this width. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--start-second",
        type=float,
        default=0.0,
        help="Start processing from this timestamp.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Process only this many seconds. Use 0 for the full remaining video.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame for faster experiments.",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=420,
        help="Width of each preview panel in the annotated video.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Frames used to warm up the background model before reporting mask activity.",
    )
    parser.add_argument(
        "--bg-history",
        type=int,
        default=250,
        help="Background-subtractor history length.",
    )
    parser.add_argument(
        "--bg-var-threshold",
        type=float,
        default=24.0,
        help="MOG2 variance threshold. Higher values make the mask less sensitive.",
    )
    parser.add_argument(
        "--bg-learning-rate",
        type=float,
        default=-1.0,
        help="MOG2 learning rate. -1 lets OpenCV choose automatically.",
    )
    parser.add_argument(
        "--diff-percentile",
        type=float,
        default=92.0,
        help="Frame-difference percentile used as the adaptive motion threshold.",
    )
    parser.add_argument(
        "--diff-min-threshold",
        type=float,
        default=8.0,
        help="Minimum grayscale difference threshold.",
    )
    parser.add_argument(
        "--seg-method",
        choices=("splash", "motion"),
        default="splash",
        help="Segmentation method. 'splash' uses fixed splash rules; 'motion' is the older motion mask.",
    )
    parser.add_argument(
        "--bright-value",
        type=int,
        default=165,
        help="HSV value threshold for the older motion segmentation method.",
    )
    parser.add_argument(
        "--bright-saturation",
        type=int,
        default=95,
        help="HSV saturation threshold for the older motion segmentation method.",
    )
    parser.add_argument(
        "--splash-min-value",
        type=int,
        default=145,
        help="Minimum HSV value for splash/foam candidates.",
    )
    parser.add_argument(
        "--splash-max-saturation",
        type=int,
        default=115,
        help="Maximum HSV saturation for splash/foam candidates.",
    )
    parser.add_argument(
        "--splash-white-score",
        type=float,
        default=105.0,
        help="Minimum whiteness score: HSV value - 0.55 * HSV saturation.",
    )
    parser.add_argument(
        "--splash-texture-threshold",
        type=float,
        default=8.0,
        help="Minimum local grayscale standard deviation for splash/foam texture.",
    )
    parser.add_argument(
        "--splash-edge-threshold",
        type=float,
        default=14.0,
        help="Minimum Laplacian edge magnitude for splash/foam texture.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=30,
        help="Remove active mask components smaller than this many pixels.",
    )
    parser.add_argument(
        "--flow-percentile",
        type=float,
        default=90.0,
        help="Percentile of optical-flow magnitude reported as flow activity.",
    )
    parser.add_argument(
        "--flow-mask",
        choices=("segmentation", "none"),
        default="segmentation",
        help="Use the segmentation mask to suppress ripple-only optical flow.",
    )
    parser.add_argument(
        "--flow-min-mask-pixels",
        type=int,
        default=50,
        help="Minimum segmentation pixels needed before masked flow is reported.",
    )
    parser.add_argument(
        "--seg-weight",
        type=float,
        default=1.0,
        help="Weight for raw segmentation percent in the total activity score.",
    )
    parser.add_argument(
        "--flow-weight",
        type=float,
        default=0.1,
        help="Weight for splash-flow energy in the total activity score.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for the output video.",
    )
    return parser.parse_args()


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] == target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def roi_bounds(width: int) -> tuple[int, int]:
    x0 = int(round(width * 0.15))
    x1 = int(round(width * 0.85))
    return x0, x1


def bottom_panel_layout(
    frame_width: int,
    frame_height: int,
    requested_preview_width: int,
) -> tuple[int, int, int]:
    text_area_width = 330
    available_for_maps = max(240, frame_width - text_area_width - 72)
    preview_width = min(requested_preview_width, max(120, available_for_maps // 2))
    preview_height = max(1, int(round(frame_height * (preview_width / frame_width))))
    panel_height = max(160, preview_height + 56)
    return panel_height, preview_width, text_area_width


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for component_id in range(1, component_count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == component_id] = 255
    return clean


def compute_segmentation_mask(
    roi_bgr: np.ndarray,
    roi_gray_blur: np.ndarray,
    prev_gray_blur: np.ndarray | None,
    subtractor: cv2.BackgroundSubtractor,
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


def compute_flow_magnitude(
    roi_gray_blur: np.ndarray,
    prev_gray_blur: np.ndarray | None,
) -> np.ndarray:
    if prev_gray_blur is None:
        return np.zeros_like(roi_gray_blur, dtype=np.float32)

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


def put_text_panel(frame: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.68
    thickness = 2
    line_height = 28
    padding = 10
    widths = []
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append(text_w)
    panel_w = max(widths) + padding * 2
    panel_h = line_height * len(lines) + padding * 2
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (255, 255, 255), 1)
    for i, line in enumerate(lines):
        baseline = y + padding + 21 + i * line_height
        cv2.putText(frame, line, (x + padding, baseline), font, font_scale, (255, 255, 255), thickness)


def make_mask_preview(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
    x0: int,
    x1: int,
    width: int,
) -> np.ndarray:
    frame_height, frame_width = frame_shape
    full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    full_mask[:, x0:x1] = mask
    height = max(1, int(round(frame_height * (width / frame_width))))
    resized = cv2.resize(full_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    preview[resized > 0] = (45, 220, 45)
    preview_x0 = int(round(x0 * (width / frame_width)))
    preview_x1 = int(round(x1 * (width / frame_width)))
    cv2.rectangle(preview, (preview_x0, 0), (preview_x1 - 1, height - 1), (0, 255, 255), 1)
    return preview


def make_flow_preview(
    mag: np.ndarray,
    frame_shape: tuple[int, int],
    x0: int,
    x1: int,
    width: int,
) -> np.ndarray:
    frame_height, frame_width = frame_shape
    full_mag = np.zeros((frame_height, frame_width), dtype=np.float32)
    full_mag[:, x0:x1] = mag

    height = max(1, int(round(frame_height * (width / frame_width))))
    positive_mag = mag[mag > 0]
    scale = float(np.percentile(positive_mag, 99.0)) if positive_mag.size else 0.0
    if scale <= 1e-6:
        mag_u8 = np.zeros((frame_height, frame_width), dtype=np.uint8)
    else:
        mag_u8 = np.clip((full_mag / scale) * 255.0, 0, 255).astype(np.uint8)

    mag_u8 = cv2.resize(mag_u8, (width, height), interpolation=cv2.INTER_AREA)
    signal = cv2.resize(
        (full_mag > 0).astype(np.uint8) * 255,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    preview = cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO)
    preview[signal == 0] = (0, 0, 0)
    preview_x0 = int(round(x0 * (width / frame_width)))
    preview_x1 = int(round(x1 * (width / frame_width)))
    cv2.rectangle(preview, (preview_x0, 0), (preview_x1 - 1, height - 1), (0, 255, 255), 1)
    return preview


def paste_preview(
    frame: np.ndarray,
    preview: np.ndarray,
    top_left: tuple[int, int],
    label: str,
) -> None:
    x, y = top_left
    h, w = preview.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return

    cv2.rectangle(frame, (x - 2, y - 28), (x + w + 2, y + h + 2), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    frame[y : y + h, x : x + w] = preview
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)


def overlay_visuals(
    frame: np.ndarray,
    x0: int,
    x1: int,
    mask: np.ndarray,
    flow_mag: np.ndarray,
    values: dict[str, float],
    time_s: float,
    preview_width: int,
    panel_height: int,
    text_area_width: int,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]

    tint = image.copy()
    tint[:, :x0] = (0, 0, 180)
    tint[:, x1:] = (0, 0, 180)
    image = cv2.addWeighted(tint, 0.38, image, 0.62, 0)

    cv2.rectangle(image, (x0, 0), (x1 - 1, height - 1), (0, 255, 255), 3)
    cv2.line(image, (x0, 0), (x0, height - 1), (0, 255, 255), 2)
    cv2.line(image, (x1 - 1, 0), (x1 - 1, height - 1), (0, 255, 255), 2)

    canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
    canvas[:height, :] = image
    canvas[height:, :] = (18, 18, 18)
    cv2.line(canvas, (0, height), (width - 1, height), (80, 80, 80), 1)

    lines = [
        f"Time: {time_s:7.2f} s",
        f"Seg score:    {values['seg_score']:7.3f}",
        f"Flow score:   {values['flow_score']:7.3f}",
        f"Total score:  {values['total_activity']:7.3f}",
    ]
    panel_y = height
    put_text_panel(canvas, lines, (18, panel_y + 16))

    mask_preview = make_mask_preview(mask, (height, width), x0, x1, preview_width)
    flow_preview = make_flow_preview(flow_mag, (height, width), x0, x1, preview_width)
    preview_y = panel_y + 36
    mask_x = text_area_width + 36
    flow_x = min(width - flow_preview.shape[1] - 18, mask_x + mask_preview.shape[1] + 18)
    paste_preview(canvas, mask_preview, (mask_x, preview_y), "Seg map - ROI signal")
    paste_preview(
        canvas,
        flow_preview,
        (flow_x, preview_y),
        "Flow map - ROI signal",
    )
    return canvas


def open_writer(output: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")
    return writer


def main() -> int:
    args = parse_args()
    if args.frame_step < 1:
        raise SystemExit("--frame-step must be >= 1")
    if not args.input.exists():
        raise SystemExit(f"Input video does not exist: {args.input}")

    output = args.output or args.input.with_name(f"{args.input.stem}_activity_v1.mp4")
    csv_output = args.csv or output.with_suffix(".csv")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {args.input}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    output_fps = fps / args.frame_step
    if args.start_second > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_second * 1000.0)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame from the video.")
    frame = resize_frame(frame, args.resize_width)
    height, width = frame.shape[:2]
    x0, x1 = roi_bounds(width)
    panel_height, preview_width, text_area_width = bottom_panel_layout(
        width,
        height,
        args.preview_width,
    )
    writer = open_writer(output, output_fps, (width, height + panel_height), args.codec)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.bg_history,
        varThreshold=args.bg_var_threshold,
        detectShadows=False,
    )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_output.open("w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "processed_frame",
            "source_frame",
            "time_s",
            "roi_x0",
            "roi_x1",
            "seg_method",
            "flow_mask",
            "seg_weight",
            "flow_weight",
            "segmentation_activity_pct",
            "segmentation_score",
            "optical_flow_activity",
            "optical_flow_score",
            "optical_flow_raw_activity",
            "total_activity",
        ],
    )
    csv_writer.writeheader()

    prev_gray_blur: np.ndarray | None = None
    processed_index = 0
    source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    max_end_time = args.start_second + args.duration if args.duration > 0 else None

    try:
        while True:
            if source_frame_index % args.frame_step == 0:
                roi = frame[:, x0:x1]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
                raw_flow_mag = compute_flow_magnitude(gray_blur, prev_gray_blur)

                mask, seg_activity_pct = compute_segmentation_mask(
                    roi,
                    gray_blur,
                    prev_gray_blur,
                    subtractor,
                    args,
                    processed_index,
                )
                flow_mask = mask if args.flow_mask == "segmentation" else None
                flow_mag, flow_activity, raw_flow_activity = score_flow(
                    raw_flow_mag,
                    args.flow_percentile,
                    flow_mask,
                    args.flow_min_mask_pixels,
                )

                seg_score = args.seg_weight * seg_activity_pct
                flow_score = args.flow_weight * flow_activity
                total_activity = seg_score + flow_score
                time_s = args.start_second + (processed_index * args.frame_step / fps)

                annotated = overlay_visuals(
                    frame,
                    x0,
                    x1,
                    mask,
                    flow_mag,
                    {
                        "seg_activity_pct": seg_activity_pct,
                        "seg_score": seg_score,
                        "flow_activity": flow_activity,
                        "flow_score": flow_score,
                        "raw_flow_activity": raw_flow_activity,
                        "total_activity": total_activity,
                    },
                    time_s,
                    preview_width,
                    panel_height,
                    text_area_width,
                )
                writer.write(annotated)
                csv_writer.writerow(
                    {
                        "processed_frame": processed_index,
                        "source_frame": source_frame_index,
                        "time_s": f"{time_s:.6f}",
                        "roi_x0": x0,
                        "roi_x1": x1,
                        "seg_method": args.seg_method,
                        "flow_mask": args.flow_mask,
                        "seg_weight": f"{args.seg_weight:.6f}",
                        "flow_weight": f"{args.flow_weight:.6f}",
                        "segmentation_activity_pct": f"{seg_activity_pct:.6f}",
                        "segmentation_score": f"{seg_score:.6f}",
                        "optical_flow_activity": f"{flow_activity:.6f}",
                        "optical_flow_score": f"{flow_score:.6f}",
                        "optical_flow_raw_activity": f"{raw_flow_activity:.6f}",
                        "total_activity": f"{total_activity:.6f}",
                    }
                )
                prev_gray_blur = gray_blur
                processed_index += 1

                if processed_index % 100 == 0:
                    print(
                        f"processed={processed_index} "
                        f"time={time_s:.1f}s "
                        f"seg_score={seg_score:.3f} "
                        f"flow_score={flow_score:.3f} "
                        f"total={total_activity:.1f}",
                        flush=True,
                    )

            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame(frame, args.resize_width)
            source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            current_time_s = args.start_second + max(0, source_frame_index) / fps
            if max_end_time is not None and current_time_s >= max_end_time:
                break
    finally:
        csv_file.close()
        writer.release()
        cap.release()

    print(f"Wrote annotated video: {output}")
    print(f"Wrote activity CSV: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

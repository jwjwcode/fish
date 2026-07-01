#!/usr/bin/env python3
"""Version 1 fish-feeding activity visualizer.

This module keeps the public V1 CLI stable while delegating config handling,
detection, scoring, rendering, and video IO to smaller package modules.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing startup guard
    raise SystemExit(
        "Missing dependency: "
        f"{exc.name}\nInstall dependencies with: python3 -m pip install -r requirements.txt"
    ) from exc

from fish_activity.config import apply_preset, load_config_values, provided_cli_dests
from fish_activity.decision import DecisionConfig, FeedingDecisionEngine
from fish_activity.detectors.unsupervised import (
    UnsupervisedSplashDetector,
    compute_flow_magnitude,
    create_flow_estimator,
)
from fish_activity.render import (
    bottom_panel_layout,
    overlay_visuals,
    resize_frame,
    roi_bounds,
)
from fish_activity.scoring import average_score_history, score_flow
from fish_activity.video_io import open_writer


CSV_FIELDNAMES = [
    "processed_frame",
    "source_frame",
    "time_s",
    "roi_x0",
    "roi_x1",
    "preset",
    "seg_method",
    "flow_method",
    "flow_mask",
    "artifact_filter",
    "seg_weight",
    "flow_weight",
    "segmentation_activity_pct",
    "segmentation_score",
    "optical_flow_activity",
    "optical_flow_score",
    "optical_flow_raw_activity",
    "total_activity",
    "previous_10_frame_count",
    "segmentation_score_prev10_avg",
    "optical_flow_score_prev10_avg",
    "total_activity_prev10_avg",
    "feeding_state",
    "feeding_command",
    "last_feeding_command",
    "last_feeding_command_time_s",
    "decision_window_avg",
    "decision_background_score",
    "decision_threshold",
    "decision_pause_count",
    "feeding_process_score",
    "feeding_finish_reason",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Create annotated fish-feeding activity video from one input video."
    )
    parser.add_argument("input", type=Path, help="Input video path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. Command-line flags override matching config keys.",
    )
    parser.add_argument(
        "--preset",
        choices=("current", "previous", "motion_raw"),
        default="current",
        help="Method preset. Explicit method/weight flags override this preset.",
    )
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
        choices=("anomaly", "splash", "motion"),
        default=None,
        help="Segmentation method. 'anomaly' learns normal water/ripples; 'splash' uses fixed splash rules; 'motion' is the older motion mask.",
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
        "--anomaly-learning-rate",
        type=float,
        default=0.03,
        help="Learning rate for the adaptive normal-water anomaly model.",
    )
    parser.add_argument(
        "--anomaly-color-z",
        type=float,
        default=2.4,
        help="Positive z-score threshold for abnormal bright/white pixels.",
    )
    parser.add_argument(
        "--anomaly-texture-z",
        type=float,
        default=2.0,
        help="Positive z-score threshold for abnormal texture/edge pixels.",
    )
    parser.add_argument(
        "--anomaly-flow-z",
        type=float,
        default=2.3,
        help="Positive z-score threshold for abnormal optical-flow pixels.",
    )
    parser.add_argument(
        "--anomaly-min-flow",
        type=float,
        default=1.0,
        help="Minimum optical-flow magnitude for flow anomaly support.",
    )
    parser.add_argument(
        "--artifact-filter",
        choices=("on", "off"),
        default=None,
        help="Reject persistent/smooth ripple, reflection, and bubble-like components.",
    )
    parser.add_argument(
        "--artifact-min-texture-mean",
        type=float,
        default=12.0,
        help="Components below this mean texture can be treated as smooth artifacts.",
    )
    parser.add_argument(
        "--artifact-min-edge-density",
        type=float,
        default=0.04,
        help="Components below this edge-density can be treated as smooth artifacts.",
    )
    parser.add_argument(
        "--artifact-min-flow-chaos",
        type=float,
        default=0.45,
        help="Minimum flow coefficient of variation expected from chaotic splash motion.",
    )
    parser.add_argument(
        "--artifact-persistence-frames",
        type=float,
        default=8.0,
        help="Mean per-pixel age after which static components are treated as artifacts.",
    )
    parser.add_argument(
        "--artifact-static-new-ratio",
        type=float,
        default=0.35,
        help="Maximum new-pixel ratio for persistent/static reflection or bubble artifacts.",
    )
    parser.add_argument(
        "--artifact-max-bubble-area-pct",
        type=float,
        default=0.25,
        help="Small persistent components under this ROI area percent can be treated as bubbles.",
    )
    parser.add_argument(
        "--artifact-min-reflection-area-pct",
        type=float,
        default=1.0,
        help="Large smooth components over this ROI area percent can be treated as reflections.",
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
        "--flow-method",
        choices=("auto", "dis", "farneback", "none"),
        default=None,
        help="Optical-flow method. 'none' skips flow calculation.",
    )
    parser.add_argument(
        "--flow-mask",
        choices=("segmentation", "none"),
        default=None,
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
        default=None,
        help="Weight for splash-flow energy in the total activity score.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for the output video.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=("on", "off"),
        default="on",
        help="Enable local feeding start/pause/finish decision logic.",
    )
    parser.add_argument(
        "--decision-background-frames",
        type=int,
        default=10,
        help="Processed frames used to estimate before-feeding background activity.",
    )
    parser.add_argument(
        "--decision-window-frames",
        type=int,
        default=10,
        help="Recent processed frames averaged for feeding decisions.",
    )
    parser.add_argument(
        "--decision-observe-seconds",
        type=float,
        default=45.0,
        help="Seconds after each start before low-activity pause/finish decisions.",
    )
    parser.add_argument(
        "--decision-pause-seconds",
        type=float,
        default=120.0,
        help="Seconds to wait after a pause before issuing another start command.",
    )
    parser.add_argument(
        "--decision-threshold-margin",
        type=float,
        default=0.5,
        help="Activity added to background score when forming the low-activity threshold.",
    )
    parser.add_argument(
        "--decision-threshold-multiplier",
        type=float,
        default=1.2,
        help="Background multiplier also considered when forming the low-activity threshold.",
    )
    parser.add_argument(
        "--decision-max-pauses",
        type=int,
        default=2,
        help="Allowed pause commands before the next low-activity decision finishes feeding.",
    )
    parser.add_argument(
        "--decision-machine-finish-second",
        type=float,
        default=0.0,
        help="Offline simulation hook: send one external finish signal at this timestamp. 0 disables it.",
    )
    args = parser.parse_args(argv)
    provided_dests = provided_cli_dests(parser, argv)
    config_values = load_config_values(parser, args.config)
    for dest, value in config_values.items():
        if dest not in provided_dests:
            setattr(args, dest, value)
    return apply_preset(args)


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

    detector = UnsupervisedSplashDetector(args)
    flow_estimator, flow_method = create_flow_estimator(args.flow_method)
    decision_engine = FeedingDecisionEngine(DecisionConfig.from_args(args))
    machine_finish_sent = False

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_output.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    csv_writer.writeheader()

    prev_gray_blur: np.ndarray | None = None
    score_history: deque[dict[str, float]] = deque(maxlen=10)
    processed_index = 0
    source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    max_end_time = args.start_second + args.duration if args.duration > 0 else None

    try:
        while True:
            if source_frame_index % args.frame_step == 0:
                roi = frame[:, x0:x1]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
                if flow_method == "none":
                    raw_flow_mag = np.zeros_like(gray_blur, dtype=np.float32)
                else:
                    raw_flow_mag = compute_flow_magnitude(
                        gray_blur,
                        prev_gray_blur,
                        flow_estimator,
                    )

                detector_result = detector.compute(
                    roi,
                    gray_blur,
                    prev_gray_blur,
                    raw_flow_mag,
                    processed_index,
                )
                mask = detector_result.mask
                seg_activity_pct = detector_result.activity_score

                if flow_method == "none":
                    flow_mag = raw_flow_mag
                    flow_activity = 0.0
                    raw_flow_activity = 0.0
                else:
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
                score_averages = average_score_history(score_history)
                time_s = args.start_second + (processed_index * args.frame_step / fps)
                external_command = None
                if (
                    args.decision_machine_finish_second > 0
                    and not machine_finish_sent
                    and time_s >= args.decision_machine_finish_second
                ):
                    external_command = "finish"
                    machine_finish_sent = True
                decision = decision_engine.update(
                    time_s,
                    total_activity,
                    external_command=external_command,
                )

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
                        "total_activity_prev10_avg": score_averages[
                            "total_activity_prev10_avg"
                        ],
                        "feeding_state": decision.state,
                        "feeding_command": decision.command,
                        "last_feeding_command": decision.last_command,
                        "last_feeding_command_time_s": decision.last_command_time_s,
                        "decision_window_avg": decision.window_avg,
                        "decision_threshold": decision.threshold,
                        "feeding_process_score": decision.process_score,
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
                        "preset": args.preset,
                        "seg_method": args.seg_method,
                        "flow_method": flow_method,
                        "flow_mask": args.flow_mask,
                        "artifact_filter": args.artifact_filter,
                        "seg_weight": f"{args.seg_weight:.6f}",
                        "flow_weight": f"{args.flow_weight:.6f}",
                        "segmentation_activity_pct": f"{seg_activity_pct:.6f}",
                        "segmentation_score": f"{seg_score:.6f}",
                        "optical_flow_activity": f"{flow_activity:.6f}",
                        "optical_flow_score": f"{flow_score:.6f}",
                        "optical_flow_raw_activity": f"{raw_flow_activity:.6f}",
                        "total_activity": f"{total_activity:.6f}",
                        "previous_10_frame_count": (
                            f"{int(score_averages['previous_10_frame_count'])}"
                        ),
                        "segmentation_score_prev10_avg": (
                            f"{score_averages['segmentation_score_prev10_avg']:.6f}"
                        ),
                        "optical_flow_score_prev10_avg": (
                            f"{score_averages['optical_flow_score_prev10_avg']:.6f}"
                        ),
                        "total_activity_prev10_avg": (
                            f"{score_averages['total_activity_prev10_avg']:.6f}"
                        ),
                        "feeding_state": decision.state,
                        "feeding_command": decision.command,
                        "last_feeding_command": decision.last_command,
                        "last_feeding_command_time_s": (
                            f"{decision.last_command_time_s:.6f}"
                        ),
                        "decision_window_avg": f"{decision.window_avg:.6f}",
                        "decision_background_score": (
                            f"{decision.background_score:.6f}"
                        ),
                        "decision_threshold": f"{decision.threshold:.6f}",
                        "decision_pause_count": decision.pause_count,
                        "feeding_process_score": f"{decision.process_score:.6f}",
                        "feeding_finish_reason": decision.finish_reason,
                    }
                )
                score_history.append(
                    {
                        "segmentation_score": seg_score,
                        "optical_flow_score": flow_score,
                        "total_activity": total_activity,
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

"""Debug video rendering helpers."""

from __future__ import annotations

import cv2
import numpy as np


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
    panel_height = max(270, preview_height + 74)
    return panel_height, preview_width, text_area_width


def put_text_panel(frame: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.68
    thickness = 2
    line_height = 24
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
        baseline = y + padding + 19 + i * line_height
        cv2.putText(
            frame,
            line,
            (x + padding, baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )


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


def put_command_banner(
    frame: np.ndarray,
    command: str,
    command_time_s: float,
    top_right: tuple[int, int],
) -> None:
    if command == "none":
        return

    colors = {
        "start": (35, 135, 45),
        "pause": (0, 145, 230),
        "finish": (35, 35, 210),
    }
    color = colors.get(command, (110, 110, 110))
    label = f"COMMAND: {command.upper()} @ {command_time_s:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    padding_x = 14
    padding_y = 10
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    box_w = text_w + padding_x * 2
    box_h = text_h + padding_y * 2
    x1, y = top_right
    x0 = max(0, x1 - box_w)
    cv2.rectangle(frame, (x0, y), (x1, y + box_h), color, -1)
    cv2.rectangle(frame, (x0, y), (x1, y + box_h), (255, 255, 255), 1)
    cv2.putText(
        frame,
        label,
        (x0 + padding_x, y + padding_y + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )


def overlay_visuals(
    frame: np.ndarray,
    x0: int,
    x1: int,
    mask: np.ndarray,
    flow_mag: np.ndarray,
    values: dict[str, object],
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
        f"Prev10 avg:   {values['total_activity_prev10_avg']:7.3f}",
    ]
    if "feeding_state" in values:
        last_command = str(values.get("last_feeding_command", "none"))
        last_command_time_s = float(values.get("last_feeding_command_time_s", -1.0))
        lines.extend(
            [
                f"State: {values['feeding_state']}",
                f"Cmd: {values['feeding_command']}",
                (
                    "Last cmd: "
                    f"{last_command} @ {last_command_time_s:5.1f}s"
                    if last_command_time_s >= 0
                    else "Last cmd: none"
                ),
                (
                    "Win10/Thr: "
                    f"{values['decision_window_avg']:5.2f}/"
                    f"{values['decision_threshold']:5.2f}"
                ),
                f"Feed avg:    {values['feeding_process_score']:7.3f}",
            ]
        )
    panel_y = height
    put_text_panel(canvas, lines, (18, panel_y + 16))

    mask_preview = make_mask_preview(mask, (height, width), x0, x1, preview_width)
    flow_preview = make_flow_preview(flow_mag, (height, width), x0, x1, preview_width)
    preview_y = panel_y + 64
    mask_x = text_area_width + 36
    flow_x = min(width - flow_preview.shape[1] - 18, mask_x + mask_preview.shape[1] + 18)
    paste_preview(canvas, mask_preview, (mask_x, preview_y), "Seg map - ROI signal")
    paste_preview(
        canvas,
        flow_preview,
        (flow_x, preview_y),
        "Flow map - ROI signal",
    )
    if "feeding_command" in values:
        command = str(values.get("feeding_command", "none"))
        last_command = str(values.get("last_feeding_command", "none"))
        last_command_time_s = float(values.get("last_feeding_command_time_s", -1.0))
        if command == "none" and 0 <= time_s - last_command_time_s <= 3.0:
            command = last_command
        if command != "none":
            put_command_banner(
                canvas,
                command,
                last_command_time_s,
                (width - 18, panel_y + 10),
            )
    return canvas

"""Video IO helpers."""

from __future__ import annotations

from pathlib import Path
import re

import cv2


STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def is_stream_source(source: object) -> bool:
    return str(source).lower().startswith(STREAM_PREFIXES)


def source_name(source: object) -> str:
    text = str(source).rstrip("/")
    if is_stream_source(text):
        for prefix in STREAM_PREFIXES:
            if text.lower().startswith(prefix):
                text = text[len(prefix) :]
                break
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = text.strip("_")
        return text or "camera_stream"
    return Path(text).stem


def open_writer(output: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")
    return writer

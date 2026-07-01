"""Video IO helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


def open_writer(output: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")
    return writer

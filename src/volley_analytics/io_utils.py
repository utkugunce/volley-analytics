"""Video I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def open_video(path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def iter_frames(path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) for every frame in `path`."""
    cap = open_video(path)
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_fps(path) -> float:
    cap = open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(f"Could not read FPS from {path}")
        return float(fps)
    finally:
        cap.release()


def first_frame(path) -> np.ndarray:
    cap = open_video(path)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read first frame of {path}")
        return frame
    finally:
        cap.release()

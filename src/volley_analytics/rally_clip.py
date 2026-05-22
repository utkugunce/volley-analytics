"""Rally segmentation (from ball-trajectory gaps) and ffmpeg clip cutting."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Rally:
    start_frame: int
    end_frame: int

    def to_seconds(self, fps: float) -> tuple[float, float]:
        return self.start_frame / fps, self.end_frame / fps


def segment_rallies(
    frames_with_ball,
    *,
    gap_frames: int = 45,
    min_rally_frames: int = 60,
) -> list[Rally]:
    """Group consecutive ball-detected frames into rallies.

    A gap larger than `gap_frames` ends the current rally. Rallies shorter
    than `min_rally_frames` are dropped (likely noise: stray detections,
    warm-ups, between-point ball handling).
    """
    frames = np.asarray(sorted(set(int(f) for f in frames_with_ball)), dtype=int)
    if len(frames) == 0:
        return []

    rallies: list[Rally] = []
    start = int(frames[0])
    prev = int(frames[0])
    for f in frames[1:]:
        f = int(f)
        if f - prev > gap_frames:
            rallies.append(Rally(start_frame=start, end_frame=prev))
            start = f
        prev = f
    rallies.append(Rally(start_frame=start, end_frame=prev))

    return [r for r in rallies if (r.end_frame - r.start_frame) >= min_rally_frames]


def cut_clip(video_path, out_path, start_sec: float, end_sec: float, *, pad: float = 1.0) -> Path:
    """Cut `[start_sec - pad, end_sec + pad]` from `video_path` via ffmpeg.

    Uses stream copy (`-c copy`) so the cut is near-instant and lossless.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — cannot cut rally clips.")

    start = max(0.0, start_sec - pad)
    duration = (end_sec + pad) - start
    if duration <= 0:
        raise ValueError(f"Non-positive clip duration ({duration}s) for [{start_sec}, {end_sec}]")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out

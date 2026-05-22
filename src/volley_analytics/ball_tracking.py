"""Ball tracking — STUB.

Contract for the rest of the pipeline (do not change):

    ball.csv columns: frame, x_px, y_px, conf

Anything that fills `ball.csv` correctly is acceptable. The recommended
integration is a TrackNet-style ONNX model:

    https://github.com/asigatchov/fast-volleyball-tracking-inference

Concrete next step (start here in Claude Code):

  1. `pip install -e ".[tracking]"`
  2. Download or train an ONNX ball model; put it under `models/`.
  3. Implement `BallTracker._load_model` and `BallTracker.track`.
  4. Verify by running:
         python scripts/02_track.py --video <test.mp4> --ball-model models/ball.onnx
     then inspecting `data/interim/ball.csv`.

Key conversion point: the model's per-frame output (heatmap or bbox) must
be reduced to one `BallDetection(frame, x_px, y_px, conf)` (or None) per
frame. argmax on a heatmap; bbox-center on a detector.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class BallDetection:
    frame: int
    x_px: float
    y_px: float
    conf: float


class BallTracker:
    """Per-frame ball detector. Stub — see module docstring."""

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.3):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._session = None  # filled by _load_model

    def _load_model(self) -> None:
        # TODO(claude-code): load ONNX model.
        # Example:
        #     import onnxruntime as ort
        #     providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        #     self._session = ort.InferenceSession(self.model_path, providers=providers)
        raise NotImplementedError(
            "BallTracker.track requires an ONNX model. See module docstring."
        )

    def track(self, frames: Iterable[np.ndarray]) -> Iterator[BallDetection | None]:
        """Yield one BallDetection per input frame, or None when not detected."""
        if self._session is None:
            self._load_model()
        # TODO(claude-code):
        #   for i, frame in enumerate(frames):
        #       inp = preprocess(frame)
        #       heatmap = self._session.run(...)
        #       y, x, conf = argmax_and_score(heatmap)
        #       if conf >= self.conf_threshold:
        #           yield BallDetection(frame=i, x_px=x, y_px=y, conf=conf)
        #       else:
        #           yield None
        raise NotImplementedError


# ──── CSV I/O ────────────────────────────────────────────────────────────────

BALL_CSV_HEADER = ("frame", "x_px", "y_px", "conf")


def write_ball_csv(detections: Iterable[BallDetection | None], out_path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BALL_CSV_HEADER)
        for d in detections:
            if d is None:
                continue
            w.writerow([d.frame, f"{d.x_px:.2f}", f"{d.y_px:.2f}", f"{d.conf:.3f}"])
    return out


def read_ball_csv(path) -> np.ndarray:
    """Return an (N, 4) array of [frame, x_px, y_px, conf]. Empty if no rows."""
    path = Path(path)
    rows: list[tuple[int, float, float, float]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        missing = set(BALL_CSV_HEADER) - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in r:
            rows.append((
                int(row["frame"]),
                float(row["x_px"]),
                float(row["y_px"]),
                float(row["conf"]),
            ))
    if not rows:
        return np.empty((0, 4))
    return np.array(rows, dtype=np.float64)

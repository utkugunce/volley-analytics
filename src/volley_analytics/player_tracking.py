"""Player tracking — STUB.

Contract for the rest of the pipeline (do not change):

    players.csv columns:
        frame, track_id, x_px, y_px, w_px, h_px, conf

    - (x_px, y_px) is the bbox CENTER in pixels.
    - For court projection use the FOOT point: (x_px, y_px + h_px / 2).
      The homography only behaves well for things resting on the floor.
    - track_id must be PERSISTENT — same player across frames → same id.

Recommended stack:
    - YOLOv8 (`ultralytics`) for `person` detection
    - ByteTrack (`ultralytics`'s built-in tracker) for ID persistence
    - Optional: SAM 3 prompts ("white-shirt player" vs "red-shirt player")
      when team-vs-background contrast confuses jersey colors.

Concrete next step:
    after ball tracking is wired, implement `PlayerTracker.track` using
    `ultralytics.YOLO("yolov8n.pt").track(source=frames, tracker="bytetrack.yaml")`
    and emit `PlayerDetection` per (frame, box). Filter `cls == 0` (person)
    only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class PlayerDetection:
    frame: int
    track_id: int
    x_px: float   # bbox center
    y_px: float   # bbox center
    w_px: float
    h_px: float
    conf: float

    @property
    def foot_px(self) -> tuple[float, float]:
        return self.x_px, self.y_px + self.h_px / 2.0


class PlayerTracker:
    """Per-frame player detector + tracker. Stub — see module docstring."""

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.4):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self._tracker = None

    def track(self, frames: Iterable[np.ndarray]) -> Iterator[PlayerDetection]:
        # TODO(claude-code):
        #   from ultralytics import YOLO
        #   model = YOLO(self.model_path or "yolov8n.pt")
        #   for i, frame in enumerate(frames):
        #       res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])
        #       for box in res[0].boxes:
        #           if box.conf < self.conf_threshold: continue
        #           x, y, w, h = box.xywh[0].tolist()
        #           yield PlayerDetection(
        #               frame=i, track_id=int(box.id),
        #               x_px=x, y_px=y, w_px=w, h_px=h,
        #               conf=float(box.conf),
        #           )
        raise NotImplementedError(
            "PlayerTracker.track is a stub. See module docstring."
        )


# ──── CSV I/O ────────────────────────────────────────────────────────────────

PLAYERS_CSV_HEADER = ("frame", "track_id", "x_px", "y_px", "w_px", "h_px", "conf")


def write_players_csv(detections: Iterable[PlayerDetection], out_path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PLAYERS_CSV_HEADER)
        for d in detections:
            w.writerow([
                d.frame, d.track_id,
                f"{d.x_px:.2f}", f"{d.y_px:.2f}",
                f"{d.w_px:.2f}", f"{d.h_px:.2f}",
                f"{d.conf:.3f}",
            ])
    return out


def read_players_csv(path) -> np.ndarray:
    """Return an (N, 7) array. Empty if no rows."""
    path = Path(path)
    rows: list[tuple[int, int, float, float, float, float, float]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        missing = set(PLAYERS_CSV_HEADER) - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in r:
            rows.append((
                int(row["frame"]),
                int(row["track_id"]),
                float(row["x_px"]),
                float(row["y_px"]),
                float(row["w_px"]),
                float(row["h_px"]),
                float(row["conf"]),
            ))
    if not rows:
        return np.empty((0, 7))
    return np.array(rows, dtype=np.float64)

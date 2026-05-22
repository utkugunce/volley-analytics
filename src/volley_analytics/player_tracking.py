"""Player tracking.

Public contract — DO NOT change without updating downstream consumers:

    players.csv columns:
        frame, track_id, x_px, y_px, w_px, h_px, conf

    - (x_px, y_px) is the bbox CENTER in pixels.
    - For court projection use the FOOT point: (x_px, y_px + h_px / 2).
      The homography only behaves well for things resting on the floor.
    - track_id must be PERSISTENT — same player across frames → same id.

The default `PlayerTracker` uses YOLOv8 (`ultralytics`) for `person`
detection plus the built-in ByteTrack tracker for ID persistence. Switch
the YOLO weights via `model_path` (the default downloads `yolov8n.pt`
on first use) and the tracker config via `tracker_config` (defaults to
the bundled `bytetrack.yaml`; `botsort.yaml` also ships with ultralytics).

When team-vs-background contrast confuses jersey colors, SAM 3 prompts
(`"white-shirt player"` vs `"red-shirt player"`) can replace or augment
this YOLO+ByteTrack layer — but the CSV contract stays the same.

Run with:
    pip install -e ".[tracking]"     # pulls ultralytics + lap
    python scripts/02_track.py --video <video> --skip-ball
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


# YOLO COCO class id for "person".
PERSON_CLASS_ID = 0


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
    """YOLOv8 + ByteTrack per-frame player detector with persistent IDs.

    Emits zero or more PlayerDetection per input frame. Detections whose
    tracker has not yet assigned an ID (first frame of life, before
    ByteTrack confirms) are dropped — only confirmed, persistent IDs hit
    the CSV.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.4,
        tracker_config: str = "bytetrack.yaml",
        verbose: bool = False,
    ):
        # None → ultralytics default yolov8n.pt (auto-downloaded).
        self.model_path = model_path or "yolov8n.pt"
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.verbose = verbose
        self._model = None

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ultralytics is required for player tracking. "
                "Install with: pip install -e \".[tracking]\""
            ) from exc

        self._model = YOLO(self.model_path)

    def track(self, frames: Iterable[np.ndarray]) -> Iterator[PlayerDetection]:
        if self._model is None:
            self._load_model()
        assert self._model is not None

        for i, frame in enumerate(frames):
            results = self._model.track(
                frame,
                persist=True,            # carry tracker state across calls
                tracker=self.tracker_config,
                classes=[PERSON_CLASS_ID],
                conf=self.conf_threshold,
                verbose=self.verbose,
            )
            if not results:
                continue
            boxes = results[0].boxes
            if boxes is None or boxes.id is None:
                continue  # no confirmed tracks this frame

            ids = boxes.id.int().cpu().tolist()
            xywh = boxes.xywh.cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for track_id, (cx, cy, w, h), conf in zip(ids, xywh, confs):
                yield PlayerDetection(
                    frame=i,
                    track_id=int(track_id),
                    x_px=float(cx),
                    y_px=float(cy),
                    w_px=float(w),
                    h_px=float(h),
                    conf=float(conf),
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

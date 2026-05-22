"""Ball tracking.

Public contract — DO NOT change without updating downstream consumers:

    ball.csv columns: frame, x_px, y_px, conf

The default `BallTracker` implements a TrackNet-style ONNX inference path
that mirrors the reference repository:

    https://github.com/asigatchov/fast-volleyball-tracking-inference

It assumes a model trained for VballNetV1 (or compatible), specifically
the configuration `seq9_grayscale_h288_w512`:

    - input  : (1, SEQ_LEN, H_MODEL, W_MODEL) float32 in [0, 1], grayscale
    - output : (1, SEQ_LEN, H_MODEL, W_MODEL) per-frame heatmaps

Run with:
    pip install -e ".[tracking]"     # pulls onnxruntime
    python scripts/02_track.py --video <video> --ball-model <model.onnx>

Other model shapes (different sequence length, model size, or single
output heatmap) work too as long as the input shape stays NCHW with
SEQ_LEN grayscale channels — `track` reads the actual session input shape
on load and adapts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


# Defaults tuned to VballNetV1 seq9 grayscale 288×512 — overridable per instance.
DEFAULT_SEQ_LEN = 9
DEFAULT_INPUT_SIZE = (288, 512)  # (height, width)


@dataclass
class BallDetection:
    frame: int
    x_px: float
    y_px: float
    conf: float


class BallTracker:
    """TrackNet-style per-frame ball detector backed by an ONNX model.

    Processes frames in non-overlapping windows of `seq_len` (matches the
    reference repo's batching pattern). Within each window, the model
    produces one heatmap per input frame; argmax + heatmap-value give the
    (x, y, conf) detection, scaled back to original resolution.

    Detections below `conf_threshold` are yielded as None so the caller
    can still align outputs with input frame indices.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.3,
        seq_len: int = DEFAULT_SEQ_LEN,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        providers: list[str] | None = None,
    ):
        if not model_path:
            raise ValueError("BallTracker needs a path to an ONNX ball model.")
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.seq_len = seq_len
        self.input_h, self.input_w = input_size
        self.providers = providers
        self._session = None
        self._input_name: str | None = None

    def _load_model(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "onnxruntime is required for ball tracking. "
                "Install with: pip install -e \".[tracking]\""
            ) from exc

        providers = self.providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.model_path, providers=providers)

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        # Adapt to the actual model shape when it's concrete.
        shape = inp.shape  # e.g. [1, 9, 288, 512] or with dynamic dims as strings
        if len(shape) == 4 and all(isinstance(d, int) for d in shape[1:]):
            self.seq_len = int(shape[1])
            self.input_h = int(shape[2])
            self.input_w = int(shape[3])

    def _preprocess(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """Stack `seq_len` BGR frames → (1, seq_len, H, W) float32 in [0,1]."""
        gray = [
            cv2.resize(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                (self.input_w, self.input_h),
            )
            for f in frames_bgr
        ]
        arr = np.stack(gray, axis=0).astype(np.float32) / 255.0
        return arr[np.newaxis, ...]  # add batch dim → (1, seq_len, H, W)

    def _postprocess(
        self,
        heatmaps: np.ndarray,
        base_frame: int,
        orig_size: tuple[int, int],
    ) -> list[BallDetection | None]:
        """Reduce per-frame heatmaps to one detection per frame.

        `heatmaps` has shape (1, seq_len, H_model, W_model) for full-sequence
        outputs, or (1, 1, H_model, W_model) when only the middle frame is
        predicted. `orig_size` is (H_orig, W_orig) for rescaling.
        """
        if heatmaps.ndim == 4 and heatmaps.shape[0] == 1:
            heatmaps = heatmaps[0]  # → (N_out, H, W)
        else:
            raise ValueError(f"Unexpected heatmap shape: {heatmaps.shape}")

        h_orig, w_orig = orig_size
        sx = w_orig / self.input_w
        sy = h_orig / self.input_h

        out: list[BallDetection | None] = []
        for i in range(self.seq_len):
            # Models that emit only the middle frame: map every input frame
            # in the window to that single heatmap (best we can do).
            hm = heatmaps[i] if i < heatmaps.shape[0] else heatmaps[-1]
            idx = int(hm.argmax())
            conf = float(hm.flat[idx])
            if conf < self.conf_threshold:
                out.append(None)
                continue
            y_h, x_h = np.unravel_index(idx, hm.shape)
            out.append(BallDetection(
                frame=base_frame + i,
                x_px=float(x_h) * sx,
                y_px=float(y_h) * sy,
                conf=conf,
            ))
        return out

    def track(self, frames: Iterable[np.ndarray]) -> Iterator[BallDetection | None]:
        if self._session is None:
            self._load_model()
        assert self._session is not None and self._input_name is not None

        buffer: list[np.ndarray] = []
        orig_size: tuple[int, int] | None = None
        base = 0

        for frame in frames:
            if orig_size is None:
                orig_size = (frame.shape[0], frame.shape[1])
            buffer.append(frame)
            if len(buffer) == self.seq_len:
                inp = self._preprocess(buffer)
                outputs = self._session.run(None, {self._input_name: inp})
                yield from self._postprocess(outputs[0], base, orig_size)
                base += self.seq_len
                buffer = []

        # Tail frames (< seq_len): not enough context for a model that needs
        # SEQ_LEN inputs. Yield None for each so frame indices stay aligned.
        for i in range(len(buffer)):
            yield None
        # update base for any caller introspection — not strictly needed
        # since the generator is exhausted at this point.


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

"""Unit tests for BallTracker that mock the ONNX session.

These verify the preprocess → inference → postprocess → CSV chain works
end-to-end without requiring onnxruntime or a real model file.
"""

from __future__ import annotations

import numpy as np
import pytest

from volley_analytics.ball_tracking import (
    BallDetection,
    BallTracker,
    read_ball_csv,
    write_ball_csv,
)


SEQ_LEN = 9
H_MODEL, W_MODEL = 288, 512
H_ORIG, W_ORIG = 720, 1280


class FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession."""

    def __init__(self, peaks):
        # peaks: list of (x_model, y_model, conf) per frame in window
        self._peaks = peaks
        self._input = type(
            "Inp", (), {"name": "input", "shape": [1, SEQ_LEN, H_MODEL, W_MODEL]}
        )()

    def get_inputs(self):
        return [self._input]

    def run(self, _output_names, feeds):
        arr = feeds["input"]
        assert arr.shape == (1, SEQ_LEN, H_MODEL, W_MODEL)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0 and arr.max() <= 1.0
        hm = np.zeros((1, SEQ_LEN, H_MODEL, W_MODEL), dtype=np.float32)
        for i, (xm, ym, conf) in enumerate(self._peaks):
            if conf > 0:
                hm[0, i, ym, xm] = conf
        return [hm]


def _make_tracker(peaks, conf_threshold=0.3):
    t = BallTracker(model_path="dummy.onnx", conf_threshold=conf_threshold)
    t._session = FakeSession(peaks)
    t._input_name = "input"
    return t


def _bgr_frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(H_ORIG, W_ORIG, 3), dtype=np.uint8)


def test_init_requires_model_path():
    with pytest.raises(ValueError):
        BallTracker()


def test_track_yields_one_detection_per_frame():
    peaks = [(100 + i, 50 + i, 0.9) for i in range(SEQ_LEN)]
    t = _make_tracker(peaks)
    frames = [_bgr_frame() for _ in range(SEQ_LEN)]
    out = list(t.track(frames))
    assert len(out) == SEQ_LEN
    for i, det in enumerate(out):
        assert det is not None
        assert det.frame == i
        # Coords scaled back to original resolution
        expected_x = (100 + i) * W_ORIG / W_MODEL
        expected_y = (50 + i) * H_ORIG / H_MODEL
        assert det.x_px == pytest.approx(expected_x)
        assert det.y_px == pytest.approx(expected_y)
        assert det.conf == pytest.approx(0.9)


def test_low_confidence_becomes_none():
    peaks = [(10, 10, 0.05)] * SEQ_LEN  # below default 0.3 threshold
    t = _make_tracker(peaks, conf_threshold=0.3)
    out = list(t.track([_bgr_frame() for _ in range(SEQ_LEN)]))
    assert len(out) == SEQ_LEN
    assert all(d is None for d in out)


def test_tail_frames_below_window_yield_none():
    peaks = [(50, 50, 0.9)] * SEQ_LEN
    t = _make_tracker(peaks)
    # 9 frames → one full window; 5 extra → tail → None x 5
    out = list(t.track([_bgr_frame() for _ in range(SEQ_LEN + 5)]))
    assert len(out) == SEQ_LEN + 5
    assert all(d is not None for d in out[:SEQ_LEN])
    assert all(d is None for d in out[SEQ_LEN:])


def test_two_windows_increment_frame_indices():
    peaks = [(50, 50, 0.9)] * SEQ_LEN
    t = _make_tracker(peaks)
    out = list(t.track([_bgr_frame() for _ in range(SEQ_LEN * 2)]))
    frames = [d.frame for d in out if d is not None]
    assert frames == list(range(SEQ_LEN * 2))


def test_track_then_csv_roundtrip(tmp_path):
    peaks = [(100 + i, 50 + i, 0.9) for i in range(SEQ_LEN)]
    t = _make_tracker(peaks)
    out_csv = tmp_path / "ball.csv"
    write_ball_csv(t.track([_bgr_frame() for _ in range(SEQ_LEN)]), out_csv)
    arr = read_ball_csv(out_csv)
    assert arr.shape == (SEQ_LEN, 4)
    np.testing.assert_array_equal(arr[:, 0], np.arange(SEQ_LEN))

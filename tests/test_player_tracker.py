"""Unit tests for PlayerTracker that mock the YOLO model.

These verify the YOLO+ByteTrack → CSV chain without requiring ultralytics
or any model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from volley_analytics.player_tracking import (
    PERSON_CLASS_ID,
    PlayerDetection,
    PlayerTracker,
    read_players_csv,
    write_players_csv,
)


class _FakeTensor:
    def __init__(self, data, dtype=float):
        self._data = np.asarray(data, dtype=dtype)

    def int(self):
        return _FakeTensor(self._data.astype(int), dtype=int)

    def cpu(self):
        return self

    def tolist(self):
        return self._data.tolist()


class _FakeBoxes:
    def __init__(self, ids, xywh, confs):
        self.id = None if ids is None else _FakeTensor(ids, dtype=int)
        self.xywh = _FakeTensor(xywh)
        self.conf = _FakeTensor(confs)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Returns scripted detections per `.track()` call."""

    def __init__(self, scripted_frames):
        self._frames = list(scripted_frames)
        self.calls: list[dict] = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        if not self._frames:
            return [_FakeResult(_FakeBoxes(ids=[], xywh=[], confs=[]))]
        spec = self._frames.pop(0)
        if spec is None:
            return [_FakeResult(_FakeBoxes(ids=None, xywh=[], confs=[]))]
        ids, xywh, confs = spec
        return [_FakeResult(_FakeBoxes(ids=ids, xywh=xywh, confs=confs))]


def _bgr(h=720, w=1280):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _make_tracker(scripted_frames, **kw):
    t = PlayerTracker(model_path="yolov8n.pt", **kw)
    t._model = FakeYOLO(scripted_frames)
    return t


def test_emits_one_detection_per_box():
    scripted = [(
        [1, 2],
        [[100, 200, 50, 120], [300, 400, 60, 130]],
        [0.9, 0.85],
    )]
    t = _make_tracker(scripted)
    out = list(t.track([_bgr()]))
    assert len(out) == 2
    assert {d.track_id for d in out} == {1, 2}
    d0 = next(d for d in out if d.track_id == 1)
    assert d0.frame == 0
    assert d0.x_px == 100 and d0.y_px == 200
    assert d0.w_px == 50 and d0.h_px == 120
    assert d0.conf == pytest.approx(0.9)


def test_foot_point_is_below_center():
    det = PlayerDetection(
        frame=0, track_id=1,
        x_px=500, y_px=400, w_px=60, h_px=120, conf=0.9,
    )
    fx, fy = det.foot_px
    assert fx == 500
    assert fy == 460  # 400 + 120/2


def test_unconfirmed_tracks_dropped():
    # boxes.id is None → ByteTrack hasn't confirmed any track yet
    scripted = [None]
    t = _make_tracker(scripted)
    out = list(t.track([_bgr()]))
    assert out == []


def test_track_passes_class_filter_and_persist():
    scripted = [([1], [[10, 20, 30, 40]], [0.9])]
    t = _make_tracker(scripted, conf_threshold=0.5)
    list(t.track([_bgr()]))
    call = t._model.calls[0]
    assert call["classes"] == [PERSON_CLASS_ID]
    assert call["persist"] is True
    assert call["conf"] == 0.5
    assert call["tracker"] == "bytetrack.yaml"


def test_frame_indices_increment_across_calls():
    scripted = [
        ([1], [[100, 100, 50, 100]], [0.9]),
        ([1, 2], [[110, 100, 50, 100], [200, 200, 50, 100]], [0.9, 0.8]),
    ]
    t = _make_tracker(scripted)
    out = list(t.track([_bgr(), _bgr()]))
    by_frame = {}
    for d in out:
        by_frame.setdefault(d.frame, []).append(d.track_id)
    assert by_frame == {0: [1], 1: [1, 2]}


def test_track_to_csv_roundtrip(tmp_path):
    scripted = [
        ([1, 2],
         [[100, 200, 50, 120], [300, 400, 60, 130]],
         [0.9, 0.85]),
    ]
    t = _make_tracker(scripted)
    out_csv = tmp_path / "players.csv"
    write_players_csv(t.track([_bgr()]), out_csv)
    arr = read_players_csv(out_csv)
    assert arr.shape == (2, 7)
    # Sorted by file order, which matches yield order.
    np.testing.assert_allclose(arr[0], [0, 1, 100, 200, 50, 120, 0.9])
    np.testing.assert_allclose(arr[1], [0, 2, 300, 400, 60, 130, 0.85])

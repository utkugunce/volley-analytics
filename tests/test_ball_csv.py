import numpy as np

from volley_analytics.ball_tracking import BallDetection, read_ball_csv, write_ball_csv


def test_write_then_read_roundtrip(tmp_path):
    dets = [
        BallDetection(frame=0, x_px=100.5, y_px=200.25, conf=0.9),
        None,  # missed frame — should be skipped
        BallDetection(frame=2, x_px=110.0, y_px=205.0, conf=0.8),
    ]
    out = tmp_path / "ball.csv"
    write_ball_csv(dets, out)
    arr = read_ball_csv(out)
    assert arr.shape == (2, 4)
    np.testing.assert_allclose(arr[0], [0, 100.5, 200.25, 0.9])
    np.testing.assert_allclose(arr[1], [2, 110.0, 205.0, 0.8])


def test_read_empty_csv(tmp_path):
    out = tmp_path / "ball.csv"
    write_ball_csv([], out)
    arr = read_ball_csv(out)
    assert arr.shape == (0, 4)

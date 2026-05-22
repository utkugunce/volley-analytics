import json

import numpy as np
import pytest

from volley_analytics.calibration import (
    COURT_H,
    COURT_W,
    Calibration,
    compute_calibration,
    default_court_corners,
)


IMAGE_CORNERS = np.array(
    [[100, 200], [1820, 250], [1900, 1000], [50, 1050]],
    dtype=float,
)


def test_corners_map_to_court_corners():
    calib = compute_calibration(IMAGE_CORNERS)
    court = calib.pixel_to_court(IMAGE_CORNERS)
    expected = default_court_corners()
    np.testing.assert_allclose(court, expected, atol=1e-6)


def test_pixel_court_roundtrip():
    calib = compute_calibration(IMAGE_CORNERS)
    sample_px = np.array([[500.0, 600.0], [1200.0, 800.0]])
    court = calib.pixel_to_court(sample_px)
    back = calib.court_to_pixel(court)
    np.testing.assert_allclose(back, sample_px, atol=1e-3)


def test_court_center_lands_on_image_interior():
    calib = compute_calibration(IMAGE_CORNERS)
    center = calib.court_to_pixel(np.array([[COURT_W / 2, COURT_H / 2]]))
    x, y = center[0]
    # The four pixel corners enclose a region; the center should be inside.
    assert 50 < x < 1900
    assert 200 < y < 1050


def test_save_and_load_roundtrip(tmp_path):
    calib = compute_calibration(IMAGE_CORNERS)
    out = tmp_path / "calib.json"
    calib.save(out)

    data = json.loads(out.read_text())
    assert set(data) == {"H", "image_corners", "court_corners"}

    loaded = Calibration.load(out)
    np.testing.assert_allclose(loaded.H, calib.H)
    np.testing.assert_allclose(loaded.image_corners, calib.image_corners)
    np.testing.assert_allclose(loaded.court_corners, calib.court_corners)


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        compute_calibration(np.zeros((3, 2)))

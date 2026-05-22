import numpy as np
import pytest

from volley_analytics.heatmap import render_heatmap


def test_renders_to_png(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(low=[1, 1], high=[8, 17], size=(500, 2))
    out = tmp_path / "heatmap.png"
    render_heatmap(pts, out)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_empty_input_raises(tmp_path):
    with pytest.raises(ValueError):
        render_heatmap(np.empty((0, 2)), tmp_path / "x.png")


def test_wrong_shape_raises(tmp_path):
    with pytest.raises(ValueError):
        render_heatmap(np.zeros((5, 3)), tmp_path / "x.png")

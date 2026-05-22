"""Render heatmaps from (x, y) court coordinates in meters."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe on headless servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from .calibration import COURT_H, COURT_W


def render_heatmap(
    court_xy,
    out_path,
    *,
    bins: tuple[int, int] = (36, 72),
    sigma: float = 1.5,
    title: str | None = None,
    cmap: str = "hot",
) -> Path:
    """Save a heatmap PNG of `court_xy` (in meters) to `out_path`.

    `bins` is (x_bins, y_bins). Default 36×72 = 0.25 m resolution.
    `sigma` smooths the histogram (in bin units).
    """
    xy = np.asarray(court_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"court_xy must be (N, 2), got {xy.shape}")
    if xy.size == 0:
        raise ValueError("court_xy is empty — nothing to render.")

    hist, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=bins,
        range=[[0.0, COURT_W], [0.0, COURT_H]],
    )
    hist = gaussian_filter(hist, sigma=sigma)

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(
        hist.T,
        origin="lower",
        extent=(0.0, COURT_W, 0.0, COURT_H),
        cmap=cmap,
        aspect="equal",
    )
    # net at y = 9, attack lines at y = 6 and 12
    ax.axhline(COURT_H / 2, color="white", lw=1.5, ls="--")
    ax.axhline(COURT_H / 2 - 3, color="white", lw=0.6, ls=":")
    ax.axhline(COURT_H / 2 + 3, color="white", lw=0.6, ls=":")
    ax.set_xlim(0.0, COURT_W)
    ax.set_ylim(0.0, COURT_H)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

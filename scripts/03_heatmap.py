#!/usr/bin/env python3
"""Render a ball heatmap from ball.csv + calibration.json.

Projects pixel detections into court coordinates via the calibration
homography, drops outliers outside the court, then writes a smoothed
2D histogram as a PNG.
"""

import argparse
from pathlib import Path

from volley_analytics.ball_tracking import read_ball_csv
from volley_analytics.calibration import COURT_H, COURT_W, Calibration
from volley_analytics.heatmap import render_heatmap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ball", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="Ball heatmap")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Drop detections below this confidence.")
    ap.add_argument("--sigma", type=float, default=1.5)
    args = ap.parse_args()

    arr = read_ball_csv(args.ball)
    if arr.size == 0:
        raise SystemExit(f"No detections in {args.ball}")

    if args.min_conf > 0:
        arr = arr[arr[:, 3] >= args.min_conf]
        if arr.size == 0:
            raise SystemExit(f"No detections survive --min-conf {args.min_conf}")

    calib = Calibration.load(args.calibration)
    pts_court = calib.pixel_to_court(arr[:, 1:3])
    mask = (
        (pts_court[:, 0] >= 0) & (pts_court[:, 0] <= COURT_W)
        & (pts_court[:, 1] >= 0) & (pts_court[:, 1] <= COURT_H)
    )
    kept = pts_court[mask]
    dropped = len(pts_court) - len(kept)
    if dropped:
        print(f"  dropped {dropped}/{len(pts_court)} detections outside court bounds")
    if kept.size == 0:
        raise SystemExit("All detections fell outside the court — check calibration.")

    render_heatmap(kept, args.out, sigma=args.sigma, title=args.title)
    print(f"Saved heatmap → {args.out}")


if __name__ == "__main__":
    main()

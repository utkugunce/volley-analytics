#!/usr/bin/env python3
"""Calibrate court from a video frame by clicking 4 corners (TL, TR, BR, BL).

Opens an OpenCV window — run on a desktop session, not on a headless server.
"""

import argparse
from pathlib import Path

from volley_analytics.calibration import click_corners, compute_calibration
from volley_analytics.io_utils import first_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    frame = first_frame(args.video)
    print("Click 4 court corners in order: TL, TR, BR, BL (ESC to abort).")
    corners = click_corners(frame)
    calib = compute_calibration(corners)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calib.save(args.out)
    print(f"Saved calibration → {args.out}")


if __name__ == "__main__":
    main()

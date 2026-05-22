#!/usr/bin/env python3
"""Segment rallies from ball.csv and cut each as a separate mp4.

Uses ffmpeg `-c copy` (stream copy) so cuts are near-instant and lossless.
Requires ffmpeg on PATH.
"""

import argparse
from pathlib import Path

from volley_analytics.ball_tracking import read_ball_csv
from volley_analytics.io_utils import video_fps
from volley_analytics.rally_clip import cut_clip, segment_rallies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--ball", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--gap-frames", type=int, default=45,
                    help="A gap larger than this ends the current rally.")
    ap.add_argument("--min-rally-frames", type=int, default=60,
                    help="Drop rallies shorter than this.")
    ap.add_argument("--pad", type=float, default=1.0,
                    help="Seconds of pre/post padding on each clip.")
    args = ap.parse_args()

    arr = read_ball_csv(args.ball)
    if arr.size == 0:
        raise SystemExit(f"No detections in {args.ball}")

    rallies = segment_rallies(
        arr[:, 0].astype(int),
        gap_frames=args.gap_frames,
        min_rally_frames=args.min_rally_frames,
    )
    if not rallies:
        raise SystemExit("No rallies survived the min-rally-frames filter.")

    fps = video_fps(args.video)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rallies, start=1):
        s, e = r.to_seconds(fps)
        out = args.out_dir / f"rally_{i:03d}.mp4"
        cut_clip(args.video, out, s, e, pad=args.pad)
        print(f"  rally {i:03d}: {s:7.2f}s → {e:7.2f}s  ({out.name})")

    print(f"Saved {len(rallies)} rally clips → {args.out_dir}")


if __name__ == "__main__":
    main()

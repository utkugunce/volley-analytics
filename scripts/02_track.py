#!/usr/bin/env python3
"""Run ball + player tracking on a video and write the two CSV contracts.

Both trackers are STUBS today — wiring them is the next concrete task.
See `src/volley_analytics/ball_tracking.py` and `player_tracking.py`.

Outputs (defaults under `data/interim/`):
    ball.csv     — frame, x_px, y_px, conf
    players.csv  — frame, track_id, x_px, y_px, w_px, h_px, conf
"""

import argparse
from pathlib import Path

from volley_analytics.ball_tracking import BallTracker, write_ball_csv
from volley_analytics.io_utils import iter_frames
from volley_analytics.player_tracking import PlayerTracker, write_players_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--ball-out", type=Path, default=Path("data/interim/ball.csv"))
    ap.add_argument("--players-out", type=Path, default=Path("data/interim/players.csv"))
    ap.add_argument("--ball-model", type=Path, help="Path to ball ONNX model.")
    ap.add_argument("--player-model", type=Path, help="Path to player detector (YOLO).")
    ap.add_argument("--skip-ball", action="store_true")
    ap.add_argument("--skip-players", action="store_true")
    args = ap.parse_args()

    if args.skip_ball and args.skip_players:
        raise SystemExit("Nothing to do: both --skip-ball and --skip-players given.")

    if not args.skip_ball:
        ball = BallTracker(model_path=str(args.ball_model) if args.ball_model else None)
        write_ball_csv(
            ball.track(f for _, f in iter_frames(args.video)),
            args.ball_out,
        )
        print(f"Saved ball detections → {args.ball_out}")

    if not args.skip_players:
        players = PlayerTracker(model_path=str(args.player_model) if args.player_model else None)
        write_players_csv(
            players.track(f for _, f in iter_frames(args.video)),
            args.players_out,
        )
        print(f"Saved player detections → {args.players_out}")


if __name__ == "__main__":
    main()

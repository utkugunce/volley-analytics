# volley-analytics — Phase 1

Computer-vision pipeline for amateur volleyball match video. This repo is
the **vision layer** of an open-ended volleyball analytics project.

Phase 1 scope: take a single-camera match video and produce
1. ball trajectory + heatmap on real court coordinates,
2. rally clips (one mp4 per rally),
3. raw player tracks (foundation for later phases).

Higher-level statistics (kill %, reception quality, rotation) are
explicitly **out of scope** for Phase 1. They live in later phases that
consume the CSV contracts this phase produces.

## Architecture

```
video ──► [01] calibrate (4-click homography) ─┐
        ├► [02] ball tracker  (ONNX)            ├─► ball.csv  ───┐
        └► [02] player tracker (YOLO+ByteTrack) ├─► players.csv  │
                                                                 │
ball.csv + calibration.json ──► [03] heatmap.png                 │
ball.csv + video           ──► [04] rally clips (mp4)             │
players.csv + calibration ──► (Phase 2: positional analytics) ◄───┘
```

## Layout

- `src/volley_analytics/`
  - `calibration.py` — **working**. 4-corner click → `cv2.findHomography` → pixel↔court (m).
  - `heatmap.py` — **working**. 2D histogram on court coordinates + gaussian smoothing.
  - `rally_clip.py` — **working**. Gap-based rally segmentation + `ffmpeg -c copy` cuts.
  - `ball_tracking.py` — **STUB**. Contract: writes `ball.csv` (frame, x_px, y_px, conf).
  - `player_tracking.py` — **STUB**. Contract: writes `players.csv` (frame, track_id, x_px, y_px, w_px, h_px, conf).
  - `io_utils.py` — frame iteration, fps lookup.
- `scripts/` — thin CLIs (`01_calibrate.py`, `02_track.py`, `03_heatmap.py`, `04_rally_clips.py`).
- `tests/` — unit tests for the working modules.
- `data/{raw,interim,out}/` — gitignored runtime data dirs.

## Data contracts (do NOT change without updating consumers)

`data/interim/ball.csv`
```
frame, x_px, y_px, conf
```
- `frame`: int, 0-based frame index
- `x_px`, `y_px`: float, pixel coords of ball center
- `conf`: float in [0, 1]

`data/interim/players.csv`
```
frame, track_id, x_px, y_px, w_px, h_px, conf
```
- `x_px`, `y_px`: bbox **center** in pixels.
  Use the foot point (`y_px + h_px/2`) when projecting to court coords —
  feet sit on the floor, the homography only behaves well for things on the floor.
- `track_id`: int, must be **persistent** across frames (same player → same id).

`data/interim/calibration.json`
```
{ "H": [[..3x3..]], "image_corners": [[..4x2..]], "court_corners": [[..4x2..]] }
```
- `H` is the pixel→court (meters) homography. Inverse for the other direction.
- Standard FIVB court is 9 × 18 m; `court_corners` defaults to that.
- Corner order is **TL, TR, BR, BL** (top-left first, clockwise).

## Next step — start here in Claude Code

Wire `BallTracker.track()` in `src/volley_analytics/ball_tracking.py` to a
TrackNet-style ONNX model. Reference:
https://github.com/asigatchov/fast-volleyball-tracking-inference

The critical glue is mapping that model's per-frame output to the
`ball.csv` contract `(frame, x_px, y_px, conf)`. Everything downstream
depends only on that contract — once it fills correctly, the chain
`02_track → 03_heatmap → 04_rally_clips` produces the first heatmap and
rally clips end-to-end on a test video.

After ball tracking is wired, the next concrete task is player tracking
in `player_tracking.py` — YOLOv8 (`ultralytics`) for `person` detection
plus ByteTrack for ID persistence. SAM 3 prompts (`"white-shirt player"`
vs `"red-shirt player"`) become useful if and only if jersey-vs-background
contrast causes team confusion.

## Why this shape

- **Trackers are isolated behind a CSV contract.** Any detector (TrackNet,
  YOLO-ball, SAM 3 prompts) can be swapped in without touching
  heatmap/rally code.
- **Calibration is decoupled from tracking.** Re-render heatmaps for a
  different court without re-tracking.
- **`ffmpeg -c copy`** for rally cuts: near-instant, lossless, no re-encode.
- **Heavy ML deps are optional** (`pip install -e ".[tracking]"`). Core
  modules import fine without `onnxruntime` or `ultralytics`, so tests
  and analysis on existing CSVs don't pull GPU stacks.

## Reference projects

- https://github.com/masouduut94/volleyball_analytics — Phase 4 backbone (action classification).
- https://github.com/shukkkur/VolleyVision — 7-class volleyball action dataset.
- https://github.com/openvolley/ovml — alternative court↔world projection (R package).
- https://github.com/asigatchov/fast-volleyball-tracking-inference — ball ONNX, Phase 1 target.
- https://github.com/facebookresearch/sam3 — optional: prompted segmentation for team split.

## Phase roadmap (context only — NOT in scope for this repo)

1. **Phase 1 (this repo)** — ball + player tracking, calibration, heatmap, rally clips.
2. **Phase 2** — positional analytics (formation, side preference) from `players.csv`.
3. **Phase 3** — action classification (serve/spike/block) on player tracks.
4. **Phase 4** — rally-level KPIs (rally length, attack side, ace detection).
5. **Phase 5+** — rotation / touch attribution. Research-grade, human-in-the-loop.

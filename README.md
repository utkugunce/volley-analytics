# volley-analytics

Computer-vision pipeline for amateur volleyball match video.
**Phase 1**: court calibration → ball / player tracking → heatmap + rally clips.

See [`CLAUDE.md`](./CLAUDE.md) for the full architecture, data contracts,
and what's implemented vs. stubbed.

## Install

```bash
pip install -e .
# Optional: ONNX + YOLO dependencies for the trackers
pip install -e ".[tracking]"
```

Requires `ffmpeg` on PATH (used by the rally clipper).

## End-to-end (once trackers are wired)

```bash
# 1. Pick 4 court corners (TL, TR, BR, BL) by clicking — opens an OpenCV window.
python scripts/01_calibrate.py \
    --video data/raw/match.mp4 \
    --out   data/interim/calibration.json

# 2. Run ball + player tracking, writing CSVs.
python scripts/02_track.py \
    --video data/raw/match.mp4 \
    --ball-model models/ball.onnx

# 3. Render heatmap from ball.csv + calibration.
python scripts/03_heatmap.py \
    --ball        data/interim/ball.csv \
    --calibration data/interim/calibration.json \
    --out         data/out/ball_heatmap.png

# 4. Cut one mp4 per rally.
python scripts/04_rally_clips.py \
    --video   data/raw/match.mp4 \
    --ball    data/interim/ball.csv \
    --out-dir data/out/rallies
```

## Status

| Module              | Status        |
|---------------------|---------------|
| court calibration   | working       |
| heatmap rendering   | working       |
| rally segmentation  | working       |
| ball tracking       | **stub (TODO)** |
| player tracking     | **stub (TODO)** |

## Tests

```bash
pip install pytest
pytest
```

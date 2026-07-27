# Offline bunny future-position evaluation

This workflow needs no robot. Create JSONL track observations from saved video,
YOLO replay, or depth replay. Every line has:

```json
{"timestamp_s": 12.34, "track_id": 1, "center_xy": [320.0, 180.0], "depth_m": 0.82, "confidence": 0.91}
```

Then evaluate the 100 ms, 200 ms, and 500 ms prediction horizons:

```bash
uv run g1 tune future-eval tracks.jsonl
```

The report is written under `runs/offline/future_prediction/`. Prioritize the
200 ms median/p95 pixel error and, when depth exists, the 3D median/p95 error.
Low-confidence observations are excluded before velocity estimation.

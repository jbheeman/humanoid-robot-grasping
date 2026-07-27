# Tracking Start

The first tracking target is a manual bounding-box tracker. This lets us test the camera/logging/tracking loop before adding object detection or GR00T.

## Install

```bash
uv sync
```

If OpenCV reports that CSRT/KCF tracking is unavailable, install the contrib build:

```bash
uv add opencv-contrib-python
```

## Run With Webcam

```bash
uv run python scripts/local/manual.py --camera 0 --output runs/object_manual_test
```

Steps:

1. A window opens with the first camera frame.
2. Drag a box around the plush object.
3. Press Enter or Space.
4. Move the object slowly.
5. Press `q` or Escape to stop.

The script writes:

```text
runs/object_manual_test/tracking.jsonl
```

Each row contains:

- `frame_id`
- `timestamp`
- `elapsed_s`
- `track_ok`
- `bbox`

## Next Step

After manual tracking works, add an automatic detector for the target plush object and use it to initialize or refresh the tracker.

## Unitree G1 Integration Note

The tracker accepts normal OpenCV sources. For G1, pass `--camera g1` and use the configured RTP/GStreamer camera source. Robot control and depth remain on ROS 2; see `docs/ARCHITECTURE.md`.

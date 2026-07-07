# Tracking Start

The first tracking target is a manual bounding-box tracker. This lets us test the camera/logging/tracking loop before adding object detection or GR00T.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If OpenCV reports that CSRT/KCF tracking is unavailable, install the contrib build:

```bash
pip install opencv-contrib-python
```

## Run With Webcam

```bash
python scripts/run_manual_tracker.py --camera 0 --output runs/object_manual_test
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

The current tracker accepts a normal OpenCV camera/video source. For the first test, that is intentional. When G1 access is available, add a camera adapter around `unitree_sdk2` or the available Unitree bindings instead of rewriting the tracker.

See `docs/UNITREE_SDK2_NOTES.md`.

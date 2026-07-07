# Humanoid Robot Grasping

## Quick Start

```bash
git switch aarav
uv sync
uv run python scripts/run_manual_tracker.py --camera 0 --output runs/object_manual_test
```

In the first video frame, drag a box around the moving plush object and press Enter. Press `q` or Escape to stop.

The run writes:

- `runs/object_manual_test/tracking.mp4`
- `runs/object_manual_test/tracking.jsonl`

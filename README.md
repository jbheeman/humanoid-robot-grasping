# Humanoid Robot Grasping

Initial code for the Unitree G1 moving plush object project.

The first milestone is not grasping or stopping. It is tracking:

> Draw or record a stable bounding box around a plush object while it moves slowly.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_manual_tracker.py --camera 0 --output runs/object_manual_test
```

See `docs/TRACKING_START.md` for details.

For future Unitree G1 integration, see `docs/UNITREE_SDK2_NOTES.md`.

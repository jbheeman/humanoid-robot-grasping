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

## Unitree G1 SDK2 Commands

This project uses the C++ SDK2 G1 loco example through a small Python wrapper. Build the SDK helper first:

```bash
cd ~/Documents/unitree_sdk2
mkdir -p build
cd build
cmake ..
make g1_loco_client
```

Send a one-shot G1 command, replacing `enp3s0` with the network interface connected to the robot:

```bash
uv run g1-loco --network-interface enp3s0 get_fsm_id
uv run g1-loco --network-interface enp3s0 stand_up
uv run g1-loco --network-interface enp3s0 move --velocity "0.2 0 0 1.0"
uv run g1-loco --network-interface enp3s0 stop_move
```

The tracker can also send optional one-shot G1 commands after tracker initialization:

```bash
uv run python scripts/run_manual_tracker.py \
  --camera 0 \
  --output runs/object_manual_test \
  --unitree-network-interface enp3s0 \
  --g1-command-on-start get_fsm_id \
  --g1-stop-on-exit
```

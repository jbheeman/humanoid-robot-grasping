# Unitree SDK2 Notes

`unitree_sdk2` is integrated through the SDK's G1 C++ loco example binary because this checkout does not include Python bindings.

## SDK Location

The wrapper defaults to:

```text
~/Documents/unitree_sdk2
```

It looks for a built G1 loco binary at:

```text
~/Documents/unitree_sdk2/build/bin/g1_loco_client
~/Documents/unitree_sdk2/build/g1_loco_client
```

Build it with:

```bash
cd ~/Documents/unitree_sdk2
mkdir -p build
cd build
cmake ..
make g1_loco_client
```

## Command Path

Python code in `object_tracking.unitree_g1` shells out to `g1_loco_client`, which uses:

```cpp
unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
unitree::robot::g1::LocoClient client;
client.Init();
```

The underlying SDK service is G1 `sport`, defined by `unitree::robot::g1::LOCO_SERVICE_NAME`.

## Standalone Commands

```bash
uv run g1-loco --network-interface enp3s0 get_fsm_id
uv run g1-loco --network-interface enp3s0 start
uv run g1-loco --network-interface enp3s0 stand_up
uv run g1-loco --network-interface enp3s0 balance_stand
uv run g1-loco --network-interface enp3s0 move --velocity "0.2 0 0 1.0"
uv run g1-loco --network-interface enp3s0 stop_move
uv run g1-loco --network-interface enp3s0 damp
```

## Tracker Integration

The manual tracker remains camera-first. SDK2 robot commands are opt-in:

```bash
uv run python scripts/run_manual_tracker.py \
  --camera 0 \
  --output runs/object_manual_test \
  --unitree-network-interface enp3s0 \
  --g1-command-on-start get_fsm_id \
  --g1-stop-on-exit
```

Available tracker startup commands:

```text
none, get_fsm_id, start, stand_up, balance_stand, stop_move, damp
```

Optional startup velocity:

```bash
--g1-velocity-on-start "VX VY OMEGA [DURATION]"
```

Example:

```bash
--g1-velocity-on-start "0.1 0 0 1.0"
```

## Vision Status

This repository still uses OpenCV camera/video sources for perception. The inspected SDK2 checkout does not contain Python files or a G1-specific vision/video client. The only video clients present there are for other Unitree models, so G1 vision should be handled through a separate G1 camera source, ROS bridge, or camera driver when available.

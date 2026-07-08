# Humanoid Robot Grasping

## Setup

From the project root:

```bash
git switch aarav
uv sync
```

This installs vision plus the Unitree SDK2 Python runtime. `uv sync --extra loco` is still accepted for the old workflow, but the SDK is now a normal dependency so plain `uv sync` will not prune it:

```bash
uv sync --extra loco
```

Loco commands are run from the server that is connected to the robot network, not necessarily on the robot itself. Pass the robot LAN IP to `uv run loco`. The local editable `unitree-sdk2py==1.0.1` checkout must exist on that server at `../unitree_sdk2_python` relative to this project. For example, if the server checkout is `/home/neel/humanoid-robot-grasping`, uv expects the SDK at `/home/neel/unitree_sdk2_python`. The robot SSH target (for example `unitree@ubuntu`) is separate from this local Python dependency path.

```bash
uv sync --extra loco
uv run loco <robot_lan_ip> stop_move
```

## Vision runbook

1) Make sure you know the robot LAN IP.
2) Use these commands from the project root on the machine that is connected to the robot network:

Headless snapshot (fastest smoke test, no display needed):

```bash
uv run vision 192.168.0.4
```

SSH-friendly form (vision can parse target args directly):

```bash
uv run vision 192.168.0.4 unitree@192.168.0.100
uv run vision 192.168.0.4 unitree 192.168.0.100
```

If reachable, this opens one frame and saves:
- `runs/vision_test/snapshot.jpg`
- `runs/vision_test/camera.json`

Run tracker (headless, scripted target box):

```bash
uv run vision 192.168.0.4 --bbox "320 180 200 200" --max-frames 300
```

If you are on a machine without GUI, `--bbox` is required.  
If you have a display, you can omit `--bbox` and add `--show` for manual ROI selection.

Debug first if it fails:

```bash
uv run vision 192.168.0.4 --list-cameras
uv run vision 192.168.0.4 --diagnose
```

Commonly useful options:

```bash
uv run vision 192.168.0.4 --camera-url "udpsrc address={ip} port=5600 ! application/x-rtp,media=(string)video ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
uv run vision 192.168.0.4 --no-ssh
uv run vision 192.168.0.4 --show
```

Notes:
- G1 default transport is GStreamer/UDP (`gstreamer` mode), not RTSP.
- `--no-ssh` disables SSH tunnel probing (safe default for GStreamer/UDP).
- If using RTSP URLs manually, use `--ssh-host`/`--ssh-user` if port forwarding is needed.
- `--diagnose` writes probe JSON including network + camera candidate status.

Example local webcam path (non-Robot testing):

```bash
uv run python scripts/run_manual_tracker.py --camera 0 --output runs/object_manual_test
```

## Unitree SDK2 loco

Run commands with robot IP:

```bash
uv run loco 192.168.0.4 stand_up
uv run loco 192.168.0.4 balance_stand
uv run loco 192.168.0.4 move --velocity "0.2 0 0 1.0"
uv run loco 192.168.0.4 stop_move
uv run loco 192.168.0.212 stop_move --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco --diagnose 192.168.0.4
```

For G1 loco, `--loco-service-name auto` is the default and patches the SDK to use `ai_sport`. Use `--loco-service-name sport` only when explicitly testing older firmware/service behavior.
For DDS init, `--dds-config-mode no_trace` is the default. The Unitree SDK's original `unitree` config mode is kept for debugging, but it has been observed to crash with `SIGABRT` during `ChannelFactoryInitialize` on the G1 test machine.

By default, loco resolves the local DDS interface with `ip route get <robot_ip>`. If DDS picks the wrong NIC, pass it explicitly:

```bash
uv run loco 192.168.0.4 stop_move --interface eno1
uv run loco 192.168.0.4 stop_move --network-interface eno1
```

`--interface auto` is the default. Use `uv run loco --diagnose <robot_ip>` to print the resolved interface, Python paths, Unitree SDK path/version, CycloneDDS path/version, and relevant DDS environment variables.

If `LocoClient` fails during DDS topic creation, probe the specific DDS topics without sending robot commands:

```bash
uv run loco --dds-probe 192.168.0.4
uv run loco --dds-probe 192.168.0.4 --interface eno1 --loco-service-name ai_sport
```

### Unitree DDS troubleshooting

If `LocoClient` fails while creating a CycloneDDS topic, first run the project-free minimal constructor test:

```bash
uv run python scripts/unitree_loco_minimal.py --robot-ip 192.168.0.4 --loco-service-name ai_sport
uv run python scripts/unitree_loco_minimal.py --interface eno1 --loco-service-name ai_sport
```

You can also run the same smoke test through the main CLI without sending movement commands:

```bash
uv run loco --smoke-loco 192.168.0.4 --interface eno1 --loco-service-name ai_sport
uv run loco --smoke-loco-subprocess 192.168.0.4 --interface eno1 --loco-service-name ai_sport
uv run loco --smoke-loco-config-sweep 192.168.0.4 --interface eno1 --loco-service-name ai_sport
```

Prefer `--smoke-loco-subprocess` when debugging native crashes such as `*** buffer overflow detected ***`; it reports whether the child process exited normally or was killed by a native signal.
Use `--smoke-loco-config-sweep` when the subprocess dies during `ChannelFactoryInitialize`. It tests `unitree`, `no_trace`, `simple`, and `autodetermine` DDS config modes in separate child processes.

The Unitree SDK config can try to write CycloneDDS tracing to `/tmp/cdds.LOG`. This CLI patches that path to `/tmp/unitree_cdds_<uid>_<pid>.log` before DDS initialization. To choose a specific writable file:

```bash
uv run loco --smoke-loco-subprocess 192.168.0.4 --interface eno1 --loco-service-name ai_sport --cyclonedds-log-file /tmp/ucdds.log
```

If one config mode passes, use it for later commands:

```bash
uv run loco 192.168.0.212 stop_move --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
```

If every config mode exits with `SIGABRT` before `Constructing G1 LocoClient`, the failure is in CycloneDDS domain initialization, not in the repo movement wrapper or G1 service name patch. Rebuild or reinstall the native CycloneDDS and `unitree_sdk2py` stack before trying robot movement again.

If the minimal script fails, the problem is below this project: fix the Unitree SDK checkout, CycloneDDS/Python environment, DDS domain/interface, or robot firmware/SDK compatibility. Inspect `sys.path`, remove duplicate SDK installs, and reinstall exactly one `unitree_sdk2py` source cleanly.

If the minimal script passes but `uv run loco ...` fails, look for mixed-client imports or fallback construction in this repo. The default backend is `g1_loco`; `go2_sport` is only constructed when explicitly requested:

```bash
uv run loco 192.168.0.4 stop_move --backend g1_loco
uv run loco 192.168.0.4 --backend g1_loco_minimal
```

During a normal `stop_move`, stderr should contain exactly one Unitree client construction line: `Constructing G1 LocoClient`.

If `stop_move` reaches `[ClientStub] send request error`, DDS and `LocoClient` construction have already succeeded. Use the read-only probe before sending more movement commands:

```bash
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name sport --dds-config-mode no_trace
```

If both probes return `3102` (`Request sending error`) on read-only methods, put the robot into high-level sport/ai-sport mode with the controller and retry. At that point the failure is the robot RPC server not responding on `rt/api/<service>/request`, not DDS initialization. The next useful checks are whether the robot firmware exposes the high-level loco RPC service at all and whether motion mode is enabled on the robot side.

If MotionSwitcher also returns `3102`, first classify the server-to-robot path from the machine that will send commands:

```bash
PYTHONPATH=$PWD/src python3 scripts/robot_eth0_rpc_probe.py \
  --interfaces enP7s7,wlan0 \
  --ping-ip 192.168.0.212 \
  --domain-id 0 \
  --timeout 15.0
```

The probe script sets `PYTHONPATH=./src` for child commands, checks `object_tracking` imports before DDS, and reports per-interface classifications. `--ping-ip` is only a network diagnostic; Unitree DDS discovery uses `--interfaces`, `--domain-id`, and the SDK service names.

Direct server-side read-only checks:

```bash
PYTHONPATH=$PWD/src python3 scripts/g1_loco.py \
  --interface enP7s7 \
  --domain-id 0 \
  --timeout 15.0 \
  --dds-config-mode no_trace \
  diagnose \
  --loco-service-name sport

PYTHONPATH=$PWD/src python3 scripts/g1_loco.py \
  --interface wlan0 \
  --domain-id 0 \
  --timeout 15.0 \
  --dds-config-mode no_trace \
  probe_loco \
  --loco-service-name sport
```

Diagnosis rules:
- Import check fails: repo packaging/import issue; DDS was not tested.
- One interface returns SDK RPC code `0`: use that interface for Unitree DDS.
- `3102`: request sending/network/interface issue.
- `3103`: API not registered or wrong service name; try `sport` and `ai_sport`.
- `3104`: timeout/discovery or service unavailable.
- Server-side fails but robot-local succeeds: do not send Unitree DDS from the server; run a robot-side motion daemon and send HTTP/WebSocket commands over Wi-Fi.

Only use robot-local checks to isolate low-level robot networking after server-side tests are exhausted:

```bash
ssh unitree@192.168.0.212
cd ~/humanoid-robot-grasping
git pull
python3 scripts/robot_eth0_rpc_probe.py --interfaces eth0,wlan0 --ping-ip 192.168.123.1
```

If the robot does not show an obvious `ai_sport`/`loco` Linux service to start manually, use the SDK motion switcher path:

```bash
uv run loco 192.168.0.212 check_motion_mode --interface enP7s7 --dds-config-mode no_trace
uv run loco 192.168.0.212 select_ai_mode --interface enP7s7 --dds-config-mode no_trace
uv run loco 192.168.0.212 probe_loco --interface enP7s7 --loco-service-name ai_sport --dds-config-mode no_trace
```

`check_motion_mode` is read-only. `select_ai_mode` calls `MotionSwitcherClient.SelectMode("ai")`, so only run it when the robot is physically safe and the controller/e-stop is ready.

After a read-only probe returns `ok: true`, a tiny movement smoke test is available but guarded:

```bash
uv run loco 192.168.0.212 smoke_move \
  --interface enP7s7 \
  --loco-service-name sport \
  --dds-config-mode no_trace \
  --i-understand-this-moves-the-robot
```

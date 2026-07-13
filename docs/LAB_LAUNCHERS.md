# Lab launchers

These wrappers keep non-secret lab network defaults in one place. They do not
store tokens and never authorize movement by default.

## Robot

Install the persistent high-FPS camera publisher once. It uses Unitree's
official `mscli` to stop `video_hub_pc4`, takes `/dev/video4`, and starts
automatically after every boot at 960x540/60 FPS:

```bash
bash scripts/robot/install-highfps-camera-service.sh
```

Restore the vendor 15-FPS camera later with:

```bash
bash scripts/robot/uninstall-highfps-camera-service.sh
```

Defaults:

- GB10 peer `192.168.0.66`
- Project ROS domain `1`; native Unitree motor DDS domain `0`
- High-FPS RGB producer and relay interface `eth0`
- Unitree control peer `192.168.123.1`
- RGB `960x540` at `60 FPS` after the boot service is installed

Override a changed peer without editing the script:

```bash
CLIENT_IP=<NEW_GB10_IP> bash scripts/robot/lab-start.sh
```

## GB10

Start the ROS client, YOLO server, research recorder, and single browser UI:

```bash
bash scripts/gb10/lab-start.sh
```

The default robot peer is `192.168.123.164`; the UI is
`http://<GB10_IP>:8000/`. Override a changed address or model with environment
variables:

```bash
ROBOT_HOST=<NEW_ROBOT_IP> bash scripts/gb10/lab-start.sh
MODEL=/path/to/best.pt bash scripts/gb10/lab-start.sh
```

This launcher is always dry-run. No robot HTTP ports or token files are
involved.

## Commissioning

Stop the normal robot node first so only one process can own `/arm_sdk`.
Start read-only commissioning:

```bash
bash scripts/robot/lab-commission.sh
```

The commissioning launcher automatically relays the boot-service camera to
the GB10 vision page. It does not open `/dev/video4` itself.

With the GB10 running, open
`http://192.168.0.66:8000/commissioning/`. Only after the full read-only
preflight, a cleared area, spotter, physical e-stop, and supported robot, use:

```bash
bash scripts/robot/lab-commission.sh move
```

The movement-capable launch still starts disarmed. A short-lived operator
session and all robot safety gates must pass before a jog can be issued.

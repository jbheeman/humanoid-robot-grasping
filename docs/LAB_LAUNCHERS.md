# Lab launchers

These wrappers keep non-secret lab network defaults in one place. They do not
store tokens and never authorize movement by default.

## Robot

If the stock camera service owns the RGB source, stop it as required by the
lab image, then start the custom RGB relay and disarmed ROS node:

```bash
sudo /unitree/sbin/mscli stopservice video_hub_pc4
bash scripts/robot/lab-start.sh
```

Defaults:

- GB10 peer `192.168.0.66`
- ROS and RGB interface `wlan0`
- Unitree control peer `192.168.123.1`
- ROS domain `0`
- RGB `960x540` at `60 FPS`

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

With the GB10 running, open
`http://192.168.0.66:8000/commissioning/`. Only after the full read-only
preflight, a cleared area, spotter, physical e-stop, and supported robot, use:

```bash
bash scripts/robot/lab-commission.sh move
```

The movement-capable launch still starts disarmed. A short-lived operator
session and all robot safety gates must pass before a jog can be issued.

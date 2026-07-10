# Lab launchers

The lab wrappers keep the working G1/GB10 paths and non-secret network
defaults in one place. They do not store bearer-token contents and they never
authorize arm movement.

## Robot

After stopping the stock `video_hub_pc4` service once with:

```bash
sudo /unitree/sbin/mscli stopservice video_hub_pc4
```

start the custom 960x540, 60 FPS RGB producer, relay, depth service, and
disarmed arm bridge with one command:

```bash
bash scripts/robot/lab-start.sh
```

The defaults are GB10 `192.168.0.66`, robot multicast interface `wlan0`, and
the mode-0600 token at `~/.config/g1-arm-token`. Override a changed GB10 IP
without editing the script:

```bash
CLIENT_IP=NEW_GB10_IP bash scripts/robot/lab-start.sh
```

## GB10

Start the 60 FPS, 960x540 YOLO11 dry-run server with a 60 FPS browser-output
target and suppressed successful HTTP access logs:

```bash
bash scripts/gb10/lab-start.sh
```

Its default robot Ethernet address is `192.168.123.164`. Override it if the
router layout changes:

```bash
ROBOT_HOST=NEW_ROBOT_IP bash scripts/gb10/lab-start.sh
```

The default checkpoint is
`models/plushie_detector/yolo11x_plushie_quality_12h_b24/weights/best.pt`.
Use a temporary YOLOv8 test without changing defaults:

```bash
MODEL=models/plushie_detector/yolov8n_plushie_mvp/weights/best.pt bash scripts/gb10/lab-start.sh
```

Arm commissioning and movement authorization remain intentionally separate;
see [ARM_COMMISSIONING_RUNBOOK.md](ARM_COMMISSIONING_RUNBOOK.md).

## Arm commissioning

Stop `lab-start.sh` first because commissioning exclusively owns the arm-bridge
port. Start a read-only preflight with:

```bash
bash scripts/robot/lab-commission.sh
```

Only with a cleared area, spotter, and physical e-stop, start the guarded
movement-capable commissioning server:

```bash
bash scripts/robot/lab-commission.sh move
```

That command requires the physical-e-stop acknowledgement to be typed again;
it does not move the robot on startup. The MacBook wizard creates and enables
the session before a guarded 0.01-rad test jog is possible.

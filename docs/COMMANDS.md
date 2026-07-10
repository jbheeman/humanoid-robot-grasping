# Project commands

The preferred operator interface is the `g1` application. It is installed by
the normal project environment and keeps the host-specific details behind one
discoverable command:

```bash
uv run g1 --help
uv run g1 <group> <workflow> --help
```

The underlying shell launchers remain in `scripts/` because the robot, GB10,
and local workstation intentionally use different environments. They are
stable automation entry points; `g1` is the shorter interactive interface.

## First-time setup

Run the setup command on the machine that will perform that role:

```bash
uv run g1 setup robot     # Unitree robot: isolated SDK/depth/arm environment
uv run g1 setup gb10      # GB10: CUDA/YOLO/IK/research server environment
uv run g1 setup vision    # lightweight vision-only environment
```

Do not install the GB10 training group into the robot environment.

## Initial arm commissioning

The server and Unitree SDK run on the robot. The MacBook only uses a browser
over an SSH tunnel:

```bash
uv run g1 arm commissioning
```

Start it read-only first. The launcher prints the tunnel command and browser
URL. Movement still requires the explicit environment acknowledgement and the
motion mode observed during preflight. See
[ARM_COMMISSIONING_RUNBOOK.md](ARM_COMMISSIONING_RUNBOOK.md).

## Perception and research

```bash
uv run g1 vision snapshot <robot-ip> --diagnose
uv run g1 vision server
uv run g1 stream viewer
uv run g1 tune joint-audit
uv run g1 tune analyze runs/research/arm_tracking
```

The `tune` workflows forward directly to the existing `g1-tune` implementation
with the subcommand inserted automatically.

## Data and training

```bash
uv run g1 data install
uv run g1 data augment --help
uv run g1 data capture --help
uv run g1 train detector --help
uv run g1 train evaluate --help
uv run g1 train guarded
```

Training and research outputs remain separated under `models/` and
`runs/research/`; this command layer does not change storage locations.

## Read-only diagnostics

```bash
uv run g1 inspect cameras
uv run g1 inspect dds
uv run g1 robot scan --help
uv run g1 robot loco <robot-ip> diagnose
```

For scripts used by systemd, CI, or an existing lab notebook, continue to call
the original `scripts/*.sh` launcher directly. Their defaults and safety gates
are unchanged.

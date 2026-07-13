# ROS 2 Rewrite Status

The active architecture is the ROS 2 design documented in
[ARCHITECTURE.md](ARCHITECTURE.md). This page replaces the superseded
robot-side HTTP/WebSocket implementation plan.

## Runtime topology

- The robot owns `/lowstate`, `/arm_sdk`, guarded arm control,
  commissioning services, and compressed-depth publication.
- GB10 owns perception, fusion, IK, the ROS client, and the only browser HTTP
  server on port 8000.
- CycloneDDS uses generated static peers; robot control and depth are not
  exposed as HTTP or WebSocket endpoints.
- Commissioning is depth-free and uses automatic session heartbeats while
  enabled.

## Safety invariants

Movement remains disabled by default and requires the explicit robot movement
launch gate, verified motion mode, exclusive controller ownership, stable
standing state, and the appropriate calibration or 29-DOF joint contract.
Targets remain bounded by joint, slew, acceleration, following-error,
freshness, and deadman checks. Any stale target, stale LowState, ownership
conflict, process loss, or operator stop enters the bounded zero-weight release
path.

Real-hardware commissioning uses 0.01 rad shoulder-only jogs with conservative
limits of 0.10 rad/s, 0.50 rad/s², kp=60, and kd=1.5. Simulation-derived gains
are not hardware defaults and require separate guarded validation.

## Operator entry points

```bash
uv run g1 robot start --client-ip <GB10_IP>
uv run g1 gb10 start --robot-host <ROBOT_IP> --dry-run
```

Open `http://<GB10_IP>:8000/`. For guarded commissioning, follow
[ARM_COMMISSIONING_RUNBOOK.md](ARM_COMMISSIONING_RUNBOOK.md). For tracking,
follow [ARM_TRACKING_RUNBOOK.md](ARM_TRACKING_RUNBOOK.md).

Hardware and cross-distribution checks remain required before movement:
Foxy/Jazzy interface builds, static-peer topic flow, dry-run state/depth,
disconnect and duplicate-publisher tests, then supervised physical
commissioning with a cleared area and physical e-stop.

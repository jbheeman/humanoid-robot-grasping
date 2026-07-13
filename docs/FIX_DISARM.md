# Diagnosing a disarmed G1 ROS arm controller

`DISARMED`, movement disabled, and zero arm weight are healthy startup values.
Do not bypass them to test video, detection, or ROS discovery.

## Read-only diagnosis

Run on the robot while its node is active:

```bash
uv run g1 inspect ros --role robot --peer <GB10_IP> --interface wlan0
ip -br addr
ip route
```

Run on GB10:

```bash
uv run g1 inspect ros --role gb10 --peer <ROBOT_IP>
```

The expected minimum is fresh `/lowstate`, `/g1/arm/state`, `/g1/depth`, and
`/g1/commissioning/state`. The UI at `http://<GB10_IP>:8000/` should show the
same state. There is no robot `/health` URL.

If discovery fails, compare these values on both launchers:

- peer addresses (`CLIENT_IP` on robot, `ROBOT_HOST` on GB10);
- `ROS_DOMAIN_ID`;
- selected network interface;
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`;
- generated `CYCLONEDDS_URI` path and static peer contents.

On the robot, also verify `UNITREE_CONTROL_PEER` (default
`192.168.123.1`) and the Unitree motion-switcher/sport services. A fresh
`/lowstate` with no motion-switch response means state transport works but
movement mode cannot be verified; movement must remain blocked.

Other movement blockers include another `/arm_sdk` publisher, unstable stance,
wrong motion mode, stale targets, unavailable motor status, failed calibration,
or a robot/profile identity mismatch.

## Before movement

Use a cleared exclusion zone, supported robot, spotter, and physical e-stop.
Complete the read-only and `0.01 rad` workflow in
[ARM_COMMISSIONING_RUNBOOK.md](ARM_COMMISSIONING_RUNBOOK.md). GB10 dry-run and
visualization never require movement permission.

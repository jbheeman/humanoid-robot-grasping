# Superseded Implementation Review

The REST/WebSocket architecture and port-based handoff previously recorded in
this file have been retired. It must not be used to launch or commission the
robot.

Use the current documents instead:

- [Architecture](ARCHITECTURE.md)
- [Commands](COMMANDS.md)
- [Arm commissioning runbook](ARM_COMMISSIONING_RUNBOOK.md)
- [Arm tracking runbook](ARM_TRACKING_RUNBOOK.md)
- [ROS 2 rewrite status](PLAN.md)

The robot exposes no project HTTP/WebSocket control or depth server. GB10 port
8000 is the only browser endpoint. Physical movement still requires the
documented explicit launch gate, verified ROS state, conservative
commissioning, and physical e-stop procedure.

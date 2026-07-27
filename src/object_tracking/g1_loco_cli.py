"""Concise ROS 2 command line client for G1 locomotion."""

from __future__ import annotations

import argparse
import sys

from object_tracking.unitree_g1 import (
    G1Ros2LocoClient,
    LOCO_SERVICE_CHOICES,
    MotionSwitcherRos2Client,
    UnitreeCommandResult,
    UnitreeG1Error,
    parse_velocity,
)


COMMAND_CHOICES = (
    "probe",
    "probe_loco",
    "check_motion_mode",
    "select_motion_mode",
    "select_ai_mode",
    "release_motion_mode",
    "start",
    "stand_up",
    "balance_stand",
    "stop_move",
    "damp",
    "move",
    "smooth_move",
)

COMMANDS_REQUIRING_EXECUTE = {
    "select_motion_mode",
    "select_ai_mode",
    "release_motion_mode",
    "start",
    "stand_up",
    "balance_stand",
    "move",
    "smooth_move",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe or command Unitree G1 locomotion through ROS 2."
    )
    parser.add_argument("command", choices=COMMAND_CHOICES)
    parser.add_argument(
        "--service",
        "--loco-service-name",
        dest="service",
        choices=LOCO_SERVICE_CHOICES,
        default="auto",
        help="Locomotion service. 'auto' read-only probes sport, then ai_sport.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for each ROS response. Default: 1.0.",
    )
    parser.add_argument(
        "--velocity",
        type=parse_velocity,
        metavar='"VX VY OMEGA [DURATION]"',
        help="Velocity and optional duration for move or smooth_move.",
    )
    parser.add_argument(
        "--move-ramp-s",
        type=float,
        default=0.25,
        help="Smooth-move acceleration/deceleration ramp. Default: 0.25 seconds.",
    )
    parser.add_argument(
        "--mode",
        default="ai",
        help="Motion mode for select_motion_mode. Default: ai.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required for commands that initiate movement or change robot mode.",
    )
    return parser


def _require_execute(args: argparse.Namespace) -> None:
    if args.command in COMMANDS_REQUIRING_EXECUTE and not args.execute:
        raise UnitreeG1Error(
            f"Refusing {args.command!r} without --execute. Keep the e-stop ready and "
            "confirm the robot has clear space."
        )


def _motion_mode_command(args: argparse.Namespace) -> UnitreeCommandResult:
    with MotionSwitcherRos2Client(timeout_s=args.timeout) as client:
        if args.command == "check_motion_mode":
            return client.check_mode()
        if args.command == "release_motion_mode":
            return client.release_mode()
        mode = "ai" if args.command == "select_ai_mode" else args.mode
        return client.select_mode(mode)


def _locomotion_command(args: argparse.Namespace) -> UnitreeCommandResult:
    with G1Ros2LocoClient(
        timeout_s=args.timeout,
        loco_service_name=args.service,
    ) as client:
        if args.command in ("probe", "probe_loco"):
            return client.probe_loco()
        if args.command in ("move", "smooth_move"):
            if args.velocity is None:
                raise UnitreeG1Error(
                    f"{args.command} requires --velocity 'vx vy omega [duration]'."
                )
            vx, vy, omega, duration = args.velocity
            if args.command == "smooth_move":
                return client.smooth_move(
                    vx,
                    vy,
                    omega,
                    duration=0.5 if duration is None else duration,
                    ramp_s=args.move_ramp_s,
                )
            return client.move(vx, vy, omega, duration)
        return client.command(args.command)


def main() -> None:
    args = build_parser().parse_args()
    try:
        _require_execute(args)
        if args.command in {
            "check_motion_mode",
            "select_motion_mode",
            "select_ai_mode",
            "release_motion_mode",
        }:
            result = _motion_mode_command(args)
        else:
            result = _locomotion_command(args)
    except (UnitreeG1Error, ValueError) as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt as exc:
        print("Command interrupted; a stop command was sent.", file=sys.stderr)
        raise SystemExit(130) from exc

    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

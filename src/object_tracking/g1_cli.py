"""Operator CLI for the realtime G1 plush-interception demo.

Examples::

    uv run g1 setup robot
    uv run g1 robot start --client-ip 192.168.0.66
    uv run g1 gb10 start --robot-host 192.168.0.213

Arguments after a workflow are forwarded unchanged to the existing command.
Use ``uv run g1 <group> <workflow> --help`` for that workflow's options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class Route:
    """How a user-facing workflow is launched."""

    kind: str
    target: str
    description: str
    prefix: tuple[str, ...] = ()


ROUTES: dict[tuple[str, str], Route] = {
    ("assets", "build"): Route(
        "module", "object_tracking.g1_asset_builder", "build offline G1 visualization assets"
    ),
    ("calibrate", "camera"): Route(
        "module", "object_tracking.arm_tracking.calibration_cli", "camera calibration workflow"
    ),
    ("calibrate", "localization"): Route(
        "module", "object_tracking.localization_cli", "validate paired D435I RGB/depth localization"
    ),
    ("calibrate", "tuner"): Route(
        "script", "scripts/robot/localization_tuner.py", "interactive local D435I localization capture"
    ),
    ("inspect", "cameras"): Route(
        "script", "scripts/dev/cameras.sh", "inspect camera device ownership"
    ),
    ("inspect", "ros"): Route(
        "script", "scripts/dev/ros.sh", "inspect ROS 2 discovery and G1 interfaces"
    ),
    ("robot", "services"): Route(
        "script", "scripts/robot/start.sh", "start the disarmed robot ROS 2 node"
    ),
    ("robot", "start"): Route(
        "script", "scripts/robot/start.sh", "start the disarmed robot ROS 2 node"
    ),
    ("robot", "bunny-test"): Route(
        "script",
        "scripts/robot/bunny-test.sh",
        "start the robot bridge and automatically run one guarded bunny-tracking session",
    ),
    ("setup", "gb10"): Route(
        "script", "scripts/gb10/setup.sh", "install the GB10 interception environment"
    ),
    ("setup", "robot"): Route(
        "script", "scripts/robot/setup.sh", "install the Foxy robot ROS 2 environment"
    ),
    ("vision", "server"): Route(
        "script", "scripts/gb10/start.sh", "run the GB10 ROS client and browser UI"
    ),
    ("gb10", "start"): Route(
        "script", "scripts/gb10/start.sh", "run the GB10 ROS client and browser UI"
    ),
    ("gb10", "relay-test"): Route(
        "script", "scripts/gb10/relay-test.sh", "decode-test the GB10 UDP video relay"
    ),
    ("gb10", "plushie"): Route(
        "script", "scripts/gb10/plushie.sh", "run the lightweight plushie stream variant"
    ),
}


def _workflow_parser(parser: argparse.ArgumentParser, group: str) -> None:
    workflows = sorted(
        (name, route) for (route_group, name), route in ROUTES.items() if route_group == group
    )
    subparsers = parser.add_subparsers(dest="workflow", metavar="WORKFLOW", required=True)
    for name, route in workflows:
        # Do not consume --help here.  Module-backed workflows should expose
        # their real parser, while script-backed workflows are handled safely
        # by _run without accidentally starting a server or installer.
        child = subparsers.add_parser(
            name, help=route.description, description=route.description, add_help=False
        )
        child.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the workflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g1",
        description="Launch and inspect the realtime G1 plush-interception demo.",
    )
    subparsers = parser.add_subparsers(dest="group", metavar="GROUP", required=True)
    groups = {
        "assets": "offline browser visualization assets",
        "calibrate": "camera/arm calibration",
        "inspect": "read-only hardware and network diagnostics",
        "robot": "robot-local interception services",
        "setup": "host-specific environment setup",
        "vision": "G1 camera and GB10 perception workflows",
        "gb10": "GB10 realtime inference and interception",
    }
    for group, help_text in groups.items():
        child = subparsers.add_parser(group, help=help_text, description=help_text)
        _workflow_parser(child, group)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _role_launcher(route: Route, args: Sequence[str]) -> tuple[Path, list[str], dict[str, str]]:
    """Translate friendly role flags into the existing launcher environment."""

    root = _repo_root()
    env = os.environ.copy()
    launcher = root / route.target
    parser = argparse.ArgumentParser(prog=f"g1 {route.target.split('/')[1]} start")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    if route.target == "scripts/robot/start.sh":
        parser.add_argument("--client-ip", default="192.168.0.66")
        parser.add_argument("--interface")
        parser.add_argument("--control-peer")
        parser.add_argument("--ros-domain-id", type=int)
        parser.add_argument("--calibration")
        parser.add_argument("--robot-id")
        parser.add_argument("--allow-movement", action="store_true")
        parser.add_argument("--expected-motion-mode")
        parser.add_argument("--depth-source", choices=("auto", "ros", "librealsense"))
        parser.add_argument("--depth-serial")
        parser.add_argument("--robot-python")
        obsolete = {"--token-file", "--arm-port", "--depth-port"}
        used_obsolete = sorted(
            token.split("=", 1)[0] for token in args if token.split("=", 1)[0] in obsolete
        )
        if used_obsolete:
            parser.error(
                f"removed robot HTTP option(s): {', '.join(used_obsolete)}; "
                "ROS 2 transport needs only --client-ip"
            )
        namespace = parser.parse_args(args)
        mappings = {
            "CLIENT_IP": namespace.client_ip,
            "ROBOT_INTERFACE": namespace.interface,
            "UNITREE_CONTROL_PEER": namespace.control_peer,
            "G1_PROJECT_ROS_DOMAIN_ID": namespace.ros_domain_id,
            "CALIBRATION": namespace.calibration,
            "G1_ROBOT_ID": namespace.robot_id,
            "ALLOW_MOVEMENT": "1" if namespace.allow_movement else None,
            "EXPECTED_MOTION_MODE": namespace.expected_motion_mode,
            "DEPTH_SOURCE": namespace.depth_source,
            "DEPTH_SERIAL": namespace.depth_serial,
            "ROBOT_PYTHON": namespace.robot_python,
        }
    elif route.target == "scripts/gb10/start.sh":
        parser.add_argument("--robot-host")
        parser.add_argument("--ros-interface")
        parser.add_argument("--ros-domain-id", type=int)
        parser.add_argument("--model")
        parser.add_argument("--host")
        parser.add_argument("--port", type=int)
        parser.add_argument("--calibration")
        parser.add_argument("--intercept-config")
        parser.add_argument("--arm-home")
        parser.add_argument("--robot-id")
        parser.add_argument("--research-root")
        parser.add_argument("--research-label")
        parser.add_argument("--research-notes")
        parser.add_argument("--no-research-record", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--execute", action="store_true")
        obsolete = {"--viewer-port", "--arm-url", "--arm-token-file", "--depth-ws"}
        used_obsolete = sorted(
            token.split("=", 1)[0] for token in args if token.split("=", 1)[0] in obsolete
        )
        if used_obsolete:
            parser.error(
                f"removed robot HTTP option(s): {', '.join(used_obsolete)}; "
                "GB10 now uses ROS 2 and serves one UI on --port"
            )
        namespace = parser.parse_args(args)
        mappings = {
            "ROBOT_HOST": namespace.robot_host,
            "ROS_INTERFACE": namespace.ros_interface,
            "ROS_DOMAIN_ID": namespace.ros_domain_id,
            "MODEL": namespace.model,
            "HOST": namespace.host,
            "PORT": namespace.port,
            "CALIBRATION": namespace.calibration,
            "INTERCEPT_CONFIG": namespace.intercept_config,
            "ARM_HOME": namespace.arm_home,
            "G1_ROBOT_ID": namespace.robot_id,
            "RESEARCH_ROOT": namespace.research_root,
            "RESEARCH_LABEL": namespace.research_label,
            "RESEARCH_NOTES": namespace.research_notes,
            "RESEARCH_RECORD": "0" if namespace.no_research_record else None,
            "EXECUTE": "1" if namespace.execute else ("0" if namespace.dry_run else None),
        }
    else:  # pragma: no cover - only role launchers call this helper.
        return launcher, list(args), env

    for key, value in mappings.items():
        if value is not None:
            env[key] = str(value)
    for assignment in namespace.set:
        if "=" not in assignment:
            parser.error(f"--set requires KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            parser.error(f"invalid environment key in --set: {key!r}")
        env[key] = value
    return launcher, [], env


def _run(route: Route, args: Sequence[str]) -> int:
    root = _repo_root()
    forwarded = [*route.prefix, *args]
    role_targets = {"scripts/robot/start.sh", "scripts/gb10/start.sh"}
    if route.kind == "script" and route.target not in role_targets and list(args) == ["--help"]:
        print(route.description)
        print("This workflow is a host-specific shell launcher; see docs/COMMANDS.md for usage.")
        return 0
    if route.kind == "module":
        module = importlib.import_module(route.target)
        old_argv = sys.argv
        sys.argv = [route.target, *forwarded]
        try:
            result = module.main()
        finally:
            sys.argv = old_argv
        return int(result or 0)
    else:
        if route.target in role_targets:
            launcher, forwarded, environment = _role_launcher(route, args)
        else:
            launcher = root / route.target
            forwarded = list(args)
            environment = None
        if launcher.suffix == ".py":
            command = [sys.executable, str(launcher), *forwarded]
        elif launcher.suffix == ".sh":
            command = ["bash", str(launcher), *forwarded]
        else:
            command = [str(launcher), *forwarded]
    try:
        completed = subprocess.run(command, cwd=root, check=False, env=environment)
    except FileNotFoundError as exc:
        raise SystemExit(f"Workflow launcher not found: {exc.filename}") from exc
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    namespace, _extras = parser.parse_known_args(raw_args)
    route = ROUTES.get((namespace.group, namespace.workflow))
    if route is None:  # pragma: no cover - argparse choices make this unreachable.
        parser.error("unknown workflow")
    # argparse splits unknown option/value pairs around REMAINDER and can
    # reverse them (``--target all`` became ``all --target``). The first two
    # tokens are the validated group/workflow; preserve everything after them.
    return _run(route, raw_args[2:])


if __name__ == "__main__":
    raise SystemExit(main())

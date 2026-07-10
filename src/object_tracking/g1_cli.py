"""Unified operator CLI for the humanoid robot grasping project.

The repository has a few intentionally host-specific launchers: the robot
must not receive the GB10 training environment, and camera/GStreamer setup is
different from arm commissioning.  This command gives those workflows one
discoverable namespace without hiding the underlying scripts or changing
their safety defaults.

Examples::

    uv run g1 setup robot
    uv run g1 arm commissioning
    uv run g1 vision server
    uv run g1 tune analyze runs/research/arm_tracking

Arguments after a workflow are forwarded unchanged to the existing command.
Use ``uv run g1 <group> <workflow> --help`` for that workflow's options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
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
    ("arm", "commissioning"): Route(
        "script", "scripts/run_robot_arm_commissioning.sh", "robot-local guarded arm commissioning"
    ),
    ("calibrate", "camera"): Route(
        "module", "object_tracking.arm_tracking.calibration_cli", "camera calibration workflow"
    ),
    ("data", "augment"): Route(
        "module", "object_tracking.augment_plushie_dataset", "train-only dataset augmentation"
    ),
    ("data", "capture"): Route(
        "script", "scripts/capture_training_frames.sh", "capture labeled frames from a running stream"
    ),
    ("data", "install"): Route(
        "script", "scripts/install_all_plushie_datasets.sh", "install public plushie datasets"
    ),
    ("inspect", "cameras"): Route(
        "script", "scripts/list_camera_bindings.sh", "inspect camera device ownership"
    ),
    ("inspect", "dds"): Route(
        "script", "scripts/dds_discovery_probe.py", "inspect Unitree DDS discovery"
    ),
    ("robot", "command"): Route(
        "script", "scripts/robot.py", "run the explicit robot command wrapper"
    ),
    ("robot", "loco"): Route(
        "module", "object_tracking.g1_loco_cli", "send a Unitree G1 locomotion command"
    ),
    ("robot", "scan"): Route(
        "module", "object_tracking.g1_scan_cli", "scan the local robot network"
    ),
    ("robot", "services"): Route(
        "script", "scripts/run_robot_grasping_services.sh", "start disarmed robot services"
    ),
    ("setup", "gb10"): Route(
        "script", "scripts/setup_gb10_vision_server.sh", "install GB10 vision/training environment"
    ),
    ("setup", "opencv"): Route(
        "script", "scripts/setup_opencv_vision_server.sh", "install system-OpenCV vision environment"
    ),
    ("setup", "robot"): Route(
        "script", "scripts/setup_robot_grasping_services.sh", "install the isolated robot environment"
    ),
    ("setup", "vision"): Route(
        "script", "scripts/setup_vision_server.sh", "install the lightweight vision environment"
    ),
    ("stream", "local"): Route(
        "script", "scripts/run_yolo_stream.sh", "serve a local camera stream"
    ),
    ("stream", "remote"): Route(
        "script", "scripts/run_remote_yolo_streams.sh", "run the GB10 research stream and viewer"
    ),
    ("stream", "viewer"): Route(
        "script", "scripts/run_dual_camera_viewer.sh", "serve the browser viewer"
    ),
    ("train", "detector"): Route(
        "module", "object_tracking.train_plushie_detector", "train the plushie detector"
    ),
    ("train", "evaluate"): Route(
        "script", "scripts/eval_plushie_detector.sh", "evaluate a detector checkpoint"
    ),
    ("train", "guarded"): Route(
        "script", "scripts/run_guarded_plushie_training.sh", "resource-capped detector training"
    ),
    ("tune", "analyze"): Route(
        "module", "object_tracking.tuning_cli", "analyze one research run", ("analyze",)
    ),
    ("tune", "compare"): Route(
        "module", "object_tracking.tuning_cli", "compare controlled research runs", ("compare",)
    ),
    ("tune", "joint-audit"): Route(
        "module", "object_tracking.tuning_cli", "audit the 29-DOF joint contract", ("joint-audit",)
    ),
    ("vision", "snapshot"): Route(
        "module", "object_tracking.g1_vision_cli", "capture or diagnose a G1 camera stream"
    ),
    ("vision", "server"): Route(
        "script", "scripts/run_remote_yolo_streams.sh", "run the GB10 vision server and browser UI"
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
        description="One discoverable command for the robot, vision, research, and training workflows.",
        epilog="Existing scripts remain available for automation; this command is their short operator interface.",
    )
    subparsers = parser.add_subparsers(dest="group", metavar="GROUP", required=True)
    groups = {
        "arm": "arm commissioning and guarded movement",
        "calibrate": "camera/arm calibration",
        "data": "dataset capture, install, and augmentation",
        "inspect": "read-only hardware and network diagnostics",
        "robot": "robot-local services and Unitree commands",
        "setup": "host-specific environment setup",
        "stream": "camera streams and browser viewers",
        "train": "detector training and evaluation",
        "tune": "offline run and joint-contract analysis",
        "vision": "G1 camera and GB10 perception workflows",
    }
    for group, help_text in groups.items():
        child = subparsers.add_parser(group, help=help_text, description=help_text)
        _workflow_parser(child, group)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(route: Route, args: Sequence[str]) -> int:
    root = _repo_root()
    forwarded = [*route.prefix, *args]
    if route.kind == "script" and list(args) == ["--help"]:
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
        launcher = root / route.target
        if launcher.suffix == ".py":
            command = [sys.executable, str(launcher), *forwarded]
        elif launcher.suffix == ".sh":
            command = ["bash", str(launcher), *forwarded]
        else:
            command = [str(launcher), *forwarded]
    try:
        completed = subprocess.run(command, cwd=root, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"Workflow launcher not found: {exc.filename}") from exc
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace, extras = parser.parse_known_args(argv)
    route = ROUTES.get((namespace.group, namespace.workflow))
    if route is None:  # pragma: no cover - argparse choices make this unreachable.
        parser.error("unknown workflow")
    return _run(route, [*namespace.args, *extras])


if __name__ == "__main__":
    raise SystemExit(main())

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
    ("arm", "commissioning"): Route(
        "script", "scripts/robot/commission.sh", "robot-local guarded arm commissioning"
    ),
    ("calibrate", "camera"): Route(
        "module", "object_tracking.arm_tracking.calibration_cli", "camera calibration workflow"
    ),
    ("data", "augment"): Route(
        "module", "object_tracking.augment_plushie_dataset", "train-only dataset augmentation"
    ),
    ("data", "capture"): Route(
        "script", "scripts/data/capture.sh", "capture labeled frames from a running stream"
    ),
    ("data", "install"): Route(
        "script", "scripts/data/install.sh", "install public plushie datasets"
    ),
    ("inspect", "cameras"): Route(
        "script", "scripts/dev/cameras.sh", "inspect camera device ownership"
    ),
    ("inspect", "dds"): Route(
        "script", "scripts/dev/dds.py", "inspect Unitree DDS discovery"
    ),
    ("robot", "command"): Route(
        "script", "scripts/robot/command.py", "run the explicit robot command wrapper"
    ),
    ("robot", "loco"): Route(
        "module", "object_tracking.g1_loco_cli", "send a Unitree G1 locomotion command"
    ),
    ("robot", "scan"): Route(
        "module", "object_tracking.g1_scan_cli", "scan the local robot network"
    ),
    ("robot", "services"): Route(
        "script", "scripts/robot/start.sh", "start disarmed robot services"
    ),
    ("robot", "start"): Route(
        "script", "scripts/robot/start.sh", "start disarmed robot services"
    ),
    ("setup", "gb10"): Route(
        "script", "scripts/gb10/setup.sh", "install GB10 vision/training environment"
    ),
    ("setup", "opencv"): Route(
        "script", "scripts/local/opencv-setup.sh", "install system-OpenCV vision environment"
    ),
    ("setup", "robot"): Route(
        "script", "scripts/robot/setup.sh", "install the isolated robot environment"
    ),
    ("setup", "sim"): Route(
        "script", "scripts/sim/setup.sh", "install the isolated MuJoCo simulation environment"
    ),
    ("setup", "vision"): Route(
        "script", "scripts/local/setup.sh", "install the lightweight vision environment"
    ),
    ("setup", "local"): Route(
        "script", "scripts/local/setup.sh", "install the lightweight vision environment"
    ),
    ("stream", "local"): Route(
        "script", "scripts/local/start.sh", "serve a local camera stream"
    ),
    ("stream", "remote"): Route(
        "script", "scripts/gb10/start.sh", "run the GB10 research stream and viewer"
    ),
    ("stream", "viewer"): Route(
        "script", "scripts/local/viewer.sh", "serve the browser viewer"
    ),
    ("train", "detector"): Route(
        "module", "object_tracking.train_plushie_detector", "train the plushie detector"
    ),
    ("train", "evaluate"): Route(
        "script", "scripts/training/evaluate.sh", "evaluate a detector checkpoint"
    ),
    ("train", "guarded"): Route(
        "script", "scripts/training/guarded.sh", "resource-capped detector training"
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
        "script", "scripts/gb10/start.sh", "run the GB10 vision server and browser UI"
    ),
    ("gb10", "start"): Route(
        "script", "scripts/gb10/start.sh", "run the GB10 vision server and browser UI"
    ),
    ("gb10", "relay-test"): Route(
        "script", "scripts/gb10/relay-test.sh", "decode-test the GB10 UDP video relay"
    ),
    ("gb10", "plushie"): Route(
        "script", "scripts/gb10/plushie.sh", "run the lightweight plushie stream variant"
    ),
    ("local", "start"): Route(
        "script", "scripts/local/start.sh", "serve a local camera stream"
    ),
    ("sim", "doctor"): Route(
        "module", "object_tracking.g1_sim_cli", "check simulation resources", ("doctor",)
    ),
    ("sim", "smoke"): Route(
        "module", "object_tracking.g1_sim_cli", "run a short deterministic simulation", ("smoke",)
    ),
    ("sim", "sweep"): Route(
        "module", "object_tracking.g1_sim_cli", "tune right-arm motion in MuJoCo", ("sweep",)
    ),
    ("sim", "status"): Route(
        "module", "object_tracking.g1_sim_cli", "show the latest sweep checkpoint", ("status",)
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
        "assets": "offline browser visualization assets",
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
        "gb10": "GB10 inference and browser server",
        "local": "local camera and replay server",
        "sim": "read-only suspended G1 simulation and tuning",
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
        parser.add_argument("--client-ip")
        parser.add_argument("--token-file")
        parser.add_argument("--calibration")
        parser.add_argument("--arm-port", type=int)
        parser.add_argument("--depth-port", type=int)
        parser.add_argument("--robot-python")
        namespace = parser.parse_args(args)
        mappings = {
            "CLIENT_IP": namespace.client_ip,
            "ARM_TOKEN_FILE": namespace.token_file,
            "CALIBRATION": namespace.calibration,
            "ARM_PORT": namespace.arm_port,
            "DEPTH_PORT": namespace.depth_port,
            "ROBOT_PYTHON": namespace.robot_python,
        }
    elif route.target == "scripts/gb10/start.sh":
        parser.add_argument("--robot-host")
        parser.add_argument("--model")
        parser.add_argument("--host")
        parser.add_argument("--port", type=int)
        parser.add_argument("--viewer-port", type=int)
        parser.add_argument("--calibration")
        parser.add_argument("--arm-url")
        parser.add_argument("--arm-home")
        parser.add_argument("--robot-id")
        parser.add_argument("--research-root")
        parser.add_argument("--research-label")
        parser.add_argument("--research-notes")
        parser.add_argument("--no-research-record", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--execute", action="store_true")
        namespace = parser.parse_args(args)
        mappings = {
            "ROBOT_HOST": namespace.robot_host,
            "MODEL": namespace.model,
            "HOST": namespace.host,
            "PORT": namespace.port,
            "VIEWER_PORT": namespace.viewer_port,
            "CALIBRATION": namespace.calibration,
            "ARM_URL": namespace.arm_url,
            "ARM_HOME": namespace.arm_home,
            "G1_ROBOT_ID": namespace.robot_id,
            "RESEARCH_ROOT": namespace.research_root,
            "RESEARCH_LABEL": namespace.research_label,
            "RESEARCH_NOTES": namespace.research_notes,
            "RESEARCH_RECORD": "0" if namespace.no_research_record else None,
            "EXECUTE": "1" if namespace.execute else ("0" if namespace.dry_run else None),
        }
    elif route.target == "scripts/local/start.sh":
        parser.add_argument("--source", choices=("camera", "opencv", "realsense", "file"), default="camera")
        parser.add_argument("--device")
        parser.add_argument("--pipeline")
        parser.add_argument("--model")
        parser.add_argument("--host")
        parser.add_argument("--port", type=int)
        parser.add_argument("--imgsz", type=int)
        parser.add_argument("--conf", type=float)
        parser.add_argument("--infer-every", type=int)
        parser.add_argument("--jpeg-quality", type=int)
        parser.add_argument("--max-det", type=int)
        namespace = parser.parse_args(args)
        if namespace.source == "opencv":
            launcher = root / "scripts/local/opencv.sh"
        mappings = {
            "DEVICE": namespace.device,
            "PIPELINE": namespace.pipeline,
            "MODEL": namespace.model,
            "HOST": namespace.host,
            "PORT": namespace.port,
            "IMGSZ": namespace.imgsz,
            "CONF": namespace.conf,
            "INFER_EVERY": namespace.infer_every,
            "JPEG_QUALITY": namespace.jpeg_quality,
            "MAX_DET": namespace.max_det,
        }
        if namespace.source == "realsense" and not namespace.pipeline:
            device = namespace.device or "/dev/video0"
            mappings["PIPELINE"] = (
                f"v4l2src device={device} io-mode=2 ! image/jpeg,width=640,height=480,framerate=30/1 "
                "! jpegdec ! videoconvert ! videoscale ! video/x-raw,width=640,height=360,format=BGR "
                "! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream "
                "! appsink sync=false drop=true max-buffers=1"
            )
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
    role_targets = {"scripts/robot/start.sh", "scripts/gb10/start.sh", "scripts/local/start.sh"}
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
        if route.target in {"scripts/robot/start.sh", "scripts/gb10/start.sh", "scripts/local/start.sh"}:
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

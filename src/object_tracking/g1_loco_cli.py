from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import json
import subprocess
import sys

from object_tracking.unitree_g1 import (
    G1LocoSdk2Client,
    UnitreeG1Error,
    UnitreeSdk2Context,
    normalize_network_interface,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send Unitree G1 commands via unitree-sdk2 Python package (version 1.0.1)."
    )
    parser.add_argument("robot_ip", nargs="?", help="Robot IP address.")
    parser.add_argument(
        "--robot-ip",
        dest="robot_ip_option",
        help="Robot IP address. Same as positional robot_ip.",
    )
    parser.add_argument(
        "--network-interface",
        "--interface",
        dest="network_interface",
        default="auto",
        help="Local DDS interface name, or 'auto' to resolve it from robot_ip. Default: auto.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain id for Unitree SDK2. Default: 0.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for each SDK command.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("stand_up", "balance_stand", "stop_move", "damp", "move", "move_arms_up"),
        help="G1 loco command to send. Omit and pass --diagnose for diagnostics only.",
    )
    parser.add_argument(
        "--velocity",
        type=str,
        metavar='"VX VY OMEGA [DURATION]"',
        help="Required for command=move.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print SDK route/proxy diagnostics and exit.",
    )
    parser.add_argument(
        "--dds-probe",
        action="store_true",
        help="Initialize DDS and probe Unitree topic creation without sending robot commands.",
    )
    return parser


def _parse_velocity(raw: str) -> tuple[float, float, float, float | None]:
    parts = raw.split()
    if len(parts) not in (3, 4):
        raise ValueError("Expected --velocity as 'vx vy omega' or 'vx vy omega duration'.")
    vx, vy, omega = (float(part) for part in parts[:3])
    duration = float(parts[3]) if len(parts) == 4 else None
    return vx, vy, omega, duration


def _diagnose_route(robot_ip: str | None) -> dict[str, object]:
    if robot_ip is None:
        return {"ok": False, "error": "robot_ip not provided"}
    try:
        completed = subprocess.run(
            ["ip", "route", "get", robot_ip],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip(),
            "raw": completed.stdout.strip(),
        }

    tokens = completed.stdout.split()
    interface: str | None = None
    if "dev" in tokens:
        idx = tokens.index("dev") + 1
        if idx < len(tokens):
            interface = tokens[idx]

    return {
        "ok": True,
        "raw": completed.stdout.strip(),
        "interface": interface,
    }


def _package_version(*names: str) -> dict[str, object]:
    errors: dict[str, str] = {}
    for name in names:
        try:
            return {"available": True, "package": name, "version": importlib.metadata.version(name)}
        except Exception as exc:
            errors[name] = str(exc)
    return {"available": False, "errors": errors}


def _module_info(module_name: str) -> dict[str, object]:
    try:
        module = __import__(module_name, fromlist=["__name__"])
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "file": getattr(module, "__file__", None),
    }


def _g1_loco_info() -> dict[str, object]:
    try:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.g1.loco.g1_loco_api import LOCO_SERVICE_NAME, LOCO_API_VERSION
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "client": f"{LocoClient.__module__}.{LocoClient.__name__}",
        "service_name": LOCO_SERVICE_NAME,
        "api_version": LOCO_API_VERSION,
        "backend": "g1_loco",
    }


def _probe_one_topic(name: str, type_path: str) -> dict[str, object]:
    module_name, attr_name = type_path.rsplit(":", 1)
    try:
        from unitree_sdk2py.core.channel import ChannelPublisher

        module = __import__(module_name, fromlist=[attr_name])
        topic_type = getattr(module, attr_name)
        publisher = ChannelPublisher(name, topic_type)
        publisher.Init()
    except Exception as exc:
        return {
            "ok": False,
            "topic": name,
            "type": type_path,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    return {"ok": True, "topic": name, "type": type_path}


def _dds_probe(
    robot_ip: str | None,
    network_interface: str | None,
    domain_id: int,
) -> None:
    if robot_ip is None and normalize_network_interface(network_interface) is None:
        print("Pass robot_ip/--robot-ip or --interface/--network-interface for DDS probe.", file=sys.stderr)
        raise SystemExit(1)

    try:
        resolved_interface = UnitreeSdk2Context.initialize(
            robot_ip=robot_ip,
            network_interface=network_interface,
            domain_id=domain_id,
        )
    except UnitreeG1Error as exc:
        report = {
            "ok": False,
            "stage": "dds_initialize",
            "robot_ip": robot_ip,
            "requested_interface": network_interface or "auto",
            "domain_id": domain_id,
            "error": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from exc

    unique_suffix = f"{os.getpid()}"
    probes = [
        (
            f"rt/unitree_probe/request_{unique_suffix}",
            "unitree_sdk2py.idl.unitree_api.msg.dds_:Request_",
        ),
        (
            "rt/api/sport/request",
            "unitree_sdk2py.idl.unitree_api.msg.dds_:Request_",
        ),
        (
            f"rt/unitree_probe/lowcmd_{unique_suffix}",
            "unitree_sdk2py.idl.unitree_hg.msg.dds_:LowCmd_",
        ),
    ]
    results = [_probe_one_topic(name, type_path) for name, type_path in probes]
    print(
        json.dumps(
            {
                "ok": all(item["ok"] for item in results),
                "robot_ip": robot_ip,
                "requested_interface": network_interface or "auto",
                "resolved_interface": resolved_interface,
                "domain_id": domain_id,
                "probes": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _diagnose_client(
    robot_ip: str | None,
    network_interface: str | None,
    timeout_s: float,
    domain_id: int,
) -> None:
    if robot_ip is None and network_interface is None:
        print("Pass robot_ip/--robot-ip or --network-interface to run diagnostics.", file=sys.stderr)
        raise SystemExit(1)

    route = _diagnose_route(robot_ip)
    explicit_interface = normalize_network_interface(network_interface)
    resolved_interface = explicit_interface
    if route.get("ok") and route.get("interface") and explicit_interface is None:
        resolved_interface = route["interface"]

    report = {
        "backend": "g1_loco",
        "robot_ip": robot_ip,
        "timeout_s": timeout_s,
        "domain_id": domain_id,
        "requested_interface": network_interface or "auto",
        "resolved_interface": resolved_interface,
        "route": route,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "CYCLONEDDS_URI": os.environ.get("CYCLONEDDS_URI"),
            "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"),
            "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION"),
        },
        "packages": {
            "unitree_sdk2": _package_version("unitree-sdk2", "unitree-sdk2py"),
            "cyclonedds": _package_version("cyclonedds"),
        },
        "modules": {
            "unitree_sdk2py": _module_info("unitree_sdk2py"),
            "cyclonedds": _module_info("cyclonedds"),
        },
        "g1_loco": _g1_loco_info(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    robot_ip = args.robot_ip_option or args.robot_ip

    if args.diagnose:
        _diagnose_client(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            timeout_s=args.timeout,
            domain_id=args.domain_id,
        )
        return

    if args.dds_probe:
        _dds_probe(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            domain_id=args.domain_id,
        )
        return

    if args.command is None:
        print(
            "Missing command. Use one of: stand_up, balance_stand, stop_move, damp, move, move_arms_up.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if robot_ip is None and normalize_network_interface(args.network_interface) is None:
        print("Pass robot_ip/--robot-ip, or --interface/--network-interface.", file=sys.stderr)
        raise SystemExit(1)

    if args.command == "move":
        if args.velocity is None:
            print('move requires --velocity "vx vy omega [duration]"', file=sys.stderr)
            raise SystemExit(1)
        try:
            vx, vy, omega, duration = _parse_velocity(args.velocity)
        except ValueError as exc:
            print(f"Invalid --velocity: {exc}", file=sys.stderr)
            raise SystemExit(1)

    try:
        client = G1LocoSdk2Client(
            network_interface=args.network_interface,
            robot_ip=robot_ip,
            timeout_s=args.timeout,
            domain_id=args.domain_id,
        )
        if args.command == "move":
            result = client.move(vx, vy, omega, duration)
        elif args.command == "move_arms_up":
            print("Moving arms to the forward test pose. Press Ctrl-C to stop holding.", file=sys.stderr)
            result = client.move_arms_up()
        else:
            result = client.command(args.command)
    except UnitreeG1Error as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        print("Tip: run `uv run loco --diagnose <ip>` to check network + Python SDK setup.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

from __future__ import annotations

import argparse
import importlib.metadata
import os
import pathlib
import platform
import json
import signal
import subprocess
import sys

from object_tracking.unitree_g1 import (
    DDS_CONFIG_MODE_CHOICES,
    DEFAULT_DDS_CONFIG_MODE,
    G1LocoSdk2Client,
    Go2SportSdk2Client,
    LOCO_SERVICE_CHOICES,
    MotionSwitcherSdk2Client,
    UnitreeG1Error,
    UnitreeSdk2Context,
    default_cyclonedds_log_file,
    effective_loco_service_name,
    loco_rpc_request_topic,
    normalize_network_interface,
    patch_g1_loco_service_name,
    patch_unitree_cyclonedds_config,
)


COMMAND_CHOICES = (
    "stand_up",
    "balance_stand",
    "stop_move",
    "damp",
    "move",
    "smooth_move",
    "move_arms_up",
    "calibrate_arms",
    "probe_loco",
    "check_motion_mode",
    "select_ai_mode",
    "diagnose",
    "smoke_move",
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
        "--backend",
        choices=("g1_loco", "g1_loco_minimal", "go2_sport"),
        default="g1_loco",
        help="Unitree SDK backend. g1_loco is default; go2_sport is explicit debugging only.",
    )
    parser.add_argument(
        "--loco-service-name",
        choices=LOCO_SERVICE_CHOICES,
        default="auto",
        help="G1 loco RPC service name. auto prefers ai_sport. Use sport only when explicitly needed.",
    )
    parser.add_argument(
        "--cyclonedds-log-file",
        default=None,
        help="Writable CycloneDDS trace log path. Default: /tmp/unitree_cdds_<uid>_<pid>.log.",
    )
    parser.add_argument(
        "--dds-config-mode",
        choices=DDS_CONFIG_MODE_CHOICES,
        default=DEFAULT_DDS_CONFIG_MODE,
        help="CycloneDDS config patch mode. Use subprocess smoke tests before changing movement commands.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMAND_CHOICES,
        help="G1 loco command to send. Omit and pass --diagnose for diagnostics only.",
    )
    parser.add_argument(
        "--velocity",
        type=str,
        metavar='"VX VY OMEGA [DURATION]"',
        help="Required for command=move or smooth_move.",
    )
    parser.add_argument(
        "--move-ramp-s",
        type=float,
        default=0.25,
        help="For smooth_move: seconds to ramp velocity up/down. Default: 0.25.",
    )
    parser.add_argument(
        "--arm-scale",
        type=float,
        default=1.0,
        help="For move_arms_up: fraction of the forward arm target to use. Default: 1.0.",
    )
    parser.add_argument(
        "--arm-ramp-s",
        type=float,
        default=2.0,
        help="For move_arms_up: seconds to ramp to the target. Default: 2.0.",
    )
    parser.add_argument(
        "--arm-hold-s",
        type=float,
        default=None,
        help="For move_arms_up: seconds to hold target before returning. Default: hold until Ctrl-C.",
    )
    parser.add_argument(
        "--arm-kp",
        type=float,
        default=None,
        help="For move_arms_up/calibrate_arms: low-level position gain.",
    )
    parser.add_argument(
        "--arm-kd",
        type=float,
        default=None,
        help="For move_arms_up/calibrate_arms: low-level damping gain.",
    )
    parser.add_argument(
        "--arm-max-step",
        type=float,
        default=None,
        help="For move_arms_up/calibrate_arms: max radians to move each controlled joint per control tick.",
    )
    parser.add_argument(
        "--arm-error-tolerance",
        type=float,
        default=None,
        help="For move_arms_up/calibrate_arms: target error threshold in radians for early settle.",
    )
    parser.add_argument(
        "--arm-settle-s",
        type=float,
        default=None,
        help="For move_arms_up/calibrate_arms: seconds inside tolerance before returning when hold_s is set.",
    )
    parser.add_argument(
        "--joint-target",
        action="append",
        default=[],
        metavar="JOINT_ID:RAD",
        help="For calibration: absolute low-level joint target in radians. Repeat for fingers or arm overrides.",
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
    parser.add_argument(
        "--smoke-loco",
        action="store_true",
        help="Initialize DDS, patch the G1 loco service name, construct LocoClient, then exit.",
    )
    parser.add_argument(
        "--smoke-loco-subprocess",
        action="store_true",
        help="Run the project-free LocoClient smoke test in a subprocess so native aborts are reported.",
    )
    parser.add_argument(
        "--smoke-loco-config-sweep",
        action="store_true",
        help="Run subprocess smoke tests once per DDS config mode.",
    )
    return parser


def _parse_velocity(raw: str) -> tuple[float, float, float, float | None]:
    parts = raw.split()
    if len(parts) not in (3, 4):
        raise ValueError("Expected --velocity as 'vx vy omega' or 'vx vy omega duration'.")
    vx, vy, omega = (float(part) for part in parts[:3])
    duration = float(parts[3]) if len(parts) == 4 else None
    return vx, vy, omega, duration


def _parse_joint_targets(raw_targets: list[str]) -> dict[int, float]:
    targets: dict[int, float] = {}
    for raw in raw_targets:
        if ":" not in raw:
            raise ValueError(f"Expected joint target as JOINT_ID:RAD, got {raw!r}.")
        raw_joint, raw_value = raw.split(":", 1)
        try:
            joint_id = int(raw_joint.strip())
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid joint target {raw!r}; expected integer id and float radians.") from exc
        targets[joint_id] = value
    return targets


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


def _sdk_path_report() -> dict[str, object]:
    imported_file: str | None = None
    try:
        import unitree_sdk2py

        imported_file = getattr(unitree_sdk2py, "__file__", None)
    except Exception:
        imported_file = None

    sys_path_unitree = [entry for entry in sys.path if "unitree" in entry.lower()]
    candidates: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = pathlib.Path(entry) / "unitree_sdk2py"
        if candidate.exists():
            candidates.append(str(candidate.resolve()))

    home_checkout = pathlib.Path.home() / "unitree_sdk2_python" / "unitree_sdk2py"
    if home_checkout.exists():
        candidates.append(str(home_checkout.resolve()))

    if imported_file:
        imported_path = pathlib.Path(imported_file).resolve().parent
        candidates.append(str(imported_path))

    unique_candidates = sorted(set(candidates))
    return {
        "imported_file": imported_file,
        "metadata": _package_version("unitree-sdk2py", "unitree-sdk2"),
        "sys_path_entries_containing_unitree": sys_path_unitree,
        "candidate_sdk_paths": unique_candidates,
        "multiple_sdk_copies_detected": len(unique_candidates) > 1,
        "warning": (
            "Multiple unitree_sdk2py copies are visible; remove duplicate installs or sys.path entries."
            if len(unique_candidates) > 1
            else None
        ),
    }


def _g1_loco_info(loco_service_name: str) -> dict[str, object]:
    try:
        import unitree_sdk2py.g1.loco.g1_loco_client as g1_loco_client
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.g1.loco.g1_loco_api import LOCO_SERVICE_NAME, LOCO_API_VERSION
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    effective_service = effective_loco_service_name(loco_service_name)
    return {
        "available": True,
        "client": f"{LocoClient.__module__}.{LocoClient.__name__}",
        "detected_api_service_name": LOCO_SERVICE_NAME,
        "detected_client_service_name": getattr(g1_loco_client, "LOCO_SERVICE_NAME", None),
        "requested_service_name": loco_service_name,
        "effective_service_name": effective_service,
        "rpc_request_topic": loco_rpc_request_topic(effective_service),
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
    loco_service_name: str,
    cyclonedds_log_file: str | None,
    dds_config_mode: str,
) -> None:
    if robot_ip is None and normalize_network_interface(network_interface) is None:
        print("Pass robot_ip/--robot-ip or --interface/--network-interface for DDS probe.", file=sys.stderr)
        raise SystemExit(1)

    try:
        resolved_interface = UnitreeSdk2Context.initialize(
            robot_ip=robot_ip,
            network_interface=network_interface,
            domain_id=domain_id,
            cyclonedds_log_file=cyclonedds_log_file,
            dds_config_mode=dds_config_mode,
        )
    except UnitreeG1Error as exc:
        report = {
            "ok": False,
            "stage": "dds_initialize",
            "robot_ip": robot_ip,
            "requested_interface": network_interface or "auto",
            "domain_id": domain_id,
            "dds_config_mode": dds_config_mode,
            "error": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from exc

    unique_suffix = f"{os.getpid()}"
    effective_service = effective_loco_service_name(loco_service_name)
    probes = [
        (
            f"rt/unitree_probe/request_{unique_suffix}",
            "unitree_sdk2py.idl.unitree_api.msg.dds_:Request_",
        ),
        (
            loco_rpc_request_topic(effective_service),
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
                "requested_loco_service_name": loco_service_name,
                "effective_loco_service_name": effective_service,
                "loco_rpc_request_topic": loco_rpc_request_topic(effective_service),
                "probes": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _construct_g1_loco_minimal(
    robot_ip: str | None,
    network_interface: str | None,
    domain_id: int,
    loco_service_name: str,
    cyclonedds_log_file: str | None,
    dds_config_mode: str,
) -> None:
    resolved_interface = UnitreeSdk2Context.initialize(
        robot_ip=robot_ip,
        network_interface=network_interface,
        domain_id=domain_id,
        cyclonedds_log_file=cyclonedds_log_file,
        dds_config_mode=dds_config_mode,
    )
    try:
        LocoClient, service_report = patch_g1_loco_service_name(loco_service_name)
    except Exception as exc:
        raise UnitreeG1Error(f"Could not import and patch G1 LocoClient: {exc}") from exc

    print(
        f"Constructing G1 LocoClient service={service_report['effective_service_name']} "
        f"topic={service_report['rpc_request_topic']}",
        file=sys.stderr,
    )
    try:
        LocoClient()
    except Exception as exc:
        raise UnitreeG1Error(
            "Minimal G1 LocoClient construction failed. This bypasses the project movement wrapper, "
            "so fix the Unitree SDK/CycloneDDS/Python environment first.\n"
            f"DDS domain: {domain_id}\n"
            f"DDS interface: {resolved_interface}\n"
            f"Loco service: {service_report['effective_service_name']}\n"
            f"DDS request topic: {service_report['rpc_request_topic']}\n"
            f"Original error: {exc}"
        ) from exc

    print(
        json.dumps(
            {
                "ok": True,
                "backend": "g1_loco_minimal",
                "domain_id": domain_id,
                "resolved_interface": resolved_interface,
                "loco_service": service_report,
                "sdk_paths": _sdk_path_report(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _signal_name(returncode: int) -> str | None:
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def _smoke_loco_subprocess(
    robot_ip: str | None,
    network_interface: str | None,
    domain_id: int,
    loco_service_name: str,
    cyclonedds_log_file: str | None,
    dds_config_mode: str,
    timeout_s: float,
) -> dict[str, object]:
    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "unitree_loco_minimal.py"
    if not script.exists():
        print(f"Minimal smoke script not found: {script}", file=sys.stderr)
        raise SystemExit(1)

    log_file = cyclonedds_log_file or default_cyclonedds_log_file()
    command = [
        sys.executable,
        str(script),
        "--domain-id",
        str(domain_id),
        "--loco-service-name",
        loco_service_name,
        "--cyclonedds-log-file",
        log_file,
        "--dds-config-mode",
        dds_config_mode,
    ]
    if robot_ip:
        command.extend(["--robot-ip", robot_ip])
    resolved_interface = normalize_network_interface(network_interface)
    if resolved_interface is not None:
        command.extend(["--interface", resolved_interface])
    elif robot_ip is None:
        print("Pass robot_ip/--robot-ip or --interface/--network-interface for subprocess smoke test.", file=sys.stderr)
        raise SystemExit(1)

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 5.0) + 5.0,
        )
    except subprocess.TimeoutExpired as exc:
        report = {
            "ok": False,
            "stage": "subprocess_timeout",
            "command": command,
            "timeout_s": exc.timeout,
            "dds_config_mode": dds_config_mode,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from exc

    report = {
        "ok": completed.returncode == 0,
        "stage": "subprocess_smoke_loco",
        "command": command,
        "returncode": completed.returncode,
        "signal": _signal_name(completed.returncode),
        "dds_config_mode": dds_config_mode,
        "cyclonedds_log_file": log_file,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    return report


def _print_subprocess_smoke_report(report: dict[str, object]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
    returncode = int(report.get("returncode", 0) or 0)
    if returncode != 0:
        if returncode < 0:
            print(
                "Subprocess was terminated by native code before Python could catch an exception. "
                "That points below this repo: CycloneDDS, unitree_sdk2py, or the linked C layer.",
                file=sys.stderr,
            )
        raise SystemExit(1)


def _smoke_loco_config_sweep(
    robot_ip: str | None,
    network_interface: str | None,
    domain_id: int,
    loco_service_name: str,
    cyclonedds_log_file: str | None,
    timeout_s: float,
) -> None:
    reports = []
    for mode in DDS_CONFIG_MODE_CHOICES:
        log_file = cyclonedds_log_file
        if log_file is None:
            uid = os.getuid() if hasattr(os, "getuid") else "user"
            log_file = f"/tmp/unitree_cdds_{uid}_{os.getpid()}_{mode}.log"
        reports.append(
            _smoke_loco_subprocess(
                robot_ip=robot_ip,
                network_interface=network_interface,
                domain_id=domain_id,
                loco_service_name=loco_service_name,
                cyclonedds_log_file=log_file,
                dds_config_mode=mode,
                timeout_s=timeout_s,
            )
        )
    print(json.dumps({"ok": all(item["ok"] for item in reports), "reports": reports}, indent=2, sort_keys=True))
    if not all(item["ok"] for item in reports):
        raise SystemExit(1)


def _diagnose_client(
    robot_ip: str | None,
    network_interface: str | None,
    timeout_s: float,
    domain_id: int,
    loco_service_name: str,
    cyclonedds_log_file: str | None,
    dds_config_mode: str,
) -> None:
    if robot_ip is None and network_interface is None:
        print("Pass robot_ip/--robot-ip or --network-interface to run diagnostics.", file=sys.stderr)
        raise SystemExit(1)

    route = _diagnose_route(robot_ip)
    explicit_interface = normalize_network_interface(network_interface)
    resolved_interface = explicit_interface
    if route.get("ok") and route.get("interface") and explicit_interface is None:
        resolved_interface = route["interface"]

    effective_service = effective_loco_service_name(loco_service_name)
    config_report = patch_unitree_cyclonedds_config(cyclonedds_log_file, dds_config_mode)
    report = {
        "backend": "g1_loco",
        "robot_ip": robot_ip,
        "timeout_s": timeout_s,
        "domain_id": domain_id,
        "dds_config_mode": dds_config_mode,
        "requested_loco_service_name": loco_service_name,
        "effective_loco_service_name": effective_service,
        "loco_rpc_request_topic": loco_rpc_request_topic(effective_service),
        "cyclonedds_config": config_report,
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
        "g1_loco": _g1_loco_info(loco_service_name),
        "sdk_paths": _sdk_path_report(),
        "topic_creation_plan": {
            "g1_loco": [
                f"Constructing G1 LocoClient service={effective_service}",
                f"G1 LocoClient creates {loco_rpc_request_topic(effective_service)} through unitree_api.msg.dds_.Request_",
                f"G1 LocoClient creates rt/api/{effective_service}/response through unitree_api.msg.dds_.Response_",
            ],
            "go2_sport": [
                "Constructing Go2 SportClient",
                "Go2 SportClient also uses rt/api/sport/request; only constructed with --backend go2_sport",
            ],
            "project_default": "Only g1_loco is constructed for stop_move unless --backend is changed.",
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_result_json(result_stdout: str) -> object:
    try:
        return json.loads(result_stdout)
    except Exception:
        return result_stdout


def _runtime_diagnose(
    robot_ip: str | None,
    network_interface: str | None,
    timeout_s: float,
    domain_id: int,
    loco_service_name: str,
    cyclonedds_log_file: str | None,
    dds_config_mode: str,
) -> None:
    effective_service = effective_loco_service_name(loco_service_name)
    report: dict[str, object] = {
        "ok": False,
        "interface": network_interface or "auto",
        "domain_id": domain_id,
        "dds_config_mode": dds_config_mode,
        "service_name": effective_service,
        "client": "unitree_sdk2py.g1.loco.g1_loco_client.LocoClient",
        "motion_switcher_client": "unitree_sdk2py.comm.motion_switcher.motion_switcher_client.MotionSwitcherClient",
        "sdk_imports": {
            "unitree_sdk2py": _module_info("unitree_sdk2py"),
            "cyclonedds": _module_info("cyclonedds"),
            "g1_loco": _g1_loco_info(loco_service_name),
        },
        "check_motion_mode": None,
        "loco_probe": None,
    }

    try:
        motion_client = MotionSwitcherSdk2Client(
            network_interface=network_interface,
            robot_ip=robot_ip,
            timeout_s=timeout_s,
            domain_id=domain_id,
            cyclonedds_log_file=cyclonedds_log_file,
            dds_config_mode=dds_config_mode,
        )
        report["check_motion_mode"] = _load_result_json(motion_client.check_mode().stdout)
    except Exception as exc:
        report["check_motion_mode"] = {"ok": False, "exception": repr(exc)}

    try:
        loco_client = G1LocoSdk2Client(
            network_interface=network_interface,
            robot_ip=robot_ip,
            timeout_s=timeout_s,
            domain_id=domain_id,
            loco_service_name=loco_service_name,
            cyclonedds_log_file=cyclonedds_log_file,
            dds_config_mode=dds_config_mode,
        )
        report["loco_probe"] = _load_result_json(loco_client.probe_loco().stdout)
    except Exception as exc:
        report["loco_probe"] = {"ok": False, "exception": repr(exc)}

    report["ok"] = any(
        isinstance(item, dict) and bool(item.get("ok"))
        for item in (report["check_motion_mode"], report["loco_probe"])
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command is None and args.robot_ip in COMMAND_CHOICES:
        args.command = args.robot_ip
        args.robot_ip = None
    robot_ip = args.robot_ip_option or args.robot_ip

    if args.diagnose:
        _diagnose_client(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            timeout_s=args.timeout,
            domain_id=args.domain_id,
            loco_service_name=args.loco_service_name,
            cyclonedds_log_file=args.cyclonedds_log_file,
            dds_config_mode=args.dds_config_mode,
        )
        return

    if args.smoke_loco:
        try:
            _construct_g1_loco_minimal(
                robot_ip=robot_ip,
                network_interface=args.network_interface,
                domain_id=args.domain_id,
                loco_service_name=args.loco_service_name,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
        except UnitreeG1Error as exc:
            print(f"Command failed: {exc}", file=sys.stderr)
            print(
                "Tip: try `uv run loco --smoke-loco <ip> --interface <dev> --loco-service-name ai_sport`.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        return

    if args.smoke_loco_config_sweep:
        _smoke_loco_config_sweep(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            domain_id=args.domain_id,
            loco_service_name=args.loco_service_name,
            cyclonedds_log_file=args.cyclonedds_log_file,
            timeout_s=args.timeout,
        )
        return

    if args.smoke_loco_subprocess:
        _print_subprocess_smoke_report(
            _smoke_loco_subprocess(
                robot_ip=robot_ip,
                network_interface=args.network_interface,
                domain_id=args.domain_id,
                loco_service_name=args.loco_service_name,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
                timeout_s=args.timeout,
            )
        )
        return

    if args.dds_probe:
        _dds_probe(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            domain_id=args.domain_id,
            loco_service_name=args.loco_service_name,
            cyclonedds_log_file=args.cyclonedds_log_file,
            dds_config_mode=args.dds_config_mode,
        )
        return

    if args.backend == "g1_loco_minimal":
        try:
            _construct_g1_loco_minimal(
                robot_ip=robot_ip,
                network_interface=args.network_interface,
                domain_id=args.domain_id,
                loco_service_name=args.loco_service_name,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
        except UnitreeG1Error as exc:
            print(f"Command failed: {exc}", file=sys.stderr)
            print("Tip: run `uv run python scripts/unitree_loco_minimal.py --interface <dev>` for a project-free check.", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    if args.command is None:
        print(
            "Missing command. Use one of: stand_up, balance_stand, stop_move, damp, move, smooth_move, move_arms_up, "
            "calibrate_arms, probe_loco, check_motion_mode, select_ai_mode, diagnose, smoke_move.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if robot_ip is None and normalize_network_interface(args.network_interface) is None:
        print("Pass robot_ip/--robot-ip, or --interface/--network-interface.", file=sys.stderr)
        raise SystemExit(1)

    if args.command in ("move", "smooth_move"):
        if args.velocity is None:
            print(f'{args.command} requires --velocity "vx vy omega [duration]"', file=sys.stderr)
            raise SystemExit(1)
        try:
            vx, vy, omega, duration = _parse_velocity(args.velocity)
        except ValueError as exc:
            print(f"Invalid --velocity: {exc}", file=sys.stderr)
            raise SystemExit(1)

    try:
        joint_targets = _parse_joint_targets(args.joint_target)
    except ValueError as exc:
        print(f"Invalid --joint-target: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.command == "diagnose":
        _runtime_diagnose(
            robot_ip=robot_ip,
            network_interface=args.network_interface,
            timeout_s=args.timeout,
            domain_id=args.domain_id,
            loco_service_name=args.loco_service_name,
            cyclonedds_log_file=args.cyclonedds_log_file,
            dds_config_mode=args.dds_config_mode,
        )
        return

    try:
        if args.command in ("check_motion_mode", "select_ai_mode"):
            client = MotionSwitcherSdk2Client(
                network_interface=args.network_interface,
                robot_ip=robot_ip,
                timeout_s=args.timeout,
                domain_id=args.domain_id,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
            if args.command == "select_ai_mode":
                print(
                    "Selecting Unitree motion mode 'ai'. Keep the robot physically safe and controller/e-stop ready.",
                    file=sys.stderr,
                )
                result = client.select_mode("ai")
            else:
                result = client.check_mode()
        elif args.command == "smoke_move":
            motion_client = MotionSwitcherSdk2Client(
                network_interface=args.network_interface,
                robot_ip=robot_ip,
                timeout_s=args.timeout,
                domain_id=args.domain_id,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
            check_result = _load_result_json(motion_client.check_mode().stdout)
            if not (isinstance(check_result, dict) and check_result.get("ok")):
                raise UnitreeG1Error(
                    "Refusing smoke_move because check_motion_mode did not return ok.\n"
                    f"check_motion_mode: {check_result}"
                )
            select_result = _load_result_json(motion_client.select_mode("ai").stdout)
            if not (isinstance(select_result, dict) and select_result.get("ok")):
                raise UnitreeG1Error(
                    "Refusing smoke_move because select_ai_mode did not return ok.\n"
                    f"select_ai_mode: {select_result}"
                )
            client = G1LocoSdk2Client(
                network_interface=args.network_interface,
                robot_ip=robot_ip,
                timeout_s=args.timeout,
                domain_id=args.domain_id,
                loco_service_name=args.loco_service_name,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
            probe_result = _load_result_json(client.probe_loco().stdout)
            if not (isinstance(probe_result, dict) and probe_result.get("ok")):
                raise UnitreeG1Error(
                    "Refusing smoke_move because probe_loco did not return ok.\n"
                    f"probe_loco: {probe_result}"
                )
            result = client.move(0.05, 0.0, 0.0, 0.3)
        elif args.backend == "go2_sport":
            client = Go2SportSdk2Client(
                network_interface=args.network_interface,
                robot_ip=robot_ip,
                timeout_s=args.timeout,
                domain_id=args.domain_id,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
        else:
            client = G1LocoSdk2Client(
                network_interface=args.network_interface,
                robot_ip=robot_ip,
                timeout_s=args.timeout,
                domain_id=args.domain_id,
                loco_service_name=args.loco_service_name,
                cyclonedds_log_file=args.cyclonedds_log_file,
                dds_config_mode=args.dds_config_mode,
            )
        if args.command in ("check_motion_mode", "select_ai_mode", "smoke_move"):
            pass
        elif args.command == "move":
            result = client.move(vx, vy, omega, duration)
        elif args.command == "smooth_move":
            result = client.smooth_move(vx, vy, omega, duration or 0.5, args.move_ramp_s)
        elif args.command in ("move_arms_up", "calibrate_arms"):
            if args.arm_hold_s is None:
                print(
                    f"Running {args.command}. Press Ctrl-C to stop holding.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Running {args.command} with scale={args.arm_scale}, "
                    f"ramp_s={args.arm_ramp_s}, hold_s={args.arm_hold_s}.",
                    file=sys.stderr,
                )
            if args.command == "calibrate_arms":
                result = client.calibrate_arms(
                    hold_s=1.0 if args.arm_hold_s is None else args.arm_hold_s,
                    scale=args.arm_scale,
                    ramp_s=args.arm_ramp_s,
                    joint_targets=joint_targets,
                    kp=18.0 if args.arm_kp is None else args.arm_kp,
                    kd=1.2 if args.arm_kd is None else args.arm_kd,
                    max_step=0.025 if args.arm_max_step is None else args.arm_max_step,
                    error_tolerance=0.025 if args.arm_error_tolerance is None else args.arm_error_tolerance,
                    settle_s=0.4 if args.arm_settle_s is None else args.arm_settle_s,
                )
            else:
                result = client.move_arms_up(
                    hold_s=args.arm_hold_s,
                    scale=args.arm_scale,
                    ramp_s=args.arm_ramp_s,
                    joint_targets=joint_targets,
                    kp=25.0 if args.arm_kp is None else args.arm_kp,
                    kd=1.0 if args.arm_kd is None else args.arm_kd,
                    max_step=0.04 if args.arm_max_step is None else args.arm_max_step,
                    error_tolerance=0.03 if args.arm_error_tolerance is None else args.arm_error_tolerance,
                    settle_s=0.25 if args.arm_settle_s is None else args.arm_settle_s,
                )
        else:
            result = client.command(args.command)
    except UnitreeG1Error as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        print("Tip: run `uv run loco --diagnose <ip>` to check network + Python SDK setup.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

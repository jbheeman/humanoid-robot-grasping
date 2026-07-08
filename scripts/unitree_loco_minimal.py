#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import traceback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal Unitree G1 LocoClient constructor smoke test with no project imports."
    )
    parser.add_argument("--interface", "--network-interface", dest="network_interface")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--robot-ip", help="Robot IP used only for route/interface diagnostics.")
    parser.add_argument(
        "--loco-service-name",
        choices=("auto", "sport", "ai_sport"),
        default="auto",
        help="G1 loco RPC service name. auto prefers ai_sport.",
    )
    parser.add_argument(
        "--cyclonedds-log-file",
        default=None,
        help="Writable CycloneDDS trace log path. Default: runs/dds/cdds_<pid>.log.",
    )
    return parser


def effective_loco_service_name(raw: str | None) -> str:
    if raw in (None, "", "auto"):
        return "ai_sport"
    return raw


def loco_rpc_request_topic(service_name: str) -> str:
    return f"rt/api/{service_name}/request"


def patch_loco_service_name(raw_service_name: str) -> dict[str, object]:
    service_name = effective_loco_service_name(raw_service_name)
    api_module = importlib.import_module("unitree_sdk2py.g1.loco.g1_loco_api")
    detected_api_service = getattr(api_module, "LOCO_SERVICE_NAME", None)
    api_module.LOCO_SERVICE_NAME = service_name

    client_module = importlib.import_module("unitree_sdk2py.g1.loco.g1_loco_client")
    detected_client_service = getattr(client_module, "LOCO_SERVICE_NAME", None)
    client_module.LOCO_SERVICE_NAME = service_name

    return {
        "requested_service_name": raw_service_name,
        "effective_service_name": service_name,
        "detected_api_service_name": detected_api_service,
        "detected_client_service_name": detected_client_service,
        "rpc_request_topic": loco_rpc_request_topic(service_name),
    }


def default_cyclonedds_log_file() -> str:
    return str((pathlib.Path.cwd() / "runs" / "dds" / f"cdds_{os.getpid()}.log").resolve())


def patch_unitree_cyclonedds_log_file(log_file: str | None) -> dict[str, object]:
    target = log_file or default_cyclonedds_log_file()
    target_path = pathlib.Path(target).expanduser()
    if not target_path.is_absolute():
        target_path = pathlib.Path.cwd() / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        channel_module = importlib.import_module("unitree_sdk2py.core.channel")
        config_module = importlib.import_module("unitree_sdk2py.core.channel_config")
    except Exception as exc:
        return {
            "log_file": str(target_path),
            "available": False,
            "error": repr(exc),
            "replacements": {},
        }
    replacements: dict[str, bool] = {}
    for module in (config_module, channel_module):
        for attr in ("ChannelConfigHasInterface", "ChannelConfigAutoDetermine"):
            if not hasattr(module, attr):
                continue
            value = getattr(module, attr)
            if not isinstance(value, str):
                continue
            updated = value.replace("/tmp/cdds.LOG", str(target_path))
            setattr(module, attr, updated)
            replacements[f"{module.__name__}.{attr}"] = updated != value

    return {"log_file": str(target_path), "available": True, "replacements": replacements}


def package_version(*names: str) -> dict[str, object]:
    errors: dict[str, str] = {}
    for name in names:
        try:
            return {"available": True, "package": name, "version": importlib.metadata.version(name)}
        except Exception as exc:
            errors[name] = str(exc)
    return {"available": False, "errors": errors}


def route_for_robot(robot_ip: str | None) -> dict[str, object]:
    if not robot_ip:
        return {"ok": False, "error": "--robot-ip not provided"}
    try:
        completed = subprocess.run(
            ["ip", "route", "get", robot_ip],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    result: dict[str, object] = {
        "ok": completed.returncode == 0,
        "raw": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    tokens = completed.stdout.split()
    if "dev" in tokens:
        idx = tokens.index("dev") + 1
        if idx < len(tokens):
            result["interface"] = tokens[idx]
    return result


def sdk_path_report() -> dict[str, object]:
    imported_file = None
    try:
        import unitree_sdk2py

        imported_file = getattr(unitree_sdk2py, "__file__", None)
    except Exception as exc:
        imported_file = f"import failed: {exc!r}"

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

    if imported_file and not imported_file.startswith("import failed:"):
        candidates.append(str(pathlib.Path(imported_file).resolve().parent))

    unique_candidates = sorted(set(candidates))
    return {
        "unitree_sdk2py_file": imported_file,
        "unitree_sdk2py_metadata": package_version("unitree-sdk2py", "unitree-sdk2"),
        "sys_path_entries_containing_unitree": sys_path_unitree,
        "candidate_sdk_paths": unique_candidates,
        "multiple_sdk_copies_detected": len(unique_candidates) > 1,
        "warning": (
            "Multiple unitree_sdk2py copies are visible; remove duplicate installs/sys.path entries."
            if len(unique_candidates) > 1
            else None
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    route = route_for_robot(args.robot_ip)
    interface = args.network_interface or route.get("interface")

    report = {
        "python": {"executable": sys.executable, "version": sys.version},
        "domain_id": args.domain_id,
        "requested_loco_service_name": args.loco_service_name,
        "effective_loco_service_name": effective_loco_service_name(args.loco_service_name),
        "loco_rpc_request_topic": loco_rpc_request_topic(effective_loco_service_name(args.loco_service_name)),
        "cyclonedds_log": patch_unitree_cyclonedds_log_file(args.cyclonedds_log_file),
        "requested_interface": args.network_interface,
        "robot_ip": args.robot_ip,
        "route": route,
        "resolved_interface": interface,
        "sdk_paths": sdk_path_report(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not interface:
        print("ERROR: pass --interface or --robot-ip so the local DDS interface can be resolved", file=sys.stderr)
        return 2

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        log_report = patch_unitree_cyclonedds_log_file(args.cyclonedds_log_file)
        print(f"using CycloneDDS log file {log_report['log_file']}", file=sys.stderr)
        print(f"initializing DDS domain={args.domain_id} interface={interface}", file=sys.stderr)
        ChannelFactoryInitialize(args.domain_id, str(interface))

        service_report = patch_loco_service_name(args.loco_service_name)
        client_module = importlib.import_module("unitree_sdk2py.g1.loco.g1_loco_client")
        LocoClient = client_module.LocoClient
        print(
            f"Constructing G1 LocoClient service={service_report['effective_service_name']} "
            f"topic={service_report['rpc_request_topic']}",
            file=sys.stderr,
        )
        LocoClient()
    except Exception as exc:
        print("FAILURE: minimal G1 LocoClient construction failed", file=sys.stderr)
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("SUCCESS: minimal G1 LocoClient constructed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

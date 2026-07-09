#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only Unitree SDK2 RPC probes across one or more DDS interfaces."
    )
    parser.add_argument(
        "--robot-ip",
        dest="legacy_robot_ip",
        default=None,
        help="Deprecated alias for --ping-ip. DDS probes use interface/domain, not this IP.",
    )
    parser.add_argument(
        "--ping-ip",
        action="append",
        default=None,
        help="Optional diagnostic IP to ping. May be passed multiple times.",
    )
    parser.add_argument("--interface", default=None, help="Single DDS interface to test.")
    parser.add_argument("--interfaces", default=None, help="Comma-separated DDS interfaces to test, e.g. eth0,wlan0.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--control-peer", default="192.168.123.1")
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable for child SDK probes. Default: .venv/bin/python3 if present, else this interpreter.",
    )
    parser.add_argument("--skip-ping", action="store_true")
    return parser


def run_command(command: list[str], timeout_s: float = 20.0, env: dict[str, str] | None = None) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "exception": repr(exc),
        }
    report: dict[str, object] = {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    try:
        report["stdout_json"] = json.loads(completed.stdout)
    except Exception:
        pass
    return report


def default_child_python(repo_root: pathlib.Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def collect_codes(value: object) -> list[int]:
    codes: list[int] = []
    if isinstance(value, dict):
        code = value.get("code")
        if isinstance(code, int):
            codes.append(code)
        for child in value.values():
            codes.extend(collect_codes(child))
    elif isinstance(value, list):
        for child in value:
            codes.extend(collect_codes(child))
    return codes


def probe_rpc_ok(result: dict[str, object]) -> bool:
    stdout_json = result.get("stdout_json")
    return isinstance(stdout_json, dict) and bool(stdout_json.get("ok"))


def probe_codes(result: dict[str, object]) -> list[int]:
    return collect_codes(result.get("stdout_json"))


def classify_interface(probes: dict[str, dict[str, object]]) -> dict[str, object]:
    check = probes.get("check_motion_mode", {})
    sport = probes.get("probe_loco_sport", {})
    ai_sport = probes.get("probe_loco_ai_sport", {})
    all_codes = [code for result in probes.values() for code in probe_codes(result)]
    any_ok = any(probe_rpc_ok(result) for result in probes.values())

    if any_ok:
        if probe_rpc_ok(check) and not probe_rpc_ok(sport) and not probe_rpc_ok(ai_sport):
            diagnosis = "MotionSwitcher reachable, but both loco services failed. Service name/API may be wrong."
        else:
            diagnosis = "At least one SDK RPC service responded on this interface."
    elif all_codes and all(code == 3102 for code in all_codes):
        diagnosis = "All SDK RPC calls returned 3102. This points to request sending/network/interface failure."
    elif all_codes and 3103 in all_codes:
        diagnosis = "One or more SDK RPC calls returned 3103. This points to API not registered or wrong service name."
    elif all_codes and 3104 in all_codes:
        diagnosis = "One or more SDK RPC calls returned 3104. This points to timeout/discovery/service unavailable."
    else:
        diagnosis = "No SDK RPC success code found. Inspect subprocess stderr/stdout for packaging or DDS errors."

    return {
        "ok": any_ok,
        "codes": sorted(set(all_codes)),
        "diagnosis": diagnosis,
    }


def repo_import_check(python: str, env: dict[str, str]) -> dict[str, object]:
    return run_command(
        [
            python,
            "-c",
            "import object_tracking; print(object_tracking.__file__)",
        ],
        timeout_s=5.0,
        env=env,
    )


def import_report() -> dict[str, object]:
    report: dict[str, object] = {}
    for module_name in ("unitree_sdk2py", "cyclonedds"):
        try:
            module = __import__(module_name, fromlist=["__name__"])
        except Exception as exc:
            report[module_name] = {"available": False, "error": repr(exc)}
        else:
            report[module_name] = {
                "available": True,
                "file": getattr(module, "__file__", None),
            }
    return report


def main() -> int:
    args = build_parser().parse_args()
    script = pathlib.Path(__file__).resolve().with_name("g1_loco.py")
    repo_root = script.parents[1]
    child_python = args.python or default_child_python(repo_root)
    src_path = repo_root / "src"
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        str(src_path)
        if not existing_pythonpath
        else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    interfaces_raw = args.interfaces or args.interface or "eth0"
    interfaces = [item.strip() for item in interfaces_raw.split(",") if item.strip()]
    ping_ips = list(args.ping_ip or [])
    if args.legacy_robot_ip:
        ping_ips.append(args.legacy_robot_ip)

    report: dict[str, object] = {
        "interfaces": interfaces,
        "ping_ips": ping_ips,
        "domain_id": args.domain_id,
        "python": child_python,
        "parent_python": sys.executable,
        "repo_root": str(repo_root),
        "child_pythonpath": child_env["PYTHONPATH"],
        "imports": import_report(),
        "repo_import_check": repo_import_check(child_python, child_env),
        "route": run_command(["ip", "route"], timeout_s=5.0),
        "addr": run_command(["ip", "-br", "addr"], timeout_s=5.0),
        "interface_results": {},
    }

    if not report["repo_import_check"]["ok"]:
        report["diagnosis"] = "Repo packaging/import issue. DDS was not tested."
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    if not args.skip_ping:
        ping_targets = [args.control_peer, *ping_ips]
        report["pings"] = {
            target: run_command(["ping", "-c", "3", target], timeout_s=8.0)
            for target in ping_targets
        }

    probe_timeout = max(args.timeout + 10.0, 20.0)
    for interface in interfaces:
        base_loco = [
            child_python,
            str(script),
            "--interface",
            interface,
            "--domain-id",
            str(args.domain_id),
            "--timeout",
            str(args.timeout),
            "--dds-config-mode",
            "no_trace",
        ]
        report["interface_results"][interface] = {
            "neighbors": run_command(["ip", "neigh", "show", "dev", interface], timeout_s=5.0),
            "probes": {},
        }
        probes = report["interface_results"][interface]["probes"]
        probes["check_motion_mode"] = run_command(
            [*base_loco, "check_motion_mode"],
            timeout_s=probe_timeout,
            env=child_env,
        )
        probes["probe_loco_sport"] = run_command(
            [*base_loco, "probe_loco", "--loco-service-name", "sport"],
            timeout_s=probe_timeout,
            env=child_env,
        )
        probes["probe_loco_ai_sport"] = run_command(
            [*base_loco, "probe_loco", "--loco-service-name", "ai_sport"],
            timeout_s=probe_timeout,
            env=child_env,
        )
        report["interface_results"][interface]["classification"] = classify_interface(probes)

    ok_interfaces = [
        interface
        for interface, result in report["interface_results"].items()
        if result["classification"]["ok"]
    ]
    if not ok_interfaces:
        report["diagnosis"] = (
            "Import check passed, but no interface had a successful SDK RPC response. "
            "3102=request sending/network/interface, 3103=API not registered/wrong service, "
            "3104=timeout/discovery or service unavailable."
        )
    elif len(ok_interfaces) == 1:
        report["diagnosis"] = f"Use interface {ok_interfaces[0]} for Unitree SDK2 DDS RPC."
    else:
        report["diagnosis"] = f"Multiple interfaces had SDK RPC responses: {', '.join(ok_interfaces)}."

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

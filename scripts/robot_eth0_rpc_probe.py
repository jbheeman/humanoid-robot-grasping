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
        description="Run read-only Unitree SDK2 RPC probes from the robot internal eth0 network."
    )
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--control-peer", default="192.168.123.1")
    parser.add_argument("--python", default=sys.executable)
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
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


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
    src_path = repo_root / "src"
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        str(src_path)
        if not existing_pythonpath
        else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    base_loco = [
        args.python,
        str(script),
        args.robot_ip,
        "--interface",
        args.interface,
        "--domain-id",
        str(args.domain_id),
        "--timeout",
        str(args.timeout),
        "--dds-config-mode",
        "no_trace",
    ]

    report: dict[str, object] = {
        "robot_ip": args.robot_ip,
        "interface": args.interface,
        "domain_id": args.domain_id,
        "python": args.python,
        "repo_root": str(repo_root),
        "child_pythonpath": child_env["PYTHONPATH"],
        "imports": import_report(),
        "route": run_command(["ip", "route"], timeout_s=5.0),
        "probes": {},
    }

    if not args.skip_ping:
        report["ping_control_peer"] = run_command(["ping", "-c", "3", args.control_peer], timeout_s=8.0)
        report["ping_robot_ip"] = run_command(["ping", "-c", "3", args.robot_ip], timeout_s=8.0)

    probe_timeout = max(args.timeout + 10.0, 20.0)
    report["probes"]["check_motion_mode"] = run_command(
        [*base_loco, "check_motion_mode"],
        timeout_s=probe_timeout,
        env=child_env,
    )
    report["probes"]["probe_loco_ai_sport"] = run_command(
        [*base_loco, "probe_loco", "--loco-service-name", "ai_sport"],
        timeout_s=probe_timeout,
        env=child_env,
    )
    report["probes"]["probe_loco_sport"] = run_command(
        [*base_loco, "probe_loco", "--loco-service-name", "sport"],
        timeout_s=probe_timeout,
        env=child_env,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

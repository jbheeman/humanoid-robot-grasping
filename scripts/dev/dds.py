#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe CycloneDDS discovery without shell XML quoting.")
    parser.add_argument("--interfaces", default="enP7s7,wlP9s9", help="Comma-separated interfaces to test.")
    parser.add_argument("--runtime", type=int, default=20, help="ddsls runtime seconds.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--ddsls", default=None, help="Path to ddsls. Default: .venv/bin/ddsls if present.")
    parser.add_argument("--no-multicast", action="store_true", help="Disable explicit AllowMulticast=true.")
    return parser


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_ddsls() -> str:
    candidate = repo_root() / ".venv" / "bin" / "ddsls"
    if candidate.exists():
        return str(candidate)
    return "ddsls"


def cyclonedds_uri(interface: str, allow_multicast: bool) -> str:
    multicast = "<AllowMulticast>true</AllowMulticast>" if allow_multicast else ""
    return (
        "<CycloneDDS>"
        "<Domain Id=\"any\">"
        "<General>"
        f"{multicast}"
        "<Interfaces>"
        f"<NetworkInterface name=\"{interface}\"/>"
        "</Interfaces>"
        "</General>"
        "</Domain>"
        "</CycloneDDS>"
    )


def run_command(command: list[str], env: dict[str, str] | None = None, timeout_s: int = 30) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except Exception as exc:
        return {"ok": False, "command": command, "exception": repr(exc)}
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    args = build_parser().parse_args()
    interfaces = [item.strip() for item in args.interfaces.split(",") if item.strip()]
    ddsls = args.ddsls or default_ddsls()
    allow_multicast = not args.no_multicast
    timeout_s = max(args.runtime + 5, 10)

    report: dict[str, object] = {
        "ok": False,
        "ddsls": ddsls,
        "domain_id": args.domain_id,
        "runtime": args.runtime,
        "allow_multicast": allow_multicast,
        "python": sys.executable,
        "addr": run_command(["ip", "-br", "addr"], timeout_s=5),
        "route": run_command(["ip", "route"], timeout_s=5),
        "interfaces": {},
    }

    for interface in interfaces:
        env = os.environ.copy()
        env["CYCLONEDDS_URI"] = cyclonedds_uri(interface, allow_multicast)
        all_result = run_command(
            [ddsls, "-a", "-r", str(args.runtime), "-i", str(args.domain_id)],
            env=env,
            timeout_s=timeout_s,
        )
        publications = run_command(
            [ddsls, "-t", "dcpspublication", "-r", str(args.runtime), "-i", str(args.domain_id)],
            env=env,
            timeout_s=timeout_s,
        )
        subscriptions = run_command(
            [ddsls, "-t", "dcpssubscription", "-r", str(args.runtime), "-i", str(args.domain_id)],
            env=env,
            timeout_s=timeout_s,
        )
        visible = any(
            bool(result.get("stdout", "").strip())
            for result in (all_result, publications, subscriptions)
        )
        report["interfaces"][interface] = {
            "ok": visible,
            "cyclonedds_uri": env["CYCLONEDDS_URI"],
            "all": all_result,
            "publications": publications,
            "subscriptions": subscriptions,
        }

    ok_interfaces = [
        interface for interface, result in report["interfaces"].items() if result["ok"]
    ]
    report["ok"] = bool(ok_interfaces)
    report["diagnosis"] = (
        f"DDS discovery visible on: {', '.join(ok_interfaces)}"
        if ok_interfaces
        else (
            "No DDS discovery output on tested interfaces. Direct Unitree SDK2 DDS from this server "
            "will not reach robot RPC services on the current network; run a robot-side motion bridge "
            "and send HTTP commands from the server."
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

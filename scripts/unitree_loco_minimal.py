#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
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
    return parser


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
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        print(f"initializing DDS domain={args.domain_id} interface={interface}", file=sys.stderr)
        ChannelFactoryInitialize(args.domain_id, str(interface))
        print("Constructing G1 LocoClient", file=sys.stderr)
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

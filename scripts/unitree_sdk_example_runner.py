#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


G1_LOCO_ACTIONS = {
    "damp": "Damp",
    "balance_stand": "BalanceStand",
    "low_stand": "LowStand",
    "high_stand": "HighStand",
    "zero_torque": "ZeroTorque",
    "stop_move": "StopMove",
    "wave_hand": "WaveHand",
    "shake_hand": "ShakeHand",
    "move_forward_tiny": "Move",
    "move_lateral_tiny": "Move",
    "rotate_tiny": "Move",
}

G1_ARM_ACTIONS = (
    "release arm",
    "shake hand",
    "high five",
    "hug",
    "high wave",
    "clap",
    "face wave",
    "left kiss",
    "heart",
    "right heart",
    "hands up",
    "x-ray",
    "right hand up",
    "reject",
    "right kiss",
    "two-hand kiss",
)

MOTION_SWITCHER_ACTIONS = ("check_mode", "release_mode", "select_ai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Allowlisted Unitree SDK2 example runner for robot-local use.")
    parser.add_argument("example", choices=("list", "g1_loco", "g1_arm_action", "motion_switcher"))
    parser.add_argument("action", nargs="?", default="list")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dds-config-mode", default="no_trace")
    parser.add_argument("--loco-service-name", default="sport", choices=("auto", "sport", "ai_sport"))
    parser.add_argument("--duration", type=float, default=0.5, help="Duration for tiny Move examples.")
    parser.add_argument("--speed", type=float, default=0.1, help="Speed for tiny Move examples.")
    parser.add_argument("--smoke", action="store_true", help="List/validate without DDS or robot commands.")
    return parser


def normalize_action(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def action_display(value: str) -> str:
    return value.replace("_", " ")


def list_report() -> dict[str, object]:
    return {
        "ok": True,
        "examples": {
            "g1_loco": sorted(G1_LOCO_ACTIONS),
            "g1_arm_action": list(G1_ARM_ACTIONS),
            "motion_switcher": list(MOTION_SWITCHER_ACTIONS),
        },
        "source_examples": {
            "g1_loco": "example/g1/high_level/g1_loco_client_example.py",
            "g1_arm_action": "example/g1/high_level/g1_arm_action_example.py",
            "motion_switcher": "example/motionSwitcher/motion_switcher_example.py",
        },
    }


def print_json(report: dict[str, object]) -> int:
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


def initialize_dds(args: argparse.Namespace) -> None:
    from object_tracking.unitree_g1 import patch_unitree_cyclonedds_config
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    patch_unitree_cyclonedds_config(dds_config_mode=args.dds_config_mode)
    ChannelFactoryInitialize(args.domain_id, args.interface)


def unitree_result(value: object) -> dict[str, object]:
    ok = True
    code = None
    if isinstance(value, tuple) and value:
        try:
            code = int(value[0])
            ok = code == 0
        except Exception:
            code = None
    elif isinstance(value, int):
        code = int(value)
        ok = code == 0
    return {
        "ok": ok,
        "code": code,
        "raw": repr(value),
    }


def run_motion_switcher(args: argparse.Namespace) -> dict[str, object]:
    action = normalize_action(args.action)
    if action == "list":
        return {"ok": True, "example": "motion_switcher", "actions": list(MOTION_SWITCHER_ACTIONS)}
    if action not in MOTION_SWITCHER_ACTIONS:
        return {"ok": False, "error": f"Unknown motion_switcher action: {args.action}", "actions": list(MOTION_SWITCHER_ACTIONS)}
    if args.smoke:
        return {"ok": True, "smoke": True, "example": "motion_switcher", "action": action}

    initialize_dds(args)
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    client = MotionSwitcherClient()
    client.SetTimeout(args.timeout)
    init_result = client.Init()
    if action == "check_mode":
        result = client.CheckMode()
    elif action == "release_mode":
        result = client.ReleaseMode()
    elif action == "select_ai":
        result = client.SelectMode("ai")
    else:
        raise AssertionError(action)

    return {
        "ok": unitree_result(result)["ok"],
        "example": "motion_switcher",
        "action": action,
        "init": repr(init_result),
        "result": unitree_result(result),
    }


def run_g1_loco(args: argparse.Namespace) -> dict[str, object]:
    action = normalize_action(args.action)
    if action == "list":
        return {"ok": True, "example": "g1_loco", "actions": sorted(G1_LOCO_ACTIONS)}
    if action not in G1_LOCO_ACTIONS:
        return {"ok": False, "error": f"Unknown g1_loco action: {args.action}", "actions": sorted(G1_LOCO_ACTIONS)}
    if args.smoke:
        return {"ok": True, "smoke": True, "example": "g1_loco", "action": action}

    initialize_dds(args)
    from object_tracking.unitree_g1 import patch_g1_loco_service_name

    LocoClient, service_report = patch_g1_loco_service_name(args.loco_service_name)
    client = LocoClient()
    client.SetTimeout(args.timeout)
    init_result = client.Init()

    if action == "move_forward_tiny":
        result = client.Move(args.speed, 0.0, 0.0)
        time.sleep(args.duration)
        stop_result = client.StopMove()
    elif action == "move_lateral_tiny":
        result = client.Move(0.0, args.speed, 0.0)
        time.sleep(args.duration)
        stop_result = client.StopMove()
    elif action == "rotate_tiny":
        result = client.Move(0.0, 0.0, args.speed)
        time.sleep(args.duration)
        stop_result = client.StopMove()
    else:
        method = getattr(client, G1_LOCO_ACTIONS[action])
        result = method()
        stop_result = None

    report = {
        "ok": unitree_result(result)["ok"],
        "example": "g1_loco",
        "action": action,
        "service": service_report,
        "init": repr(init_result),
        "result": unitree_result(result),
    }
    if stop_result is not None:
        report["stop_result"] = unitree_result(stop_result)
    return report


def run_g1_arm_action(args: argparse.Namespace) -> dict[str, object]:
    action = action_display(normalize_action(args.action))
    if action == "list":
        return {"ok": True, "example": "g1_arm_action", "actions": list(G1_ARM_ACTIONS)}
    if action not in G1_ARM_ACTIONS:
        return {"ok": False, "error": f"Unknown g1_arm_action action: {args.action}", "actions": list(G1_ARM_ACTIONS)}
    if args.smoke:
        return {"ok": True, "smoke": True, "example": "g1_arm_action", "action": action}

    initialize_dds(args)
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

    action_id = action_map.get(action)
    if action_id is None:
        return {"ok": False, "error": f"Action not in SDK action_map: {action}"}

    client = G1ArmActionClient()
    client.SetTimeout(args.timeout)
    init_result = client.Init()
    result = client.ExecuteAction(action_id)
    return {
        "ok": unitree_result(result)["ok"],
        "example": "g1_arm_action",
        "action": action,
        "action_id": action_id,
        "init": repr(init_result),
        "result": unitree_result(result),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.example == "list":
        return print_json(list_report())
    if args.example == "motion_switcher":
        return print_json(run_motion_switcher(args))
    if args.example == "g1_loco":
        return print_json(run_g1_loco(args))
    if args.example == "g1_arm_action":
        return print_json(run_g1_arm_action(args))
    raise AssertionError(args.example)


if __name__ == "__main__":
    raise SystemExit(main())

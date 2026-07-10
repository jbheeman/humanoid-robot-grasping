#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
G1_LOCO_SCRIPT = REPO_ROOT / "src" / "object_tracking" / "g1_loco_cli.py"
SDK_EXAMPLE_SCRIPT = REPO_ROOT / "scripts" / "vendor" / "sdk_runner.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HTTP bridge for sending Unitree SDK2 commands from a server without server-side DDS."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765")
    parser.add_argument("--interface", default="wlan0", help="Robot-local DDS interface. Default: wlan0.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--dds-config-mode", default="no_trace")
    parser.add_argument("--loco-service-name", default="sport", choices=("auto", "sport", "ai_sport"))
    parser.add_argument("--python", default=None, help="Python executable for child SDK commands.")
    parser.add_argument(
        "--allow-movement",
        action="store_true",
        default=True,
        help="Allow endpoints that can move the robot: /move, /move_arms_up, and /smoke_move.",
    )
    parser.add_argument(
        "--read-only",
        action="store_false",
        dest="allow_movement",
        help="Disable endpoints that can move the robot.",
    )
    return parser


def default_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    raw_length = handler.headers.get("Content-Length")
    if not raw_length:
        return {}
    try:
        length = int(raw_length)
    except ValueError:
        return {}
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = args.python or default_python()
        self.started_at = time.time()
        self.command_lock = threading.Lock()

    def base_command(self) -> list[str]:
        return [
            self.python,
            str(G1_LOCO_SCRIPT),
            "--interface",
            self.args.interface,
            "--domain-id",
            str(self.args.domain_id),
            "--timeout",
            str(self.args.timeout),
            "--dds-config-mode",
            self.args.dds_config_mode,
        ]

    def run_loco(self, command: list[str], timeout_extra_s: float = 10.0) -> dict[str, object]:
        return self.run_child([*self.base_command(), *command], timeout_extra_s=timeout_extra_s)

    def run_child(self, full_command: list[str], timeout_extra_s: float = 10.0) -> dict[str, object]:
        timeout_s = max(self.args.timeout + timeout_extra_s, 20.0)
        with self.command_lock:
            try:
                completed = subprocess.run(
                    full_command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    env=child_env(),
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "ok": False,
                    "stage": "timeout",
                    "command": full_command,
                    "timeout_s": exc.timeout,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "stage": "spawn",
                    "command": full_command,
                    "exception": repr(exc),
                }

        report: dict[str, object] = {
            "ok": completed.returncode == 0,
            "command": full_command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        try:
            report["stdout_json"] = json.loads(completed.stdout)
        except Exception:
            pass
        return report

    def run_sdk_example(self, body: dict[str, object]) -> dict[str, object]:
        example = str(body.get("example", body.get("name", "list")))
        action = str(body.get("action", "list"))
        command = [
            self.python,
            str(SDK_EXAMPLE_SCRIPT),
            example,
            action,
            "--interface",
            self.args.interface,
            "--domain-id",
            str(self.args.domain_id),
            "--timeout",
            str(self.args.timeout),
            "--dds-config-mode",
            self.args.dds_config_mode,
            "--loco-service-name",
            self.args.loco_service_name,
        ]
        if "duration" in body:
            command.extend(["--duration", str(body["duration"])])
        if "speed" in body:
            command.extend(["--speed", str(body["speed"])])
        if body.get("smoke"):
            command.append("--smoke")
        return self.run_child(command, timeout_extra_s=float(body.get("timeout_extra_s", 20.0)))

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "bridge": "unitree_motion_bridge",
            "repo_root": str(REPO_ROOT),
            "python": self.python,
            "interface": self.args.interface,
            "domain_id": self.args.domain_id,
            "dds_config_mode": self.args.dds_config_mode,
            "loco_service_name": self.args.loco_service_name,
            "allow_movement": self.args.allow_movement,
            "uptime_s": round(time.time() - self.started_at, 3),
        }

    def movement_allowed(self) -> tuple[bool, dict[str, object] | None]:
        if self.args.allow_movement:
            return True, None
        return False, {"ok": False, "error": "Bridge is running read-only. Restart without --read-only to send movement commands."}

    def command_stop(self) -> dict[str, object]:
        return self.run_loco(
            [
                "stop_move",
                "--loco-service-name",
                self.args.loco_service_name,
            ]
        )

    def command_move(self, body: dict[str, object]) -> dict[str, object]:
        allowed, error = self.movement_allowed()
        if not allowed:
            return error or {"ok": False, "error": "Movement is disabled."}
        vx = float(body.get("vx", 0.0))
        vy = float(body.get("vy", 0.0))
        omega = float(body.get("omega", body.get("wz", 0.0)))
        duration = float(body.get("duration", body.get("duration_s", 0.3)))
        ramp_s = float(body.get("ramp_s", body.get("ramp", min(0.2, duration / 2.0))))
        smooth = bool(body.get("smooth", True))
        velocity = f"{vx} {vy} {omega} {duration}"
        command_name = "smooth_move" if smooth else "move"
        return self.run_loco(
            [
                command_name,
                "--velocity",
                velocity,
                "--loco-service-name",
                self.args.loco_service_name,
                "--move-ramp-s",
                str(ramp_s),
            ],
            timeout_extra_s=max(duration + 5.0, 10.0),
        )

    def command_estop(self) -> dict[str, object]:
        return self.command_stop()

    def command_arms_up(self, body: dict[str, object]) -> dict[str, object]:
        allowed, error = self.movement_allowed()
        if not allowed:
            return error or {"ok": False, "error": "Movement is disabled."}
        scale = float(body.get("scale", body.get("amount", 0.15)))
        ramp_s = float(body.get("ramp_s", body.get("ramp", 1.5)))
        hold_s = float(body.get("hold_s", body.get("hold", 1.0)))
        return self.run_loco(
            [
                "move_arms_up",
                "--loco-service-name",
                self.args.loco_service_name,
                "--arm-scale",
                str(scale),
                "--arm-ramp-s",
                str(ramp_s),
                "--arm-hold-s",
                str(hold_s),
            ],
            timeout_extra_s=max(ramp_s + hold_s + 5.0, 10.0),
        )

    def command_smoke_move(self) -> dict[str, object]:
        allowed, error = self.movement_allowed()
        if not allowed:
            return error or {"ok": False, "error": "Movement is disabled."}
        return self.run_loco(
            [
                "smoke_move",
                "--loco-service-name",
                self.args.loco_service_name,
            ],
            timeout_extra_s=20.0,
        )

    def dispatch_command(self, body: dict[str, object]) -> dict[str, object]:
        command = str(body.get("command", body.get("cmd", ""))).strip().lower().replace("-", "_")
        aliases = {
            "arms": "arms_up",
            "arm_up": "arms_up",
            "arms_up": "arms_up",
            "move_arms_up": "arms_up",
            "forward": "forward",
            "move": "move",
            "stop": "stop",
            "stop_move": "stop",
            "smoke_move": "smoke_move",
        }
        command = aliases.get(command, command)
        if command == "arms_up":
            return self.command_arms_up(body)
        if command == "stop":
            return self.command_stop()
        if command == "smoke_move":
            return self.command_smoke_move()
        if command == "forward":
            command_body = dict(body)
            command_body.setdefault("vx", body.get("speed", 0.05))
            command_body.setdefault("vy", 0.0)
            command_body.setdefault("omega", 0.0)
            command_body.setdefault("duration", body.get("duration", 0.3))
            return self.command_move(command_body)
        if command == "move":
            return self.command_move(body)
        return {"ok": False, "error": f"Unknown command: {command or '<missing>'}"}


def make_handler(bridge: Bridge) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "UnitreeMotionBridge/0.1"

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

        def send_json(self, status: int, payload: dict[str, object]) -> None:
            raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(200, bridge.health())
                return
            if path == "/diagnose":
                result = bridge.run_loco(
                    [
                        "diagnose",
                        "--loco-service-name",
                        bridge.args.loco_service_name,
                    ],
                    timeout_extra_s=20.0,
                )
                self.send_json(200 if result["ok"] else 502, result)
                return
            if path == "/probe_loco":
                result = bridge.run_loco(
                    [
                        "probe_loco",
                        "--loco-service-name",
                        bridge.args.loco_service_name,
                    ]
                )
                self.send_json(200 if result["ok"] else 502, result)
                return
            if path == "/check_motion_mode":
                result = bridge.run_loco(["check_motion_mode"])
                self.send_json(200 if result["ok"] else 502, result)
                return
            if path == "/sdk/examples":
                result = bridge.run_sdk_example({"example": "list", "action": "list"})
                self.send_json(200 if result["ok"] else 502, result)
                return
            self.send_json(404, {"ok": False, "error": f"Unknown GET path: {path}"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = parse_json_body(self)
            except Exception as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})
                return

            if path == "/stop_move":
                result = bridge.command_stop()
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path in ("/estop", "/stop"):
                result = bridge.command_estop()
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path == "/select_ai_mode":
                result = bridge.run_loco(["select_ai_mode"])
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path in ("/sdk/example", "/sdk/run-example"):
                result = bridge.run_sdk_example(body)
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path in ("/command", "/cmd"):
                try:
                    result = bridge.dispatch_command(body)
                except (TypeError, ValueError) as exc:
                    self.send_json(400, {"ok": False, "error": f"Invalid command body: {exc}"})
                    return
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path == "/move":
                try:
                    result = bridge.command_move(body)
                except (TypeError, ValueError) as exc:
                    self.send_json(400, {"ok": False, "error": f"Invalid movement body: {exc}"})
                    return
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path in ("/move_arms_up", "/arms/up"):
                try:
                    result = bridge.command_arms_up(body)
                except (TypeError, ValueError) as exc:
                    self.send_json(400, {"ok": False, "error": f"Invalid arm movement body: {exc}"})
                    return
                self.send_json(200 if result["ok"] else 502, result)
                return

            if path == "/smoke_move":
                result = bridge.command_smoke_move()
                self.send_json(200 if result["ok"] else 502, result)
                return

            self.send_json(404, {"ok": False, "error": f"Unknown POST path: {path}"})

    return Handler


def main() -> int:
    args = build_parser().parse_args()
    bridge = Bridge(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge))
    print(
        json.dumps(
            {
                "ok": True,
                "listening": f"http://{args.host}:{args.port}",
                "interface": args.interface,
                "domain_id": args.domain_id,
                "dds_config_mode": args.dds_config_mode,
                "loco_service_name": args.loco_service_name,
                "python": bridge.python,
                "repo_root": str(REPO_ROOT),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

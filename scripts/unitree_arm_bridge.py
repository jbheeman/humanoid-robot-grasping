#!/usr/bin/env python3
"""Authenticated robot-local REST bridge for a persistent ``rt/arm_sdk`` publisher."""

from __future__ import annotations

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sys
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from object_tracking.arm_tracking.arm_auth import bearer_is_valid, load_token  # noqa: E402
from object_tracking.arm_tracking.arm_bridge import (  # noqa: E402
    ArmBridgeConfig,
    ArmBridgeController,
    ArmBridgeError,
)
from object_tracking.arm_tracking.arm_unitree import UnitreeArmHardware  # noqa: E402
from object_tracking.arm_tracking.calibration import load_calibration  # noqa: E402


MAX_BODY_BYTES = 64 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent authenticated Unitree G1 right-arm bridge"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument(
        "--allow-movement", action="store_true", help="Permit arming after all safety gates pass"
    )
    parser.add_argument("--calibration-id", default=os.environ.get("ARM_CALIBRATION_ID"))
    parser.add_argument(
        "--calibration", type=Path, help="Validated calibration YAML required for movement"
    )
    parser.add_argument("--token-env", default="ARM_BRIDGE_TOKEN")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--controller-lock", type=Path, default=Path("/tmp/unitree_arm_bridge.lock")
    )
    return parser


def acquire_controller_lock(path: Path) -> object:
    """Exclude a second bridge publisher on this robot for this process lifetime."""
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another arm bridge owns the controller lock: {path}") from exc
    return handle


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        return {}
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ArmBridgeError("Content-Length must be an integer") from exc
    if length < 0 or length > MAX_BODY_BYTES:
        raise ArmBridgeError(f"Request body must be no larger than {MAX_BODY_BYTES} bytes")
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArmBridgeError("Request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ArmBridgeError("Request JSON must be an object")
    return value


def make_handler(controller: ArmBridgeController, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "UnitreeArmBridge/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            # BaseHTTPRequestHandler never receives the Authorization value.
            print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

        def send_json(self, status: int, payload: dict[str, object]) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(200, controller.health_report())
            elif path == "/state":
                self.send_json(200, controller.state_report())
            else:
                self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if not bearer_is_valid(self.headers.get("Authorization"), token):
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            path = urlparse(self.path).path
            try:
                body = parse_json_body(self)
                if path == "/arm/enable":
                    report = controller.enable(
                        session_id=body.get("session_id"),
                        calibration_id=body.get("calibration_id"),
                    )
                elif path == "/arm/target":
                    report = controller.set_target(
                        session_id=body.get("session_id"),
                        sequence=body.get("sequence"),
                        calibration_id=body.get("calibration_id"),
                        right_arm_q=body.get("right_arm_q", body.get("joint_positions")),
                        source_timestamp=body.get("source_timestamp"),
                    )
                elif path == "/arm/stop":
                    report = controller.stop()
                else:
                    self.send_json(404, {"ok": False, "error": "not_found"})
                    return
            except ArmBridgeError as exc:
                self.send_json(
                    409 if exc.code in {"not_armed", "invalid_state"} else 400,
                    {"ok": False, "error": exc.code, "message": str(exc)},
                )
                return
            except Exception as exc:
                # Do not expose SDK internals or request secrets through HTTP.
                print(f"arm bridge request failed: {type(exc).__name__}", file=sys.stderr)
                controller.stop("request_failure")
                self.send_json(503, {"ok": False, "error": "bridge_failure"})
                return
            self.send_json(200, report)

    return Handler


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.allow_movement and args.calibration is None:
        parser.error("--allow-movement requires --calibration")
    try:
        calibration = None if args.calibration is None else load_calibration(args.calibration)
        calibration_id = (
            calibration.calibration_id if calibration is not None else args.calibration_id
        )
        token = load_token(env_name=args.token_env, token_file=args.token_file)
        controller_lock = acquire_controller_lock(args.controller_lock)
        config = ArmBridgeConfig(
            allow_movement=args.allow_movement,
            calibration_id=calibration_id,
            control_hz=args.control_hz,
            waist_reference_rad=(None if calibration is None else calibration.waist_reference_rad),
        )
        hardware = UnitreeArmHardware(interface=args.interface, domain_id=args.domain_id)
        controller = ArmBridgeController(hardware, config)
        controller.start()
    except Exception as exc:
        print(f"arm bridge startup failed: {exc}", file=sys.stderr)
        return 2

    server = ThreadingHTTPServer((args.host, args.port), make_handler(controller, token))
    server.daemon_threads = True

    def request_shutdown(signum: int, frame: object) -> None:
        del signum, frame
        # shutdown() must be called from a different thread than serve_forever;
        # setting the private flag through a signal callback is unsafe, so use a
        # short daemon thread.
        import threading

        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        json.dumps(
            {
                "ok": True,
                "listening": f"http://{args.host}:{args.port}",
                "interface": args.interface,
                "domain_id": args.domain_id,
                "control_hz": args.control_hz,
                "allow_movement": args.allow_movement,
                "calibration_configured": bool(calibration_id),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        controller.close()
        controller_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

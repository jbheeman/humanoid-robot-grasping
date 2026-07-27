#!/usr/bin/env python3
"""Native-SDK G1 arm worker, intentionally isolated from every ROS import."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socketserver
import stat
import threading
from typing import Any

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmBridgeError,
)
from object_tracking.arm_tracking.arm_unitree import UnitreeArmHardware
from object_tracking.arm_tracking.calibration import load_calibration


MAX_REQUEST_BYTES = 16 * 1024


def _safe_json(value: object) -> str:
    def finite(item: object) -> object:
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {str(key): finite(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(nested) for nested in item]
        return item

    return json.dumps(finite(value), separators=(",", ":"), sort_keys=True, allow_nan=False)


class NativeArmService:
    """Translate bounded local JSON requests into the existing safety controller."""

    def __init__(self, controller: ArmBridgeController) -> None:
        self.controller = controller

    def dispatch(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ArmBridgeError("Worker request must be a JSON object")
        operation = str(request.get("operation", "")).strip().lower()
        payload = request.get("payload", {})
        if not isinstance(payload, dict):
            raise ArmBridgeError("Worker request payload must be a JSON object")
        if operation == "enable":
            return self.controller.enable(
                session_id=payload.get("session_id"),
                calibration_id=payload.get("calibration_id"),
            )
        if operation == "heartbeat":
            return self.controller.heartbeat(session_id=payload.get("session_id"))
        if operation == "target":
            return self.controller.set_target(
                session_id=payload.get("session_id"),
                sequence=payload.get("sequence"),
                calibration_id=payload.get("calibration_id"),
                right_arm_q=payload.get("right_arm_q"),
                source_timestamp=payload.get("source_timestamp"),
                pipeline_age_ms=payload.get("pipeline_age_ms"),
            )
        if operation == "stop":
            return self.controller.stop(str(payload.get("reason") or "operator_stop"))
        if operation == "state":
            return self.controller.state_report()
        raise ArmBridgeError("Unknown native arm operation", code="invalid_request")


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(1.0)
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        request_id = ""
        try:
            if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                raise ArmBridgeError("Worker request is empty, oversized, or unterminated")
            request = json.loads(raw.decode("utf-8"))
            if isinstance(request, dict):
                request_id = str(request.get("request_id", ""))
            report = self.server.service.dispatch(request)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {
                "request_id": request_id,
                "ok": False,
                "error_code": str(getattr(exc, "code", "worker_failure")),
                "message": str(exc),
                "report": {},
            }
        else:
            response = {
                "request_id": request_id,
                "ok": True,
                "error_code": "",
                "message": "",
                "report": report,
            }
        self.wfile.write((_safe_json(response) + "\n").encode("utf-8"))


class NativeArmServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, socket_path: str, service: NativeArmService) -> None:
        self.service = service
        super().__init__(socket_path, _RequestHandler)


def _prepare_socket(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_socket():
        mode = path.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"Refusing to replace non-socket IPC path: {path}")
        path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS-free native Unitree arm worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--allow-movement", action="store_true")
    parser.add_argument("--expected-motion-mode")
    parser.add_argument("--hardware-interface", default="eth0")
    parser.add_argument("--hardware-domain-id", type=int, default=0)
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=5.0,
        help="maximum absolute IMU roll/pitch allowed before arm motion is blocked",
    )
    parser.add_argument(
        "--max-waist-deviation-deg",
        type=float,
        default=3.0,
        help="maximum dynamic waist deviation from the calibrated pose",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.allow_movement and not args.expected_motion_mode:
        raise SystemExit("--allow-movement requires --expected-motion-mode")
    if args.allow_movement and not args.calibration:
        raise SystemExit("--allow-movement requires --calibration")
    if not math.isfinite(args.max_tilt_deg) or not 1.0 <= args.max_tilt_deg <= 15.0:
        raise SystemExit("--max-tilt-deg must be between 1 and 15 degrees")
    if (
        not math.isfinite(args.max_waist_deviation_deg)
        or not 3.0 <= args.max_waist_deviation_deg <= 15.0
    ):
        raise SystemExit("--max-waist-deviation-deg must be between 3 and 15 degrees")
    calibration = None if args.calibration is None else load_calibration(args.calibration)
    hardware = UnitreeArmHardware(
        interface=args.hardware_interface,
        domain_id=args.hardware_domain_id,
        max_tilt_rad=math.radians(args.max_tilt_deg),
        expected_motion_mode=args.expected_motion_mode,
    )
    controller = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            allow_movement=args.allow_movement,
            calibration_id=None if calibration is None else calibration.calibration_id,
            waist_reference_rad=(
                None if calibration is None else calibration.waist_reference_rad
            ),
            # The standing controller naturally pitches the waist while the
            # arm's mass moves forward/up.  Use the operator-selected live
            # tilt envelope here too; the old fixed 3 degree gate rejected
            # normal balance compensation midway through the escape path.
            max_waist_deviation_rad=math.radians(args.max_waist_deviation_deg),
        ),
    )
    socket_path = Path(args.socket)
    _prepare_socket(socket_path)
    controller.start()
    server = NativeArmServer(str(socket_path), NativeArmService(controller))
    os.chmod(socket_path, 0o600)
    stopping = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(_safe_json({"ok": True, "worker": "native_arm", "socket": str(socket_path)}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        controller.close()
        if socket_path.is_socket():
            socket_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

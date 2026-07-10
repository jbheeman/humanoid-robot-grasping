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


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from object_tracking.arm_tracking.arm_auth import bearer_is_valid, load_token  # noqa: E402
from object_tracking.arm_tracking.arm_bridge import (  # noqa: E402
    ArmBridgeConfig,
    ArmBridgeController,
    ArmBridgeError,
    ArmControlMode,
)
from object_tracking.arm_tracking.arm_commissioning import (  # noqa: E402
    CommissioningConfig,
    CommissioningController,
)
from object_tracking.arm_tracking.arm_unitree import UnitreeArmHardware  # noqa: E402
from object_tracking.arm_tracking.calibration import load_calibration  # noqa: E402
from object_tracking.arm_tracking.joints import joint_contract_id  # noqa: E402


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
        "--control-mode",
        choices=tuple(mode.value for mode in ArmControlMode),
        default=ArmControlMode.TRACKING.value,
    )
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
    parser.add_argument("--joint-contract-id", default=joint_contract_id())
    parser.add_argument("--expected-motion-mode")
    parser.add_argument(
        "--commissioning-root", type=Path, default=Path("runs/research/arm_commissioning")
    )
    parser.add_argument(
        "--commissioning-profile",
        type=Path,
        default=Path.home() / ".config/g1-grasping/right-arm-home.json",
    )
    parser.add_argument("--robot-id", default=os.environ.get("G1_ROBOT_ID", ""))
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


def make_handler(
    controller: ArmBridgeController,
    token: str,
    commissioning: CommissioningController | None = None,
    *,
    wizard_path: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
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

        def send_html(self, payload: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_empty(self, status: int = 204) -> None:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def authorized(self) -> bool:
            return bearer_is_valid(self.headers.get("Authorization"), token)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/favicon.ico":
                self.send_empty()
            elif path in {"/commissioning", "/commissioning/"}:
                if wizard_path is None or not wizard_path.is_file():
                    self.send_json(404, {"ok": False, "error": "wizard_not_found"})
                else:
                    self.send_html(wizard_path.read_bytes())
            elif path == "/health":
                self.send_json(200, controller.health_report())
            elif path == "/state":
                self.send_json(200, controller.state_report())
            elif path.startswith("/api/v1/") and not self.authorized():
                self.send_json(401, {"ok": False, "error": "unauthorized"})
            elif path == "/api/v1/info":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "control_mode": controller.control_mode.value,
                        "commissioning_available": commissioning is not None,
                        "health": controller.health_report(),
                    },
                )
            elif path == "/api/v1/state":
                self.send_json(200, controller.state_report())
            elif path == "/api/v1/commissioning/state" and commissioning is not None:
                self.send_json(200, commissioning.report())
            else:
                self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if not self.authorized():
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
                elif path == "/api/v1/commissioning/sessions" and commissioning is not None:
                    report = commissioning.create_session(
                        operator_ack=body.get("operator_ack"),
                        operator=body.get("operator"),
                        client_id=body.get("client_id"),
                    )
                elif (
                    path.startswith("/api/v1/commissioning/sessions/") and commissioning is not None
                ):
                    report = self.dispatch_commissioning(path, body)
                else:
                    self.send_json(404, {"ok": False, "error": "not_found"})
                    return
            except ArmBridgeError as exc:
                if exc.code in {
                    "movement_disabled",
                    "mode_mismatch",
                    "operator_ack_required",
                }:
                    status = 403
                elif exc.code in {
                    "not_armed",
                    "invalid_state",
                    "session_conflict",
                    "motion_busy",
                    "stale_sequence",
                }:
                    status = 409
                elif exc.code in {
                    "robot_state_stale",
                    "motion_mode_unverified",
                    "controller_ownership_unverified",
                    "motor_status_unverified",
                }:
                    status = 503
                elif exc.code in {
                    "not_standing",
                    "standing_not_stable",
                    "motor_state_fault",
                    "stage_limit",
                    "session_limit",
                    "sign_checks_incomplete",
                    "sign_measurement_mismatch",
                }:
                    status = 422
                else:
                    status = 400
                self.send_json(
                    status,
                    {
                        "ok": False,
                        "error": exc.code,
                        "message": str(exc),
                        "retryable": status in {409, 503},
                        "state": controller.state.value,
                        "session_id": controller.session_id,
                    },
                )
                return
            except Exception as exc:
                # Do not expose SDK internals or request secrets through HTTP.
                print(f"arm bridge request failed: {type(exc).__name__}", file=sys.stderr)
                controller.stop("request_failure")
                self.send_json(503, {"ok": False, "error": "bridge_failure"})
                return
            self.send_json(200, report)

        def dispatch_commissioning(self, path: str, body: dict[str, object]) -> dict[str, object]:
            prefix = "/api/v1/commissioning/sessions/"
            remainder = path[len(prefix) :]
            session_id, separator, action = remainder.partition("/")
            if not session_id or not separator:
                raise ArmBridgeError("Session action is required", code="invalid_request")
            assert commissioning is not None
            if action == "enable":
                return commissioning.enable(session_id)
            if action == "heartbeat":
                return commissioning.heartbeat(session_id)
            if action == "jogs":
                return commissioning.jog(
                    session_id,
                    sequence=body.get("sequence"),
                    joint_name=body.get("joint_name"),
                    direction=body.get("direction"),
                    kind=str(body.get("kind") or "jog"),
                )
            if action == "confirmations":
                return commissioning.confirm_motion(
                    session_id,
                    sequence=body.get("sequence"),
                    outcome=body.get("outcome"),
                    notes=body.get("notes", ""),
                )
            if action == "checkpoints":
                return commissioning.checkpoint(session_id, label=body.get("label"))
            if action == "candidate":
                return commissioning.capture_candidate(session_id, label=body.get("label"))
            if action == "replay-steps":
                return commissioning.replay_step(
                    session_id,
                    sequence=body.get("sequence"),
                )
            if action == "replay-validations":
                return commissioning.validate_replay(session_id)
            if action == "stop":
                return commissioning.stop(
                    session_id, reason=str(body.get("reason") or "operator_stop")
                )
            if action == "promote":
                return commissioning.promote(session_id)
            raise ArmBridgeError("Unknown commissioning action", code="invalid_request")

    return Handler


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    control_mode = ArmControlMode(args.control_mode)
    if args.allow_movement and control_mode is ArmControlMode.TRACKING and args.calibration is None:
        parser.error("--allow-movement requires --calibration")
    if args.allow_movement and not args.expected_motion_mode:
        parser.error("--allow-movement requires --expected-motion-mode")
    try:
        calibration = None if args.calibration is None else load_calibration(args.calibration)
        calibration_id = (
            calibration.calibration_id if calibration is not None else args.calibration_id
        )
        token = load_token(env_name=args.token_env, token_file=args.token_file)
        controller_lock = acquire_controller_lock(args.controller_lock)
        config = ArmBridgeConfig(
            control_mode=control_mode,
            allow_movement=args.allow_movement,
            calibration_id=calibration_id,
            joint_contract_id=(
                args.joint_contract_id if control_mode is ArmControlMode.COMMISSIONING else None
            ),
            control_hz=args.control_hz,
            waist_reference_rad=(None if calibration is None else calibration.waist_reference_rad),
            max_velocity_rad_s=(0.10 if control_mode is ArmControlMode.COMMISSIONING else 0.50),
            max_acceleration_rad_s2=(0.50 if control_mode is ArmControlMode.COMMISSIONING else 2.0),
            max_following_error_rad=(
                0.05 if control_mode is ArmControlMode.COMMISSIONING else 0.35
            ),
        )
        hardware = UnitreeArmHardware(
            interface=args.interface,
            domain_id=args.domain_id,
            expected_motion_mode=args.expected_motion_mode,
        )
        controller = ArmBridgeController(hardware, config)
        controller.start()
        commissioning = (
            CommissioningController(
                controller,
                CommissioningConfig(
                    research_root=args.commissioning_root,
                    profile_path=args.commissioning_profile,
                ),
                robot_identity={"robot_id": args.robot_id, "model": "g1-29dof"},
            )
            if control_mode is ArmControlMode.COMMISSIONING
            else None
        )
    except Exception as exc:
        print(f"arm bridge startup failed: {exc}", file=sys.stderr)
        return 2

    wizard_path = REPO_ROOT / "scripts" / "robot" / "web" / "arm_commissioning.html"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(controller, token, commissioning, wizard_path=wizard_path),
    )
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
                "control_mode": control_mode.value,
                "calibration_configured": bool(calibration_id),
                "joint_contract_id": args.joint_contract_id,
                "commissioning_url": (
                    f"http://{args.host}:{args.port}/commissioning/"
                    if commissioning is not None
                    else None
                ),
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

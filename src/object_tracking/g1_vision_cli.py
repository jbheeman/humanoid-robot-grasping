from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import cv2
import numpy as np

from object_tracking.manual_tracker import open_camera, opencv_gstreamer_enabled, parse_bbox, run
from object_tracking.unitree_g1 import UnitreeG1Error, g1_camera_candidates, get_sdk2_video_sample, is_gstreamer_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read camera data from a G1 vision stream.")
    parser.add_argument("robot_ip", help="Robot IP address.")
    parser.add_argument(
        "ssh_login",
        nargs="?",
        help=(
            "Optional SSH login target. Can be user@host (for example unitree@192.168.0.4) "
            "or just host (for example 192.168.0.4)."
        ),
    )
    parser.add_argument(
        "ssh_host_arg",
        nargs="?",
        help=(
            "Optional SSH host when passing username and host as separate arguments: "
            "`vision <robot_ip> <ssh_user> <ssh_host>`."
        ),
    )
    parser.add_argument(
        "--camera-url",
        help="Explicit camera source. May be a gstreamer pipeline or URL. {ip} is substituted for plain IP templates.",
    )
    parser.add_argument(
        "--sdk2-video",
        action="store_true",
        help="Read one visual frame through Unitree SDK2 videohub GetImageSample instead of URL/GStreamer probing.",
    )
    parser.add_argument(
        "--sdk2-interface",
        help="Network interface for SDK2 video. Defaults to route-derived interface for robot_ip.",
    )
    parser.add_argument(
        "--sdk2-timeout",
        type=float,
        default=3.0,
        help="SDK2 video request timeout in seconds.",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        metavar='"X Y WIDTH HEIGHT"',
        help="Initial tracking box. If omitted, vision only saves a snapshot and exits.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many tracked frames.",
    )
    parser.add_argument(
        "--output",
        default="runs/vision_test",
        help="Directory for camera logs, snapshots, and tracking output.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=1.5,
        help="Timeout in seconds for socket reachability checks during diagnostics.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a GUI window for ROI selection/display.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable annotated MP4 recording.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print candidate camera URLs and exit.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run network + camera URL diagnostics and exit.",
    )
    parser.add_argument(
        "--ssh",
        action="store_true",
        default=True,
        help="Enable SSH probing and RTSP forwarding fallback (default: true). Has no effect for GStreamer/UDP candidates.",
    )
    parser.add_argument(
        "--no-ssh",
        action="store_true",
        help="Disable SSH probing/fallback.",
    )
    parser.add_argument(
        "--ssh-host",
        help="SSH host to use for tunnel. Defaults to robot_ip when omitted.",
    )
    parser.add_argument(
        "--ssh-user",
        help="SSH user for tunnel. Defaults to local user / system ssh config if omitted.",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port for tunnel (default: 22).",
    )
    parser.add_argument(
        "--ssh-key",
        help="Optional SSH private key path for tunnel.",
    )
    parser.add_argument(
        "--ssh-remote-host",
        help="Override host inside ssh tunnel (default: resolved camera host).",
    )
    return parser


def _route_info(robot_ip: str) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["ip", "route", "get", robot_ip],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"robot_ip": robot_ip, "ok": False, "error": str(exc)}

    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return {
            "robot_ip": robot_ip,
            "ok": False,
            "error": completed.stderr.strip(),
            "raw": completed.stdout.strip(),
        }

    tokens = completed.stdout.split()
    route: dict[str, object] = {
        "robot_ip": robot_ip,
        "ok": True,
        "raw": completed.stdout.strip(),
        "interface": None,
        "via": None,
        "src": None,
    }
    if "dev" in tokens:
        idx = tokens.index("dev") + 1
        if idx < len(tokens):
            route["interface"] = tokens[idx]
    if "via" in tokens:
        idx = tokens.index("via") + 1
        if idx < len(tokens):
            route["via"] = tokens[idx]
    if "src" in tokens:
        idx = tokens.index("src") + 1
        if idx < len(tokens):
            route["src"] = tokens[idx]
    return route


def _apply_ssh_positionals(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.ssh_host_arg is not None and args.ssh_login is None:
        parser.error("ssh_host provided without a preceding ssh_login argument (user).")

    parsed_user: str | None = None
    parsed_host: str | None = None

    if args.ssh_login is not None:
        if "@" in args.ssh_login and args.ssh_host_arg is not None:
            parser.error("cannot pass both user@host and a separate ssh_host")
        if "@" in args.ssh_login:
            parsed_user, parsed_host = args.ssh_login.rsplit("@", 1)
            if not parsed_user or not parsed_host:
                parser.error("ssh_login must be in the form user@host.")
        elif args.ssh_host_arg is not None:
            parsed_user = args.ssh_login
            parsed_host = args.ssh_host_arg
        else:
            parsed_host = args.ssh_login

    if parsed_user is not None:
        if args.ssh_user is not None and args.ssh_user != parsed_user:
            parser.error(
                "ssh user specified in positional arguments does not match --ssh-user"
            )
        args.ssh_user = parsed_user

    if parsed_host is not None:
        if args.ssh_host is not None and args.ssh_host != parsed_host:
            parser.error(
                "ssh host specified in positional arguments does not match --ssh-host"
            )
        args.ssh_host = parsed_host


def _probe_camera_url(camera_url: str, timeout_s: float) -> dict[str, object]:
    if is_gstreamer_pipeline(camera_url):
        host_match = re.search(r"address=([^\s!]+)", camera_url)
        port_match = re.search(r"port=(\d+)", camera_url)
        if port_match is None:
            return {
                "url": camera_url,
                "ok": False,
                "transport": "gstreamer",
                "error": "Could not parse UDP port from gstreamer source string",
            }

        bind_host = host_match.group(1) if host_match is not None else "0.0.0.0"
        port = int(port_match.group(1))
        has_gstreamer = opencv_gstreamer_enabled()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(timeout_s)
                sock.bind((bind_host, port))
            bind_ok = True
            bind_result = "local UDP bind available"
        except OSError as exc:
            bind_ok = False
            bind_result = str(exc)

        return {
            "url": camera_url,
            "ok": has_gstreamer and bind_ok,
            "transport": "gstreamer_udp",
            "bind_host": bind_host,
            "port": port,
            "opencv_gstreamer": has_gstreamer,
            "result": bind_result,
            "note": "UDP/GStreamer must be run on the host that receives the robot stream; SSH tunneling does not carry this default transport.",
        }

    parsed = urlparse(camera_url)
    host = parsed.hostname
    if host is None:
        return {
            "url": camera_url,
            "ok": False,
            "error": "Could not parse hostname.",
        }

    defaults = {"rtsp": 554, "http": 80, "https": 443}
    port = parsed.port
    if port is None:
        port = defaults.get(parsed.scheme)
    if port is None:
        return {
            "url": camera_url,
            "ok": False,
            "host": host,
            "error": f"Unknown scheme for implicit port: {parsed.scheme!r}",
        }

    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return {
                "url": camera_url,
                "ok": True,
                "transport": parsed.scheme,
                "host": host,
                "port": int(port),
                "result": "connected",
            }
    except OSError as exc:
        return {
            "url": camera_url,
            "ok": False,
            "transport": parsed.scheme,
            "host": host,
            "port": int(port),
            "result": str(exc),
        }


def _build_local_rtsp_url(remote_url: str, local_port: int) -> str:
    parsed = urlparse(remote_url)
    suffix = parsed.path or ""
    if parsed.query:
        suffix = f"{suffix}?{parsed.query}"
    return f"rtsp://127.0.0.1:{local_port}{suffix}"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ssh_target(args: argparse.Namespace) -> tuple[str, str | None]:
    target_host = args.ssh_host or args.robot_ip
    return target_host, args.ssh_user


def _ssh_login_ok(target_host: str, user: str | None, port: int, key_path: str | None) -> tuple[bool, str]:
    target = f"{user + '@' if user else ''}{target_host}"
    command = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=4",
        "-o",
        "PasswordAuthentication=no",
        target,
        "true",
    ]
    if key_path:
        command.insert(1, "-i")
        command.insert(2, str(key_path))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode == 0:
        return True, "ssh reachable"
    return (
        False,
        (completed.stderr.strip() or completed.stdout.strip() or "ssh returned non-zero exit").strip(),
    )


@contextmanager
def _ssh_tunnel(
    target_host: str,
    user: str | None,
    port: int,
    local_port: int,
    remote_host: str,
    remote_port: int,
    key_path: str | None,
) -> Iterator[None]:
    target = f"{user + '@' if user else ''}{target_host}"
    bind_arg = f"{local_port}:{remote_host}:{remote_port}"
    command = [
        "ssh",
        "-N",
        "-L",
        bind_arg,
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=4",
        "-p",
        str(port),
        target,
    ]
    if key_path:
        command.insert(1, "-i")
        command.insert(2, key_path)

    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.4)
        if proc.poll() is not None:
            error = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"SSH tunnel exited early: {error.strip()}")
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def _probe_ssh_tunnel(
    target_host: str,
    user: str | None,
    ssh_port: int,
    remote_url: str,
    key_path: str | None,
    timeout_s: float,
) -> dict[str, object]:
    parsed = urlparse(remote_url)
    remote_host = parsed.hostname or target_host
    remote_port = parsed.port or 554
    local_port = _free_tcp_port()

    login_ok, login_result = _ssh_login_ok(target_host, user, ssh_port, key_path)
    if not login_ok:
        return {
            "remote_url": remote_url,
            "ok": False,
            "stage": "ssh_login",
            "result": login_result,
        }

    try:
        with _ssh_tunnel(target_host, user, ssh_port, local_port, remote_host, remote_port, key_path):
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=timeout_s):
                    return {
                        "remote_url": remote_url,
                        "ok": True,
                        "stage": "tunnel",
                        "local_url": _build_local_rtsp_url(remote_url, local_port),
                        "local_port": local_port,
                        "remote_host": remote_host,
                        "remote_port": remote_port,
                        "result": "connected",
                    }
            except OSError as exc:
                return {
                    "remote_url": remote_url,
                    "ok": False,
                    "stage": "tunnel",
                    "local_port": local_port,
                    "result": str(exc),
                }
    except RuntimeError as exc:
        return {
            "remote_url": remote_url,
            "ok": False,
            "stage": "tunnel_start",
            "result": str(exc),
        }


def _pick_camera_source(robot_ip: str, robot_camera_url: str | None, args: argparse.Namespace) -> tuple[str, bool]:
    candidates = g1_camera_candidates(robot_ip, robot_camera_url)
    use_ssh = args.ssh and not args.no_ssh
    errors: list[str] = []
    gstreamer_only = all(is_gstreamer_pipeline(candidate) for candidate in candidates)

    if not use_ssh:
        return "g1", True

    try:
        cap, _frame, _source = open_camera("g1", robot_ip, robot_camera_url)
        cap.release()
        return "g1", True
    except RuntimeError as exc:
        errors.append(f"direct: {exc}")

    if gstreamer_only:
        errors.append(
            "ssh fallback is not used for GStreamer/UDP camera candidates."
        )
        if args.ssh_host is not None or args.ssh_user is not None:
            errors.append(
                "provided --ssh-host/--ssh-user (or positional SSH args) do not change G1 UDP stream transport."
            )
            errors.append(
                "run vision on the same network as the robot (or on the remote host itself), then use `--no-ssh`."
            )
        raise RuntimeError("Could not open camera source. Tried:\n" + "\n".join(f"  - {error}" for error in errors))

    target_host, user = _ssh_target(args)
    for remote_url in candidates:
        if is_gstreamer_pipeline(remote_url):
            errors.append(f"ssh:{remote_url}: skipped (GStreamer candidate; unsupported for SSH tunnel fallback)")
            continue
        parsed = urlparse(remote_url)
        remote_host = args.ssh_remote_host or parsed.hostname or target_host
        remote_port = int(parsed.port or 554)
        local_port = _free_tcp_port()

        try:
            with _ssh_tunnel(
                target_host,
                user,
                args.ssh_port,
                local_port,
                remote_host,
                remote_port,
                args.ssh_key,
            ):
                local_url = _build_local_rtsp_url(remote_url, local_port)
                cap = cv2.VideoCapture(local_url)
                if not cap.isOpened():
                    cap.release()
                    errors.append(f"ssh:{remote_url} -> {local_url}: could not open")
                    continue
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    errors.append(f"ssh:{remote_url} -> {local_url}: opened but no frame")
                    continue
                return local_url, False
        except RuntimeError as exc:
            errors.append(f"ssh:{remote_url}: {exc}")

    raise RuntimeError("Could not open camera source. Tried:\n" + "\n".join(f"  - {error}" for error in errors))


def _open_camera_for_vision(robot_ip: str, camera_url: str | None, args: argparse.Namespace):
    source, use_robot_ip = _pick_camera_source(robot_ip, camera_url, args)
    if source == "g1":
        return open_camera("g1", robot_ip, camera_url), use_robot_ip, source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source. Tried:\n  - {source}: could not open")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Could not open camera source. Tried:\n  - {source}: opened but did not return a frame")
    return (cap, frame, source), use_robot_ip, source


def _diagnose(robot_ip: str, camera_url: str | None, output_dir: Path, probe_timeout: float, args: argparse.Namespace) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = g1_camera_candidates(robot_ip, camera_url)
    use_ssh = args.ssh and not args.no_ssh

    report: dict[str, object] = {
        "timestamp": time.time(),
        "robot_ip": robot_ip,
        "route": _route_info(robot_ip),
        "candidates": candidates,
        "probe_timeout": probe_timeout,
        "ssh_enabled": use_ssh,
        "opencv": {
            "version": cv2.__version__,
            "gstreamer": opencv_gstreamer_enabled(),
        },
        "camera_probe": [_probe_camera_url(url, probe_timeout) for url in candidates],
    }

    if use_ssh:
        target_host, user = _ssh_target(args)
        report["ssh_target"] = {
            "host": target_host,
            "user": user or "(default)",
            "port": args.ssh_port,
        }
        report["ssh_probe"] = [
            _probe_ssh_tunnel(
                target_host=target_host,
                user=user,
                ssh_port=args.ssh_port,
                remote_url=url,
                key_path=args.ssh_key,
                timeout_s=probe_timeout,
            )
            if not is_gstreamer_pipeline(url)
            else {
                "remote_url": url,
                "ok": False,
                "stage": "skipped",
                "result": "GStreamer/UDP source does not use SSH tunnel",
            }
            for url in candidates
        ]

    diag_path = output_dir / "camera_diagnostics.json"
    diag_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _diagnose_help_if_needed(failed_reason: str) -> None:
    print(f"Could not open camera stream: {failed_reason}", file=sys.stderr)
    print("Hint: run with --diagnose for network + camera probe output.", file=sys.stderr)


def save_sdk2_video_snapshot(robot_ip: str, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        image_bytes, network_interface = get_sdk2_video_sample(
            robot_ip=robot_ip,
            network_interface=args.sdk2_interface,
            timeout_s=args.sdk2_timeout,
        )
    except UnitreeG1Error as exc:
        print(f"SDK2 video failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        print(
            f"SDK2 video returned {len(image_bytes)} bytes, but OpenCV could not decode them as an image.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    snapshot_path = output_dir / "snapshot.jpg"
    metadata_path = output_dir / "camera.json"
    if not cv2.imwrite(str(snapshot_path), frame):
        raise RuntimeError(f"Could not write snapshot to {snapshot_path}")

    metadata = {
        "timestamp": time.time(),
        "robot_ip": robot_ip,
        "camera_source": "unitree_sdk2:videohub:GetImageSample",
        "network_interface": network_interface,
        "snapshot": str(snapshot_path),
        "frame_shape": list(frame.shape),
        "image_sample_bytes": len(image_bytes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("SDK2 video frame opened: videohub GetImageSample")
    print(f"Using network interface: {network_interface}")
    print(f"Wrote snapshot to {snapshot_path}")
    print(f"Wrote camera metadata to {metadata_path}")


def save_snapshot(robot_ip: str, camera_url: str | None, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        (cap, frame, camera_source), _use_robot_ip, _source = _open_camera_for_vision(robot_ip, camera_url, args)
    except RuntimeError as exc:
        _diagnose_help_if_needed(str(exc))
        raise SystemExit(1)

    try:
        snapshot_path = output_dir / "snapshot.jpg"
        metadata_path = output_dir / "camera.json"
        if not cv2.imwrite(str(snapshot_path), frame):
            raise RuntimeError(f"Could not write snapshot to {snapshot_path}")

        metadata = {
            "timestamp": time.time(),
            "robot_ip": robot_ip,
            "camera_source": str(camera_source),
            "snapshot": str(snapshot_path),
            "frame_shape": list(frame.shape),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"Camera stream opened: {camera_source}")
        print(f"Wrote snapshot to {snapshot_path}")
        print(f"Wrote camera metadata to {metadata_path}")
    finally:
        cap.release()


def _run_tracking(robot_ip: str, camera_url: str | None, output_dir: Path, args: argparse.Namespace) -> None:
    camera = "g1"
    use_robot_ip = True
    if args.ssh and not args.no_ssh:
        try:
            source, use_robot_ip = _pick_camera_source(robot_ip, camera_url, args)
            camera = source
        except RuntimeError as exc:
            _diagnose_help_if_needed(str(exc))
            raise SystemExit(1)

    try:
        run(
            camera=camera,
            output_dir=output_dir,
            show=args.show,
            record=not args.no_record,
            robot_ip=robot_ip if use_robot_ip else None,
            robot_camera_url=None if camera != "g1" else camera_url,
            bbox=args.bbox,
            max_frames=args.max_frames,
        )
    except RuntimeError as exc:
        print(f"Vision run failed: {exc}", file=sys.stderr)
        print("Try: uv run vision <robot_ip> --diagnose", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _apply_ssh_positionals(args, parser)
    output_dir = Path(args.output)

    if args.no_ssh:
        args.ssh = False

    if args.sdk2_video:
        if args.bbox is not None or args.show:
            print("--sdk2-video currently captures a visual snapshot only; run it without --bbox/--show first.", file=sys.stderr)
            raise SystemExit(1)
        save_sdk2_video_snapshot(args.robot_ip, output_dir, args)
        return

    if args.list_cameras:
        for i, candidate in enumerate(g1_camera_candidates(args.robot_ip, args.camera_url), start=1):
            print(f"{i:02d}: {candidate}")
        return

    if args.diagnose:
        report = _diagnose(args.robot_ip, args.camera_url, output_dir, args.probe_timeout, args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.bbox is None and not args.show:
        save_snapshot(args.robot_ip, args.camera_url, output_dir, args)
        return

    _run_tracking(args.robot_ip, args.camera_url, output_dir, args)

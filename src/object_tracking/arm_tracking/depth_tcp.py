"""Latest-frame TCP transport for cross-version ROS depth streams."""

from __future__ import annotations

import socket
import struct
import threading
import time


_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 4 * 1024 * 1024


class DepthTcpSender:
    """Non-blocking producer that drops superseded frames while reconnecting."""

    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self._condition = threading.Condition()
        self._latest: bytes | None = None
        self._stop = False
        self._connected = False
        self._frames_sent = 0
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="depth-tcp-send")
        self._thread.start()

    def publish(self, frame: bytes) -> None:
        if not 0 < len(frame) <= _MAX_FRAME_BYTES:
            raise ValueError("depth TCP frame is empty or oversized")
        with self._condition:
            self._latest = frame
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)

    def diagnostics(self) -> dict[str, object]:
        with self._condition:
            return {
                "connected": self._connected,
                "frames_sent": self._frames_sent,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        connection: socket.socket | None = None
        while True:
            with self._condition:
                while self._latest is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    break
                frame, self._latest = self._latest, None
            try:
                if connection is None:
                    connection = socket.create_connection(self.address, timeout=0.5)
                    connection.settimeout(0.5)
                    with self._condition:
                        self._connected = True
                        self._last_error = None
                connection.sendall(_HEADER.pack(len(frame)) + frame)
                with self._condition:
                    self._frames_sent += 1
            except OSError as exc:
                if connection is not None:
                    connection.close()
                connection = None
                with self._condition:
                    self._connected = False
                    self._last_error = f"{type(exc).__name__}: {exc}"
        if connection is not None:
            connection.close()


class DepthTcpReceiver:
    """One-client latest-frame server with bounded framing and freshness."""

    def __init__(self, port: int, host: str = "0.0.0.0") -> None:
        self.address = (host, port)
        self._condition = threading.Condition()
        self._latest: tuple[bytes, float] | None = None
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._connections = 0
        self._frames_received = 0
        self._last_receipt_at: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.address)
        server.listen(1)
        server.settimeout(0.5)
        self._server = server
        self._thread = threading.Thread(target=self._run, daemon=True, name="depth-tcp-recv")
        self._thread.start()

    def receive(self, timeout_s: float) -> tuple[bytes, float] | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            item, self._latest = self._latest, None
            return item

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.close()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def diagnostics(self) -> dict[str, object]:
        with self._condition:
            age_ms = (
                None
                if self._last_receipt_at is None
                else round((time.monotonic() - self._last_receipt_at) * 1000.0, 3)
            )
            return {
                "depth_transport": "tcp",
                "depth_tcp_connected": self._connected,
                "depth_tcp_connections": self._connections,
                "depth_tcp_frames_received": self._frames_received,
                "depth_tcp_last_frame_age_ms": age_ms,
            }

    def connection_count(self) -> int:
        """Return the connection generation for sender-restart detection."""

        with self._condition:
            return self._connections

    @staticmethod
    def _read(connection: socket.socket, size: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = connection.recv(size - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                assert self._server is not None
                connection, _ = self._server.accept()
                connection.settimeout(1.0)
                with self._condition:
                    self._connected = True
                    self._connections += 1
                with connection:
                    while not self._stop.is_set():
                        header = self._read(connection, _HEADER.size)
                        if header is None:
                            break
                        size = _HEADER.unpack(header)[0]
                        if not 0 < size <= _MAX_FRAME_BYTES:
                            break
                        frame = self._read(connection, size)
                        if frame is None:
                            break
                        with self._condition:
                            now = time.monotonic()
                            self._latest = (frame, now)
                            self._last_receipt_at = now
                            self._frames_received += 1
                            self._condition.notify()
            except (OSError, socket.timeout):
                continue
            finally:
                with self._condition:
                    self._connected = False

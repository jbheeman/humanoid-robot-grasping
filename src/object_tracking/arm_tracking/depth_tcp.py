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
                connection.sendall(_HEADER.pack(len(frame)) + frame)
            except OSError:
                if connection is not None:
                    connection.close()
                connection = None
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
                            self._latest = (frame, time.monotonic())
                            self._condition.notify()
            except (OSError, socket.timeout):
                continue

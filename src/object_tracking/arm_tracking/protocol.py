"""Versioned and bounded Z16 depth-frame wire protocol.

An envelope is a four-byte, network-order JSON header length followed by the
UTF-8 header and a zstd payload.  The checksum covers uncompressed Z16 bytes so
corruption is detected after decompression as well as in transit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from typing import Any


PROTOCOL_VERSION = 1
_HEADER_LENGTH = struct.Struct("!I")


class DepthProtocolError(ValueError):
    """Base class for malformed or unsafe depth envelopes."""


class UnsupportedProtocolVersion(DepthProtocolError):
    pass


class PayloadTooLarge(DepthProtocolError):
    pass


class ChecksumMismatch(DepthProtocolError):
    pass


@dataclass(frozen=True)
class DepthFrameHeader:
    sequence: int
    width: int
    height: int
    depth_scale: float
    sensor_timestamp_ms: float
    timestamp_domain: str
    calibration_id: str
    payload_size: int
    uncompressed_size: int
    checksum_sha256: str
    version: int = PROTOCOL_VERSION
    encoding: str = "z16"
    byte_order: str = "little"
    registered_to_rgb: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DepthFrameHeader":
        try:
            return cls(
                sequence=int(value["sequence"]),
                width=int(value["width"]),
                height=int(value["height"]),
                depth_scale=float(value["depth_scale"]),
                sensor_timestamp_ms=float(value["sensor_timestamp_ms"]),
                timestamp_domain=str(value["timestamp_domain"]),
                calibration_id=str(value["calibration_id"]),
                payload_size=int(value["payload_size"]),
                uncompressed_size=int(value["uncompressed_size"]),
                checksum_sha256=str(value["checksum_sha256"]),
                version=int(value.get("version", 0)),
                encoding=str(value.get("encoding", "")),
                byte_order=str(value.get("byte_order", "")),
                registered_to_rgb=bool(value.get("registered_to_rgb", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DepthProtocolError(f"invalid depth header: {exc}") from exc

    def validate(self, *, max_pixels: int, max_payload_size: int) -> None:
        if self.version != PROTOCOL_VERSION:
            raise UnsupportedProtocolVersion(f"unsupported depth protocol version {self.version}")
        if self.encoding != "z16" or self.byte_order != "little":
            raise DepthProtocolError("only little-endian Z16 frames are supported")
        if self.sequence < 0 or self.width <= 0 or self.height <= 0:
            raise DepthProtocolError("sequence and frame dimensions must be positive")
        pixels = self.width * self.height
        if pixels > max_pixels:
            raise PayloadTooLarge(f"frame contains {pixels} pixels; limit is {max_pixels}")
        expected_size = pixels * 2
        if self.uncompressed_size != expected_size:
            raise DepthProtocolError(
                f"Z16 size {self.uncompressed_size} does not match {expected_size}"
            )
        if not 0 < self.payload_size <= max_payload_size:
            raise PayloadTooLarge(
                f"compressed payload is {self.payload_size} bytes; limit is {max_payload_size}"
            )
        if not 0.0 < self.depth_scale < 1.0:
            raise DepthProtocolError("depth_scale must be between zero and one metre/unit")
        if not math.isfinite(self.sensor_timestamp_ms) or self.sensor_timestamp_ms < 0:
            raise DepthProtocolError("sensor_timestamp_ms must be finite and non-negative")
        if not self.calibration_id or not self.timestamp_domain:
            raise DepthProtocolError("calibration_id and timestamp_domain are required")
        if len(self.checksum_sha256) != 64:
            raise DepthProtocolError("checksum_sha256 must be a hexadecimal SHA-256 digest")
        try:
            int(self.checksum_sha256, 16)
        except ValueError as exc:
            raise DepthProtocolError("checksum_sha256 is not hexadecimal") from exc


@dataclass(frozen=True)
class DecodedDepthFrame:
    header: DepthFrameHeader
    z16_le: bytes

    def as_numpy(self) -> Any:
        """Return a native NumPy ``uint16`` view, importing NumPy lazily."""

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - base project includes NumPy
            raise RuntimeError("NumPy is required to materialize a Z16 array") from exc
        return np.frombuffer(self.z16_le, dtype="<u2").reshape(
            self.header.height, self.header.width
        )


class DepthEnvelopeCodec:
    """Encode/decode depth frames with strict allocation and payload bounds."""

    def __init__(
        self,
        *,
        max_pixels: int = 1920 * 1080,
        max_payload_size: int = 8 * 1024 * 1024,
        max_header_size: int = 16 * 1024,
        compression_level: int = 3,
    ) -> None:
        if min(max_pixels, max_payload_size, max_header_size) <= 0:
            raise ValueError("codec bounds must be positive")
        self.max_pixels = max_pixels
        self.max_payload_size = max_payload_size
        self.max_header_size = max_header_size
        self.compression_level = compression_level

    @staticmethod
    def _zstd() -> Any:
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "zstandard is required for depth transport; install the depth dependency group"
            ) from exc
        return zstandard

    def encode(
        self,
        z16: Any,
        *,
        sequence: int,
        width: int,
        height: int,
        depth_scale: float,
        sensor_timestamp_ms: float,
        timestamp_domain: str,
        calibration_id: str,
        registered_to_rgb: bool = False,
    ) -> bytes:
        if width <= 0 or height <= 0:
            raise DepthProtocolError("frame dimensions must be positive")
        if width * height > self.max_pixels:
            raise PayloadTooLarge(
                f"frame contains {width * height} pixels; limit is {self.max_pixels}"
            )
        raw = self._coerce_z16(z16, width=width, height=height)
        zstandard = self._zstd()
        payload = zstandard.ZstdCompressor(level=self.compression_level).compress(raw)
        header = DepthFrameHeader(
            sequence=sequence,
            width=width,
            height=height,
            depth_scale=depth_scale,
            sensor_timestamp_ms=sensor_timestamp_ms,
            timestamp_domain=timestamp_domain,
            calibration_id=calibration_id,
            payload_size=len(payload),
            uncompressed_size=len(raw),
            checksum_sha256=hashlib.sha256(raw).hexdigest(),
            registered_to_rgb=registered_to_rgb,
        )
        header.validate(max_pixels=self.max_pixels, max_payload_size=self.max_payload_size)
        encoded_header = json.dumps(asdict(header), separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        if len(encoded_header) > self.max_header_size:
            raise PayloadTooLarge("depth header exceeds configured bound")
        return _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + payload

    def decode(self, envelope: bytes | bytearray | memoryview) -> DecodedDepthFrame:
        view = memoryview(envelope)
        if len(view) < _HEADER_LENGTH.size:
            raise DepthProtocolError("truncated depth envelope")
        (header_size,) = _HEADER_LENGTH.unpack(view[: _HEADER_LENGTH.size])
        if not 0 < header_size <= self.max_header_size:
            raise PayloadTooLarge(f"depth header size {header_size} is invalid")
        payload_offset = _HEADER_LENGTH.size + header_size
        if len(view) < payload_offset:
            raise DepthProtocolError("truncated depth header")
        try:
            header_value = json.loads(bytes(view[_HEADER_LENGTH.size : payload_offset]))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DepthProtocolError("depth header is not valid UTF-8 JSON") from exc
        if not isinstance(header_value, dict):
            raise DepthProtocolError("depth header must be a JSON object")
        header = DepthFrameHeader.from_dict(header_value)
        header.validate(max_pixels=self.max_pixels, max_payload_size=self.max_payload_size)
        payload = bytes(view[payload_offset:])
        if len(payload) != header.payload_size:
            raise DepthProtocolError(
                f"payload length {len(payload)} does not match header {header.payload_size}"
            )
        zstandard = self._zstd()
        try:
            parameters = zstandard.get_frame_parameters(payload)
            if parameters.content_size != header.uncompressed_size:
                raise DepthProtocolError(
                    "zstd frame content size does not match the bounded Z16 size"
                )
            raw = zstandard.ZstdDecompressor().decompress(
                payload, max_output_size=header.uncompressed_size
            )
        except zstandard.ZstdError as exc:
            raise DepthProtocolError("invalid zstd depth payload") from exc
        if len(raw) != header.uncompressed_size:
            raise DepthProtocolError("decompressed Z16 payload has the wrong size")
        if hashlib.sha256(raw).hexdigest() != header.checksum_sha256:
            raise ChecksumMismatch("Z16 checksum mismatch")
        return DecodedDepthFrame(header=header, z16_le=raw)

    def _coerce_z16(self, value: Any, *, width: int, height: int) -> bytes:
        expected = width * height * 2
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
        else:
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("NumPy is required to encode a depth array") from exc
            array = np.asarray(value)
            if array.shape != (height, width):
                raise DepthProtocolError(
                    f"depth shape {array.shape} does not match {(height, width)}"
                )
            if array.dtype.kind != "u" or array.dtype.itemsize != 2:
                raise DepthProtocolError("depth array must contain unsigned 16-bit values")
            raw = array.astype("<u2", copy=False).tobytes(order="C")
        if len(raw) != expected:
            raise DepthProtocolError(f"Z16 data is {len(raw)} bytes; expected {expected}")
        return raw

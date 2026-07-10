from __future__ import annotations

import importlib.util
import json
import struct
import unittest

import numpy as np

from object_tracking.arm_tracking.protocol import (
    ChecksumMismatch,
    DepthEnvelopeCodec,
    DepthProtocolError,
    PayloadTooLarge,
)


class DepthEnvelopeCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = DepthEnvelopeCodec(max_pixels=100, max_payload_size=1024)

    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "zstandard is optional")
    def test_round_trip_z16(self) -> None:
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
        envelope = self.codec.encode(
            depth,
            sequence=9,
            width=4,
            height=3,
            depth_scale=0.001,
            sensor_timestamp_ms=123.5,
            timestamp_domain="hardware_clock",
            calibration_id="calibration-a",
        )

        decoded = self.codec.decode(envelope)

        self.assertEqual(decoded.header.sequence, 9)
        np.testing.assert_array_equal(decoded.as_numpy(), depth)

    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "zstandard is optional")
    def test_rejects_checksum_tampering(self) -> None:
        envelope = self.codec.encode(
            np.arange(4, dtype=np.uint16).reshape(2, 2),
            sequence=1,
            width=2,
            height=2,
            depth_scale=0.001,
            sensor_timestamp_ms=1.0,
            timestamp_domain="hardware_clock",
            calibration_id="calibration-a",
        )
        (header_size,) = struct.unpack("!I", envelope[:4])
        header = json.loads(envelope[4 : 4 + header_size])
        header["checksum_sha256"] = "0" * 64
        encoded_header = json.dumps(header).encode()
        tampered = (
            struct.pack("!I", len(encoded_header)) + encoded_header + envelope[4 + header_size :]
        )

        with self.assertRaises(ChecksumMismatch):
            self.codec.decode(tampered)

    def test_rejects_oversized_dimensions_before_decompression(self) -> None:
        header = {
            "version": 1,
            "sequence": 1,
            "width": 1000,
            "height": 1000,
            "depth_scale": 0.001,
            "sensor_timestamp_ms": 1.0,
            "timestamp_domain": "hardware_clock",
            "calibration_id": "calibration-a",
            "payload_size": 1,
            "uncompressed_size": 2_000_000,
            "checksum_sha256": "0" * 64,
            "encoding": "z16",
            "byte_order": "little",
        }
        encoded = json.dumps(header).encode()
        envelope = struct.pack("!I", len(encoded)) + encoded + b"x"

        with self.assertRaises(PayloadTooLarge):
            self.codec.decode(envelope)

    def test_rejects_payload_length_mismatch_before_importing_zstd(self) -> None:
        header = {
            "version": 1,
            "sequence": 1,
            "width": 2,
            "height": 2,
            "depth_scale": 0.001,
            "sensor_timestamp_ms": 1.0,
            "timestamp_domain": "hardware_clock",
            "calibration_id": "calibration-a",
            "payload_size": 10,
            "uncompressed_size": 8,
            "checksum_sha256": "0" * 64,
            "encoding": "z16",
            "byte_order": "little",
        }
        encoded = json.dumps(header).encode()
        envelope = struct.pack("!I", len(encoded)) + encoded + b"x"

        with self.assertRaisesRegex(DepthProtocolError, "payload length"):
            self.codec.decode(envelope)

    def test_bad_array_shape_is_rejected_before_compression(self) -> None:
        with self.assertRaisesRegex(DepthProtocolError, "shape"):
            self.codec.encode(
                np.zeros((2, 3), dtype=np.uint16),
                sequence=1,
                width=2,
                height=2,
                depth_scale=0.001,
                sensor_timestamp_ms=1.0,
                timestamp_domain="hardware_clock",
                calibration_id="calibration-a",
            )

    def test_oversized_encode_is_rejected_before_compression(self) -> None:
        with self.assertRaises(PayloadTooLarge):
            self.codec.encode(
                b"\0" * 202,
                sequence=1,
                width=101,
                height=1,
                depth_scale=0.001,
                sensor_timestamp_ms=1.0,
                timestamp_domain="hardware_clock",
                calibration_id="calibration-a",
            )


if __name__ == "__main__":
    unittest.main()

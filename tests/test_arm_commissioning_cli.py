from object_tracking.arm_commissioning_cli import _invoke, build_parser


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def commissioning(
        self, operation: str, payload: dict[str, object]
    ) -> dict[str, object]:
        self.calls.append((operation, payload))
        return {"ok": True, "operation": operation}


def test_start_uses_ros_commissioning_operation() -> None:
    transport = FakeTransport()
    args = build_parser().parse_args(
        ["start", "--operator", "Aarav", "--client-id", "gb10"]
    )

    report = _invoke(args, transport)

    assert report == {"ok": True, "operation": "create_session"}
    assert transport.calls == [
        (
            "create_session",
            {"operator_ack": True, "operator": "Aarav", "client_id": "gb10"},
        )
    ]


def test_jog_maps_to_bounded_ros_payload() -> None:
    transport = FakeTransport()
    args = build_parser().parse_args(
        ["jog", "session-1", "right_shoulder_pitch_joint", "+", "--sequence", "3"]
    )

    _invoke(args, transport)

    assert transport.calls == [
        (
            "jog",
            {
                "session_id": "session-1",
                "sequence": 3,
                "joint_name": "right_shoulder_pitch_joint",
                "direction": 1,
                "kind": "jog",
            },
        )
    ]

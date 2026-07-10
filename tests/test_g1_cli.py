from __future__ import annotations

import pytest

from object_tracking.g1_cli import ROUTES, _role_launcher, build_parser


def test_unified_cli_exposes_expected_workflows() -> None:
    parser = build_parser()
    namespace, extras = parser.parse_known_args(["arm", "commissioning", "--help"])

    assert namespace.group == "arm"
    assert namespace.workflow == "commissioning"
    assert extras == ["--help"]
    assert ("vision", "server") in ROUTES
    assert ("setup", "robot") in ROUTES


@pytest.mark.parametrize(
    ("group", "workflow"),
    [
        ("arm", "commissioning"),
        ("data", "augment"),
        ("robot", "loco"),
        ("setup", "gb10"),
        ("stream", "remote"),
        ("train", "detector"),
        ("tune", "analyze"),
        ("tune", "joint-audit"),
        ("vision", "snapshot"),
        ("gb10", "plushie"),
    ],
)
def test_routes_are_explicit(group: str, workflow: str) -> None:
    route = ROUTES[(group, workflow)]
    assert route.kind in {"module", "script"}
    assert route.target


def test_robot_role_flags_configure_existing_launcher() -> None:
    launcher, forwarded, env = _role_launcher(
        ROUTES[("robot", "start")],
        ["--client-ip", "192.168.0.66", "--token-file", "/tmp/token", "--arm-port", "9000"],
    )

    assert launcher.name == "start.sh"
    assert forwarded == []
    assert env["CLIENT_IP"] == "192.168.0.66"
    assert env["ARM_TOKEN_FILE"] == "/tmp/token"
    assert env["ARM_PORT"] == "9000"


def test_local_realsense_role_builds_pipeline() -> None:
    launcher, _forwarded, env = _role_launcher(
        ROUTES[("local", "start")], ["--source", "realsense", "--device", "/dev/video9"]
    )

    assert launcher.name == "start.sh"
    assert "v4l2src device=/dev/video9" in env["PIPELINE"]

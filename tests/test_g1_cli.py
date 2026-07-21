from __future__ import annotations

import pytest

from object_tracking.g1_cli import ROUTES, _role_launcher, build_parser
from scripts.robot.ros_node import build_parser as build_robot_parser


def test_robot_ros_node_parser_supports_depth_free_commissioning() -> None:
    args = build_robot_parser().parse_args(
        [
            "--control-mode",
            "commissioning",
            "--disable-depth",
            "--manual-control-profile",
            "xr",
        ]
    )

    assert args.control_mode == "commissioning"
    assert args.disable_depth is True
    assert args.manual_control_profile == "xr"


def test_unified_cli_exposes_expected_workflows() -> None:
    parser = build_parser()
    namespace, extras = parser.parse_known_args(["arm", "commissioning", "--help"])

    assert namespace.group == "arm"
    assert namespace.workflow == "commissioning"
    assert extras == ["--help"]
    assert ("vision", "server") in ROUTES
    assert ("setup", "robot") in ROUTES
    assert ("inspect", "ros") in ROUTES
    assert ("inspect", "dds") not in ROUTES


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
        [
            "--client-ip",
            "192.168.0.66",
            "--interface",
            "wlan0",
            "--ros-domain-id",
            "7",
        ],
    )

    assert launcher.name == "start.sh"
    assert forwarded == []
    assert env["CLIENT_IP"] == "192.168.0.66"
    assert env["ROBOT_INTERFACE"] == "wlan0"
    assert env["G1_PROJECT_ROS_DOMAIN_ID"] == "7"


def test_robot_role_uses_lab_gb10_default() -> None:
    _launcher, forwarded, env = _role_launcher(ROUTES[("robot", "start")], [])

    assert forwarded == []
    assert env["CLIENT_IP"] == "192.168.0.66"


def test_gb10_vla_preview_configures_observation_only_launcher() -> None:
    launcher, forwarded, env = _role_launcher(
        ROUTES[("gb10", "start")],
        ["--robot-host", "192.168.0.213", "--dry-run", "--vla-preview"],
    )

    assert launcher.name == "start.sh"
    assert forwarded == []
    assert env["ROBOT_HOST"] == "192.168.0.213"
    assert env["EXECUTE"] == "0"
    assert env["VLA_PREVIEW"] == "1"


def test_removed_robot_http_flags_have_migration_error() -> None:
    with pytest.raises(SystemExit):
        _role_launcher(
            ROUTES[("robot", "start")],
            ["--client-ip", "192.168.0.66", "--token-file", "/tmp/token"],
        )


def test_local_realsense_role_builds_pipeline() -> None:
    launcher, _forwarded, env = _role_launcher(
        ROUTES[("local", "start")], ["--source", "realsense", "--device", "/dev/video9"]
    )

    assert launcher.name == "start.sh"
    assert "v4l2src device=/dev/video9" in env["PIPELINE"]

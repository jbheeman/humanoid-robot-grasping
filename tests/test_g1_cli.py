from __future__ import annotations

import pytest

from object_tracking.g1_cli import ROUTES, _role_launcher, build_parser
from scripts.robot.ros_node import build_parser as build_robot_parser


def test_robot_ros_node_parser_defaults_to_tracking() -> None:
    args = build_robot_parser().parse_args([])
    assert args.control_mode == "tracking"


def test_unified_cli_exposes_expected_workflows() -> None:
    parser = build_parser()
    namespace, extras = parser.parse_known_args(["robot", "start", "--help"])

    assert namespace.group == "robot"
    assert namespace.workflow == "start"
    assert extras == ["--help"]
    assert ("vision", "server") in ROUTES
    assert ("setup", "robot") in ROUTES
    assert ("inspect", "ros") in ROUTES
    assert ("inspect", "dds") not in ROUTES


@pytest.mark.parametrize(
    ("group", "workflow"),
    [
        ("robot", "start"),
        ("robot", "bunny-test"),
        ("setup", "gb10"),
        ("setup", "robot"),
        ("vision", "server"),
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


def test_bunny_test_requires_operator_enter_before_readiness_and_enable() -> None:
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "robot"
        / "bunny-test.sh"
    ).read_text()

    terminal_gate = script.index("if [[ ! -t 0 ]]")
    enter_gate = script.index('read -r -p "Press Enter')
    readiness = script.index('python3 - "${CLIENT_IP}" "${READY_TIMEOUT_S}"')
    profile_selection = script.index(
        'f"http://{host}:8000/api/v1/arm/follow-profile"'
    )
    enable = script.index('f"http://{host}:8000/api/v1/arm/enable"')
    assert terminal_gate < enter_gate < readiness < profile_selection < enable
    assert '"follow_profile": follow_profile' in script
    assert 'DEPTH_PUBLISH_FPS="${DEPTH_PUBLISH_FPS:-20}"' in script


def test_gb10_intercept_profile_is_forwarded_without_implying_execute() -> None:
    launcher, forwarded, env = _role_launcher(
        ROUTES[("gb10", "start")],
        [
            "--calibration",
            "/tmp/camera-calibration.yaml",
            "--intercept-config",
            "/tmp/demo-lane.yaml",
            "--dry-run",
        ],
    )

    assert launcher.name == "start.sh"
    assert forwarded == []
    assert env["CALIBRATION"] == "/tmp/camera-calibration.yaml"
    assert env["INTERCEPT_CONFIG"] == "/tmp/demo-lane.yaml"
    assert env["EXECUTE"] == "0"


def test_gb10_default_does_not_enable_interception() -> None:
    _launcher, _forwarded, env = _role_launcher(
        ROUTES[("gb10", "start")],
        ["--dry-run"],
    )

    assert "INTERCEPT_CONFIG" not in env


def test_gb10_follow_profile_is_forwarded_to_launcher() -> None:
    _launcher, forwarded, env = _role_launcher(
        ROUTES[("gb10", "start")],
        ["--follow-profile", "aggressive", "--dry-run"],
    )

    assert forwarded == []
    assert env["FOLLOW_PROFILE"] == "aggressive"


def test_removed_robot_http_flags_have_migration_error() -> None:
    with pytest.raises(SystemExit):
        _role_launcher(
            ROUTES[("robot", "start")],
            ["--client-ip", "192.168.0.66", "--token-file", "/tmp/token"],
        )

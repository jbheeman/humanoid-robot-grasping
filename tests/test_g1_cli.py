from __future__ import annotations

import pytest

from object_tracking.g1_cli import ROUTES, build_parser


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
    ],
)
def test_routes_are_explicit(group: str, workflow: str) -> None:
    route = ROUTES[(group, workflow)]
    assert route.kind in {"module", "script"}
    assert route.target

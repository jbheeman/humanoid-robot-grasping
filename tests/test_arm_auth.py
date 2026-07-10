from __future__ import annotations

import pytest

from object_tracking.arm_tracking.arm_auth import (
    TokenConfigurationError,
    bearer_is_valid,
    load_token,
)


def test_load_token_from_environment_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_ARM_TOKEN", "secret-token")
    assert load_token(env_name="TEST_ARM_TOKEN") == "secret-token"
    assert bearer_is_valid("Bearer secret-token", "secret-token")
    assert bearer_is_valid("bearer secret-token", "secret-token")
    assert not bearer_is_valid("Bearer wrong", "secret-token")
    assert not bearer_is_valid(None, "secret-token")


def test_token_file_must_be_private(tmp_path: object) -> None:
    path = tmp_path / "token"  # type: ignore[operator]
    path.write_text("secret-token\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(TokenConfigurationError, match="0600"):
        load_token(token_file=path)
    path.chmod(0o600)
    assert load_token(token_file=path) == "secret-token"


def test_missing_token_error_does_not_contain_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_ARM_TOKEN", raising=False)
    with pytest.raises(TokenConfigurationError) as rejected:
        load_token(env_name="MISSING_ARM_TOKEN")
    assert "MISSING_ARM_TOKEN" in str(rejected.value)
    assert "secret-token" not in str(rejected.value)

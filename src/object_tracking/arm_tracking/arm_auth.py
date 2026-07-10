"""Authentication helpers shared by the robot arm HTTP service."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
import stat


class TokenConfigurationError(ValueError):
    pass


def load_token(*, env_name: str = "ARM_BRIDGE_TOKEN", token_file: str | Path | None = None) -> str:
    """Load a token without ever placing its value in an error or report."""
    if token_file is not None:
        path = Path(token_file)
        try:
            details = path.stat()
        except OSError as exc:
            raise TokenConfigurationError(f"Could not read token file metadata: {path}") from exc
        if not stat.S_ISREG(details.st_mode):
            raise TokenConfigurationError(f"Token path is not a regular file: {path}")
        if stat.S_IMODE(details.st_mode) != 0o600:
            raise TokenConfigurationError(f"Token file must have mode 0600: {path}")
        if hasattr(os, "getuid") and details.st_uid != os.getuid():
            raise TokenConfigurationError(f"Token file must be owned by the bridge user: {path}")
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TokenConfigurationError(f"Could not read token file: {path}") from exc
    else:
        token = os.environ.get(env_name, "").strip()
    if not token:
        source = str(token_file) if token_file is not None else f"environment variable {env_name}"
        raise TokenConfigurationError(f"A non-empty bearer token is required in {source}")
    return token


def bearer_is_valid(header_value: str | None, expected_token: str) -> bool:
    if not header_value:
        return False
    scheme, separator, supplied = header_value.partition(" ")
    if not separator or scheme.lower() != "bearer" or not supplied:
        return False
    return hmac.compare_digest(supplied, expected_token)

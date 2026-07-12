"""Compact cross-host candidate exchange helpers."""
from __future__ import annotations

import json
import subprocess
from typing import Any


def exchange_candidates(reference: str) -> list[dict[str, float]]:
    """Read published candidates without changing the current worktree."""
    try:
        completed = subprocess.run(
            ["git", "show", reference],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    candidates = payload.get("candidates", [])
    required = ("kp", "kd", "vmax", "amax")
    return [
        {name: float(candidate[name]) for name in required}
        for candidate in candidates
        if isinstance(candidate, dict) and all(name in candidate for name in required)
    ]


def merge_elites(*groups: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    """Deduplicate parameter vectors while preserving score-ranked order."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for group in groups:
        for candidate in group:
            try:
                key = tuple(round(float(candidate[name]), 6) for name in ("kp", "kd", "vmax", "amax"))
            except (KeyError, TypeError, ValueError):
                continue
            if key not in seen:
                seen.add(key)
                merged.append(candidate)
                if len(merged) >= limit:
                    return merged
    return merged

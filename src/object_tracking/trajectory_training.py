"""Small, dependency-light model definition shared by trainer and runtime."""

from __future__ import annotations


def _torch():
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - train environment only.
        raise RuntimeError("Torch is required for trajectory prediction") from exc
    return nn


class TrajectoryGRU(_torch().Module):
    def __init__(self, *, hidden_size: int = 64, outputs: int = 3) -> None:
        super().__init__()
        nn = _torch()
        self.encoder = nn.GRU(input_size=6, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, outputs * 3))
        self.outputs = outputs

    def forward(self, features):  # type: ignore[no-untyped-def]
        _, hidden = self.encoder(features)
        return self.head(hidden[-1]).reshape(-1, self.outputs, 3)

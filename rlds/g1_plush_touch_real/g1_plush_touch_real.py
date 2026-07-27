"""TFDS entry point for curated XR plush-touch episodes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_plush_touch_common import G1PlushTouchBuilderBase  # noqa: E402


class G1PlushTouchReal(G1PlushTouchBuilderBase):
    SOURCE = "real"

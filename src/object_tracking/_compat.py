"""Small runtime compatibility helpers shared with the Python 3.8 robot."""

from __future__ import annotations

from itertools import zip_longest
from typing import Any, Iterable, Iterator, Tuple


def strict_zip(*iterables: Iterable[Any]) -> Iterator[Tuple[Any, ...]]:
    """Backport ``zip(..., strict=True)`` without weakening length checks."""

    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values

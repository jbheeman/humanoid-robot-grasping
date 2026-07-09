#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    script = Path(__file__).resolve().with_name("robot.py")
    return subprocess.call([sys.executable, str(script), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

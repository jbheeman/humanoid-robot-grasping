from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    app_path = Path(__file__).with_name("streamlit_viewer.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
    ]
    streamlit_cli.main()

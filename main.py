#!/usr/bin/env python3
"""Entry point: launches the Streamlit UI."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    frontend = str(Path(__file__).parent / "frontend" / "front.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", frontend])

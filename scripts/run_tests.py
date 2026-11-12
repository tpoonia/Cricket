#!/usr/bin/env python
"""Pytest wrapper. Keep-it-green gate: `python scripts/run_tests.py` must pass
after every task (Hard rule #2).

Usage:
    python scripts/run_tests.py            # run the whole suite, quiet
    python scripts/run_tests.py -k milp    # pass extra args straight to pytest
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    args = sys.argv[1:] or ["-q"]
    cmd = [sys.executable, "-m", "pytest", *args]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

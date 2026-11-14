#!/usr/bin/env python3
"""Dependency and script-level smoke test; does not require the large input tables."""

from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = [PACKAGE / "run_pipeline.py", *sorted((PACKAGE / "src").glob("*.py"))]


def main() -> None:
    import matplotlib
    import numpy
    import openpyxl
    import pandas

    for script in SCRIPTS:
        py_compile.compile(str(script), doraise=True)
        completed = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"--help failed for {script.name}: {completed.stderr}")
    result = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "openpyxl": openpyxl.__version__,
        "scripts_checked": [x.name for x in SCRIPTS],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

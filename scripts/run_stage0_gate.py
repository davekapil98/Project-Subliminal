#!/usr/bin/env python3
"""Run the Stage 0 acceptance suite and write a machine-readable local report."""

from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    report = {
        "gate": "stage0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "passed": completed.returncode == 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pytest_exit_code": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
    }
    report_path = Path("artifacts/reports/stage0_gate.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    print(f"Stage 0 gate report: {report_path}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

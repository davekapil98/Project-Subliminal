from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_config_query_needs_only_the_python_standard_library() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/stage1_host.py"),
            "config-value",
            "--config",
            str(ROOT / "configs/simulation/stage1_gb100.toml"),
            "total_episodes",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "12288"


def test_full_audit_runs_in_container_before_resume_marker_removal() -> None:
    text = (ROOT / ".launch_stage1_impl.sh").read_text(encoding="utf-8")
    validation = text.index("/workspace/subliminal/scripts/stage1_host.py validate")
    marker_removal = text.index('rm -f -- "$ACTIVE_MARKER"')
    assert validation < marker_removal

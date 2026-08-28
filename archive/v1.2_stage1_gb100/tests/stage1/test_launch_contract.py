from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_launcher_shell_syntax_and_confirmation_gate() -> None:
    launcher = ROOT / "launch_stage1.sh"
    implementation = ROOT / ".launch_stage1_impl.sh"
    for script in (launcher, implementation):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    text = implementation.read_text(encoding="utf-8")
    verify_position = text.index("--mode verify")
    prompt_position = text.index("Press ENTER to start simulation data recording")
    record_position = text.index("--mode record")
    assert verify_position < prompt_position < record_position
    assert "--preflight-only" in text
    assert ".stage1_active_run" in text
    assert "ISAAC_PYTHON" in launcher.read_text(encoding="utf-8")


def test_scene_declares_exactly_the_requested_primary_sensors() -> None:
    text = (ROOT / "src/sim/stage1/isaac_worlds.py").read_text(encoding="utf-8")
    assert "camera_ego = None" in text
    assert "camera_external_D455 = None" in text
    assert text.count("rear_camera =") == 1
    assert text.count("gripper_imu =") == 1
    assert "{ENV_REGEX_NS}/Robot/gripper" in text


def test_container_is_reproducibly_pinned() -> None:
    dockerfile = (ROOT / "docker/stage1/Dockerfile").read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/isaac-lab:2.3.2" in dockerfile
    assert "ce807d99724cb65671abec01f908a2fcb4a6eab7" in dockerfile
    assert "e670ac5daf9b76" in dockerfile
    assert "lfs pull" in dockerfile
    assert "SO-ARM101-USD.usd" in dockerfile

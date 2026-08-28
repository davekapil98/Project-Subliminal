"""Reproducibility metadata captured inside the Isaac container."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from .config import Stage1Config


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def hash_asset_tree(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def collect_provenance(config: Stage1Config) -> dict[str, Any]:
    workshop = Path("/workspace/Sim-to-Real-SO-101-Workshop")
    lerobot = Path("/workspace/lerobot")
    assets = workshop / "source/sim_to_real_so101/assets"
    if not assets.is_dir():
        raise RuntimeError(f"pinned workshop assets are missing: {assets}")
    asset_hash, asset_count = hash_asset_tree(assets)
    workshop_actual = _command("git", "-C", str(workshop), "rev-parse", "HEAD")
    lerobot_actual = _command("git", "-C", str(lerobot), "rev-parse", "HEAD")
    if workshop_actual != config.runtime.workshop_commit:
        raise RuntimeError("actual NVIDIA workshop revision does not match the configured pin")
    if lerobot_actual is None or not lerobot_actual.startswith(config.runtime.lerobot_commit):
        raise RuntimeError("actual LeRobot revision does not match the configured pin")
    return {
        "isaac_lab_version": config.runtime.isaac_lab_version,
        "workshop_repository": config.runtime.workshop_repository,
        "workshop_commit_configured": config.runtime.workshop_commit,
        "workshop_commit_actual": workshop_actual,
        "lerobot_commit_configured": config.runtime.lerobot_commit,
        "lerobot_commit_actual": lerobot_actual,
        "asset_tree_sha256": asset_hash,
        "asset_file_count": asset_count,
        "container_image": os.environ.get("SUBLIMINAL_IMAGE_REF"),
        "container_image_id": os.environ.get("SUBLIMINAL_IMAGE_ID"),
        "source_git_commit": os.environ.get("SUBLIMINAL_SOURCE_COMMIT"),
        "source_git_dirty": os.environ.get("SUBLIMINAL_SOURCE_DIRTY") == "1",
        "nvidia_smi": _command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
    }

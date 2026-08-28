"""Full integrity audit for a completed or interrupted Stage 1 raw run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_run_directory(path: str | Path, verify_payload: bool = True) -> dict[str, Any]:
    # NumPy is provided by the pinned Isaac image. Defer the payload-layer
    # import so dependency-free host preflight/config commands stay usable.
    from .writer import verify_episode_directory

    run = Path(path).expanduser().resolve()
    manifest_path = run / "run_manifest.json"
    state_path = run / "run_state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError(f"not a Stage 1 run directory: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    episodes_dir = run / "episodes"
    episode_paths = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    metadata = [verify_episode_directory(path, verify_payload=verify_payload) for path in episode_paths]
    ids = [item["episode_id"] for item in metadata]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode ids")
    if state.get("committed_episodes") != len(ids):
        raise ValueError("run_state committed episode count is stale or incorrect")
    index_path = run / "episodes.jsonl"
    index_ids = []
    if index_path.exists():
        index_ids = [json.loads(line)["episode_id"] for line in index_path.read_text(encoding="utf-8").splitlines()]
    if index_ids != sorted(ids):
        raise ValueError("episodes.jsonl does not exactly match committed episode directories")
    return {
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "status": state["status"],
        "episodes": len(ids),
        "planned_episodes": state["planned_episodes"],
        "verified_payloads": verify_payload,
    }

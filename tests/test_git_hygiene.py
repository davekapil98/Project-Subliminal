from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_git_hygiene", ROOT / "scripts/check_git_hygiene.py")
assert SPEC is not None and SPEC.loader is not None
HYGIENE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HYGIENE)


def test_rejects_secrets_and_training_artifacts() -> None:
    rejected = (
        ".env",
        "config/.env.production",
        "production.env",
        "data/raw/episode.jsonl",
        "datasets/lerobot/chunk.parquet",
        "artifacts/runs/model.bin",
        "elsewhere/checkpoint.safetensors",
        "preview.mp4",
    )
    for path in rejected:
        assert HYGIENE.violation_reason(path) is not None, path


def test_allows_source_docs_and_placeholders() -> None:
    allowed = (
        "src/data/canonical_schema.py",
        "src/models/body_dynamics/model.py",
        "data/raw/.gitkeep",
        "data/manifests/README.md",
        "artifacts/reports/README.md",
        "docs/stage1_data_contract.md",
    )
    for path in allowed:
        assert HYGIENE.violation_reason(path) is None, path


def test_rejects_oversized_source_file() -> None:
    assert HYGIENE.violation_reason("docs/huge.bin", HYGIENE.MAX_SOURCE_FILE_BYTES + 1) is not None

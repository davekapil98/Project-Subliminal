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
        "data/raw/source/README.md",
        "datasets/lerobot/chunk.parquet",
        "artifacts/runs/model.bin",
        "elsewhere/checkpoint.safetensors",
        "preview.mp4",
        "data/manifests/episodes.parquet",
    )
    for path in rejected:
        assert HYGIENE.violation_reason(path) is not None, path


def test_allows_source_docs_placeholders_and_small_metadata() -> None:
    allowed = (
        "src/data/canonical_schema.py",
        "src/models/body_dynamics/model.py",
        "data/raw/.gitkeep",
        "data/manifests/README.md",
        "data/manifests/dataset_registry.csv",
        "data/manifests/preprocessing_runs.jsonl",
        "data/splits/stage1_v1.json",
        "artifacts/reports/README.md",
        "docs/tensor_shapes.md",
    )
    for path in allowed:
        assert HYGIENE.violation_reason(path) is None, path


def test_rejects_oversized_source_and_metadata_files() -> None:
    assert HYGIENE.violation_reason("docs/huge.bin", HYGIENE.MAX_SOURCE_FILE_BYTES + 1) is not None
    assert (
        HYGIENE.violation_reason(
            "data/manifests/huge.json", HYGIENE.MAX_METADATA_FILE_BYTES + 1
        )
        is not None
    )

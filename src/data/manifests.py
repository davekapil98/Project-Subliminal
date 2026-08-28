"""Manifest records required before a dataset enters training."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    source_url: str
    license: str
    redistribution_terms: str
    checksum: str


def write_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest(**json.loads(path.read_text(encoding="utf-8")))

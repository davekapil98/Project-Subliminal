"""Pinned source manifests required before a public dataset enters training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile


VALID_DOMAINS = frozenset({"sim", "real"})
VALID_PRIORITIES = frozenset({"A+", "A", "B", "C"})
VALID_STATUSES = frozenset({"planned", "validated", "admitted", "rejected"})


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    revision: str
    source_url: str
    license: str
    redistribution_terms: str
    domain: str
    robot_id: str
    embodiment: str
    data_format: str
    modalities: tuple[str, ...]
    task_families: tuple[str, ...]
    native_action_semantics: str
    unit_conventions: dict[str, str]
    coordinate_frames: dict[str, str]
    priority: str
    status: str = "planned"
    checksum: str | None = None
    fps: float | None = None
    camera_names: tuple[str, ...] = ()
    simulator_family: str | None = None
    simulator_version: str | None = None
    known_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "source_url": self.source_url,
            "license": self.license,
            "redistribution_terms": self.redistribution_terms,
            "robot_id": self.robot_id,
            "embodiment": self.embodiment,
            "data_format": self.data_format,
            "native_action_semantics": self.native_action_semantics,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"required manifest fields are empty: {', '.join(missing)}")
        if self.domain not in VALID_DOMAINS:
            raise ValueError("domain must be explicitly 'sim' or 'real'")
        if self.priority not in VALID_PRIORITIES:
            raise ValueError("priority must be one of A+, A, B or C")
        if self.status not in VALID_STATUSES:
            raise ValueError("invalid dataset status")
        if self.status in {"validated", "admitted"} and not self.checksum:
            raise ValueError("validated/admitted datasets require a checksum")
        if self.domain == "sim" and not self.simulator_family:
            raise ValueError("simulation manifests require simulator_family")
        if not self.modalities or not self.task_families:
            raise ValueError("modalities and task_families cannot be empty")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique")


def write_manifest(path: Path, manifest: DatasetManifest) -> None:
    """Write once; an existing manifest may only be rewritten identically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite versioned manifest {path}")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_manifest(path: Path) -> DatasetManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for field_name in ("modalities", "task_families", "camera_names", "known_limitations"):
        if field_name in raw:
            raw[field_name] = tuple(raw[field_name])
    return DatasetManifest(**raw)

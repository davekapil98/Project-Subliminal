#!/usr/bin/env python3
"""Download and qualify the pinned Stage 1.1 Project IRA SO-101 subset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from data.adapters.project_ira_so101 import ProjectIRASO101Adapter, ProjectIRASourceSpec
from data.manifests import DatasetManifest, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs/datasets/registry/project_ira_so101_v1.toml"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/project_ira_so101_v1.json"
QUALIFICATION_PATH = PROJECT_ROOT / "data/manifests/project_ira_so101_v1.qualification.json"
SPLIT_PATH = PROJECT_ROOT / "data/splits/project_ira_so101_v1.json"
RAW_BASE = PROJECT_ROOT / "data/raw/public_real/project_ira_so101"
CLEANED_BASE = PROJECT_ROOT / "data/cleaned/public_real/project_ira_so101"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite reproducibility record {path}")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _download_url(spec: ProjectIRASourceSpec, relative_path: str) -> str:
    encoded_path = quote(relative_path, safe="/")
    return f"https://huggingface.co/datasets/{spec.repository_id}/resolve/{spec.revision}/{encoded_path}?download=true"


def download_qualified_files(raw_root: Path, spec: ProjectIRASourceSpec) -> None:
    """Download only pinned qualified files; never overwrite an existing raw object."""

    for source_file in spec.qualified_files:
        destination = raw_root / source_file.path
        if destination.exists():
            if destination.stat().st_size != source_file.size or sha256_file(destination) != source_file.sha256:
                raise ValueError(f"existing immutable raw file failed its pin: {source_file.path}")
            print(f"verified existing {source_file.path}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.part")
        partial.unlink(missing_ok=True)
        request = Request(_download_url(spec, source_file.path), headers={"User-Agent": "Project-Subliminal/0.1"})
        print(f"downloading {source_file.path} ({source_file.size} bytes)")
        with urlopen(request, timeout=120) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size != source_file.size or sha256_file(partial) != source_file.sha256:
            raise ValueError(f"downloaded file failed its pin: {source_file.path}")
        os.replace(partial, destination)


def verify_qualified_files(raw_root: Path, spec: ProjectIRASourceSpec) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for source_file in spec.qualified_files:
        path = raw_root / source_file.path
        if not path.is_file():
            raise FileNotFoundError(f"missing qualified source file: {source_file.path}")
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_size != source_file.size:
            raise ValueError(f"size mismatch for {source_file.path}: {actual_size} != {source_file.size}")
        if actual_sha256 != source_file.sha256:
            raise ValueError(f"SHA-256 mismatch for {source_file.path}")
        verified.append(asdict(source_file))
    return verified


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    contiguous = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(contiguous.dtype).removeprefix("torch."),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
    }


def canonical_sample_payload(episode: Any) -> dict[str, Any]:
    return {
        "metadata": asdict(episode.metadata),
        "language": list(episode.language),
        "observations": [
            {
                "timestamp": observation.timestamp,
                "q": observation.q.tolist(),
                "qdot": observation.qdot.tolist(),
                "previous_command": observation.previous_command.tolist(),
                "rgb": {name: _tensor_summary(frame) for name, frame in observation.rgb.items()},
                "validity": observation.validity,
            }
            for observation in episode.observations
        ],
        "actions": [
            {"timestamp": action.timestamp, "native": action.native.tolist()} for action in episode.actions
        ],
    }


def build_split_record(adapter: ProjectIRASO101Adapter) -> dict[str, Any]:
    task_splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    family_counts: dict[str, dict[str, int]] = {}
    if set(adapter.spec.task_family_ranges) != set(adapter.spec.task_families):
        raise ValueError("task-family ranges must match the declared task families")
    covered_tasks: set[int] = set()
    for family, (first, last) in adapter.spec.task_family_ranges.items():
        family_tasks = list(range(first, last + 1))
        overlap = covered_tasks.intersection(family_tasks)
        if overlap:
            raise ValueError(f"task-family ranges overlap at {sorted(overlap)}")
        covered_tasks.update(family_tasks)
        ranked = sorted(
            family_tasks,
            key=lambda task_index: hashlib.sha256(
                f"{adapter.spec.dataset_id}:{adapter.spec.revision}:{family}:{task_index}".encode()
            ).digest(),
        )
        heldout_count = max(1, round(len(ranked) * 0.1))
        if 2 * heldout_count >= len(ranked):
            raise ValueError(f"task family {family!r} is too small for three non-empty splits")
        assignments = {
            "test": ranked[:heldout_count],
            "validation": ranked[heldout_count : 2 * heldout_count],
            "train": ranked[2 * heldout_count :],
        }
        family_counts[family] = {}
        for split, task_indices in assignments.items():
            task_splits[split].extend(task_indices)
            family_counts[family][split] = len(task_indices)
    source_tasks = set(adapter.tasks())
    if covered_tasks != source_tasks:
        raise ValueError("task-family ranges must cover every source task exactly once")
    task_splits = {name: sorted(indices) for name, indices in task_splits.items()}
    episode_counts = {name: len(task_indices) * 10 for name, task_indices in task_splits.items()}
    if not all(episode_counts.values()) or sum(episode_counts.values()) != adapter.spec.total_episodes:
        raise ValueError("task-group split must produce non-empty train/validation/test partitions")
    return {
        "schema_version": 1,
        "dataset_id": adapter.spec.dataset_id,
        "source_revision": adapter.spec.revision,
        "strategy": "family-stratified SHA-256 ranking with approximately 0.1 validation and 0.1 test per family",
        "group_key": "task_index",
        "leakage_rule": "all ten episodes for an exact prompt stay in one split",
        "task_family_ranges_inclusive": {
            family: [first, last]
            for family, (first, last) in adapter.spec.task_family_ranges.items()
        },
        "task_family_prompt_counts": family_counts,
        "task_indices": task_splits,
        "episode_counts": episode_counts,
    }


def build_dataset_manifest(spec: ProjectIRASourceSpec) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        source_url=spec.source_url,
        license=spec.license,
        redistribution_terms=spec.redistribution_terms,
        domain=spec.domain,
        robot_id=spec.robot_id,
        embodiment=spec.embodiment,
        data_format=spec.data_format,
        modalities=spec.modalities,
        task_families=spec.task_families,
        native_action_semantics=spec.native_action_semantics,
        unit_conventions=spec.units,
        coordinate_frames=spec.coordinate_frames,
        priority=spec.priority,
        status=spec.status,
        checksum=f"sha256:{spec.primary_trajectory_sha256}",
        fps=spec.fps,
        camera_names=spec.camera_names,
        known_limitations=spec.known_limitations,
    )


def qualify(*, download: bool) -> dict[str, Any]:
    spec = ProjectIRASourceSpec.from_toml(REGISTRY_PATH)
    raw_root = RAW_BASE / spec.revision
    if download:
        download_qualified_files(raw_root, spec)
    files = verify_qualified_files(raw_root, spec)
    adapter = ProjectIRASO101Adapter(raw_root, spec)
    source_validation = adapter.validate_source()

    selected_episode = spec.qualified_episode_indices[0]
    video_validation: dict[str, Any] = {}
    for camera_name in spec.camera_names:
        segment = adapter.video_segment(selected_episode, camera_name)
        video_validation[camera_name] = {
            "segment_from_timestamp": segment.from_timestamp,
            "segment_to_timestamp": segment.to_timestamp,
            **adapter.validate_video(segment),
        }

    canonical = adapter.canonical_episode(
        selected_episode,
        max_transitions=2,
        include_rgb=True,
    )
    sample_payload = canonical_sample_payload(canonical)
    sample_path = CLEANED_BASE / spec.revision / "episode_000000_first_2_transitions.json"
    _write_json_once(sample_path, sample_payload)

    manifest = build_dataset_manifest(spec)
    write_manifest(MANIFEST_PATH, manifest)
    split_record = build_split_record(adapter)
    _write_json_once(SPLIT_PATH, split_record)

    report = {
        "schema_version": 1,
        "gate": "stage1.1_source_qualification",
        "passed": True,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "source_status": spec.status,
        "full_repository_bytes": spec.full_repository_bytes,
        "qualified_subset_bytes": sum(source_file["size"] for source_file in files),
        "qualified_episode_indices": list(spec.qualified_episode_indices),
        "verified_files": files,
        "source_validation": source_validation,
        "video_validation": video_validation,
        "canonical_sample": {
            "path": sample_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(sample_path),
            "observations": len(canonical.observations),
            "actions": len(canonical.actions),
            "camera_names": list(canonical.metadata.camera_names),
            "decoded_rgb_frames": sum(len(observation.rgb) for observation in canonical.observations),
        },
        "split_record": split_record,
        "admission_decision": "validated_for_stage1; not_yet_admitted_to_training",
        "admission_blockers": [
            "No target-model held-out improvement result yet; that is part of the later public-data gate.",
            "No camera calibration or task-space frame is published, so task-space actions must not be inferred.",
            "Task success and per-episode quality labels are absent and remain unknown.",
        ],
    }
    _write_json_once(QUALIFICATION_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download only the exact pinned qualification files that are missing",
    )
    args = parser.parse_args()
    report = qualify(download=args.download)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

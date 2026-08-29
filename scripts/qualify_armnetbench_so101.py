#!/usr/bin/env python3
"""Download and qualify the pinned Stage 1.2 ArmnetBench SO-101 subset."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

from data.adapters.armnetbench_so101 import ArmnetBenchSO101Adapter, ArmnetBenchSourceSpec
from data.manifests import DatasetManifest, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/armnetbench_so101_v01.json"
QUALIFICATION_PATH = PROJECT_ROOT / "data/manifests/armnetbench_so101_v01.qualification.json"
SPLIT_PATH = PROJECT_ROOT / "data/splits/armnetbench_so101_v01.json"
RAW_BASE = PROJECT_ROOT / "data/raw/public_real/armnetbench_so101"
CLEANED_BASE = PROJECT_ROOT / "data/cleaned/public_real/armnetbench_so101"
PROJECT_IRA_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/project_ira_so101_v1.json"


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


def _download_url(spec: ArmnetBenchSourceSpec, relative_path: str) -> str:
    encoded_path = quote(relative_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{spec.repository_id}/resolve/"
        f"{spec.revision}/{encoded_path}?download=true"
    )


def _download_one(raw_root: Path, spec: ArmnetBenchSourceSpec, source_object: Any) -> str:
    destination = raw_root / source_object.path
    if destination.exists():
        if destination.stat().st_size != source_object.size:
            raise ValueError(f"existing immutable raw file has wrong size: {source_object.path}")
        if sha256_file(destination) != source_object.sha256:
            raise ValueError(f"existing immutable raw file failed its pin: {source_object.path}")
        return f"verified existing {source_object.path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    request = Request(
        _download_url(spec, source_object.path),
        headers={"User-Agent": "Project-Subliminal/0.1"},
    )
    with urlopen(request, timeout=300) as response, partial.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size != source_object.size:
        raise ValueError(f"downloaded file has wrong size: {source_object.path}")
    if sha256_file(partial) != source_object.sha256:
        raise ValueError(f"downloaded file failed its pin: {source_object.path}")
    os.replace(partial, destination)
    return f"downloaded {source_object.path} ({source_object.size} bytes)"


def download_qualified_objects(raw_root: Path, spec: ArmnetBenchSourceSpec) -> None:
    """Download only pinned qualification objects; never overwrite immutable raw data."""

    raw_root.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(
        item.size for item in spec.qualified_objects if not (raw_root / item.path).exists()
    )
    free_bytes = shutil.disk_usage(raw_root.parent).free
    if free_bytes < required_bytes + 1024**3:
        raise OSError(
            f"qualification requires {required_bytes} bytes plus a 1 GiB reserve, "
            f"only {free_bytes} free"
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        messages = executor.map(
            lambda source_object: _download_one(raw_root, spec, source_object),
            spec.qualified_objects,
        )
        for message in messages:
            print(message)


def verify_qualified_objects(raw_root: Path, spec: ArmnetBenchSourceSpec) -> dict[str, Any]:
    if sha256_file(spec.object_manifest_path) != spec.object_manifest_sha256:
        raise ValueError("ArmnetBench object manifest differs from the registry pin")
    role_counts: Counter[str] = Counter()
    role_bytes: Counter[str] = Counter()
    for source_object in spec.qualified_objects:
        path = raw_root / source_object.path
        if not path.is_file():
            raise FileNotFoundError(f"missing qualified source object: {source_object.path}")
        if path.stat().st_size != source_object.size:
            raise ValueError(f"size mismatch for {source_object.path}")
        if sha256_file(path) != source_object.sha256:
            raise ValueError(f"SHA-256 mismatch for {source_object.path}")
        role_counts[source_object.role] += 1
        role_bytes[source_object.role] += source_object.size
    return {
        "object_manifest_path": spec.object_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "object_manifest_sha256": spec.object_manifest_sha256,
        "verified_object_count": len(spec.qualified_objects),
        "verified_bytes": sum(item.size for item in spec.qualified_objects),
        "role_counts": dict(sorted(role_counts.items())),
        "role_bytes": dict(sorted(role_bytes.items())),
    }


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
            {"timestamp": action.timestamp, "native": action.native.tolist()}
            for action in episode.actions
        ],
    }


def build_split_record(adapter: ArmnetBenchSO101Adapter) -> dict[str, Any]:
    spec = adapter.spec
    tasks = adapter.tasks()
    reverse_tasks = {text: index for index, text in tasks.items()}
    policy_index = {name: index for index, name in enumerate(spec.policy_types)}
    if len(policy_index) != len(spec.policy_types) or len(spec.policy_types) != spec.total_tasks:
        raise ValueError("balanced task-policy splitting requires an 8x8 task-policy matrix")

    cells: dict[str, set[tuple[int, str]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    episode_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    frame_counts: Counter[str] = Counter()
    outcome_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    task_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    policy_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    for record in adapter.episode_records():
        task_index = reverse_tasks[record["tasks"][0]]
        policy_type = str(record["policy_type"])
        index = policy_index[policy_type]
        if index == (task_index + spec.test_policy_offset) % len(spec.policy_types):
            split = "test"
        elif index == (task_index + spec.validation_policy_offset) % len(spec.policy_types):
            split = "validation"
        else:
            split = "train"
        cells[split].add((task_index, policy_type))
        episode_indices[split].append(int(record["episode_index"]))
        frame_counts[split] += int(record["length"])
        outcome_counts[split][str(record["success_class"])] += 1
        task_counts[split][spec.task_families[task_index]] += 1
        policy_counts[split][policy_type] += 1

    all_cells = set().union(*cells.values())
    expected_cells = {
        (task_index, policy_type)
        for task_index in range(spec.total_tasks)
        for policy_type in spec.policy_types
    }
    split_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    if all_cells != expected_cells or any(cells[a] & cells[b] for a, b in split_pairs):
        raise ValueError("task-policy cells must be complete and disjoint")
    for split in ("validation", "test"):
        if set(outcome_counts[split]) != set(spec.outcome_classes):
            raise ValueError(f"{split} must contain every outcome class")
        if set(task_counts[split]) != set(spec.task_families):
            raise ValueError(f"{split} must contain every task")
        if set(policy_counts[split]) != set(spec.policy_types):
            raise ValueError(f"{split} must contain every policy type")
    if sum(len(values) for values in episode_indices.values()) != spec.total_episodes:
        raise ValueError("split episode counts differ from the pinned source")
    if sum(frame_counts.values()) != spec.total_frames:
        raise ValueError("split frame counts differ from the pinned source")

    return {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "strategy": "balanced Latin-square assignment over the complete 8x8 task-policy matrix",
        "group_keys": ["task_index", "policy_type"],
        "leakage_rule": "every rollout sharing an exact task and policy family stays in one split",
        "offsets": {
            "test": spec.test_policy_offset,
            "validation": spec.validation_policy_offset,
        },
        "task_order": [tasks[index] for index in range(spec.total_tasks)],
        "task_family_order": list(spec.task_families),
        "policy_order": list(spec.policy_types),
        "cells": {
            split: [
                {"task_index": task_index, "policy_type": policy_type}
                for task_index, policy_type in sorted(values)
            ]
            for split, values in cells.items()
        },
        "episode_indices": episode_indices,
        "episode_counts": {
            split: len(values) for split, values in episode_indices.items()
        },
        "frame_counts": dict(frame_counts),
        "outcome_counts": {
            split: dict(sorted(values.items())) for split, values in outcome_counts.items()
        },
        "task_episode_counts": {
            split: dict(sorted(values.items())) for split, values in task_counts.items()
        },
        "policy_episode_counts": {
            split: dict(sorted(values.items())) for split, values in policy_counts.items()
        },
    }


def build_dataset_manifest(spec: ArmnetBenchSourceSpec) -> DatasetManifest:
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
        checksum=f"sha256:{spec.object_manifest_sha256}",
        fps=spec.fps,
        camera_names=spec.camera_names,
        known_limitations=spec.known_limitations,
    )


def comparison_to_project_ira(spec: ArmnetBenchSourceSpec) -> dict[str, Any]:
    project_ira = json.loads(PROJECT_IRA_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "shared_value": [
            "real direct SO-101 embodiment",
            "LeRobot v3 trajectories with native absolute joint-position actions",
            "multi-camera RGB, proprioception, actions and language instructions",
        ],
        "armnetbench_complement": [
            "three-way successful/failure/suboptimal outcomes",
            "seven learned-policy families plus human teleoperation",
            "eight precision, insertion, cable, object-placement and stacking tasks",
            "three AV1 cameras including top and wrist views",
        ],
        "project_ira_complement": [
            "93 prompt variants across four task families",
            "930 purely human-teleoperated episodes for language grounding",
            "smaller 9.33 GB source with a different desk, objects and camera setup",
        ],
        "source_revisions": {
            spec.dataset_id: spec.revision,
            project_ira["dataset_id"]: project_ira["revision"],
        },
        "decision": (
            "complementary and worth qualifying; do not bulk-download or mix until "
            "held-out target-model value is measured"
        ),
    }


def qualify(*, download: bool) -> dict[str, Any]:
    spec = ArmnetBenchSourceSpec.from_toml(REGISTRY_PATH)
    raw_root = RAW_BASE / spec.revision
    if download:
        download_qualified_objects(raw_root, spec)
    verified = verify_qualified_objects(raw_root, spec)
    adapter = ArmnetBenchSO101Adapter(raw_root, spec)
    source_validation = adapter.validate_source()

    packed_last_episode = 627
    video_validation: dict[str, Any] = {}
    for camera_name in spec.camera_names:
        segment = adapter.video_segment(packed_last_episode, camera_name)
        video_validation[camera_name] = {
            "verified_through_episode": packed_last_episode,
            "segment_from_timestamp": segment.from_timestamp,
            "segment_to_timestamp": segment.to_timestamp,
            "path": segment.path.relative_to(PROJECT_ROOT).as_posix(),
            **adapter.validate_video(segment),
        }

    canonical_episodes = [
        adapter.canonical_episode(
            episode_index,
            max_transitions=2,
            include_rgb=True,
        )
        for episode_index in spec.qualified_episode_indices
    ]
    sample_payload = {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "episodes": [
            canonical_sample_payload(episode) for episode in canonical_episodes
        ],
    }
    sample_path = CLEANED_BASE / spec.revision / "outcome_samples_first_2_transitions.json"
    _write_json_once(sample_path, sample_payload)

    manifest = build_dataset_manifest(spec)
    write_manifest(MANIFEST_PATH, manifest)
    split_record = build_split_record(adapter)
    _write_json_once(SPLIT_PATH, split_record)
    comparison = comparison_to_project_ira(spec)

    report = {
        "schema_version": 1,
        "gate": "stage1.2_source_qualification",
        "passed": True,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "source_release_tag": spec.release_tag,
        "source_status": spec.status,
        "full_snapshot_bytes": spec.full_snapshot_bytes,
        "hub_used_storage_bytes": spec.hub_used_storage_bytes,
        "qualified_subset": verified,
        "qualified_episode_indices": list(spec.qualified_episode_indices),
        "source_validation": source_validation,
        "video_validation": video_validation,
        "canonical_sample": {
            "path": sample_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(sample_path),
            "episodes": len(canonical_episodes),
            "outcome_classes": [
                episode.metadata.extra["success_class"] for episode in canonical_episodes
            ],
            "observations": sum(
                len(episode.observations) for episode in canonical_episodes
            ),
            "actions": sum(len(episode.actions) for episode in canonical_episodes),
            "decoded_rgb_frames": sum(
                len(observation.rgb)
                for episode in canonical_episodes
                for observation in episode.observations
            ),
        },
        "split_record": split_record,
        "comparison_to_project_ira": comparison,
        "admission_decision": "validated_for_stage1; not_yet_admitted_to_training",
        "admission_blockers": [
            "No target-model held-out improvement result yet; that remains part of the public-data gate.",
            "Published aggregate statistics are stale and must not replace recomputation from pinned trajectories.",
            "No camera calibration or task-space frame is published, so task-space actions must not be inferred.",
            "The full 60.59 GB snapshot is not justified until subset value is measured against Project IRA.",
        ],
    }
    _write_json_once(QUALIFICATION_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download only the exact pinned qualification objects that are missing",
    )
    args = parser.parse_args()
    report = qualify(download=args.download)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

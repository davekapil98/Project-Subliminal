#!/usr/bin/env python3
"""Download and qualify the pinned Stage 1.3 SO101 MA MultiTask 700 subset."""

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

from data.adapters.so101_ma_multitask_700 import (
    SO101MAMultiTaskAdapter,
    SO101MAMultiTaskSourceSpec,
)
from data.manifests import DatasetManifest, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    PROJECT_ROOT / "configs/datasets/registry/so101_ma_multitask_700.toml"
)
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/so101_ma_multitask_700.json"
QUALIFICATION_PATH = (
    PROJECT_ROOT / "data/manifests/so101_ma_multitask_700.qualification.json"
)
VALUE_GATE_PATH = (
    PROJECT_ROOT / "data/manifests/so101_ma_multitask_700.value_gate.json"
)
SPLIT_PATH = PROJECT_ROOT / "data/splits/so101_ma_multitask_700.json"
RAW_BASE = PROJECT_ROOT / "data/raw/public_sim/so101_ma_multitask_700"
CLEANED_BASE = PROJECT_ROOT / "data/cleaned/public_sim/so101_ma_multitask_700"
PROJECT_IRA_MANIFEST_PATH = (
    PROJECT_ROOT / "data/manifests/project_ira_so101_v1.json"
)
ARMNETBENCH_MANIFEST_PATH = (
    PROJECT_ROOT / "data/manifests/armnetbench_so101_v01.json"
)


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
            raise FileExistsError(
                f"refusing to overwrite reproducibility record {path}"
            )
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _download_url(
    spec: SO101MAMultiTaskSourceSpec, relative_path: str
) -> str:
    encoded_path = quote(relative_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{spec.repository_id}/resolve/"
        f"{spec.revision}/{encoded_path}?download=true"
    )


def _download_one(
    raw_root: Path,
    spec: SO101MAMultiTaskSourceSpec,
    source_object: Any,
) -> str:
    destination = raw_root / source_object.path
    if destination.exists():
        if destination.stat().st_size != source_object.size:
            raise ValueError(
                f"existing immutable raw file has wrong size: {source_object.path}"
            )
        if sha256_file(destination) != source_object.sha256:
            raise ValueError(
                f"existing immutable raw file failed its pin: {source_object.path}"
            )
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


def download_qualified_objects(
    raw_root: Path, spec: SO101MAMultiTaskSourceSpec
) -> None:
    """Download only pinned qualification objects; never overwrite raw data."""

    raw_root.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(
        item.size
        for item in spec.qualified_objects
        if not (raw_root / item.path).exists()
    )
    free_bytes = shutil.disk_usage(raw_root.parent).free
    if free_bytes < required_bytes + 1024**3:
        raise OSError(
            f"qualification requires {required_bytes} bytes plus a 1 GiB "
            f"reserve, only {free_bytes} free"
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        messages = executor.map(
            lambda source_object: _download_one(
                raw_root, spec, source_object
            ),
            spec.qualified_objects,
        )
        for message in messages:
            print(message)


def verify_qualified_objects(
    raw_root: Path, spec: SO101MAMultiTaskSourceSpec
) -> dict[str, Any]:
    if sha256_file(spec.object_manifest_path) != spec.object_manifest_sha256:
        raise ValueError("SO101 MA object manifest differs from the registry pin")
    role_counts: Counter[str] = Counter()
    role_bytes: Counter[str] = Counter()
    for source_object in spec.qualified_objects:
        path = raw_root / source_object.path
        if not path.is_file():
            raise FileNotFoundError(
                f"missing qualified source object: {source_object.path}"
            )
        if path.stat().st_size != source_object.size:
            raise ValueError(f"size mismatch for {source_object.path}")
        if sha256_file(path) != source_object.sha256:
            raise ValueError(f"SHA-256 mismatch for {source_object.path}")
        role_counts[source_object.role] += 1
        role_bytes[source_object.role] += source_object.size
    return {
        "object_manifest_path": spec.object_manifest_path.relative_to(
            PROJECT_ROOT
        ).as_posix(),
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
                "rgb": {
                    name: _tensor_summary(frame)
                    for name, frame in observation.rgb.items()
                },
                "validity": observation.validity,
            }
            for observation in episode.observations
        ],
        "actions": [
            {
                "timestamp": action.timestamp,
                "native": action.native.tolist(),
            }
            for action in episode.actions
        ],
    }


def build_split_record(
    adapter: SO101MAMultiTaskAdapter,
) -> dict[str, Any]:
    spec = adapter.spec
    tasks = adapter.tasks()
    reverse_tasks = {text: index for index, text in tasks.items()}
    episode_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    frame_counts: Counter[str] = Counter()
    task_episode_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    task_frame_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    blocks: dict[str, set[tuple[int, int]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    episode_blocks: dict[tuple[int, int], str] = {}
    for record in adapter.episode_records():
        task_index = reverse_tasks[record["tasks"][0]]
        local_episode_index = int(record["episode_index"]) % 100
        block_index = (
            local_episode_index // spec.source_episode_block_size
        )
        test_block = (
            task_index + spec.test_block_offset
        ) % 10
        validation_block = (
            task_index + spec.validation_block_offset
        ) % 10
        if block_index == test_block:
            split = "test"
        elif block_index == validation_block:
            split = "validation"
        else:
            split = "train"
        key = (task_index, block_index)
        previous_split = episode_blocks.setdefault(key, split)
        if previous_split != split:
            raise ValueError("a source episode block crosses split boundaries")
        blocks[split].add(key)
        episode_index = int(record["episode_index"])
        length = int(record["length"])
        task_family = spec.task_families[task_index]
        episode_indices[split].append(episode_index)
        frame_counts[split] += length
        task_episode_counts[split][task_family] += 1
        task_frame_counts[split][task_family] += length

    expected_blocks = {
        (task_index, block_index)
        for task_index in range(spec.total_tasks)
        for block_index in range(10)
    }
    all_blocks = set().union(*blocks.values())
    pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    if all_blocks != expected_blocks or any(
        blocks[a] & blocks[b] for a, b in pairs
    ):
        raise ValueError("source episode blocks must be complete and disjoint")
    for split in ("validation", "test"):
        if set(task_episode_counts[split]) != set(spec.task_families):
            raise ValueError(f"{split} does not retain every task family")
    if sum(len(values) for values in episode_indices.values()) != spec.total_episodes:
        raise ValueError("split episode counts differ from the source")
    if sum(frame_counts.values()) != spec.total_frames:
        raise ValueError("split frame counts differ from the source")

    return {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "strategy": (
            "rotating held-out contiguous ten-episode blocks within each "
            "upstream task source"
        ),
        "group_keys": ["task_index", "source_episode_block_10"],
        "leakage_rule": (
            "all ten adjacent episodes in a source-order block remain in one split"
        ),
        "block_size": spec.source_episode_block_size,
        "offsets": {
            "test": spec.test_block_offset,
            "validation": spec.validation_block_offset,
        },
        "task_order": [tasks[index] for index in range(spec.total_tasks)],
        "task_family_order": list(spec.task_families),
        "blocks": {
            split: [
                {
                    "task_index": task_index,
                    "source_episode_block_10": block_index,
                }
                for task_index, block_index in sorted(values)
            ]
            for split, values in blocks.items()
        },
        "episode_indices": episode_indices,
        "episode_counts": {
            split: len(values) for split, values in episode_indices.items()
        },
        "frame_counts": dict(frame_counts),
        "task_episode_counts": {
            split: dict(sorted(values.items()))
            for split, values in task_episode_counts.items()
        },
        "task_frame_counts": {
            split: dict(sorted(values.items()))
            for split, values in task_frame_counts.items()
        },
    }


def build_dataset_manifest(
    spec: SO101MAMultiTaskSourceSpec,
) -> DatasetManifest:
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
        simulator_family=spec.simulator_family,
        simulator_version=spec.simulator_version,
        known_limitations=spec.known_limitations,
    )


def comparison_to_real_sources(
    spec: SO101MAMultiTaskSourceSpec,
) -> dict[str, Any]:
    project_ira = json.loads(
        PROJECT_IRA_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    armnetbench = json.loads(
        ARMNETBENCH_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    return {
        "complementary_value": [
            "direct SO-101 geometry in Isaac simulation",
            "seven task families with dual-camera synthetic observations",
            "source-local numeric dynamics for bounded body/world evaluation",
        ],
        "restrictions": [
            "no published success or quality labels",
            "native calibration and physical action units are not proven",
            "no cross-source motor targets or task-space actions are allowed",
            "full video acquisition requires a separate visual-model value result",
        ],
        "source_revisions": {
            spec.dataset_id: spec.revision,
            project_ira["dataset_id"]: project_ira["revision"],
            armnetbench["dataset_id"]: armnetbench["revision"],
        },
    }


def qualify(*, download: bool) -> dict[str, Any]:
    spec = SO101MAMultiTaskSourceSpec.from_toml(REGISTRY_PATH)
    if spec.status != "validated":
        raise ValueError(
            "registry status must be validated only after source checks pass"
        )
    if not VALUE_GATE_PATH.is_file():
        raise FileNotFoundError(
            "run scripts/evaluate_stage1_3_body_value.py before freezing qualification"
        )
    value_gate = json.loads(VALUE_GATE_PATH.read_text(encoding="utf-8"))
    raw_root = RAW_BASE / spec.revision
    if download:
        download_qualified_objects(raw_root, spec)
    verified = verify_qualified_objects(raw_root, spec)
    adapter = SO101MAMultiTaskAdapter(raw_root, spec)
    source_validation = adapter.validate_source()

    video_validation: dict[str, Any] = {}
    canonical_episodes = []
    for episode_index in spec.qualified_episode_indices:
        video_validation[str(episode_index)] = {}
        for camera_name in spec.camera_names:
            segment = adapter.video_segment(episode_index, camera_name)
            video_validation[str(episode_index)][camera_name] = {
                "segment_from_timestamp": segment.from_timestamp,
                "aligned_to_timestamp": segment.aligned_to_timestamp,
                "declared_to_timestamp": segment.declared_to_timestamp,
                "aligned_frames": segment.aligned_frames,
                "unused_frames": segment.unused_frames,
                "path": segment.path.relative_to(PROJECT_ROOT).as_posix(),
                **adapter.validate_video(segment),
            }
        canonical_episodes.append(
            adapter.canonical_episode(
                episode_index,
                max_transitions=2,
                include_rgb=True,
            )
        )

    sample_payload = {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "episodes": [
            canonical_sample_payload(episode)
            for episode in canonical_episodes
        ],
    }
    sample_path = (
        CLEANED_BASE
        / spec.revision
        / "edge_case_samples_first_2_transitions.json"
    )
    _write_json_once(sample_path, sample_payload)
    split_record = build_split_record(adapter)
    _write_json_once(SPLIT_PATH, split_record)
    write_manifest(MANIFEST_PATH, build_dataset_manifest(spec))

    report = {
        "schema_version": 1,
        "gate": "stage1.3_source_qualification",
        "passed": True,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "source_status": spec.status,
        "full_tree_files": spec.full_tree_files,
        "full_tree_bytes": spec.full_tree_bytes,
        "hub_used_storage_bytes": spec.hub_used_storage_bytes,
        "qualified_subset": verified,
        "upstream_sources": [
            asdict(source) for source in spec.upstream_sources
        ],
        "source_validation": source_validation,
        "video_validation": video_validation,
        "canonical_sample": {
            "path": sample_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(sample_path),
            "episodes": len(canonical_episodes),
            "episode_indices": list(spec.qualified_episode_indices),
            "observations": sum(
                len(episode.observations)
                for episode in canonical_episodes
            ),
            "actions": sum(
                len(episode.actions) for episode in canonical_episodes
            ),
            "decoded_rgb_frames": sum(
                len(observation.rgb)
                for episode in canonical_episodes
                for observation in episode.observations
            ),
        },
        "split_record": split_record,
        "value_gate": value_gate,
        "comparison_to_real_sources": comparison_to_real_sources(spec),
        "admission_decision": value_gate["admission_decision"],
        "admission_blockers": value_gate["admission_blockers"],
    }
    _write_json_once(QUALIFICATION_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download only exact pinned qualification objects that are missing",
    )
    args = parser.parse_args()
    report = qualify(download=args.download)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

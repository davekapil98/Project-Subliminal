#!/usr/bin/env python3
"""Download and qualify the pinned Stage 1.4 DROID raw v1.0.1 subset."""

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

from data.adapters.droid_raw import DROIDObject, DROIDRawAdapter, DROIDSourceSpec, OUTCOMES
from data.manifests import DatasetManifest, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs/datasets/registry/droid_raw_1_0_1.toml"
OBJECTS_PATH = PROJECT_ROOT / "configs/datasets/registry/droid_raw_1_0_1.objects.json"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/droid_raw_1_0_1.json"
QUALIFICATION_PATH = PROJECT_ROOT / "data/manifests/droid_raw_1_0_1.qualification.json"
SPLIT_PATH = PROJECT_ROOT / "data/splits/droid_raw_1_0_1.json"
RAW_BASE = PROJECT_ROOT / "data/raw/public_real/droid_raw_1_0_1"
CLEANED_BASE = PROJECT_ROOT / "data/cleaned/public_real/droid_raw_1_0_1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
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
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _download_url(spec: DROIDSourceSpec, source_object: DROIDObject) -> str:
    encoded_name = quote(source_object.object_name, safe="")
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{spec.bucket}/o/"
        f"{encoded_name}?alt=media&generation={source_object.generation}"
    )


def _verify_object(path: Path, source_object: DROIDObject) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing qualified source object: {source_object.role}")
    if path.stat().st_size != source_object.size:
        raise ValueError(f"size mismatch for qualified {source_object.role} object")
    if md5_file(path) != source_object.md5:
        raise ValueError(f"MD5 mismatch for qualified {source_object.role} object")
    if sha256_file(path) != source_object.sha256:
        raise ValueError(f"SHA-256 mismatch for qualified {source_object.role} object")


def _download_one(
    raw_root: Path, spec: DROIDSourceSpec, source_object: DROIDObject
) -> str:
    destination = raw_root / source_object.path
    if destination.exists():
        _verify_object(destination, source_object)
        return f"verified existing {source_object.role} object"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    request = Request(
        _download_url(spec, source_object),
        headers={"User-Agent": "Project-Subliminal/0.1"},
    )
    with urlopen(request, timeout=300) as response, partial.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    _verify_object(partial, source_object)
    os.replace(partial, destination)
    return f"downloaded {source_object.role} object ({source_object.size} bytes)"


def download_qualified_objects(raw_root: Path, spec: DROIDSourceSpec) -> None:
    """Download exact immutable generations only; never overwrite raw data."""

    raw_root.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(
        item.size for item in spec.objects if not (raw_root / item.path).exists()
    )
    free_bytes = shutil.disk_usage(raw_root.parent).free
    if free_bytes < required_bytes + 1024**3:
        raise OSError(
            f"qualification requires {required_bytes} bytes plus a 1 GiB reserve, "
            f"only {free_bytes} bytes are free"
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        messages = executor.map(
            lambda item: _download_one(raw_root, spec, item), spec.objects
        )
        for message in messages:
            print(message)


def verify_qualified_objects(
    raw_root: Path, spec: DROIDSourceSpec
) -> dict[str, Any]:
    if sha256_file(OBJECTS_PATH) != spec.object_manifest_sha256:
        raise ValueError("DROID object manifest differs from the registry pin")
    role_counts: Counter[str] = Counter()
    role_bytes: Counter[str] = Counter()
    for source_object in spec.objects:
        _verify_object(raw_root / source_object.path, source_object)
        role_counts[source_object.role] += 1
        role_bytes[source_object.role] += source_object.size
    return {
        "object_manifest_path": OBJECTS_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "object_manifest_sha256": spec.object_manifest_sha256,
        "verified_object_count": len(spec.objects),
        "verified_bytes": sum(item.size for item in spec.objects),
        "role_counts": dict(sorted(role_counts.items())),
        "role_bytes": dict(sorted(role_bytes.items())),
        "generation_pins_present": all(bool(item.generation) for item in spec.objects),
        "size_md5_sha256_verified": True,
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
        "scene_metadata": list(episode.scene_metadata),
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
            {"timestamp": action.timestamp, "native": action.native.tolist()}
            for action in episode.actions
        ],
    }


def build_split_record(spec: DROIDSourceSpec) -> dict[str, Any]:
    lab_splits = {
        "train": spec.train_labs,
        "validation": spec.validation_labs,
        "test": spec.test_labs,
    }
    sets = {name: set(labs) for name, labs in lab_splits.items()}
    if any(
        sets[left] & sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("DROID split labs overlap")
    if set().union(*sets.values()) != set(spec.labs):
        raise ValueError("DROID splits do not cover the inventory labs")
    split_summaries: dict[str, Any] = {}
    for split, labs in lab_splits.items():
        successes = sum(spec.lab_outcome_counts[lab]["success"] for lab in labs)
        failures = sum(spec.lab_outcome_counts[lab]["failure"] for lab in labs)
        total = successes + failures
        qualified_cells = [
            f"{spec.dataset_id}:{lab}:{outcome}"
            for lab in labs
            for outcome in OUTCOMES
        ]
        split_summaries[split] = {
            "labs": list(labs),
            "lab_count": len(labs),
            "episode_count": total,
            "outcome_counts": {"success": successes, "failure": failures},
            "failure_rate": failures / total,
            "qualification_cells": qualified_cells,
        }
    if sum(item["episode_count"] for item in split_summaries.values()) != spec.indexed_episodes:
        raise ValueError("DROID split counts do not sum to the complete inventory")
    return {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "strategy": (
            "Hold complete collection laboratories out; choose validation and test lab "
            "pairs with similar sizes and failure rates."
        ),
        "group_key": "lab",
        "leakage_rule": (
            "A collection lab, its operators, scenes, cameras and robot instance occur in "
            "exactly one split."
        ),
        "inventory_episode_count": spec.indexed_episodes,
        "inventory_outcome_counts": {
            "success": spec.indexed_successes,
            "failure": spec.indexed_failures,
        },
        "splits": split_summaries,
    }


def build_dataset_manifest(spec: DROIDSourceSpec) -> DatasetManifest:
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
        task_families=("distributed_in_the_wild_manipulation",),
        native_action_semantics=spec.native_action_semantics,
        unit_conventions=spec.units,
        coordinate_frames=spec.coordinate_frames,
        priority=spec.priority,
        status=spec.status,
        checksum=f"sha256:{spec.object_manifest_sha256}",
        fps=spec.nominal_control_hz,
        camera_names=spec.camera_names,
        known_limitations=spec.known_limitations,
    )


def validate_videos(adapter: DROIDRawAdapter) -> dict[str, Any]:
    results: dict[str, Any] = {}
    decoded_frames = 0
    for outcome in OUTCOMES:
        results[outcome] = {}
        for camera in adapter.spec.camera_names:
            probe = adapter.validate_video("IPRL", outcome, camera)
            item = adapter.video_object("IPRL", outcome, camera)
            indices = (0, int(probe["frames"]) // 2, int(probe["frames"]) - 1)
            decoded = {
                str(index): _tensor_summary(
                    adapter.decode_rgb_frame(adapter.root / item.path, index)
                )
                for index in indices
            }
            decoded_frames += len(decoded)
            results[outcome][camera] = probe | {"decoded_frame_indices": decoded}
    return {
        "qualified_lab": "IPRL",
        "alignment": "frame_index_not_container_timestamp",
        "validated_streams": 6,
        "decoded_frames": decoded_frames,
        "streams": results,
    }


def qualify(*, download: bool) -> dict[str, Any]:
    spec = DROIDSourceSpec.from_toml(REGISTRY_PATH)
    if download:
        download_qualified_objects(RAW_BASE, spec)
    verified = verify_qualified_objects(RAW_BASE, spec)
    adapter = DROIDRawAdapter(RAW_BASE, spec)
    source_validation = adapter.validate_source()
    video_validation = validate_videos(adapter)

    canonical_samples: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        episode = adapter.canonical_episode(
            "IPRL", outcome, max_transitions=2, include_rgb=True
        )
        sample_path = CLEANED_BASE / f"IPRL_{outcome}_first_2_transitions.json"
        _write_json_once(sample_path, canonical_sample_payload(episode))
        canonical_samples.append(
            {
                "qualification_id": episode.metadata.episode_id,
                "path": sample_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(sample_path),
                "observations": len(episode.observations),
                "actions": len(episode.actions),
                "decoded_rgb_frames": sum(
                    len(observation.rgb) for observation in episode.observations
                ),
                "state_width": episode.observations[0].q.numel(),
                "native_action_width": episode.actions[0].native.numel(),
                "language_annotations": len(episode.language),
                "scene_metadata_records": len(episode.scene_metadata),
                "camera_calibration_roles": sorted(
                    episode.scene_metadata[0]["camera_extrinsics"]
                ),
                "original_timestamps_preserved": all(
                    "source_timestamps" in item for item in episode.scene_metadata
                ),
                "recorded_joint_velocity_preserved": all(
                    "source_recorded_joint_velocity_rad_s" in item
                    for item in episode.scene_metadata
                ),
            }
        )

    split_record = build_split_record(spec)
    manifest = build_dataset_manifest(spec)
    write_manifest(MANIFEST_PATH, manifest)
    _write_json_once(SPLIT_PATH, split_record)
    report = {
        "schema_version": 1,
        "gate": "stage1.4_source_qualification",
        "passed": True,
        "dataset_id": spec.dataset_id,
        "source_revision": spec.revision,
        "collection_code_revision": spec.collection_code_revision,
        "source_status": spec.status,
        "source_priority": spec.priority,
        "source_inventory": {
            "metadata_listing_pages": spec.inventory_pages,
            "metadata_listing_bytes": spec.inventory_metadata_bytes,
            "episodes": spec.indexed_episodes,
            "successes": spec.indexed_successes,
            "failures": spec.indexed_failures,
            "labs": len(spec.labs),
            "full_raw_snapshot_bytes_approx": 8_700_000_000_000,
        },
        "verified_objects": verified,
        "source_validation": source_validation,
        "video_validation": video_validation,
        "canonical_samples": canonical_samples,
        "split_record": split_record,
        "privacy": {
            "raw_identity_fields_present": True,
            "redacted_fields": ["user", "user_id"],
            "committed_evidence_contains_identity_values": False,
            "derived_episode_id_policy": "dataset_id:lab:outcome",
        },
        "admission_decision": "validated_for_stage1; not_yet_admitted_to_training",
        "admission_blockers": [
            "No held-out target-model value improvement result exists yet.",
            "DROID is Franka Panda data, not direct SO-101 embodiment data.",
            "The approximately 8.7 TB full raw release is outside this bounded qualification.",
            "Raw metadata contains contributor identity fields and requires redaction.",
            "Native controller actions must not be treated as SO-101 or canonical task-space actions.",
        ],
    }
    _write_json_once(QUALIFICATION_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download only missing objects from their exact immutable GCS generations",
    )
    args = parser.parse_args()
    report = qualify(download=args.download)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

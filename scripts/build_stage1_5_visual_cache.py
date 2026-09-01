#!/usr/bin/env python3
"""Build the ignored, action-free Stage 1.5 visual JEPA caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Iterable

import numpy as np

from data.adapters.armnetbench_so101 import (
    ArmnetBenchSO101Adapter,
    ArmnetBenchSourceSpec,
)
from data.adapters.project_ira_so101 import (
    ProjectIRASO101Adapter,
    ProjectIRASourceSpec,
)
from data.dataloaders.stage1_5_visual import Stage15VisualSamples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/training/stage1_5_droid_visual.toml"
ACTIVE_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs/training/stage1_5_droid_visual_protocol_v3.toml"
)
OBJECTS_PATH = (
    PROJECT_ROOT
    / "configs/datasets/registry/stage1_5_visual_subset_v2.objects.json"
)
DROID_REGISTRY_PATH = (
    PROJECT_ROOT / "configs/datasets/registry/droid_raw_1_0_1.toml"
)
PROJECT_REGISTRY_PATH = (
    PROJECT_ROOT / "configs/datasets/registry/project_ira_so101_v1.toml"
)
ARM_REGISTRY_PATH = (
    PROJECT_ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"
)
CACHE_ROOT = PROJECT_ROOT / "data/cache/stage1_5_visual"
CACHE_MANIFEST_PATH = CACHE_ROOT / "cache_manifest.json"

DROID_RAW = PROJECT_ROOT / "data/raw/public_real/droid_raw_1_0_1"
PROJECT_RAW = PROJECT_ROOT / "data/raw/public_real/project_ira_so101"
ARM_RAW = PROJECT_ROOT / "data/raw/public_real/armnetbench_so101"

VIEW_ORDER = ("exterior_primary", "exterior_secondary", "wrist")
def _video() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Stage 1.5 cache building requires PyAV") from error
    return av


def _hdf5() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Stage 1.5 cache building requires h5py") from error
    return h5py


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to replace cache record {path}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        Stage15VisualSamples.load(path)
        return
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".npz.part", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        Stage15VisualSamples.load(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def uniform_temporal_pairs(
    timestamps_seconds: np.ndarray[Any, np.dtype[np.float64]],
    *,
    horizon_seconds: float,
    maximum_horizon_seconds: float,
    sample_count: int,
    maximum_video_index: int,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Choose uniform contexts whose first >=horizon future frame is decodable."""

    if timestamps_seconds.ndim != 1 or len(timestamps_seconds) < 2:
        raise ValueError("timestamps must be a vector with at least two entries")
    if not np.isfinite(timestamps_seconds).all() or not np.all(
        np.diff(timestamps_seconds) > 0
    ):
        raise ValueError("timestamps must be finite and strictly increasing")
    indices = np.arange(len(timestamps_seconds), dtype=np.int64)
    future = np.searchsorted(
        timestamps_seconds,
        timestamps_seconds + horizon_seconds,
        side="left",
    ).astype(np.int64)
    candidate = (future < len(indices)) & (future <= maximum_video_index)
    deltas = np.full(len(indices), np.inf, dtype=np.float64)
    deltas[candidate] = (
        timestamps_seconds[future[candidate]] - timestamps_seconds[candidate]
    )
    valid = indices[candidate & (deltas <= maximum_horizon_seconds + 1e-9)]
    if len(valid) < sample_count:
        raise ValueError(
            f"episode has only {len(valid)} valid temporal contexts for {sample_count} samples"
        )
    positions = np.rint(np.linspace(0, len(valid) - 1, sample_count)).astype(
        np.int64
    )
    context = valid[positions]
    target = future[context]
    if len(np.unique(context)) != sample_count or not np.all(target > context):
        raise ValueError("uniform temporal sampling produced duplicate/non-future pairs")
    if np.any(timestamps_seconds[target] - timestamps_seconds[context] < horizon_seconds - 1e-9):
        raise ValueError("sampled temporal pair is below the frozen horizon")
    if np.any(
        timestamps_seconds[target] - timestamps_seconds[context]
        > maximum_horizon_seconds + 1e-9
    ):
        raise ValueError("sampled temporal pair exceeds the frozen maximum horizon")
    return context, target


def _resize_rgb(frame: Any, image_size: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    resized = frame.reformat(width=image_size, height=image_size, format="rgb24")
    array = resized.to_ndarray()
    if array.shape != (image_size, image_size, 3) or array.dtype != np.uint8:
        raise ValueError("decoded RGB frame has an unexpected resized representation")
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def decode_video_indices(
    path: Path,
    indices: Iterable[int],
    *,
    image_size: int,
) -> dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
    wanted = tuple(sorted(set(int(value) for value in indices)))
    if not wanted or wanted[0] < 0:
        raise ValueError("video frame indices must be non-empty and non-negative")
    result: dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    wanted_set = set(wanted)
    av = _video()
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted_set:
                result[index] = _resize_rgb(frame, image_size)
                if len(result) == len(wanted):
                    break
    missing = set(wanted) - set(result)
    if missing:
        raise ValueError(f"{path} lacks requested frames {sorted(missing)}")
    return result


def declared_video_frames(path: Path) -> int:
    av = _video()
    with av.open(str(path), mode="r") as container:
        frames = int(container.streams.video[0].frames or 0)
    if frames < 1:
        raise ValueError(f"{path} does not declare a positive video frame count")
    return frames


def _empty_sample_arrays(count: int, image_size: int) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "context_rgb": np.zeros(
            (count, 3, 3, image_size, image_size), dtype=np.uint8
        ),
        "future_rgb": np.zeros(
            (count, 3, 3, image_size, image_size), dtype=np.uint8
        ),
        "camera_valid": np.zeros((count, 3), dtype=np.bool_),
        "proprio": np.zeros((count, 24), dtype=np.float32),
        "episode_key": np.empty(count, dtype="S64"),
        "sample_index": np.arange(count, dtype=np.int64),
        "context_index": np.empty(count, dtype=np.int64),
        "future_index": np.empty(count, dtype=np.int64),
        "context_time_seconds": np.empty(count, dtype=np.float64),
        "future_time_seconds": np.empty(count, dtype=np.float64),
    }


def _merge_episode_arrays(
    episodes: list[dict[str, np.ndarray[Any, Any]]], image_size: int
) -> dict[str, np.ndarray[Any, Any]]:
    if not episodes:
        raise ValueError("cannot merge an empty episode list")
    keys = tuple(_empty_sample_arrays(0, image_size))
    merged = {key: np.concatenate([item[key] for item in episodes], axis=0) for key in keys}
    merged["sample_index"] = np.arange(len(merged["sample_index"]), dtype=np.int64)
    Stage15VisualSamples(**merged).validate(image_size=image_size)
    return merged


def _build_droid_episode(
    episode: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    image_size: int,
    horizon_seconds: float,
    maximum_horizon_seconds: float,
    sample_count: int,
) -> dict[str, np.ndarray[Any, Any]]:
    roles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        roles[str(item["role"])].append(item)
    if {name: len(roles[name]) for name in ("metadata", "trajectory", "video")} != {
        "metadata": 1,
        "trajectory": 1,
        "video": 3,
    }:
        raise ValueError("DROID selected episode does not contain the five-object contract")

    metadata_path = PROJECT_ROOT / roles["metadata"][0]["local_path"]
    trajectory_path = PROJECT_ROOT / roles["trajectory"][0]["local_path"]
    metadata = _json(metadata_path)
    if metadata.get("lab") != episode["lab"]:
        raise ValueError("DROID local metadata lab differs from the frozen selector")
    if bool(metadata.get("success")) != (episode["outcome"] == "success"):
        raise ValueError("DROID local metadata outcome differs from the frozen selector")
    serial_by_view = {
        0: str(metadata["ext1_cam_serial"]),
        1: str(metadata["ext2_cam_serial"]),
        2: str(metadata["wrist_cam_serial"]),
    }
    video_by_serial = {
        Path(str(item["path"])).stem: PROJECT_ROOT / item["local_path"]
        for item in roles["video"]
    }
    if set(video_by_serial) != set(serial_by_view.values()):
        raise ValueError("DROID camera serials differ between metadata and selected videos")

    h5py = _hdf5()
    with h5py.File(trajectory_path, "r") as handle:
        q_joint = np.asarray(handle[schema["state_joint_path"]], dtype=np.float32)
        q_gripper = np.asarray(handle[schema["state_gripper_path"]], dtype=np.float32)
        qdot_joint = np.asarray(
            handle[schema["state_joint_velocity_path"]], dtype=np.float32
        )
        command_joint = np.asarray(
            handle[schema["command_joint_position_path"]], dtype=np.float32
        )
        command_gripper = np.asarray(
            handle[schema["command_gripper_position_path"]], dtype=np.float32
        )
        timestamps_ms = np.asarray(
            handle[schema["timestamp_path"]], dtype=np.float64
        )
    length = len(q_joint)
    expected_shapes = (
        q_joint.shape == (length, 7),
        q_gripper.shape == (length,),
        qdot_joint.shape == (length, 7),
        command_joint.shape == (length, 7),
        command_gripper.shape == (length,),
        timestamps_ms.shape == (length,),
    )
    if not all(expected_shapes) or length != int(metadata["trajectory_length"]):
        raise ValueError("DROID trajectory arrays differ from the selected metadata contract")
    timestamps = (timestamps_ms - timestamps_ms[0]) / 1000.0
    declared_frames = {
        serial: declared_video_frames(path) for serial, path in video_by_serial.items()
    }
    shared_maximum_video_index = min(
        length - 2,
        min(declared_frames.values()) - 1,
    )
    context, future = uniform_temporal_pairs(
        timestamps,
        horizon_seconds=horizon_seconds,
        maximum_horizon_seconds=maximum_horizon_seconds,
        sample_count=sample_count,
        maximum_video_index=shared_maximum_video_index,
    )
    q = np.column_stack((q_joint, q_gripper)).astype(np.float32)
    gripper_velocity = np.empty(length, dtype=np.float32)
    gripper_velocity[1:] = np.diff(q_gripper) / np.diff(timestamps).astype(np.float32)
    gripper_velocity[0] = gripper_velocity[1]
    qdot = np.column_stack((qdot_joint, gripper_velocity)).astype(np.float32)
    commands = np.column_stack((command_joint, command_gripper)).astype(np.float32)
    previous = np.empty_like(commands)
    previous[0] = q[0]
    previous[1:] = commands[:-1]
    proprio = np.concatenate((q[context], qdot[context], previous[context]), axis=1)
    if proprio.shape != (sample_count, 24) or not np.isfinite(proprio).all():
        raise ValueError("DROID Stage 1.5 proprioception is not finite width 24")

    arrays = _empty_sample_arrays(sample_count, image_size)
    arrays["proprio"] = proprio.astype(np.float32)
    arrays["episode_key"][:] = str(episode["selector_sha256"]).encode("ascii")
    arrays["context_index"] = context
    arrays["future_index"] = future
    arrays["context_time_seconds"] = timestamps[context]
    arrays["future_time_seconds"] = timestamps[future]
    arrays["camera_valid"][:] = True
    requested = np.concatenate((context, future))
    for view, serial in serial_by_view.items():
        frames = decode_video_indices(
            video_by_serial[serial], requested, image_size=image_size
        )
        arrays["context_rgb"][:, view] = np.stack(
            [frames[int(index)] for index in context]
        )
        arrays["future_rgb"][:, view] = np.stack(
            [frames[int(index)] for index in future]
        )
    Stage15VisualSamples(**arrays).validate(image_size=image_size)
    return arrays


def build_droid_caches(
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, dict[str, np.ndarray[Any, Any]]]:
    schema = _toml(DROID_REGISTRY_PATH)["schema"]
    selection = config["selection"]["droid"]
    image_size = int(config["representation"]["image_size"])
    horizon = float(selection["temporal_horizon_seconds"])
    maximum_horizon = float(selection["maximum_temporal_horizon_seconds"])
    samples = int(selection["samples_per_episode"])
    objects_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in plan["objects"]:
        selector = str(item.get("episode_selector", ""))
        if item["dataset_id"] == selection["dataset_id"] and selector:
            objects_by_episode[selector].append(item)

    episodes_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in plan["selection"]["droid_episodes"]:
        episodes_by_split[str(episode["split_role"])].append(episode)
    if smoke:
        episodes_by_split = {
            split: values[:1] for split, values in episodes_by_split.items()
        }

    output: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    workers = int(config["acquisition"]["max_workers"])
    for split in ("train", "validation", "test"):
        selected = episodes_by_split[split]
        results: list[dict[str, np.ndarray[Any, Any]] | None] = [None] * len(selected)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _build_droid_episode,
                    episode,
                    objects_by_episode[str(episode["selector_sha256"])],
                    schema=schema,
                    image_size=image_size,
                    horizon_seconds=horizon,
                    maximum_horizon_seconds=maximum_horizon,
                    sample_count=samples,
                ): index
                for index, episode in enumerate(selected)
            }
            completed = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                completed += 1
                if completed % 16 == 0 or completed == len(selected):
                    print(f"decoded DROID {split}: {completed}/{len(selected)} episodes")
        output[split] = _merge_episode_arrays(
            [item for item in results if item is not None], image_size
        )
    return output


def _packed_timestamp_key(value: float) -> int:
    return round(value * 1_000_000_000)


def decode_packed_timestamps(
    path: Path,
    timestamps: Iterable[float],
    *,
    fps: float,
    image_size: int,
) -> dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
    requested = {
        _packed_timestamp_key(float(value)): float(value) for value in timestamps
    }
    result: dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    av = _video()
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        for key, timestamp in sorted(requested.items()):
            container.seek(
                int(timestamp / stream.time_base),
                stream=stream,
                any_frame=False,
                backward=True,
            )
            for frame in container.decode(stream):
                frame_time = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None
                    else timestamp
                )
                if frame_time + 0.5 / fps < timestamp:
                    continue
                result[key] = _resize_rgb(frame, image_size)
                break
    missing = set(requested) - set(result)
    if missing:
        raise ValueError(f"{path} could not decode {len(missing)} packed timestamps")
    return result


def _so101_episode_arrays(
    adapter: Any,
    episode_index: int,
    *,
    sample_count: int,
    horizon_seconds: float,
    maximum_horizon_seconds: float,
    image_size: int,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    record = adapter.episode_record(episode_index)
    start = int(record["dataset_from_index"])
    length = int(record["length"])
    rows = adapter._data_table().slice(start, length)
    q = np.asarray(rows["observation.state"].to_pylist(), dtype=np.float32)
    commands = np.asarray(rows["action"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(rows["timestamp"].to_numpy(), dtype=np.float64)
    if q.shape != (length, 6) or commands.shape != (length, 6):
        raise ValueError("SO-101 selected state/command arrays must be width six")
    context, future = uniform_temporal_pairs(
        timestamps,
        horizon_seconds=horizon_seconds,
        maximum_horizon_seconds=maximum_horizon_seconds,
        sample_count=sample_count,
        maximum_video_index=length - 1,
    )
    qdot = np.empty_like(q)
    qdot[1:] = np.diff(q, axis=0) / np.diff(timestamps)[:, None]
    qdot[0] = qdot[1]
    previous = np.empty_like(q)
    previous[0] = q[0]
    previous[1:] = commands[:-1]
    proprio18 = np.concatenate((q[context], qdot[context], previous[context]), axis=1)
    proprio = np.pad(proprio18, ((0, 0), (0, 6))).astype(np.float32)
    if not np.isfinite(proprio).all():
        raise ValueError("SO-101 Stage 1.5 proprioception contains non-finite values")

    arrays = _empty_sample_arrays(sample_count, image_size)
    key = f"{adapter.spec.dataset_id}:{episode_index:06d}".encode("ascii")
    arrays["episode_key"][:] = key
    arrays["proprio"] = proprio
    arrays["context_index"] = context
    arrays["future_index"] = future
    arrays["context_time_seconds"] = timestamps[context]
    arrays["future_time_seconds"] = timestamps[future]
    return arrays, {
        "episode_index": episode_index,
        "context": context,
        "future": future,
        "timestamps": timestamps,
    }


def build_so101_cache(
    *,
    adapter: Any,
    episode_indices: list[int],
    camera_to_view: dict[str, int],
    allowed_video_paths: set[Path],
    sample_count: int,
    horizon_seconds: float,
    maximum_horizon_seconds: float,
    image_size: int,
) -> dict[str, np.ndarray[Any, Any]]:
    episode_arrays: list[dict[str, np.ndarray[Any, Any]]] = []
    requests: dict[Path, list[tuple[float, int, bool, int]]] = defaultdict(list)
    offset = 0
    for episode_index in episode_indices:
        arrays, selected = _so101_episode_arrays(
            adapter,
            episode_index,
            sample_count=sample_count,
            horizon_seconds=horizon_seconds,
            maximum_horizon_seconds=maximum_horizon_seconds,
            image_size=image_size,
        )
        episode_arrays.append(arrays)
        for camera, view in camera_to_view.items():
            segment = adapter.video_segment(episode_index, camera)
            resolved = segment.path.resolve()
            if resolved not in allowed_video_paths:
                raise ValueError(
                    f"episode {episode_index}/{camera} uses a video outside the frozen plan"
                )
            arrays["camera_valid"][:, view] = True
            for is_future, indices in (
                (False, selected["context"]),
                (True, selected["future"]),
            ):
                for local_sample, frame_index in enumerate(indices):
                    local_time = float(selected["timestamps"][int(frame_index)])
                    timestamp = segment.from_timestamp + local_time
                    if not segment.from_timestamp <= timestamp < segment.to_timestamp:
                        raise ValueError("SO-101 packed timestamp lies outside its episode")
                    requests[resolved].append(
                        (timestamp, offset + local_sample, is_future, view)
                    )
        offset += sample_count

    merged = _merge_episode_arrays(episode_arrays, image_size)
    filled_context = np.zeros((len(merged["sample_index"]), 3), dtype=np.bool_)
    filled_future = np.zeros_like(filled_context)
    decoded_by_path: dict[Path, dict[int, np.ndarray[Any, np.dtype[np.uint8]]]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(requests))) as executor:
        futures = {
            executor.submit(
                decode_packed_timestamps,
                path,
                [item[0] for item in values],
                fps=float(adapter.spec.fps),
                image_size=image_size,
            ): path
            for path, values in requests.items()
        }
        for future in as_completed(futures):
            path = futures[future]
            decoded_by_path[path] = future.result()
            print(f"decoded packed video {path.name} ({len(requests[path])} assignments)")

    for path, values in requests.items():
        decoded = decoded_by_path[path]
        for timestamp, sample_index, is_future, view in values:
            frame = decoded[_packed_timestamp_key(timestamp)]
            target = merged["future_rgb"] if is_future else merged["context_rgb"]
            filled = filled_future if is_future else filled_context
            target[sample_index, view] = frame
            filled[sample_index, view] = True
    if not np.array_equal(filled_context, merged["camera_valid"]):
        raise ValueError("SO-101 context video assignments are incomplete")
    if not np.array_equal(filled_future, merged["camera_valid"]):
        raise ValueError("SO-101 future video assignments are incomplete")
    Stage15VisualSamples(**merged).validate(image_size=image_size)
    return merged


def _allowed_video_paths(
    plan: dict[str, Any], dataset_id: str, split: str
) -> set[Path]:
    return {
        (PROJECT_ROOT / item["local_path"]).resolve()
        for item in plan["objects"]
        if item["dataset_id"] == dataset_id
        and item["role"] == "video"
        and item["split_role"] == split
    }


def build_so101_caches(
    plan: dict[str, Any], config: dict[str, Any], *, smoke: bool
) -> dict[str, dict[str, dict[str, np.ndarray[Any, Any]]]]:
    image_size = int(config["representation"]["image_size"])
    output: dict[str, dict[str, dict[str, np.ndarray[Any, Any]]]] = {}
    specifications = (
        (
            "project_ira",
            ProjectIRASourceSpec.from_toml(PROJECT_REGISTRY_PATH),
            ProjectIRASO101Adapter,
            PROJECT_RAW,
            {"desk_view": 0, "wrist_left": 2},
        ),
        (
            "armnetbench",
            ArmnetBenchSourceSpec.from_toml(ARM_REGISTRY_PATH),
            ArmnetBenchSO101Adapter,
            ARM_RAW,
            {"front": 0, "top": 1, "wrist": 2},
        ),
    )
    for selection_name, spec, adapter_type, raw_root, camera_to_view in specifications:
        selection = config["selection"][selection_name]
        adapter = adapter_type(raw_root / spec.revision, spec)
        source_output: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
        for split in ("train", "test"):
            first = int(selection[f"{split}_episode_first"])
            last = int(selection[f"{split}_episode_last"])
            episode_indices = list(range(first, last + 1))
            if smoke:
                episode_indices = episode_indices[:1]
            source_output[split] = build_so101_cache(
                adapter=adapter,
                episode_indices=episode_indices,
                camera_to_view=camera_to_view,
                allowed_video_paths=_allowed_video_paths(
                    plan, str(selection["dataset_id"]), split
                ),
                sample_count=int(selection["samples_per_episode"]),
                horizon_seconds=float(config["selection"]["droid"]["temporal_horizon_seconds"]),
                maximum_horizon_seconds=float(
                    config["selection"]["droid"]["maximum_temporal_horizon_seconds"]
                ),
                image_size=image_size,
            )
            print(
                f"built {selection['dataset_id']} {split}: "
                f"{len(source_output[split]['sample_index'])} samples"
            )
        output[str(selection["dataset_id"])] = source_output
    return output


def _cache_path(dataset_id: str, split: str) -> Path:
    return CACHE_ROOT / f"{dataset_id}.{split}.npz"


def _cache_evidence(path: Path, samples: Stage15VisualSamples) -> dict[str, Any]:
    unique_episodes = len(np.unique(samples.episode_key))
    horizon = samples.future_time_seconds - samples.context_time_seconds
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "samples": len(samples),
        "episodes": unique_episodes,
        "views": int(samples.context_rgb.shape[1]),
        "valid_view_counts": sorted(
            set(samples.camera_valid.sum(axis=1).astype(int).tolist())
        ),
        "minimum_horizon_seconds": float(horizon.min()),
        "maximum_horizon_seconds": float(horizon.max()),
        "contains_action_fields": False,
    }


def build(*, smoke: bool) -> dict[str, Any]:
    config = _toml(CONFIG_PATH)
    active_protocol = _toml(ACTIVE_PROTOCOL_PATH)
    plan = _json(OBJECTS_PATH)
    if sha256_file(CONFIG_PATH) != active_protocol["base"]["config_sha256"]:
        raise ValueError("Stage 1.5 base config differs from the active protocol pin")
    if sha256_file(OBJECTS_PATH) != active_protocol["base"]["object_manifest_sha256"]:
        raise ValueError("Stage 1.5 object manifest differs from the active protocol pin")
    config["selection"]["droid"]["maximum_temporal_horizon_seconds"] = float(
        active_protocol["sampling"]["maximum_temporal_horizon_seconds"]
    )
    if plan["config_sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("Stage 1.5 object plan differs from the frozen config")
    if plan["acquisition"]["selected_bytes"] > config["acquisition"]["cap_bytes"]:
        raise ValueError("Stage 1.5 object plan exceeds the frozen cap")

    droid = build_droid_caches(plan, config, smoke=smoke)
    so101 = build_so101_caches(plan, config, smoke=smoke)
    all_arrays: dict[tuple[str, str], dict[str, np.ndarray[Any, Any]]] = {
        ("droid_raw_1_0_1", split): arrays for split, arrays in droid.items()
    }
    for dataset_id, splits in so101.items():
        all_arrays.update(
            {(dataset_id, split): arrays for split, arrays in splits.items()}
        )
    if smoke:
        return {
            "smoke": True,
            "caches": {
                f"{dataset_id}:{split}": {
                    "samples": len(arrays["sample_index"]),
                    "episodes": len(np.unique(arrays["episode_key"])),
                }
                for (dataset_id, split), arrays in sorted(all_arrays.items())
            },
        }

    cache_records: dict[str, Any] = {}
    for (dataset_id, split), arrays in sorted(all_arrays.items()):
        path = _cache_path(dataset_id, split)
        _write_npz_once(path, arrays)
        samples = Stage15VisualSamples.load(path)
        cache_records[f"{dataset_id}:{split}"] = _cache_evidence(path, samples)
    manifest = {
        "schema_version": 1,
        "protocol_revision": int(active_protocol["protocol_revision"]),
        "active_protocol_path": ACTIVE_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "active_protocol_sha256": sha256_file(ACTIVE_PROTOCOL_PATH),
        "gate": config["gate"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "object_manifest_sha256": sha256_file(OBJECTS_PATH),
        "view_order": list(VIEW_ORDER),
        "image_size": int(config["representation"]["image_size"]),
        "temporal_horizon_seconds": float(
            config["selection"]["droid"]["temporal_horizon_seconds"]
        ),
        "maximum_temporal_horizon_seconds": float(
            config["selection"]["droid"]["maximum_temporal_horizon_seconds"]
        ),
        "action_fields_included": False,
        "privacy": {
            "metadata_values_included": False,
            "droid_episode_keys": "SHA-256 selectors only",
        },
        "caches": cache_records,
    }
    _write_json_once(CACHE_MANIFEST_PATH, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="decode one episode per source/split without writing caches",
    )
    args = parser.parse_args()
    print(json.dumps(build(smoke=args.smoke), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plan, download, and verify the frozen Stage 1.5 visual subset.

Planning reads public object metadata only. It writes an exact, compact object
manifest before acquisition. Raw metadata, trajectories, and video always stay
under Git-ignored ``data/raw``.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/training/stage1_5_droid_visual.toml"
PLAN_PATH = (
    PROJECT_ROOT
    / "configs/datasets/registry/stage1_5_visual_subset.objects.json"
)
DROID_SPLIT_PATH = PROJECT_ROOT / "data/splits/droid_raw_1_0_1.json"

GCS_LIST_ENDPOINT = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
USER_AGENT = "Project-Subliminal/0.1"


@dataclass(frozen=True)
class SourceObject:
    dataset_id: str
    provider: str
    role: str
    path: str
    local_path: str
    size: int
    split_role: str
    generation: str = ""
    md5: str = ""
    sha256: str = ""
    object_name: str = ""
    repository_id: str = ""
    revision: str = ""
    lab: str = ""
    outcome: str = ""
    episode_selector: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        return self.dataset_id, self.path


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
            raise FileExistsError(
                f"refusing to replace frozen reproducibility record {path}"
            )
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _request_json(url: str, *, attempts: int = 5) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=180) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def gcs_list(*, prefix: str, match_glob: str | None = None) -> list[dict[str, Any]]:
    """List immutable metadata needed for selection from the public GCS API."""

    items: list[dict[str, Any]] = []
    page_token = ""
    while True:
        query = {
            "prefix": prefix,
            "maxResults": "1000",
            "fields": "items(name,size,generation,md5Hash),nextPageToken",
        }
        if match_glob:
            query["matchGlob"] = match_glob
        if page_token:
            query["pageToken"] = page_token
        payload = _request_json(f"{GCS_LIST_ENDPOINT}?{urlencode(query)}")
        items.extend(payload.get("items", ()))
        page_token = str(payload.get("nextPageToken", ""))
        if not page_token:
            break
    return items


def _gcs_md5(item: dict[str, Any]) -> str:
    encoded = str(item.get("md5Hash", ""))
    if not encoded:
        raise ValueError(f"GCS object lacks MD5: {item.get('name')}")
    return base64.b64decode(encoded, validate=True).hex()


def _episode_prefix(object_name: str) -> str:
    if "/recordings/" in object_name:
        return object_name.split("/recordings/", 1)[0]
    return object_name.rsplit("/", 1)[0]


def _split_by_lab() -> dict[str, str]:
    split = _json(DROID_SPLIT_PATH)
    mapping: dict[str, str] = {}
    for split_name, record in split["splits"].items():
        for lab in record["labs"]:
            if lab in mapping:
                raise ValueError(f"DROID lab occurs in multiple splits: {lab}")
            mapping[str(lab)] = str(split_name)
    return mapping


def _excluded_droid_prefixes(config: dict[str, Any]) -> set[str]:
    support = config["acquisition"]["pinned_support"]
    path = PROJECT_ROOT / support["droid_object_manifest"]
    if sha256_file(path) != support["droid_object_manifest_sha256"]:
        raise ValueError("DROID support object manifest differs from its frozen pin")
    return {
        _episode_prefix(str(item["object_name"]))
        for item in _json(path)["objects"]
        if item["role"] in {"metadata", "trajectory", "video"}
    }


def rank_episode_prefix(
    dataset_id: str, lab: str, outcome: str, episode_prefix: str
) -> bytes:
    value = f"{dataset_id}:stage1.5:{lab}:{outcome}:{episode_prefix}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def select_episode_prefixes(
    *,
    dataset_id: str,
    lab: str,
    outcome: str,
    listed_videos: Iterable[dict[str, Any]],
    excluded_prefixes: set[str],
    quota: int,
) -> tuple[list[str], dict[str, int]]:
    """Select complete non-stereo episodes by the frozen SHA-256 ranking."""

    complete, inventory = rank_complete_episode_prefixes(
        dataset_id=dataset_id,
        lab=lab,
        outcome=outcome,
        listed_videos=listed_videos,
        excluded_prefixes=excluded_prefixes,
    )
    if len(complete) < quota:
        raise ValueError(
            f"{lab}/{outcome} has only {len(complete)} complete eligible episodes, "
            f"below quota {quota}"
        )
    return complete[:quota], inventory


def rank_complete_episode_prefixes(
    *,
    dataset_id: str,
    lab: str,
    outcome: str,
    listed_videos: Iterable[dict[str, Any]],
    excluded_prefixes: set[str],
) -> tuple[list[str], dict[str, int]]:
    """Rank all three-video candidates; the caller checks other required objects."""

    videos_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stereo_objects = 0
    for item in listed_videos:
        name = str(item["name"])
        if not name.endswith(".mp4") or "/recordings/MP4/" not in name:
            continue
        if "-stereo" in name.rsplit("/", 1)[-1]:
            stereo_objects += 1
            continue
        videos_by_episode[_episode_prefix(name)].append(item)

    complete = sorted(
        (
            prefix
            for prefix, videos in videos_by_episode.items()
            if len(videos) == 3 and prefix not in excluded_prefixes
        ),
        key=lambda prefix: (
            rank_episode_prefix(dataset_id, lab, outcome, prefix),
            prefix,
        ),
    )
    return complete, {
        "listed_video_objects": sum(len(items) for items in videos_by_episode.values())
        + stereo_objects,
        "stereo_video_objects": stereo_objects,
        "non_stereo_episode_prefixes": len(videos_by_episode),
        "complete_eligible_episode_prefixes": len(complete),
    }


def _scan_cell(
    release_prefix: str,
    dataset_id: str,
    lab: str,
    outcome: str,
    excluded: set[str],
    quota: int,
) -> tuple[str, str, list[str], dict[str, int]]:
    cell_prefix = f"{release_prefix}/{lab}/{outcome}/"
    objects = gcs_list(
        prefix=cell_prefix,
        match_glob=f"{cell_prefix}**/recordings/MP4/*.mp4",
    )
    selected, inventory = rank_complete_episode_prefixes(
        dataset_id=dataset_id,
        lab=lab,
        outcome=outcome,
        listed_videos=objects,
        excluded_prefixes=excluded,
    )
    if len(selected) < quota:
        raise ValueError(
            f"{lab}/{outcome} has only {len(selected)} three-video candidates, "
            f"below quota {quota}"
        )
    return lab, outcome, selected, inventory


def _selected_episode_objects(
    *,
    dataset_id: str,
    release_prefix: str,
    lab: str,
    outcome: str,
    split_role: str,
    episode_prefix: str,
) -> tuple[dict[str, Any], list[SourceObject]]:
    listed = gcs_list(prefix=f"{episode_prefix}/")
    metadata = [
        item
        for item in listed
        if _episode_prefix(str(item["name"])) == episode_prefix
        and str(item["name"]).rsplit("/", 1)[-1].startswith("metadata_")
        and str(item["name"]).endswith(".json")
    ]
    trajectories = [
        item
        for item in listed
        if str(item["name"]) == f"{episode_prefix}/trajectory.h5"
    ]
    videos = [
        item
        for item in listed
        if str(item["name"]).startswith(f"{episode_prefix}/recordings/MP4/")
        and str(item["name"]).endswith(".mp4")
        and "-stereo" not in str(item["name"]).rsplit("/", 1)[-1]
    ]
    if (len(metadata), len(trajectories), len(videos)) != (1, 1, 3):
        raise ValueError(
            f"selected episode {episode_prefix} does not have exactly 1 metadata, "
            f"1 trajectory, and 3 non-stereo video objects"
        )

    selector = hashlib.sha256(
        f"{dataset_id}:stage1.5:{lab}:{outcome}:{episode_prefix}".encode("utf-8")
    ).hexdigest()
    selected: list[SourceObject] = []
    for role, items in (
        ("metadata", metadata),
        ("trajectory", trajectories),
        ("video", videos),
    ):
        for item in items:
            object_name = str(item["name"])
            relative = object_name.removeprefix(f"{release_prefix}/")
            selected.append(
                SourceObject(
                    dataset_id=dataset_id,
                    provider="gcs",
                    role=role,
                    path=relative,
                    local_path=(
                        Path("data/raw/public_real/droid_raw_1_0_1") / relative
                    ).as_posix(),
                    size=int(item["size"]),
                    split_role=split_role,
                    generation=str(item["generation"]),
                    md5=_gcs_md5(item),
                    object_name=object_name,
                    lab=lab,
                    outcome=outcome,
                    episode_selector=selector,
                )
            )
    episode = {
        "episode_prefix": episode_prefix,
        "lab": lab,
        "outcome": outcome,
        "selector_sha256": selector,
        "split_role": split_role,
        "bytes": sum(item.size for item in selected),
        "objects": len(selected),
    }
    return episode, selected


def _droid_objects(
    config: dict[str, Any], *, max_workers: int
) -> tuple[list[dict[str, Any]], list[SourceObject], dict[str, Any]]:
    selection = config["selection"]["droid"]
    registry = _toml(PROJECT_ROOT / "configs/datasets/registry/droid_raw_1_0_1.toml")
    dataset = registry["dataset"]
    if selection["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("DROID Stage 1.5 dataset ID differs from source registry")
    if selection["revision"] != dataset["revision"]:
        raise ValueError("DROID Stage 1.5 revision differs from source registry")

    dataset_id = str(selection["dataset_id"])
    release_prefix = str(dataset["release_prefix"])
    labs = tuple(str(value) for value in dataset["labs"])
    outcomes = tuple(str(value) for value in selection["expected_outcomes"])
    quota = int(selection["quota_per_lab_outcome"])
    if len(labs) != int(selection["expected_labs"]):
        raise ValueError("DROID lab count differs from frozen Stage 1.5 contract")
    excluded = _excluded_droid_prefixes(config)
    split_by_lab = _split_by_lab()
    if set(split_by_lab) != set(labs):
        raise ValueError("DROID source split does not cover the frozen labs exactly")

    cell_results: list[tuple[str, str, list[str], dict[str, int]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _scan_cell,
                release_prefix,
                dataset_id,
                lab,
                outcome,
                excluded,
                quota,
            ): (lab, outcome)
            for lab in labs
            for outcome in outcomes
        }
        for future in as_completed(futures):
            result = future.result()
            print(f"selected {len(result[2])} episodes from {result[0]}/{result[1]}")
            cell_results.append(result)
    cell_results.sort(key=lambda item: (item[0], item[1]))

    candidates = {
        (lab, outcome): prefixes
        for lab, outcome, prefixes, _ in cell_results
    }
    accepted: dict[tuple[str, str], list[tuple[dict[str, Any], list[SourceObject]]]] = {
        cell: [] for cell in candidates
    }
    rejected: Counter[tuple[str, str]] = Counter()
    next_index: Counter[tuple[str, str]] = Counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Any, tuple[str, str]] = {}

        def submit_next(cell: tuple[str, str]) -> None:
            index = next_index[cell]
            if index >= len(candidates[cell]):
                raise ValueError(
                    f"{cell[0]}/{cell[1]} exhausted candidates below quota {quota}"
                )
            next_index[cell] += 1
            lab, outcome = cell
            future = executor.submit(
                _selected_episode_objects,
                dataset_id=dataset_id,
                release_prefix=release_prefix,
                lab=lab,
                outcome=outcome,
                split_role=split_by_lab[lab],
                episode_prefix=candidates[cell][index],
            )
            futures[future] = cell

        for cell in candidates:
            submit_next(cell)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                cell = futures.pop(future)
                try:
                    accepted[cell].append(future.result())
                    completed += 1
                    if completed % 32 == 0 or completed == int(selection["expected_episodes"]):
                        print(
                            f"pinned {completed}/{selection['expected_episodes']} "
                            "eligible DROID episodes"
                        )
                except ValueError:
                    rejected[cell] += 1
                if len(accepted[cell]) < quota:
                    submit_next(cell)

    episodes = [episode for values in accepted.values() for episode, _ in values]
    objects = [item for values in accepted.values() for _, items in values for item in items]
    if len(episodes) != int(selection["expected_episodes"]):
        raise ValueError("selected DROID episode count differs from frozen contract")

    episodes.sort(key=lambda item: (item["lab"], item["outcome"], item["selector_sha256"]))
    objects.sort(key=lambda item: item.identity)
    if len(objects) != int(selection["expected_episodes"]) * int(
        selection["objects_per_episode"]
    ):
        raise ValueError("DROID object count differs from frozen contract")
    inventory = {
        f"{lab}:{outcome}": values
        | {"rejected_missing_required_objects": rejected[(lab, outcome)]}
        for lab, outcome, _, values in cell_results
    }
    return episodes, objects, inventory


def _support_objects(config: dict[str, Any]) -> list[SourceObject]:
    support = config["acquisition"]["pinned_support"]
    result: list[SourceObject] = []

    droid_path = PROJECT_ROOT / support["droid_object_manifest"]
    if sha256_file(droid_path) != support["droid_object_manifest_sha256"]:
        raise ValueError("DROID support manifest failed its pin")
    droid_manifest = _json(droid_path)
    for item in droid_manifest["objects"]:
        if item["role"] != "license":
            continue
        result.append(
            SourceObject(
                dataset_id="droid_raw_1_0_1",
                provider="gcs",
                role="license",
                path=str(item["path"]),
                local_path=(
                    Path("data/raw/public_real/droid_raw_1_0_1") / item["path"]
                ).as_posix(),
                size=int(item["size"]),
                split_role="support",
                generation=str(item["generation"]),
                md5=str(item["md5"]),
                sha256=str(item["sha256"]),
                object_name=str(item["object_name"]),
            )
        )

    project_path = PROJECT_ROOT / support["project_ira_registry"]
    if sha256_file(project_path) != support["project_ira_registry_sha256"]:
        raise ValueError("Project IRA support registry failed its pin")
    project = _toml(project_path)
    project_dataset = project["dataset"]
    for item in project["qualified_files"]:
        result.append(
            SourceObject(
                dataset_id="project_ira_so101_v1",
                provider="huggingface",
                role=str(item["role"]),
                path=str(item["path"]),
                local_path=(
                    Path("data/raw/public_real/project_ira_so101")
                    / project_dataset["revision"]
                    / item["path"]
                ).as_posix(),
                size=int(item["size"]),
                split_role="support",
                sha256=str(item["sha256"]),
                repository_id=str(project_dataset["repository_id"]),
                revision=str(project_dataset["revision"]),
            )
        )

    arm_path = PROJECT_ROOT / support["armnetbench_object_manifest"]
    if sha256_file(arm_path) != support["armnetbench_object_manifest_sha256"]:
        raise ValueError("ArmnetBench support object manifest failed its pin")
    arm_registry = _toml(
        PROJECT_ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"
    )
    arm_dataset = arm_registry["dataset"]
    for item in _json(arm_path)["objects"]:
        result.append(
            SourceObject(
                dataset_id="armnetbench_so101_v01",
                provider="huggingface",
                role=str(item["role"]),
                path=str(item["path"]),
                local_path=(
                    Path("data/raw/public_real/armnetbench_so101")
                    / arm_dataset["revision"]
                    / item["path"]
                ).as_posix(),
                size=int(item["size"]),
                split_role="support",
                sha256=str(item["sha256"]),
                repository_id=str(arm_dataset["repository_id"]),
                revision=str(arm_dataset["revision"]),
            )
        )
    return result


def _configured_so101_objects(config: dict[str, Any]) -> list[SourceObject]:
    registries = {
        "project_ira_so101_v1": _toml(
            PROJECT_ROOT / "configs/datasets/registry/project_ira_so101_v1.toml"
        )["dataset"],
        "armnetbench_so101_v01": _toml(
            PROJECT_ROOT / "configs/datasets/registry/armnetbench_so101_v01.toml"
        )["dataset"],
    }
    roots = {
        "project_ira_so101_v1": Path("data/raw/public_real/project_ira_so101"),
        "armnetbench_so101_v01": Path("data/raw/public_real/armnetbench_so101"),
    }
    result: list[SourceObject] = []
    for item in config["so101_video_objects"]:
        dataset_id = str(item["dataset_id"])
        registry = registries[dataset_id]
        revision = str(registry["revision"])
        selection_revision = str(config["selection"]["project_ira" if dataset_id.startswith("project") else "armnetbench"]["revision"])
        if revision != selection_revision:
            raise ValueError(f"{dataset_id} Stage 1.5 revision differs from its registry")
        result.append(
            SourceObject(
                dataset_id=dataset_id,
                provider="huggingface",
                role="video",
                path=str(item["path"]),
                local_path=(roots[dataset_id] / revision / item["path"]).as_posix(),
                size=int(item["size"]),
                split_role=str(item["split_role"]),
                sha256=str(item["sha256"]),
                repository_id=str(registry["repository_id"]),
                revision=revision,
            )
        )
    return result


def deduplicate_objects(objects: Iterable[SourceObject]) -> list[SourceObject]:
    deduplicated: dict[tuple[str, str], SourceObject] = {}
    for item in objects:
        previous = deduplicated.get(item.identity)
        if previous is not None:
            pins = (item.size, item.generation, item.md5, item.sha256)
            previous_pins = (
                previous.size,
                previous.generation,
                previous.md5,
                previous.sha256,
            )
            if pins != previous_pins:
                raise ValueError(f"conflicting pins for {item.identity}")
            if previous.split_role == "support" and item.split_role != "support":
                deduplicated[item.identity] = item
            continue
        deduplicated[item.identity] = item
    return sorted(deduplicated.values(), key=lambda item: item.identity)


def build_plan(config: dict[str, Any], *, max_workers: int) -> dict[str, Any]:
    episodes, droid, inventory = _droid_objects(config, max_workers=max_workers)
    objects = deduplicate_objects(
        [*_support_objects(config), *_configured_so101_objects(config), *droid]
    )
    cap = int(config["acquisition"]["cap_bytes"])
    total_bytes = sum(item.size for item in objects)
    if total_bytes > cap:
        raise ValueError(f"frozen acquisition is {total_bytes} bytes, above cap {cap}")

    role_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    dataset_bytes: Counter[str] = Counter()
    split_bytes: Counter[str] = Counter()
    for item in objects:
        role_counts[item.role] += 1
        dataset_counts[item.dataset_id] += 1
        dataset_bytes[item.dataset_id] += item.size
        split_bytes[f"{item.dataset_id}:{item.split_role}"] += item.size

    return {
        "schema_version": 1,
        "gate": str(config["gate"]),
        "status": "frozen_before_acquisition",
        "frozen_at": str(config["frozen_at"]),
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "privacy": {
            "raw_metadata_contents_committed": False,
            "identity_values_copied_from_metadata": False,
            "object_names_are_public_provider_pins": True,
        },
        "selection": {
            "algorithm": config["selection"]["droid"]["algorithm"],
            "droid_episode_count": len(episodes),
            "droid_episodes": episodes,
            "droid_cell_inventory": inventory,
            "stage1_4_episode_prefixes_excluded": 26,
        },
        "acquisition": {
            "cap_bytes": cap,
            "selected_bytes": total_bytes,
            "headroom_bytes": cap - total_bytes,
            "object_count": len(objects),
            "dataset_object_counts": dict(sorted(dataset_counts.items())),
            "dataset_bytes": dict(sorted(dataset_bytes.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "dataset_split_bytes": dict(sorted(split_bytes.items())),
        },
        "objects": [asdict(item) for item in objects],
    }


def _verify_source_object(item: SourceObject, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != item.size:
        raise ValueError(f"size mismatch for {item.local_path}")
    if item.md5 and md5_file(path) != item.md5:
        raise ValueError(f"MD5 mismatch for {item.local_path}")
    if item.sha256 and sha256_file(path) != item.sha256:
        raise ValueError(f"SHA-256 mismatch for {item.local_path}")


def _download_url(item: SourceObject) -> str:
    if item.provider == "gcs":
        return (
            "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/"
            f"{quote(item.object_name, safe='')}?alt=media&generation={item.generation}"
        )
    if item.provider == "huggingface":
        return (
            f"https://huggingface.co/datasets/{item.repository_id}/resolve/"
            f"{item.revision}/{quote(item.path, safe='/')}?download=true"
        )
    raise ValueError(f"unsupported provider: {item.provider}")


def _download_one(item: SourceObject) -> str:
    destination = PROJECT_ROOT / item.local_path
    if destination.exists():
        _verify_source_object(item, destination)
        return f"verified existing {item.dataset_id}:{item.path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.stage1_5.part")
    if partial.exists():
        raise FileExistsError(
            f"partial download exists; inspect it before retrying: {partial}"
        )
    request = Request(_download_url(item), headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=600) as response, partial.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    _verify_source_object(item, partial)
    os.replace(partial, destination)
    return f"downloaded {item.dataset_id}:{item.path} ({item.size} bytes)"


def _objects_from_plan(plan: dict[str, Any]) -> list[SourceObject]:
    return [SourceObject(**item) for item in plan["objects"]]


def verify_plan_contract(plan: dict[str, Any], config: dict[str, Any]) -> None:
    if plan["config_sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("Stage 1.5 plan was generated from a different config")
    objects = _objects_from_plan(plan)
    if sum(item.size for item in objects) != plan["acquisition"]["selected_bytes"]:
        raise ValueError("Stage 1.5 plan byte total is inconsistent")
    if len(objects) != plan["acquisition"]["object_count"]:
        raise ValueError("Stage 1.5 plan object count is inconsistent")
    if plan["acquisition"]["selected_bytes"] > int(config["acquisition"]["cap_bytes"]):
        raise ValueError("Stage 1.5 plan exceeds the current acquisition cap")
    if len(plan["selection"]["droid_episodes"]) != int(
        config["selection"]["droid"]["expected_episodes"]
    ):
        raise ValueError("Stage 1.5 plan episode count is inconsistent")


def download_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    objects = _objects_from_plan(plan)
    missing = [item for item in objects if not (PROJECT_ROOT / item.local_path).exists()]
    required_bytes = sum(item.size for item in missing)
    reserve = int(config["acquisition"]["minimum_free_reserve_bytes"])
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < required_bytes + reserve:
        raise OSError(
            f"download requires {required_bytes} bytes plus {reserve} reserve; "
            f"only {free_bytes} bytes are free"
        )
    workers = int(config["acquisition"]["max_workers"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, item): item for item in objects}
        completed = 0
        for future in as_completed(futures):
            print(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"verified/downloaded {completed}/{len(futures)} objects")


def verify_local(plan: dict[str, Any], *, require_all: bool) -> dict[str, Any]:
    objects = _objects_from_plan(plan)
    verified = 0
    verified_bytes = 0
    missing: list[str] = []
    for item in objects:
        path = PROJECT_ROOT / item.local_path
        if not path.exists():
            missing.append(item.local_path)
            continue
        _verify_source_object(item, path)
        verified += 1
        verified_bytes += item.size
    if require_all and missing:
        raise FileNotFoundError(f"{len(missing)} Stage 1.5 objects are missing")
    return {
        "verified_objects": verified,
        "verified_bytes": verified_bytes,
        "missing_objects": len(missing),
        "complete": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="query public object metadata and write the exact frozen object manifest",
    )
    mode.add_argument(
        "--download",
        action="store_true",
        help="download missing objects from the already-frozen object manifest",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="require and verify every object in the frozen manifest",
    )
    args = parser.parse_args()
    config = _toml(CONFIG_PATH)
    if args.plan:
        plan = build_plan(
            config, max_workers=int(config["acquisition"]["max_workers"])
        )
        _write_json_once(PLAN_PATH, plan)
        result = verify_local(plan, require_all=False)
    else:
        if not PLAN_PATH.is_file():
            raise FileNotFoundError("run --plan and freeze its output before acquisition")
        plan = _json(PLAN_PATH)
        verify_plan_contract(plan, config)
        if args.download:
            download_plan(plan, config)
        result = verify_local(plan, require_all=args.download or args.verify)
    print(
        json.dumps(
            {"plan": plan["acquisition"], "local": result},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

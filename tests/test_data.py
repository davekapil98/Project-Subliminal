from pathlib import Path

import pytest
import torch

from data.canonical_schema import Action, CanonicalEpisode, EpisodeMetadata, Observation
from data.manifests import DatasetManifest, read_manifest, write_manifest
from data.splits import split_for_episode


def metadata(*, domain: str = "sim") -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_id="toy-1",
        source_dataset="synthetic-fixture",
        source_version="revision-1",
        source_url="https://example.invalid/dataset",
        license="internal-test",
        redistribution_terms="not for redistribution",
        domain=domain,
        robot_id="mock-seven-axis" if domain == "sim" else "mock-real-arm",
        embodiment="seven-axis-arm",
        task="reach",
        success=True,
        quality=1.0,
        collection_method="synthetic-test" if domain == "sim" else "teleop",
        native_action_semantics="relative joint position",
        simulator_family="test-simulator" if domain == "sim" else None,
        simulator_version="1" if domain == "sim" else None,
        fps=10.0,
        camera_names=("rear",),
    )


def observation(timestamp: float, joints: int = 7) -> Observation:
    return Observation(
        timestamp=timestamp,
        q=torch.zeros(joints),
        qdot=torch.zeros(joints),
        previous_command=torch.zeros(joints),
    )


def test_canonical_episode_is_multi_embodiment_and_transition_aligned() -> None:
    episode = CanonicalEpisode(
        metadata(),
        (observation(0.0), observation(0.1)),
        (Action(timestamp=0.05, native=torch.zeros(7)),),
        language=("reach the target",),
    )
    episode.validate()
    assert episode.metadata.failure is False


def test_domain_is_explicit_and_simulator_metadata_is_required() -> None:
    values = metadata().__dict__ | {"domain": "unknown"}
    with pytest.raises(ValueError, match="explicitly"):
        EpisodeMetadata(**values)
    values = metadata().__dict__ | {"simulator_family": None}
    with pytest.raises(ValueError, match="simulator_family"):
        EpisodeMetadata(**values)


def test_manifest_round_trip_is_pinned_and_write_once(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset_id="priority-a-fixture",
        revision="exact-revision",
        source_url="https://example.invalid/card",
        license="Apache-2.0",
        redistribution_terms="retain attribution",
        domain="real",
        robot_id="so101-public",
        embodiment="SO-101",
        data_format="lerobot-parquet-video",
        modalities=("rgb", "joint_state", "action", "language"),
        task_families=("pick", "place"),
        native_action_semantics="absolute SO-101 joint position",
        unit_conventions={"joint_position": "radian"},
        coordinate_frames={"camera": "source-calibrated"},
        priority="A",
        status="validated",
        checksum="sha256:fixture",
        fps=30.0,
        camera_names=("rear",),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert read_manifest(path) == manifest
    write_manifest(path, manifest)
    changed = DatasetManifest(**(manifest.__dict__ | {"revision": "different"}))
    with pytest.raises(FileExistsError, match="overwrite"):
        write_manifest(path, changed)


def test_episode_splits_are_stable_and_episode_level() -> None:
    assert split_for_episode("episode-123") == split_for_episode("episode-123")
    assert split_for_episode("episode-123") in {"train", "validation", "test"}

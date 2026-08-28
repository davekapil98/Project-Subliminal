import torch

from data.canonical_schema import Action, CanonicalEpisode, EpisodeMetadata, Observation
from data.splits import split_for_episode


def observation(timestamp: float) -> Observation:
    return Observation(
        timestamp=timestamp,
        q=torch.zeros(6),
        qdot=torch.zeros(6),
        previous_command=torch.zeros(6),
    )


def test_canonical_episode_validates_transition_alignment() -> None:
    metadata = EpisodeMetadata(
        episode_id="toy-1",
        source_dataset="synthetic",
        source_version="1",
        license="internal-test",
        robot_id="mock-so101",
        embodiment="so101",
        task="reach",
        success=True,
        quality=1.0,
    )
    episode = CanonicalEpisode(
        metadata,
        (observation(0.0), observation(0.1)),
        (Action(timestamp=0.05, native=torch.zeros(6)),),
    )
    episode.validate()


def test_episode_splits_are_stable_and_episode_level() -> None:
    assert split_for_episode("episode-123") == split_for_episode("episode-123")
    assert split_for_episode("episode-123") in {"train", "validation", "test"}

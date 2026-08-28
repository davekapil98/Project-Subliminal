"""Leakage-resistant deterministic episode-level splitting."""

import hashlib


def split_for_episode(episode_id: str, *, validation: float = 0.1, test: float = 0.1) -> str:
    if not 0.0 <= validation < 1.0 or not 0.0 <= test < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if validation + test >= 1.0:
        raise ValueError("validation and test fractions must sum below one")
    digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test:
        return "test"
    if value < test + validation:
        return "validation"
    return "train"

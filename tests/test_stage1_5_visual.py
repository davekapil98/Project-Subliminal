import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from data.dataloaders.stage1_5_visual import (
    FrozenVisualTarget,
    SourceNormalization,
    Stage15VisualSamples,
)


ROOT = Path(__file__).resolve().parents[1]


def _cache_builder():
    path = ROOT / "scripts/build_stage1_5_visual_cache.py"
    spec = importlib.util.spec_from_file_location("stage1_5_cache_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arrays(*, samples: int = 4, project_views: bool = False) -> dict[str, np.ndarray]:
    context = np.arange(
        samples * 3 * 3 * 64 * 64, dtype=np.uint32
    ).reshape(samples, 3, 3, 64, 64).astype(np.uint8)
    future = np.flip(context, axis=-1).copy()
    camera_valid = np.ones((samples, 3), dtype=np.bool_)
    if project_views:
        camera_valid[:, 1] = False
        context[:, 1] = 0
        future[:, 1] = 0
    return {
        "context_rgb": context,
        "future_rgb": future,
        "camera_valid": camera_valid,
        "proprio": np.arange(samples * 24, dtype=np.float32).reshape(samples, 24),
        "episode_key": np.asarray([b"safe-episode"] * samples, dtype="S64"),
        "sample_index": np.arange(samples, dtype=np.int64),
        "context_index": np.arange(samples, dtype=np.int64),
        "future_index": np.arange(samples, dtype=np.int64) + 8,
        "context_time_seconds": np.arange(samples, dtype=np.float64),
        "future_time_seconds": np.arange(samples, dtype=np.float64) + 0.5,
    }


def test_cache_round_trip_preserves_three_view_action_free_contract(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    arrays = _arrays(project_views=True)
    np.savez_compressed(path, **arrays)
    loaded = Stage15VisualSamples.load(path)

    assert len(loaded) == 4
    assert loaded.context_rgb.shape == (4, 3, 3, 64, 64)
    assert loaded.proprio.shape == (4, 24)
    assert loaded.camera_valid.sum(axis=1).tolist() == [2, 2, 2, 2]
    assert not any("action" in key for key in arrays)


def test_cache_rejects_action_field_and_nonzero_missing_camera(tmp_path: Path) -> None:
    action_path = tmp_path / "action.npz"
    arrays = _arrays()
    np.savez_compressed(action_path, **arrays, action=np.zeros((4, 6)))
    with pytest.raises(ValueError, match="unexpected"):
        Stage15VisualSamples.load(action_path)

    invalid_path = tmp_path / "invalid-camera.npz"
    arrays = _arrays(project_views=True)
    arrays["context_rgb"][0, 1, 0, 0, 0] = 1
    np.savez_compressed(invalid_path, **arrays)
    with pytest.raises(ValueError, match="zero-filled"):
        Stage15VisualSamples.load(invalid_path)


def test_fixed_visual_target_and_normalization_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, **_arrays(samples=8))
    samples = Stage15VisualSamples.load(path)
    first = FrozenVisualTarget(seed=1515)
    second = FrozenVisualTarget(seed=1515)
    future = torch.from_numpy(samples.future_rgb)

    assert sum(parameter.numel() for parameter in first.parameters()) == 0
    assert torch.equal(first.projection, second.projection)
    assert torch.equal(first(future), second(future))
    assert first(future).shape == (8, 4, 16)

    normalization = SourceNormalization.fit(samples, first, target_batch_size=3)
    normalized_proprio = normalization.normalize_proprio(
        torch.from_numpy(samples.proprio)
    )
    normalized_target = normalization.normalize_target(first(future))
    assert torch.allclose(normalized_proprio.mean(dim=0), torch.zeros(24), atol=1e-6)
    assert torch.allclose(normalized_target.mean(dim=0), torch.zeros(4, 16), atol=1e-5)


def test_uniform_temporal_pairs_choose_first_frame_at_or_after_horizon() -> None:
    builder = _cache_builder()
    timestamps = np.arange(30, dtype=np.float64) / 10.0
    context, future = builder.uniform_temporal_pairs(
        timestamps,
        horizon_seconds=0.5,
        sample_count=8,
        maximum_video_index=28,
    )
    assert len(np.unique(context)) == 8
    assert np.allclose(timestamps[future] - timestamps[context], 0.5)
    assert future[-1] <= 28

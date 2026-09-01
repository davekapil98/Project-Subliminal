"""Model-specific, action-free samples for the Stage 1.5 visual JEPA gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


VIEW_COUNT = 3
IMAGE_CHANNELS = 3
PROPRIO_WIDTH = 24
ALLOWED_CACHE_KEYS = frozenset(
    {
        "context_rgb",
        "future_rgb",
        "camera_valid",
        "proprio",
        "episode_key",
        "sample_index",
        "context_index",
        "future_index",
        "context_time_seconds",
        "future_time_seconds",
    }
)


@dataclass(frozen=True)
class Stage15VisualSamples:
    context_rgb: np.ndarray[Any, np.dtype[np.uint8]]
    future_rgb: np.ndarray[Any, np.dtype[np.uint8]]
    camera_valid: np.ndarray[Any, np.dtype[np.bool_]]
    proprio: np.ndarray[Any, np.dtype[np.float32]]
    episode_key: np.ndarray[Any, np.dtype[np.bytes_]]
    sample_index: np.ndarray[Any, np.dtype[np.int64]]
    context_index: np.ndarray[Any, np.dtype[np.int64]]
    future_index: np.ndarray[Any, np.dtype[np.int64]]
    context_time_seconds: np.ndarray[Any, np.dtype[np.float64]]
    future_time_seconds: np.ndarray[Any, np.dtype[np.float64]]

    @classmethod
    def load(cls, path: Path, *, image_size: int = 64) -> "Stage15VisualSamples":
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            unexpected = keys - ALLOWED_CACHE_KEYS
            missing = ALLOWED_CACHE_KEYS - keys
            if unexpected or missing:
                raise ValueError(
                    f"Stage 1.5 cache keys differ; missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            if any("action" in key.lower() for key in keys):
                raise ValueError("Stage 1.5 visual cache must not contain action fields")
            samples = cls(**{key: archive[key].copy() for key in ALLOWED_CACHE_KEYS})
        samples.validate(image_size=image_size)
        return samples

    def validate(self, *, image_size: int = 64) -> None:
        count = len(self.sample_index)
        image_shape = (count, VIEW_COUNT, IMAGE_CHANNELS, image_size, image_size)
        if self.context_rgb.shape != image_shape or self.future_rgb.shape != image_shape:
            raise ValueError(f"context/future RGB must both have shape {image_shape}")
        if self.context_rgb.dtype != np.uint8 or self.future_rgb.dtype != np.uint8:
            raise ValueError("context/future RGB must be uint8")
        if self.camera_valid.shape != (count, VIEW_COUNT):
            raise ValueError("camera_valid must have shape [samples, 3]")
        if self.camera_valid.dtype != np.bool_:
            raise ValueError("camera_valid must be boolean")
        if self.proprio.shape != (count, PROPRIO_WIDTH):
            raise ValueError("proprio must have shape [samples, 24]")
        if self.proprio.dtype != np.float32 or not np.isfinite(self.proprio).all():
            raise ValueError("proprio must be finite float32")
        if self.episode_key.shape != (count,) or self.episode_key.dtype.kind != "S":
            raise ValueError("episode_key must be a fixed-width byte-string vector")
        integer_arrays = (self.sample_index, self.context_index, self.future_index)
        if any(value.shape != (count,) or value.dtype != np.int64 for value in integer_arrays):
            raise ValueError("sample/context/future indices must be int64 vectors")
        time_arrays = (self.context_time_seconds, self.future_time_seconds)
        if any(value.shape != (count,) or value.dtype != np.float64 for value in time_arrays):
            raise ValueError("context/future times must be float64 vectors")
        if not all(np.isfinite(value).all() for value in time_arrays):
            raise ValueError("context/future times must be finite")
        if not np.all(self.future_index > self.context_index):
            raise ValueError("every future frame index must follow its context index")
        horizon = self.future_time_seconds - self.context_time_seconds
        if np.any(horizon < 0.5 - 1e-9):
            raise ValueError("future samples violate the frozen 0.5-second horizon")
        invalid = ~self.camera_valid
        if np.any(self.context_rgb[invalid]) or np.any(self.future_rgb[invalid]):
            raise ValueError("invalid camera slots must be exactly zero-filled")
        if count and not np.array_equal(
            self.sample_index, np.arange(count, dtype=np.int64)
        ):
            raise ValueError("sample_index must be contiguous and ordered")

    def __len__(self) -> int:
        return len(self.sample_index)


class FrozenVisualTarget(nn.Module):
    """Frozen seeded projection from future RGB to four 16-D target tokens."""

    def __init__(
        self,
        *,
        image_size: int = 64,
        pool_size: int = 16,
        target_tokens: int = 4,
        target_width: int = 16,
        seed: int = 1515,
    ) -> None:
        super().__init__()
        if image_size % pool_size:
            raise ValueError("image_size must be divisible by pool_size")
        self.image_size = image_size
        self.pool_size = pool_size
        self.target_tokens = target_tokens
        self.target_width = target_width
        input_width = VIEW_COUNT * IMAGE_CHANNELS * pool_size * pool_size
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(
            input_width,
            target_tokens * target_width,
            generator=generator,
            dtype=torch.float64,
        ) / input_width**0.5
        self.register_buffer("projection", projection, persistent=True)

    def forward(self, future_rgb: Tensor) -> Tensor:
        expected_tail = (
            VIEW_COUNT,
            IMAGE_CHANNELS,
            self.image_size,
            self.image_size,
        )
        if future_rgb.ndim != 5 or tuple(future_rgb.shape[1:]) != expected_tail:
            raise ValueError(f"future_rgb must have shape [B, {expected_tail}]")
        images = future_rgb.to(dtype=torch.float64) / 255.0
        pooled = F.adaptive_avg_pool2d(
            images.flatten(0, 1), (self.pool_size, self.pool_size)
        ).reshape(images.shape[0], -1)
        target = pooled @ self.projection
        return target.reshape(
            images.shape[0], self.target_tokens, self.target_width
        ).to(torch.float32)


@dataclass(frozen=True)
class SourceNormalization:
    proprio_mean: Tensor
    proprio_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    @classmethod
    def fit(
        cls,
        samples: Stage15VisualSamples,
        target_projector: FrozenVisualTarget,
        *,
        target_batch_size: int = 256,
    ) -> "SourceNormalization":
        if not len(samples):
            raise ValueError("cannot fit normalization from an empty training cache")
        proprio = torch.from_numpy(samples.proprio).to(torch.float64)
        proprio_mean = proprio.mean(dim=0).to(torch.float32)
        proprio_std = proprio.std(dim=0, correction=0).clamp_min(1e-6).to(torch.float32)
        targets = []
        with torch.no_grad():
            for start in range(0, len(samples), target_batch_size):
                future = torch.from_numpy(
                    samples.future_rgb[start : start + target_batch_size]
                )
                targets.append(target_projector(future).to(torch.float64))
        all_targets = torch.cat(targets, dim=0)
        target_mean = all_targets.mean(dim=0).to(torch.float32)
        target_std = (
            all_targets.std(dim=0, correction=0).clamp_min(1e-6).to(torch.float32)
        )
        return cls(
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            target_mean=target_mean,
            target_std=target_std,
        )

    def normalize_proprio(self, value: Tensor) -> Tensor:
        return (value - self.proprio_mean.to(value.device)) / self.proprio_std.to(
            value.device
        )

    def normalize_target(self, value: Tensor) -> Tensor:
        return (value - self.target_mean.to(value.device)) / self.target_std.to(
            value.device
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "proprio_mean": self.proprio_mean.tolist(),
            "proprio_std": self.proprio_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "fit_scope": "frozen source training cache only",
        }

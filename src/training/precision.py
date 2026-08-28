"""Device-aware mixed-precision selection with safe CPU fallback."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    device: torch.device
    parameter_dtype: torch.dtype
    autocast_dtype: torch.dtype | None

    def autocast(self) -> ContextManager[None]:
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
        )


def resolve_precision(
    *,
    device: str = "auto",
    precision: str = "auto",
) -> PrecisionPolicy:
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
    )
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if precision == "fp32" or selected_device.type == "cpu":
        return PrecisionPolicy(selected_device, torch.float32, None)
    if precision == "bf16":
        return PrecisionPolicy(selected_device, torch.float32, torch.bfloat16)
    if precision == "fp16":
        return PrecisionPolicy(selected_device, torch.float32, torch.float16)
    # Automatic CUDA selection favors BF16 when the device reports support.
    autocast = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return PrecisionPolicy(selected_device, torch.float32, autocast)

"""Rotary positional embeddings for temporal and token order."""

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Apply RoPE to query/key tensors shaped ``[B, heads, T, head_dim]``."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        if query.shape[-1] != self.inverse_frequency.numel() * 2:
            raise ValueError("query/key head dimension does not match RoPE")
        query_positions = torch.arange(query.shape[-2], device=query.device)
        key_positions = torch.arange(key.shape[-2], device=key.device)
        query_angles = torch.outer(query_positions.float(), self.inverse_frequency)
        key_angles = torch.outer(key_positions.float(), self.inverse_frequency)
        return self._rotate(query, query_angles), self._rotate(key, key_angles)

    @staticmethod
    def _rotate(inputs: Tensor, angles: Tensor) -> Tensor:
        angles = angles.to(device=inputs.device, dtype=inputs.dtype)
        cosine = angles.cos()[None, None, :, :]
        sine = angles.sin()[None, None, :, :]
        even, odd = inputs[..., 0::2], inputs[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(-2)

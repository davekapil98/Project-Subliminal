"""Attention mask construction utilities."""

import torch
from torch import Tensor


def causal_attention_mask(length: int, *, device: torch.device) -> Tensor:
    """Return a boolean mask where ``True`` entries are allowed to attend."""

    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def combine_attention_masks(
    sequence_length: int,
    *,
    device: torch.device,
    causal: bool,
    valid_tokens: Tensor | None,
) -> Tensor | None:
    """Combine causal and per-batch validity masks for attention logits."""

    mask: Tensor | None = None
    if causal:
        mask = causal_attention_mask(sequence_length, device=device)[None, None]
    if valid_tokens is not None:
        if valid_tokens.ndim != 2 or valid_tokens.shape[1] != sequence_length:
            raise ValueError("valid_tokens must have shape [B, T]")
        key_mask = valid_tokens[:, None, None, :].bool()
        mask = key_mask if mask is None else mask & key_mask
    return mask

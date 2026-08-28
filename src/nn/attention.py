"""Reference grouped-query self-attention implementation."""

import math

import torch
from torch import Tensor, nn

from nn.masking import combine_attention_masks
from nn.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Self-attention with fewer key/value heads than query heads.

    Inputs and outputs use ``[batch, tokens, d_model]``. This implementation is
    intentionally explicit for Stage 0 inspection; optimized attention can be
    placed behind the same interface later.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        dropout: float = 0.0,
        rope: bool = True,
    ) -> None:
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_dim = d_model // num_heads
        if rope and head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.query = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.key = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.value = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.output = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(head_dim) if rope else None

    def forward(
        self,
        inputs: Tensor,
        *,
        causal: bool = False,
        valid_tokens: Tensor | None = None,
    ) -> Tensor:
        batch, tokens, _ = inputs.shape
        query = self._split(self.query(inputs), self.num_heads)
        key = self._split(self.key(inputs), self.num_kv_heads)
        value = self._split(self.value(inputs), self.num_kv_heads)
        if self.rope is not None:
            query, key = self.rope(query, key)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = combine_attention_masks(
            tokens,
            device=inputs.device,
            causal=causal,
            valid_tokens=valid_tokens,
        )
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probabilities = torch.dropout(probabilities, self.dropout, self.training)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().view(batch, tokens, self.d_model)
        return self.output(context)

    def _split(self, projection: Tensor, heads: int) -> Tensor:
        batch, tokens, _ = projection.shape
        return projection.view(batch, tokens, heads, self.head_dim).transpose(1, 2)

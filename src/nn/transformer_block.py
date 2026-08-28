"""Pre-normalized Transformer block shared across tiny models."""

from torch import Tensor, nn

from nn.attention import GroupedQueryAttention
from nn.rmsnorm import RMSNorm
from nn.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim or int(d_model * 8 / 3)
        self.attention_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout=dropout)

    def forward(
        self,
        inputs: Tensor,
        *,
        causal: bool = False,
        valid_tokens: Tensor | None = None,
    ) -> Tensor:
        hidden = inputs + self.attention(
            self.attention_norm(inputs), causal=causal, valid_tokens=valid_tokens
        )
        return hidden + self.ffn(self.ffn_norm(hidden))

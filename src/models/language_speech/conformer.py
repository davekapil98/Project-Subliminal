"""Small readable Conformer block for Stage 0 speech experiments."""

from torch import Tensor, nn


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, d_model * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = self.glu(self.pointwise_in(hidden))
        hidden = self.activation(self.batch_norm(self.depthwise(hidden)))
        return self.dropout(self.pointwise_out(hidden).transpose(1, 2))


class TinyConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn_one = ConformerFeedForward(d_model, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.convolution = ConformerConvolution(d_model, dropout=dropout)
        self.ffn_two = ConformerFeedForward(d_model, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor, *, valid_tokens: Tensor | None = None) -> Tensor:
        hidden = inputs + 0.5 * self.ffn_one(inputs)
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=(~valid_tokens.bool()) if valid_tokens is not None else None,
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.convolution(hidden)
        hidden = hidden + 0.5 * self.ffn_two(hidden)
        return self.output_norm(hidden)

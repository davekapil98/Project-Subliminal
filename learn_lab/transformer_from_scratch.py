"""Minimal pre-norm Transformer assembled from readable components."""

from torch import Tensor, nn

from learn_lab.attention_from_scratch import LearningAttention
from learn_lab.rmsnorm_from_scratch import LearningRMSNorm


class LearningTransformerBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm_one = LearningRMSNorm(width)
        self.attention = LearningAttention(width)
        self.norm_two = LearningRMSNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )

    def forward(self, tokens: Tensor, *, causal: bool = False) -> Tensor:
        attended, _ = self.attention(self.norm_one(tokens), causal=causal)
        tokens = tokens + attended
        return tokens + self.ffn(self.norm_two(tokens))

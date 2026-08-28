"""Text/semantic Transformer used after token or speech encoding."""

from torch import Tensor, nn

from nn.rmsnorm import RMSNorm
from nn.transformer_block import TransformerBlock


class SemanticTransformer(nn.Module):
    def __init__(self, d_model: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, tokens: Tensor, valid_tokens: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens, valid_tokens=valid_tokens)
        return self.norm(tokens)

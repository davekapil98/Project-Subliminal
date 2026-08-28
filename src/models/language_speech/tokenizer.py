"""Deterministic byte tokenizer used only for local Stage 0 plumbing."""

import torch
from torch import Tensor


class ByteTokenizer:
    PAD = 0
    BOS = 1
    BYTE_OFFSET = 2
    vocab_size = 258

    def encode(self, texts: list[str], *, max_length: int = 64) -> tuple[Tensor, Tensor]:
        sequences: list[list[int]] = []
        for text in texts:
            byte_ids = [byte + self.BYTE_OFFSET for byte in text.encode("utf-8")]
            sequences.append(([self.BOS] + byte_ids)[:max_length])
        width = max(len(sequence) for sequence in sequences)
        token_ids = torch.full((len(texts), width), self.PAD, dtype=torch.long)
        valid = torch.zeros((len(texts), width), dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            token_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            valid[row, : len(sequence)] = True
        return token_ids, valid

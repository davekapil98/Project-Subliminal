"""Tiny local text/speech-to-structured-task model."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.language_speech.audio_frontend import AudioFrontend
from models.language_speech.conformer import TinyConformerBlock
from models.language_speech.intent_head import IntentEntityHead
from models.language_speech.semantic_transformer import SemanticTransformer


@dataclass
class LanguageSpeechOutput:
    semantic_token: Tensor
    intent_logits: Tensor
    entity_logits: Tensor
    attribute_logits: Tensor


class TinyLanguageSpeech(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 258,
        mel_bins: int = 80,
        bus_dim: int = 64,
        d_model: int = 192,
        semantic_depth: int = 4,
        conformer_depth: int = 2,
        num_heads: int = 6,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.audio_frontend = AudioFrontend(mel_bins, d_model)
        self.conformer = nn.ModuleList(
            [TinyConformerBlock(d_model, num_heads) for _ in range(conformer_depth)]
        )
        self.modality_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.semantic = SemanticTransformer(d_model, semantic_depth, num_heads)
        self.to_bus = nn.Linear(d_model, bus_dim)
        self.heads = IntentEntityHead(d_model)

    def forward(
        self,
        *,
        text_ids: Tensor | None = None,
        text_valid: Tensor | None = None,
        log_mel: Tensor | None = None,
        audio_valid: Tensor | None = None,
    ) -> LanguageSpeechOutput:
        if text_ids is None and log_mel is None:
            raise ValueError("at least one of text_ids or log_mel is required")
        sequences: list[Tensor] = []
        masks: list[Tensor] = []
        if text_ids is not None:
            text = self.text_embedding(text_ids) + self.modality_embedding[0]
            sequences.append(text)
            masks.append(
                text_ids.ne(0) if text_valid is None else text_valid.bool()
            )
        if log_mel is not None:
            audio = self.audio_frontend(log_mel)
            for block in self.conformer:
                audio = block(audio, valid_tokens=audio_valid)
            audio = audio + self.modality_embedding[1]
            sequences.append(audio)
            masks.append(
                torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
                if audio_valid is None
                else audio_valid.bool()
            )
        tokens = torch.cat(sequences, dim=1)
        valid = torch.cat(masks, dim=1)
        hidden = self.semantic(tokens, valid)
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        intent, entity, attribute = self.heads(pooled)
        return LanguageSpeechOutput(
            semantic_token=self.to_bus(pooled),
            intent_logits=intent,
            entity_logits=entity,
            attribute_logits=attribute,
        )

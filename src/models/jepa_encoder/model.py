"""Tiny multimodal JEPA context encoder.

Shapes:
    images: [B, V, 3, height, width]
    proprioception: [B, 18] (q, qdot, previous command)
    world_tokens: [B, N, bus_dim]
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nn.rmsnorm import RMSNorm
from nn.transformer_block import TransformerBlock


@dataclass
class JEPAEncoderOutput:
    world_tokens: Tensor
    object_logits: Tensor
    pose: Tensor


class TinyJEPAEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 8,
        max_views: int = 2,
        proprio_dim: int = 18,
        d_model: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        world_tokens: int = 16,
        bus_dim: int = 64,
        num_object_classes: int = 16,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.max_views = max_views
        self.world_token_count = world_tokens
        self.patch_embedding = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.proprio_embedding = nn.Sequential(
            nn.Linear(proprio_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.view_embedding = nn.Parameter(torch.zeros(max_views, d_model))
        self.type_embedding = nn.Parameter(torch.zeros(2, d_model))
        self.context_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(depth)]
        )
        self.context_norm = RMSNorm(d_model)
        self.resampler_queries = nn.Parameter(torch.randn(world_tokens, d_model) * 0.02)
        self.resampler = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.to_bus = nn.Linear(d_model, bus_dim)
        self.object_head = nn.Linear(bus_dim, num_object_classes)
        self.pose_head = nn.Linear(bus_dim, 7)

    def forward(
        self,
        images: Tensor,
        proprioception: Tensor,
        *,
        camera_valid: Tensor | None = None,
    ) -> JEPAEncoderOutput:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B, V, 3, H, W]")
        batch, views, channels, height, width = images.shape
        if views > self.max_views:
            raise ValueError(f"at most {self.max_views} camera views are supported")
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected square {self.image_size}px images")

        patches = self.patch_embedding(images.reshape(batch * views, channels, height, width))
        patches = patches.flatten(2).transpose(1, 2)
        patches_per_view = patches.shape[1]
        patches = patches.reshape(batch, views, patches_per_view, -1)
        patches = patches + self.view_embedding[:views][None, :, None]
        patches = patches + self.type_embedding[0]
        patches = patches.flatten(1, 2)
        proprio_token = self.proprio_embedding(proprioception).unsqueeze(1)
        proprio_token = proprio_token + self.type_embedding[1]
        hidden = torch.cat((patches, proprio_token), dim=1)

        valid_tokens: Tensor | None = None
        if camera_valid is not None:
            if camera_valid.shape != (batch, views):
                raise ValueError("camera_valid must have shape [B, V]")
            patch_valid = camera_valid.bool().repeat_interleave(patches_per_view, dim=1)
            proprio_valid = torch.ones(batch, 1, dtype=torch.bool, device=images.device)
            valid_tokens = torch.cat((patch_valid, proprio_valid), dim=1)

        for block in self.context_blocks:
            hidden = block(hidden, valid_tokens=valid_tokens)
        hidden = self.context_norm(hidden)
        queries = self.resampler_queries.unsqueeze(0).expand(batch, -1, -1)
        key_padding_mask = ~valid_tokens if valid_tokens is not None else None
        resampled, _ = self.resampler(
            queries, hidden, hidden, key_padding_mask=key_padding_mask, need_weights=False
        )
        world = self.to_bus(resampled)
        return JEPAEncoderOutput(
            world_tokens=world,
            object_logits=self.object_head(world),
            pose=self.pose_head(world),
        )

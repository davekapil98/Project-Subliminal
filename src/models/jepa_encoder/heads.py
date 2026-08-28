"""Optional explicit grounding heads for object attributes and pose."""

from torch import Tensor, nn


class GroundingHeads(nn.Module):
    def __init__(self, bus_dim: int, classes: int, attributes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(bus_dim, classes)
        self.attributes = nn.Linear(bus_dim, attributes)
        self.pose = nn.Linear(bus_dim, 7)

    def forward(self, world_tokens: Tensor) -> dict[str, Tensor]:
        return {
            "class_logits": self.classifier(world_tokens),
            "attribute_logits": self.attributes(world_tokens),
            "pose": self.pose(world_tokens),
        }

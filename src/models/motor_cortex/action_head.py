"""Action-space vector-field projection."""

from torch import Tensor, nn


class ActionVectorFieldHead(nn.Module):
    def __init__(self, d_model: int, joints: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, joints)

    def forward(self, action_hidden: Tensor) -> Tensor:
        return self.projection(action_hidden)

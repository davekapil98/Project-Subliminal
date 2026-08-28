"""Interface boundary for future physical SO-101 drivers."""

from typing import Protocol

from torch import Tensor


class SO101Interface(Protocol):
    def state(self) -> Tensor: ...

    def execute_relative(self, action_prefix: Tensor) -> Tensor: ...

    def hold(self) -> None: ...

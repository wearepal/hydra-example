"""Optimisation and training code."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from torch.nn import Parameter
from torch.optim import Optimizer

__all__ = ["OptimisationCfg"]


@dataclass
class OptimisationCfg:
    """Config class for the optimisation."""

    lr: float = 5.0e-5
    weight_decay: float = 0.0
    optimizer_cls: str = "torch.optim.AdamW"
    optimizer_kwargs: dict[str, Any] | None = None

    def build(self, params: Iterator[tuple[str, Parameter]]) -> Optimizer:
        # Instantiate the optimizer for the given parameters.
        raise NotImplementedError()

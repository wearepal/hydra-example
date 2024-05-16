"""Neural network models definitions."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing_extensions import override

from torch import nn


class Activation(Enum):
    """Activation functions for neural networks as a handy enum."""

    RELU = (nn.ReLU,)
    GELU = (nn.GELU,)
    SELU = (nn.SELU,)
    SILU = (nn.SiLU,)

    def __init__(self, init: Callable[[], nn.Module]) -> None:
        self.init = init


class NormType(Enum):
    """Normalization layers for neural networks as a handy enum."""

    BN = (nn.BatchNorm1d,)
    LN = (nn.LayerNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init


# Even though the following abstract class doesn't have any fields, we nevertheless need to mark it
# as a dataclass, because otherwise Hydra will complain when we use it as a type annotation in the
# `Config` class.


@dataclass(eq=False)
class ModelFactory(ABC):
    """Interface for model factories."""

    @abstractmethod
    def build(self, in_dim: int, *, out_dim: int) -> nn.Module:
        raise NotImplementedError()


@dataclass(eq=False, kw_only=True)
class FcnFactory(ModelFactory):
    """Factory for fully-connected neural networks."""

    num_hidden: int = 1
    hidden_dim: int | None = None
    activation: Activation = Activation.GELU
    norm: NormType | None = NormType.LN
    dropout_prob: float = 0.0
    final_bias: bool = True
    input_norm: bool = False

    def _make_block(self, in_features: int, *, out_features: int) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        if self.norm is not None:
            block.append(self.norm.init(out_features))
        block.append(self.activation.init())
        if self.dropout_prob > 0:
            block.append(nn.Dropout(p=self.dropout_prob))
        return block

    @override
    def build(self, in_dim: int, *, out_dim: int, with_flatten: bool = False) -> nn.Sequential:
        predictor = nn.Sequential()
        if with_flatten:
            predictor.append(nn.Flatten())
        if self.input_norm and (self.norm is not None):
            predictor.append(self.norm.init(in_dim))
        curr_dim = in_dim
        if self.num_hidden > 0:
            hidden_dim = in_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(self._make_block(in_features=curr_dim, out_features=hidden_dim))
                curr_dim = hidden_dim
        predictor.append(
            nn.Linear(in_features=curr_dim, out_features=out_dim, bias=self.final_bias)
        )
        return predictor


@dataclass(eq=False, kw_only=True)
class SimpleCNNFactory(ModelFactory):
    """Factory for a very simple CNN."""

    kernel_size: int = 5
    pool_stride: int = 2
    activation: Activation = Activation.RELU

    @override
    def build(self, in_dim: int, *, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=6, kernel_size=self.kernel_size),
            self.activation.init(),
            nn.MaxPool2d(kernel_size=2, stride=self.pool_stride),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=self.kernel_size),
            self.activation.init(),
            nn.MaxPool2d(kernel_size=2, stride=self.pool_stride),
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            self.activation.init(),
            nn.Linear(in_features=120, out_features=84),
            self.activation.init(),
            nn.Linear(in_features=84, out_features=out_dim),
        )

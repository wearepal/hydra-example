"""Neural network models definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

from torch import nn

type Activation = Literal["relu", "gelu", "selu", "silu"]


def _get_activation(activation: Activation) -> nn.Module:
    match activation:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "selu":
            return nn.SELU()
        case "silu":
            return nn.SiLU()


type NormType = Literal["BN", "LN"]


def _get_norm(norm_type: NormType, num_features: int) -> nn.Module:
    match norm_type:
        case "BN":
            return nn.BatchNorm1d(num_features)
        case "LN":
            return nn.LayerNorm(num_features)


# Even though the following abstract class doesn't have any fields, we nevertheless need
# to mark it as a dataclass, because otherwise Hydra will complain when we use it as a
# type annotation in the `Config` class.


@dataclass
class ModelFactory(ABC):
    """Interface for model factories."""

    @abstractmethod
    def build(self, in_dim: int, *, out_dim: int) -> nn.Module:
        raise NotImplementedError()


@dataclass(kw_only=True)
class FcnFactory(ModelFactory):
    """Factory for fully-connected neural networks."""

    num_hidden: int = 1
    hidden_dim: int | None = None
    activation: Activation = "gelu"
    norm: NormType | None = "LN"
    dropout_prob: float = 0.0
    final_bias: bool = True
    input_norm: bool = False

    def _make_block(self, in_features: int, *, out_features: int) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        if self.norm is not None:
            block.append(_get_norm(self.norm, out_features))
        block.append(_get_activation(self.activation))
        if self.dropout_prob > 0:
            block.append(nn.Dropout(p=self.dropout_prob))
        return block

    @override
    def build(
        self, in_dim: int, *, out_dim: int, with_flatten: bool = False
    ) -> nn.Sequential:
        predictor = nn.Sequential()
        if with_flatten:
            predictor.append(nn.Flatten())
        if self.input_norm and (self.norm is not None):
            predictor.append(_get_norm(self.norm, in_dim))
        curr_dim = in_dim
        if self.num_hidden > 0:
            hidden_dim = in_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(
                    self._make_block(in_features=curr_dim, out_features=hidden_dim)
                )
                curr_dim = hidden_dim
        predictor.append(
            nn.Linear(in_features=curr_dim, out_features=out_dim, bias=self.final_bias)
        )
        return predictor


@dataclass(kw_only=True)
class SimpleCNNFactory(ModelFactory):
    """Factory for a very simple CNN."""

    kernel_size: int = 5
    pool_stride: int = 2
    activation: Activation = "relu"

    @override
    def build(self, in_dim: int, *, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=6, kernel_size=self.kernel_size),
            _get_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=self.pool_stride),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=self.kernel_size),
            _get_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=self.pool_stride),
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            _get_activation(self.activation),
            nn.Linear(in_features=120, out_features=84),
            _get_activation(self.activation),
            nn.Linear(in_features=84, out_features=out_dim),
        )

"""Data loading and preprocessing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing_extensions import override

from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["Dataset", "DataModule", "CelebADataModule", "ColoredMNISTDataModule"]


@dataclass(kw_only=True, eq=False)
class DataModule(ABC):
    """Base class for data modules."""

    root: Path = Path("/srv/galene0/shared/data")

    @property
    @abstractmethod
    def train_data(self) -> Dataset[tuple[Tensor, Tensor]]: ...

    @property
    @abstractmethod
    def in_dim(self) -> int: ...

    @property
    @abstractmethod
    def out_dim(self) -> int: ...

    @abstractmethod
    def prepare(self, seed: int) -> None: ...


class CelebAttr(Enum):
    BLOND_HAIR = "Blond_Hair"
    MALE = "Male"
    SMILING = "Smiling"


@dataclass(kw_only=True, eq=False)
class CelebADataModule(DataModule):
    """Data-module for the CelebA dataset."""

    default_res: int = 224
    superclass: CelebAttr = CelebAttr.BLOND_HAIR
    subclass: CelebAttr = CelebAttr.MALE
    download: bool = False
    split_seed: int | None = None

    @property
    @override
    def train_data(self) -> Dataset[tuple[Tensor, Tensor]]:
        # Load data from disk.
        # Preprocess data.
        raise NotImplementedError()

    @property
    @override
    def in_dim(self) -> int:
        return self.default_res

    @property
    @override
    def out_dim(self) -> int:
        return len(CelebAttr)

    @override
    def prepare(self, seed: int) -> None:
        # Download data if necessary.
        pass


@dataclass(kw_only=True, eq=False)
class ColoredMNISTDataModule(DataModule):
    """Data-module for the ColoredMNIST dataset."""

    default_res: int = 28

    # Colorization settings
    label_map: dict[int, int] | None = None
    colors: list[int] | None = None
    num_colors: int = 10
    val_prop: float = 0.2

    @property
    @override
    def train_data(self) -> Dataset[tuple[Tensor, Tensor]]:
        # Load data from disk.
        # Preprocess data.
        raise NotImplementedError()

    @property
    @override
    def in_dim(self) -> int:
        return self.default_res

    @property
    @override
    def out_dim(self) -> int:
        return 10

    @override
    def prepare(self, seed: int) -> None:
        # Download data if necessary.
        pass

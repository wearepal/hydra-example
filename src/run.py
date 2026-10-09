"""Main run function and config class."""

from dataclasses import dataclass, field
from typing import Any, Final

import torch

from src.datasets import CelebADataModule, ColoredMNISTDataModule, DataModule
from src.logging import WandbCfg
from src.models import FcnFactory, ModelFactory, SimpleCNNFactory
from src.optimisation import OptimisationCfg

__all__ = ["Config", "CONFIG_GROUPS"]

# Config groups enable us to have different configurations for different subcomponents.
# For example, one subcomponent is the data module, and the different data modules,
# CelebA and ColoredMNIST, need different keys and values to be configured.
CONFIG_GROUPS: Final[dict[str, dict[str, type]]] = {
    "dm": {"celeba": CelebADataModule, "cmnist": ColoredMNISTDataModule},
    "model": {"fcn": FcnFactory, "cnn": SimpleCNNFactory},
}


@dataclass(kw_only=True)
class Config:
    """Main configuration class for the code base."""

    # Hydra's defaults list. It is not a real config field: Hydra removes it from the
    # composed config, but `OmegaConf.to_object()` then fills it in again from
    # `default_factory`, so the instantiated `Config` still holds this list. (It has to
    # be a dataclass field; Hydra doesn't see a `ClassVar`.)
    defaults: list[Any] = field(
        default_factory=lambda: [
            # `_self_` (this class) has to come first. Otherwise Hydra merges this class
            # last, and the abstract types of `dm` and `model` overwrite the types of
            # the selected config group options.
            "_self_",
            # Then we choose defaults for the config groups.
            # The keys and values are those of the config groups defined in
            # `CONFIG_GROUPS` above.
            {"dm": "cmnist"},
            {"model": "fcn"},
            # For subconfigs that aren't config groups, we can just set `None` here and
            # hydra then uses the default values from the dataclass. This makes it
            # possible to write `opt=...` on the command line instead of `+opt=...`.
            {"opt": None},
        ]
    )

    # The first two fields refer to configuration groups.
    # This is why hydra doesn't let us specify a default for them here.
    # The defaults are instead specified in the `defaults` list above.
    dm: DataModule
    model: ModelFactory

    # These are normal subconfigs, for which we can specify defaults, but note that in
    # dataclasses, the default may not be mutable, so we use `default_factory`.
    opt: OptimisationCfg = field(default_factory=OptimisationCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)

    # These are normal fields, for which we can specify defaults.
    seed: int = 42
    gpu: int = 0  # Set to -1 to use CPU.

    def run(self, config_for_logging: dict[str, Any]) -> float:
        """Run the experiment."""
        print(f"Starting a run with seed {self.seed} and GPU {self.gpu}.")
        # Set the seed for reproducibility.
        torch.manual_seed(self.seed)

        # Initialize the logger.
        wandb_run = self.wandb.init(config_for_logging, reinit=True)
        if wandb_run is not None:
            wandb_run.log({"accuracy": 0.5})

        # Prepare the data module.
        self.dm.prepare(seed=self.seed)

        # Build the model.
        model = self.model.build(in_dim=self.dm.in_dim, out_dim=self.dm.out_dim)

        print("Model architecture:")
        print(model)

        # At the end, we return a value representing how well the model performed on the
        # validation set. That can be the validation loss or validation accuracy, for
        # example. This value is used for hyperparameter optimization.
        # If you don't intend to perform hyperparameter optimization, you don't have to
        # return anything.
        return 0.5

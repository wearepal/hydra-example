"""Main run function and config class."""

from dataclasses import dataclass, field
from typing import Any

import torch

from src.datasets import CelebADataModule, ColoredMNISTDataModule, DataModule
from src.logging import WandbCfg
from src.models import FcnFactory, ModelFactory, SimpleCNNFactory
from src.optimisation import OptimisationCfg

__all__ = ["Config", "CONFIG_GROUPS"]

# Configuration groups enable us to have different configurations for different subcomponents.
# For example, one subcomponent is the data module, and we can have different data modules, CelebA
# and ColoredMNIST, which need different keys and values to be configured.
CONFIG_GROUPS = {
    "dm": {"celeba": CelebADataModule, "cmnist": ColoredMNISTDataModule},
    "model": {"fcn": FcnFactory, "cnn": SimpleCNNFactory},
}


@dataclass(kw_only=True)
class Config:
    """Main configuration class for the code base."""

    # The first two fields refer to configuration groups.
    # This is why we cannot specify a default for them here.
    # The defaults can be specified in the main config yaml file (`conf/config.yaml`).
    dm: DataModule
    model: ModelFactory

    # These are normal subconfigs, for which we can specify defaults,
    # but note that in dataclasses, the default may not be mutable, so we use `default_factory`.
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

        # At the end, we return a value representing how well the model performed on the validation
        # set. That can be the validation loss or validation accuracy, for example. This value is
        # used for hyperparameter optimization.
        # If you don't intend to perform hyperparameter optimization, you don't have to return
        # anything.
        return 0.5

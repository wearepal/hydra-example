from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, TypeAlias, Union

import wandb

__all__ = ["WandbCfg"]

WandbMode = Enum("WandbMode", ["online", "offline", "disabled"])
Run: TypeAlias = Union[  # noqa: UP007
    wandb.sdk.wandb_run.Run,  # type: ignore
    wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
    None,
]


@dataclass
class WandbCfg:
    """Dataclass wrapper around the parameters for `wandb.init()`.

    This allows us to directly configure `wandb.init()` from the config file and the cmd line.
    """

    name: str | None = None
    dir: str = "./local_logging"
    id: str | None = None
    anonymous: bool | None = None
    project: str = "hydra-example"
    group: str | None = None
    entity: str = "predictive-analytics-lab"
    tags: list[str] | None = None
    job_type: str | None = None
    mode: WandbMode = WandbMode.disabled
    resume: str | None = None

    def init(self, cfg: dict[str, Any], *, reinit: bool) -> Run:
        """Call `wandb.init()` with the parameters from the config."""
        kwargs = asdict(self)
        kwargs["mode"] = kwargs["mode"].name  # Convert enum to string.
        return wandb.init(config=cfg, reinit=reinit, **kwargs)

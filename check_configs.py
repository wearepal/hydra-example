"""Check that all YAML config files in `configs/` are compatible with the config classes.

The YAML files are not checked by the type checker, so if, for example, a field in one
of the dataclasses is renamed, a YAML file that still uses the old name would only fail
once someone tries to use it. This script catches that without running any experiments:
it composes every config file with Hydra (which validates keys and value types against
the registered dataclasses) and then instantiates the `Config` object from the result.

Usage:
    python check_configs.py
"""

from pathlib import Path
import sys
import traceback
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from ranzen.hydra import register_hydra_config

from src.run import CONFIG_GROUPS, Config

CONFIG_DIR = Path(__file__).parent / "configs"
PRIMARY_CONFIG = "base"
SCHEMA_NAME = "config_schema"  # Has to match the name used in `main.py`.

_MISSING = object()


def to_object(cfg: DictConfig) -> Any:
    """Instantiate the `Config` object from the given config dict."""

    # Keys that are swept over by the sweeper are just strings in the sweeper config,
    # so we check here that they actually exist in the config.
    if (params := OmegaConf.select(cfg, "hydra.sweeper.params")) is not None:
        for key in params:
            if OmegaConf.select(cfg, key, default=_MISSING) is _MISSING:
                raise KeyError(f"Swept-over key '{key}' does not exist in the config.")

    # Remove the hydra node (as `@hydra.main` does) and instantiate the dataclasses.
    with open_dict(cfg):
        del cfg["hydra"]
    return OmegaConf.to_object(cfg)


def check(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    schema_name: str,
    config_dir: Path,
    primary_config: str,
) -> int:
    register_hydra_config(main_cls, groups, schema_name=schema_name)

    failures: list[str] = []
    with initialize_config_dir(config_dir=str(config_dir.resolve())):
        # Groups that are already in the defaults list are overridden with
        # `group=option`, all others have to be appended with `+group=option`.
        base_cfg = compose(primary_config, return_hydra_config=True)
        groups_in_defaults = set(base_cfg.hydra.runtime.choices)

        for path in sorted(config_dir.rglob("*.y*ml")):
            rel_path = path.relative_to(config_dir)
            config_name, override = construct_override(
                path, rel_path, primary_config, groups_in_defaults
            )

            description = (
                f"{rel_path} ({override if override is not None else config_name})"
            )
            try:
                cfg = compose(
                    config_name,
                    overrides=[override] if override is not None else [],
                    return_hydra_config=True,
                )
                config_obj = to_object(cfg)
                assert isinstance(config_obj, main_cls)
            except Exception:
                failures.append(description)
                print(f"FAIL  {description}")
                traceback.print_exc(file=sys.stdout)
                print()
            else:
                print(f"OK    {description}")

    if failures:
        print(f"\n{len(failures)} config file(s) are invalid:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll config files are valid.")
    return 0


def construct_override(
    path: Path, rel_path: Path, primary_config: str, groups_in_defaults: set[str]
) -> tuple[str, str | None]:
    if len(rel_path.parts) == 1:
        # A config at the top level is a primary config.
        return path.stem, None
    else:
        # Any other config is an option in a config group. Groups can be
        # nested (e.g. `hydra/launcher`) and options can contain slashes
        # (e.g. `experiment=cmnist/cnn`), so we pick the longest known group
        # that is a prefix of the path and fall back to the top-level directory.
        name_parts = rel_path.with_suffix("").parts
        split = next(
            (
                i
                for i in range(len(name_parts) - 1, 0, -1)
                if "/".join(name_parts[:i]) in groups_in_defaults
            ),
            1,
        )
        group = "/".join(name_parts[:split])
        option = "/".join(name_parts[split:])
        prefix = "" if group in groups_in_defaults else "+"
        return primary_config, f"{prefix}{group}={option}"


if __name__ == "__main__":
    sys.exit(
        check(
            main_cls=Config,
            groups=CONFIG_GROUPS,
            schema_name=SCHEMA_NAME,
            config_dir=CONFIG_DIR,
            primary_config=PRIMARY_CONFIG,
        )
    )

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

from hydra_qol import check_configs

from src.run import CONFIG_GROUPS, Config

if __name__ == "__main__":
    sys.exit(
        check_configs(
            main_cls=Config,
            groups=CONFIG_GROUPS,
            # The same `schema_name` as in `register_hydra_config()` in `main.py`.
            schema_name="base_config",
            config_dir=Path(__file__).parent / "configs",
            # A primary config that selects a default for every config group; here,
            # the same as `config_name` in `@hydra.main()`.
            config_with_defaults="default_config",
        )
    )

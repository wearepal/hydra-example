"""Check that all YAML config files in `configs/` are compatible with the config classes.

The YAML files are not checked by the type checker, so if, for example, a field in one
of the dataclasses is renamed, a YAML file that still uses the old name would only fail
once someone tries to use it. This test catches that without running any experiments:
it composes every config file with Hydra (which validates keys and value types against
the registered dataclasses) and then instantiates the `Config` object from the result.

Usage:
    pytest
"""

from collections.abc import Iterator
from pathlib import Path

from hydra_qol import ValidationContext, setup_validation, validate_config
import pytest

from src.run import CONFIG_GROUPS, Config

CONFIG_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def context() -> Iterator[ValidationContext]:
    # Hydra is initialized only once and then used for all config files.
    with setup_validation(
        main_cls=Config,
        groups=CONFIG_GROUPS,
        # The same `schema_name` as in `register_hydra_config()` in `main.py`.
        schema_name="base_config",
        config_dir=CONFIG_DIR,
        # A primary config that selects a default for every config group; here,
        # the same as `config_name` in `@hydra.main()`.
        config_with_defaults="default_config",
        # Configs in these subdirectories are primary configs (selected with
        # `--config-name experiments/<name>`), not options of a config group.
        primary_config_dirs=["experiments"],
    ) as context:
        yield context


# One test case per config file, so that every broken file is reported separately.
@pytest.mark.parametrize(
    "path",
    sorted(path.relative_to(CONFIG_DIR) for path in CONFIG_DIR.rglob("*.y*ml")),
    ids=Path.as_posix,
)
def test_config(path: Path, context: ValidationContext) -> None:
    result = validate_config(path, context)
    if result.error is not None:
        raise result.error

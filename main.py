import hydra
from hydra.utils import instantiate
import omegaconf
from ranzen.hydra import prepare_for_logging, reconstruct_cmd, register_hydra_config

from src.run import CONFIG_GROUPS, Config

# This is the main entry point for the script.
# Meaning of the parameters to @hydra.main():
#     config_path: The path to the directory containing the yaml config files.
#     config_name: The name of the config file without the ".yaml" extension.
#     version_base: The version of hydra semantics to use.
#                   If None, the latest version is used.


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(hydra_config: omegaconf.DictConfig) -> None:
    # The `hydra_config` object we get is essentially a dictionary.
    # We can convert it to an object of the `Config` class using the `instantiate` function.
    # That is what is meant by the `_convert_="object"` argument.
    # The `_recursive_=True` argument tells Hydra to recursively instantiate nested objects.

    config = instantiate(hydra_config, _convert_="object", _recursive_=True)
    assert isinstance(config, Config)

    # `prepare_for_logging` takes a hydra config dict and makes it prettier for logging.
    # Things this function does: turn enums to strings, resolve any references, etc.
    config_for_logging = prepare_for_logging(hydra_config)
    # We add the command that was used to start the program to the config.
    config_for_logging["cmd"] = reconstruct_cmd()

    # Finally, we call the `run` method of the `Config` object.
    config.run(config_for_logging)


if __name__ == "__main__":
    # Before calling the main function, we need to register the main `Config` class and
    # the configuration groups.
    # Without this, hydra doesn't know which keys and values are valid in the configuration.
    register_hydra_config(Config, CONFIG_GROUPS, schema_name="config_schema")
    main()

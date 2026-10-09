import hydra
import omegaconf
from ranzen.hydra import prepare_for_logging, reconstruct_cmd, register_hydra_config

from src.run import CONFIG_GROUPS, Config

# This is the main entry point for the script.
# Meaning of the parameters to @hydra.main():
#     config_path: The path to the directory containing the yaml config files.
#     config_name: The name of the primary config. This can be a yaml file in
#         `config_path` (without the ".yaml" extension), but here it's the `Config`
#         class, which we register under the name "main_config" below.


@hydra.main(config_path="configs", config_name="main_config")
def main(hydra_config: omegaconf.DictConfig) -> float:
    # The `hydra_config` object we get is essentially a dictionary.
    # We convert it to an object of the `Config` class using `OmegaConf.to_object()`.
    config = omegaconf.OmegaConf.to_object(hydra_config)
    assert isinstance(config, Config)

    # `prepare_for_logging` takes a hydra config dict and makes it prettier for logging.
    # Things this function does: turn enums to strings, resolve any references, etc.
    config_for_logging = prepare_for_logging(hydra_config)
    # We add the command that was used to start the program to the config.
    config_for_logging["cmd"] = reconstruct_cmd()

    # Finally, we call the `run` method of the `Config` object.
    return config.run(config_for_logging)


if __name__ == "__main__":
    # Before calling the main function, we need to register the main `Config` class and
    # the configuration groups. Without this, hydra doesn't know which keys and values
    # are valid in the configuration.
    # Whatever you set here as `schema_name` has to match the `config_name` passed to
    # `@hydra.main()` above.
    register_hydra_config(Config, CONFIG_GROUPS, schema_name="main_config")
    main()
